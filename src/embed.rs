//! Embedded kernel extraction and runtime loading via dlopen.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

pub struct KernelTable {
    pub i2_dot_i8: unsafe extern "C" fn(*const u8, *const i8, i32) -> i32,
    pub i2_dot_i8_4row: unsafe extern "C" fn(
        *const u8, *const u8, *const u8, *const u8,
        *const i8, *mut i32, i32,
    ),
    pub i2_dot_i8_4row_dual: unsafe extern "C" fn(
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *const i8, *mut i32, *mut i32, i32,
    ),
    pub quant_f32_i8: unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32),
    pub rmsnorm_f32: unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, f32),
    pub fused_attention_f32: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *mut f32, i32, i32, f32,
    ),
    pub i8dot_1row: unsafe extern "C" fn(*const i8, *const u8, i32) -> i32,
    pub i8dot_4row: unsafe extern "C" fn(
        *const i8, *const u8, *const u8, *const u8, *const u8, *mut i32, i32,
    ),
    pub squared_relu_mul_f32: unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32),
    pub vecadd_f32: unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32),
    pub quant_f32_q8k: unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32),
    pub q4k_dot_q8k: unsafe extern "C" fn(
        *const u8, *const i8, *const i32, *const u8, *const u8,
        i32, f32, f32,
    ) -> f32,
    pub q4k_dot_q8k_4row: unsafe extern "C" fn(
        *const u8, *const u8, *const u8, *const u8,
        *const i8, *const i32,
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *mut f32, i32,
        f32, f32, f32, f32,
        f32, f32, f32, f32,
    ),
    pub q4k_dot_q8k_4row_dual: unsafe extern "C" fn(
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *const i8, *const i32,
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *mut f32, *mut f32, i32,
        f32, f32, f32, f32, f32, f32, f32, f32,
        f32, f32, f32, f32, f32, f32, f32, f32,
    ),
    pub q6k_dot_q8k: unsafe extern "C" fn(
        *const u8, *const u8, *const i8, *const i8, *const i32,
        i32, f32,
    ) -> f32,
    pub q6k_dot_q8k_4row: unsafe extern "C" fn(
        *const u8, *const u8, *const u8, *const u8,
        *const u8, *const u8, *const u8, *const u8,
        *const i8, *const i8, *const i8, *const i8,
        *const i8, *const i32,
        *mut f32, i32,
        f32, f32, f32, f32,
    ),
    pub apply_rope_f32: unsafe extern "C" fn(
        *const f32, *const f32, *mut f32, i32, i32,
    ),
}

// Safety: function pointers are Send+Sync (they point to loaded .so code).
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

static KERNELS: OnceLock<KernelTable> = OnceLock::new();

/// Returns the loaded kernel table, initializing on first call.
pub fn k() -> &'static KernelTable {
    KERNELS.get_or_init(|| {
        let dir = extract().expect("kernel extraction failed");
        load(&dir).expect("kernel load failed")
    })
}

/// Extract embedded .so files to ~/.cougar/lib/{VERSION}/ if not already done.
pub fn extract() -> Result<PathBuf, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let dir = PathBuf::from(home)
        .join(".cougar")
        .join("lib")
        .join(embedded::VERSION);

    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("mkdir {}: {e}", dir.display()))?;

    for (filename, bytes) in embedded::FILES {
        let path = dir.join(filename);
        std::fs::write(&path, bytes)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }

    std::fs::write(&marker, b"ok")
        .map_err(|e| format!("write marker: {e}"))?;

    eprintln!("cougar> extracted kernels to {}", dir.display());
    Ok(dir)
}

/// Load all kernel symbols via dlopen/dlsym.
fn load(dir: &PathBuf) -> Result<KernelTable, String> {
    unsafe {
        let i2s = open_lib(dir, "libbitnet_i2s.so")?;
        let quant = open_lib(dir, "libbitnet_quant.so")?;
        let rms = open_lib(dir, "libbitnet_rmsnorm.so")?;
        let attn = open_lib(dir, "libbitnet_fused_attn.so")?;
        let i8d = open_lib(dir, "libbitnet_i8dot.so")?;
        let act = open_lib(dir, "libbitnet_activate.so")?;
        let vadd = open_lib(dir, "libbitnet_vecadd.so")?;
        let q4kq = open_lib(dir, "libq4k_quant.so")?;
        let q4kd = open_lib(dir, "libq4k_dot.so")?;
        let q6kd = open_lib(dir, "libq6k_dot.so")?;
        let rope = open_lib(dir, "librope.so")?;

        Ok(KernelTable {
            i2_dot_i8: sym(i2s, "i2_dot_i8\0")?,
            i2_dot_i8_4row: sym(i2s, "i2_dot_i8_4row\0")?,
            i2_dot_i8_4row_dual: sym(i2s, "i2_dot_i8_4row_dual\0")?,
            quant_f32_i8: sym(quant, "quant_f32_i8\0")?,
            rmsnorm_f32: sym(rms, "rmsnorm_f32\0")?,
            fused_attention_f32: sym(attn, "fused_attention_f32\0")?,
            i8dot_1row: sym(i8d, "i8dot_1row\0")?,
            i8dot_4row: sym(i8d, "i8dot_4row\0")?,
            squared_relu_mul_f32: sym(act, "squared_relu_mul_f32\0")?,
            vecadd_f32: sym(vadd, "vecadd_f32\0")?,
            quant_f32_q8k: sym(q4kq, "quant_f32_q8k\0")?,
            q4k_dot_q8k: sym(q4kd, "q4k_dot_q8k\0")?,
            q4k_dot_q8k_4row: sym(q4kd, "q4k_dot_q8k_4row\0")?,
            q4k_dot_q8k_4row_dual: sym(q4kd, "q4k_dot_q8k_4row_dual\0")?,
            q6k_dot_q8k: sym(q6kd, "q6k_dot_q8k\0")?,
            q6k_dot_q8k_4row: sym(q6kd, "q6k_dot_q8k_4row\0")?,
            apply_rope_f32: sym(rope, "apply_rope_f32\0")?,
        })
    }
}

unsafe fn open_lib(dir: &PathBuf, name: &str) -> Result<*mut c_void, String> {
    let path = dir.join(name);
    let path_str = format!("{}\0", path.display());
    let handle = dlopen(path_str.as_ptr(), RTLD_NOW);
    if handle.is_null() {
        return Err(format!("dlopen failed: {}", path.display()));
    }
    Ok(handle)
}

unsafe fn sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let ptr = dlsym(handle, name.as_ptr());
    if ptr.is_null() {
        return Err(format!("dlsym failed: {name}"));
    }
    Ok(std::mem::transmute_copy(&ptr))
}

/// Initialize kernels (extract + load). Safe to call multiple times.
pub fn init() -> Result<(), String> {
    // Force initialization via k()
    let _ = k();
    Ok(())
}
