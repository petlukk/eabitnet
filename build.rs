use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let lib_dir = if target.starts_with("aarch64") {
        Path::new("build/lib-arm")
    } else {
        Path::new("build/lib")
    };
    println!("cargo:rerun-if-changed=build/lib");
    println!("cargo:rerun-if-changed=build/lib-arm");
    println!("cargo:rustc-link-lib=dl");

    let kernels = [
        ("bitnet_activate", "libbitnet_activate.so"),
        ("bitnet_fused_attn", "libbitnet_fused_attn.so"),
        ("bitnet_i2s", "libbitnet_i2s.so"),
        ("bitnet_i8dot", "libbitnet_i8dot.so"),
        ("bitnet_quant", "libbitnet_quant.so"),
        ("bitnet_rmsnorm", "libbitnet_rmsnorm.so"),
        ("bitnet_vecadd", "libbitnet_vecadd.so"),
        ("q4k_quant", "libq4k_quant.so"),
        ("q4k_dot", "libq4k_dot.so"),
        ("q6k_dot", "libq6k_dot.so"),
    ];

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("embedded_kernels.rs");

    let mut hasher = DefaultHasher::new();
    let mut code = String::new();

    for (name, file) in &kernels {
        let abs = fs::canonicalize(lib_dir.join(file))
            .unwrap_or_else(|e| panic!("cannot find {file}: {e}"));
        let bytes = fs::read(&abs).unwrap_or_else(|e| panic!("cannot read {file}: {e}"));
        bytes.hash(&mut hasher);

        code.push_str(&format!(
            "pub const {}: &[u8] = include_bytes!(\"{}\");\n",
            name.to_uppercase(),
            abs.display(),
        ));
    }

    let hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    code.push_str(&format!(
        "\npub const VERSION: &str = \"v{version}-{hash}\";\n"
    ));

    // Filenames array for extraction
    code.push_str("\npub const FILES: &[(&str, &[u8])] = &[\n");
    for (name, file) in &kernels {
        code.push_str(&format!("    (\"{file}\", {}),\n", name.to_uppercase()));
    }
    code.push_str("];\n");

    fs::write(&out_path, code).unwrap();
}
