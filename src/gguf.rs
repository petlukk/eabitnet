use std::collections::HashMap;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" as little-endian u32
const ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_map: HashMap<String, usize>,
    pub data_offset: u64,
    raw: Vec<u8>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("n_tensors", &self.tensors.len())
            .field("n_metadata", &self.metadata.len())
            .field("data_offset", &self.data_offset)
            .field("raw_len", &self.raw.len())
            .finish()
    }
}

fn read_u8(buf: &[u8], pos: usize) -> Result<(u8, usize), String> {
    if pos >= buf.len() {
        return Err("unexpected EOF reading u8".into());
    }
    Ok((buf[pos], pos + 1))
}

fn read_i8(buf: &[u8], pos: usize) -> Result<(i8, usize), String> {
    let (v, p) = read_u8(buf, pos)?;
    Ok((v as i8, p))
}

fn read_u16(buf: &[u8], pos: usize) -> Result<(u16, usize), String> {
    if pos + 2 > buf.len() {
        return Err("unexpected EOF reading u16".into());
    }
    Ok((u16::from_le_bytes([buf[pos], buf[pos + 1]]), pos + 2))
}

fn read_i16(buf: &[u8], pos: usize) -> Result<(i16, usize), String> {
    let (v, p) = read_u16(buf, pos)?;
    Ok((v as i16, p))
}

fn read_u32(buf: &[u8], pos: usize) -> Result<(u32, usize), String> {
    if pos + 4 > buf.len() {
        return Err("unexpected EOF reading u32".into());
    }
    let val = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
    Ok((val, pos + 4))
}

fn read_i32(buf: &[u8], pos: usize) -> Result<(i32, usize), String> {
    let (v, p) = read_u32(buf, pos)?;
    Ok((v as i32, p))
}

fn read_u64(buf: &[u8], pos: usize) -> Result<(u64, usize), String> {
    if pos + 8 > buf.len() {
        return Err("unexpected EOF reading u64".into());
    }
    let val = u64::from_le_bytes([
        buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3],
        buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7],
    ]);
    Ok((val, pos + 8))
}

fn read_i64(buf: &[u8], pos: usize) -> Result<(i64, usize), String> {
    let (v, p) = read_u64(buf, pos)?;
    Ok((v as i64, p))
}

fn read_f32(buf: &[u8], pos: usize) -> Result<(f32, usize), String> {
    let (bits, p) = read_u32(buf, pos)?;
    Ok((f32::from_bits(bits), p))
}

fn read_f64(buf: &[u8], pos: usize) -> Result<(f64, usize), String> {
    let (bits, p) = read_u64(buf, pos)?;
    Ok((f64::from_bits(bits), p))
}

fn read_string(buf: &[u8], pos: usize) -> Result<(String, usize), String> {
    let (len, mut p) = read_u64(buf, pos)?;
    let len = len as usize;
    if p + len > buf.len() {
        return Err("unexpected EOF reading string".into());
    }
    let s = String::from_utf8_lossy(&buf[p..p + len]).into_owned();
    p += len;
    Ok((s, p))
}

fn read_meta_value(buf: &[u8], pos: usize, vtype: u32) -> Result<(MetaValue, usize), String> {
    match vtype {
        0 => { let (v, p) = read_u8(buf, pos)?; Ok((MetaValue::U8(v), p)) }
        1 => { let (v, p) = read_i8(buf, pos)?; Ok((MetaValue::I8(v), p)) }
        2 => { let (v, p) = read_u16(buf, pos)?; Ok((MetaValue::U16(v), p)) }
        3 => { let (v, p) = read_i16(buf, pos)?; Ok((MetaValue::I16(v), p)) }
        4 => { let (v, p) = read_u32(buf, pos)?; Ok((MetaValue::U32(v), p)) }
        5 => { let (v, p) = read_i32(buf, pos)?; Ok((MetaValue::I32(v), p)) }
        6 => { let (v, p) = read_f32(buf, pos)?; Ok((MetaValue::F32(v), p)) }
        7 => {
            let (v, p) = read_u8(buf, pos)?;
            Ok((MetaValue::Bool(v != 0), p))
        }
        8 => {
            let (s, p) = read_string(buf, pos)?;
            Ok((MetaValue::Str(s), p))
        }
        9 => {
            let (elem_type, p) = read_u32(buf, pos)?;
            let (count, mut p) = read_u64(buf, p)?;
            let mut arr = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let (v, np) = read_meta_value(buf, p, elem_type)?;
                arr.push(v);
                p = np;
            }
            Ok((MetaValue::Array(arr), p))
        }
        10 => { let (v, p) = read_u64(buf, pos)?; Ok((MetaValue::U64(v), p)) }
        11 => { let (v, p) = read_i64(buf, pos)?; Ok((MetaValue::I64(v), p)) }
        12 => { let (v, p) = read_f64(buf, pos)?; Ok((MetaValue::F64(v), p)) }
        _ => Err(format!("unknown metadata value type {vtype}")),
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Byte size per element for GGUF tensor types.
/// Returns (bits_per_element, block_size) for quantized types.
fn gguf_type_size(dtype: u32) -> Result<(usize, usize), String> {
    match dtype {
        0 => Ok((32, 1)),   // F32
        1 => Ok((16, 1)),   // F16
        2 => Ok((5, 32)),   // Q4_0: 2+16 bytes per 32 elements = 18 bytes/32 = 4.5 bits
        3 => Ok((5, 32)),   // Q4_1: 2+2+16 bytes per 32 = 20 bytes/32
        6 => Ok((5, 32)),   // Q5_0
        7 => Ok((6, 32)),   // Q5_1
        8 => Ok((9, 32)),   // Q8_0: 2+32 bytes per 32 = 34 bytes/32
        9 => Ok((9, 32)),   // Q8_1
        10 => Ok((3, 256)), // Q2_K
        11 => Ok((4, 256)), // Q3_K
        12 => Ok((5, 256)), // Q4_K
        13 => Ok((6, 256)), // Q5_K
        14 => Ok((7, 256)), // Q6_K
        15 => Ok((9, 256)), // Q8_K
        16 => Ok((2, 32)),  // IQ2_XXS
        17 => Ok((2, 32)),  // IQ2_XS
        18 => Ok((3, 32)),  // IQ3_XXS
        19 => Ok((2, 256)), // IQ1_S
        20 => Ok((4, 32)),  // IQ4_NL
        21 => Ok((3, 32)),  // IQ3_S
        22 => Ok((2, 32)),  // IQ2_S
        23 => Ok((4, 256)), // IQ4_XS
        24 => Ok((8, 1)),   // I8
        25 => Ok((16, 1)),  // I16
        26 => Ok((32, 1)),  // I32
        27 => Ok((64, 1)),  // I64
        28 => Ok((64, 1)),  // F64
        29 => Ok((2, 256)), // IQ1_M
        30 => Ok((16, 1)),  // BF16
        31 => Ok((2, 32)),  // TQ1_0: ternary
        32 => Ok((2, 32)),  // TQ2_0: ternary
        36 => Ok((2, 1)),   // I2_S: 2 bits per element
        _ => Err(format!("unknown tensor dtype {dtype}")),
    }
}

/// Compute raw byte size for a tensor given dims and dtype.
fn tensor_byte_size(dims: &[u64], dtype: u32) -> Result<usize, String> {
    if dims.is_empty() {
        return Ok(0);
    }
    let n_elements: u64 = dims.iter().product();
    if dtype == 36 {
        // I2_S: 2 bits per element + 32 bytes trailing per-tensor scale
        return Ok((n_elements as usize) / 4 + 32);
    }
    let (bits, block) = gguf_type_size(dtype)?;
    let n_blocks = (n_elements as usize + block - 1) / block;
    Ok(n_blocks * bits * block / 8)
}

impl GgufFile {
    pub fn open(path: &str) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        if raw.len() < 24 {
            return Err("file too small for GGUF header".into());
        }
        let (magic, pos) = read_u32(&raw, 0)?;
        if magic != GGUF_MAGIC {
            return Err(format!("bad magic 0x{magic:08X}, expected GGUF"));
        }
        let (version, pos) = read_u32(&raw, pos)?;
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let (n_tensors, pos) = read_u64(&raw, pos)?;
        let (n_kv, mut pos) = read_u64(&raw, pos)?;

        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let (key, p) = read_string(&raw, pos)?;
            let (vtype, p) = read_u32(&raw, p)?;
            let (val, p) = read_meta_value(&raw, p, vtype)?;
            metadata.insert(key, val);
            pos = p;
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_map = HashMap::new();
        for i in 0..n_tensors as usize {
            let (name, p) = read_string(&raw, pos)?;
            let (n_dims, p) = read_u32(&raw, p)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            let mut p = p;
            for _ in 0..n_dims {
                let (d, np) = read_u64(&raw, p)?;
                dims.push(d);
                p = np;
            }
            let (dtype, p) = read_u32(&raw, p)?;
            let (offset, p) = read_u64(&raw, p)?;
            tensor_map.insert(name.clone(), i);
            tensors.push(TensorInfo { name, dims, dtype, offset });
            pos = p;
        }

        let data_offset = align_up(pos as u64, ALIGNMENT);

        Ok(GgufFile { version, metadata, tensors, tensor_map, data_offset, raw })
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let idx = *self.tensor_map.get(name)?;
        let ti = &self.tensors[idx];
        let start = self.data_offset as usize + ti.offset as usize;
        let size = tensor_byte_size(&ti.dims, ti.dtype).ok()?;
        if start + size > self.raw.len() {
            return None;
        }
        Some(&self.raw[start..start + size])
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            MetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            MetaValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.metadata.get(key)? {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::Str(s) => out.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key)? {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::F32(f) => out.push(*f),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_missing_file() {
        let result = GgufFile::open("/tmp/nonexistent_gguf_test.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn test_open_invalid_magic() {
        std::fs::write("/tmp/bad_magic.gguf", &[0x42, 0x41, 0x41, 0x44, 0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]).unwrap();
        let result = GgufFile::open("/tmp/bad_magic.gguf");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));
        std::fs::remove_file("/tmp/bad_magic.gguf").ok();
    }

    #[test]
    fn test_open_too_small() {
        std::fs::write("/tmp/tiny.gguf", &[0u8; 10]).unwrap();
        let result = GgufFile::open("/tmp/tiny.gguf");
        assert!(result.is_err());
        std::fs::remove_file("/tmp/tiny.gguf").ok();
    }

    #[test]
    fn test_roundtrip_minimal_gguf() {
        // Build a minimal valid GGUF v3 file: 0 tensors, 1 metadata entry
        let mut buf: Vec<u8> = Vec::new();
        // magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // n_kv = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // key: "test.key"
        let key = b"test.key";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // value type = STRING (8)
        buf.extend_from_slice(&8u32.to_le_bytes());
        // value: "hello"
        let val = b"hello";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);

        let path = "/tmp/minimal_test.gguf";
        std::fs::write(path, &buf).unwrap();
        let gf = GgufFile::open(path).unwrap();
        assert_eq!(gf.version, 3);
        assert_eq!(gf.tensors.len(), 0);
        assert_eq!(gf.get_str("test.key"), Some("hello"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_roundtrip_with_tensor() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // n_kv = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // tensor info: name "w"
        let name = b"w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        // n_dims = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        // dims[0] = 4
        buf.extend_from_slice(&4u64.to_le_bytes());
        // dtype = F32 (0)
        buf.extend_from_slice(&0u32.to_le_bytes());
        // offset = 0 (relative to data section)
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Pad to 32-byte alignment for data section
        let header_end = buf.len() as u64;
        let data_start = align_up(header_end, ALIGNMENT) as usize;
        buf.resize(data_start, 0);

        // Write 4 f32 values: [1.0, 2.0, 3.0, 4.0]
        for &v in &[1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let path = "/tmp/tensor_test.gguf";
        std::fs::write(path, &buf).unwrap();
        let gf = GgufFile::open(path).unwrap();
        assert_eq!(gf.tensors.len(), 1);
        assert_eq!(gf.tensors[0].name, "w");
        assert_eq!(gf.tensors[0].dims, vec![4]);

        let data = gf.tensor_data("w").unwrap();
        assert_eq!(data.len(), 16); // 4 * f32
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);

        assert!(gf.tensor_data("nonexistent").is_none());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_metadata_types() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_kv = 3

        // U32 entry
        let k = b"num_heads";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type U32
        buf.extend_from_slice(&32u32.to_le_bytes());

        // F32 entry
        let k = b"rope_freq";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k);
        buf.extend_from_slice(&6u32.to_le_bytes()); // type F32
        buf.extend_from_slice(&10000.0f32.to_le_bytes());

        // Array of strings
        let k = b"vocab";
        buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
        buf.extend_from_slice(k);
        buf.extend_from_slice(&9u32.to_le_bytes()); // type ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // elem type STRING
        buf.extend_from_slice(&2u64.to_le_bytes()); // count = 2
        let t = b"hello";
        buf.extend_from_slice(&(t.len() as u64).to_le_bytes());
        buf.extend_from_slice(t);
        let t = b"world";
        buf.extend_from_slice(&(t.len() as u64).to_le_bytes());
        buf.extend_from_slice(t);

        let path = "/tmp/meta_types_test.gguf";
        std::fs::write(path, &buf).unwrap();
        let gf = GgufFile::open(path).unwrap();

        assert_eq!(gf.get_u32("num_heads"), Some(32));
        assert_eq!(gf.get_f32("rope_freq"), Some(10000.0));
        let vocab = gf.get_str_array("vocab").unwrap();
        assert_eq!(vocab, vec!["hello", "world"]);

        assert!(gf.get_str("num_heads").is_none());
        assert!(gf.get_u32("nonexistent").is_none());
        std::fs::remove_file(path).ok();
    }
}
