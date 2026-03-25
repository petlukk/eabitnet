//! BitNet b1.58 2B-4T model structure and weight loading from GGUF.

use crate::gguf::GgufFile;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType { I2S, Q4K }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation { SquaredReLU, SiLU }

pub struct BitNetModel {
    pub quant_type: QuantType,
    pub activation: Activation,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,

    pub layers: Vec<LayerWeights>,
    pub embed_weight_f16: *const u8,
    pub embed_weight_i8: Vec<u8>,       // per-row quantized (u8 = i8 + 128 bias)
    pub embed_row_scales: Vec<f32>,     // per-row absmax scale
    #[cfg(target_arch = "aarch64")]
    pub embed_sketch: Vec<u8>,          // subsampled (every 4th byte) for speculative top-k
    #[cfg(target_arch = "aarch64")]
    pub embed_sketch_dim: usize,        // hidden_dim / SKETCH_STRIDE
    pub norm_weight: *const f32,
    /// Owned repacked I2_S weight data (keeps pointers valid)
    pub(crate) _weight_data: Vec<Vec<u8>>,

    // Q4K fields (empty for I2S models)
    pub q4k_layers: Vec<Q4KLayerWeights>,
    pub output_weight: *const u8,
    /// Embed tensor dtype (12 = Q4_K, 14 = Q6_K, 1 = F16, etc.)
    pub embed_dtype: u32,
    pub output_block_bytes: usize,
}

pub struct Q4KLayerWeights {
    pub attn_norm: *const f32,
    pub wq: *const u8,
    pub wk: *const u8,
    pub wv: *const u8,
    pub wv_block_bytes: usize,
    pub wo: *const u8,
    pub ffn_norm: *const f32,
    pub w_gate: *const u8,
    pub w_up: *const u8,
    pub w_down: *const u8,
    pub w_down_block_bytes: usize,
}

pub struct LayerWeights {
    pub attn_norm: *const f32,
    pub attn_sub_norm: *const f32,
    pub wq: *const u8,
    pub wk: *const u8,
    pub wv: *const u8,
    pub wo: *const u8,
    pub ffn_norm: *const f32,
    pub ffn_sub_norm: *const f32,
    pub w_gate: *const u8,
    pub w_up: *const u8,
    pub w_down: *const u8,
    pub wq_scale: f32,
    pub wk_scale: f32,
    pub wv_scale: f32,
    pub wo_scale: f32,
    pub w_gate_scale: f32,
    pub w_up_scale: f32,
    pub w_down_scale: f32,
}

// Safe because GgufFile owns the backing data and must outlive BitNetModel.
unsafe impl Send for BitNetModel {}
unsafe impl Sync for BitNetModel {}

fn load_q4k_tensor(gguf: &GgufFile, name: &str) -> Result<(*const u8, usize), String> {
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    let n_blocks = data.len() / 144;
    Ok((data.as_ptr(), n_blocks))
}

/// Load a tensor that may be Q4_K (dtype 12) or Q6_K (dtype 14).
/// Returns (pointer, n_blocks, block_bytes).
fn load_qk_tensor(gguf: &GgufFile, name: &str) -> Result<(*const u8, usize, usize), String> {
    let idx = *gguf.tensor_map.get(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    let dtype = gguf.tensors[idx].dtype;
    let data = gguf.tensor_data(name)
        .ok_or_else(|| format!("missing tensor data: {name}"))?;
    let block_bytes = match dtype {
        12 => 144,  // Q4_K
        14 => 210,  // Q6_K
        _ => return Err(format!("{name}: unsupported dtype {dtype} (expected Q4_K or Q6_K)")),
    };
    let n_blocks = data.len() / block_bytes;
    Ok((data.as_ptr(), n_blocks, block_bytes))
}

fn tensor_ptr<T>(gguf: &GgufFile, name: &str) -> Result<*const T, String> {
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    Ok(data.as_ptr() as *const T)
}

fn get_meta_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    use crate::gguf::MetaValue;
    match gguf.metadata.get(key)? {
        MetaValue::U32(v) => Some(*v),
        MetaValue::I32(v) => Some(*v as u32),
        MetaValue::U64(v) => Some(*v as u32),
        MetaValue::I64(v) => Some(*v as u32),
        MetaValue::U16(v) => Some(*v as u32),
        MetaValue::U8(v) => Some(*v as u32),
        _ => None,
    }
}

fn get_meta_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
    use crate::gguf::MetaValue;
    match gguf.metadata.get(key)? {
        MetaValue::F32(v) => Some(*v),
        MetaValue::F64(v) => Some(*v as f32),
        _ => None,
    }
}

fn load_i2s_tensor(
    gguf: &GgufFile,
    name: &str,
    bufs: &mut Vec<Vec<u8>>,
) -> Result<(*const u8, f32), String> {
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    if data.len() < 32 {
        return Err(format!("{name}: tensor too small for I2_S scale"));
    }
    let scale_off = data.len() - 32;
    let sb = &data[scale_off..scale_off + 4];
    let scale = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);

    // GGUF I2_S encoding matches our kernel: 0=-1, 1=0, 2=+1. No remap needed.
    let weight_bytes = &data[..scale_off];
    let mut repacked = Vec::with_capacity(weight_bytes.len());
    repacked.extend_from_slice(weight_bytes);
    let ptr = repacked.as_ptr();
    bufs.push(repacked);
    Ok((ptr, scale))
}

impl BitNetModel {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.get_str("general.architecture").unwrap_or("llama");
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let n_layers = get_meta_u32(gguf, &key("block_count"))
            .ok_or("missing metadata: block_count")? as usize;

        let hidden_dim = get_meta_u32(gguf, &key("embedding_length"))
            .ok_or("missing metadata: embedding_length")? as usize;

        let n_heads = get_meta_u32(gguf, &key("attention.head_count"))
            .ok_or("missing metadata: head_count")? as usize;

        let n_kv_heads = get_meta_u32(gguf, &key("attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;

        let head_dim = hidden_dim / n_heads;
        let kv_dim = n_kv_heads * head_dim;

        let ffn_dim = get_meta_u32(gguf, &key("feed_forward_length"))
            .ok_or("missing metadata: feed_forward_length")? as usize;

        let rope_theta = get_meta_f32(gguf, &key("rope.freq_base"))
            .unwrap_or(10000.0);

        let rms_eps = get_meta_f32(gguf, &key("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        // Detect quant type from first weight tensor
        let quant_type = {
            let first_weight = gguf.tensor_map.get("blk.0.attn_q.weight")
                .or_else(|| gguf.tensor_map.get("blk.0.attn_qkv.weight"));
            match first_weight {
                Some(&idx) => match gguf.tensors[idx].dtype {
                    36 => QuantType::I2S,
                    12 => QuantType::Q4K,
                    dt => return Err(format!("unsupported weight quant type: {dt}")),
                },
                None => return Err("cannot find weight tensor to detect quant type".into()),
            }
        };

        // Detect activation type
        let activation = match gguf.get_str(&key("activation_function")) {
            Some("squared_relu") => Activation::SquaredReLU,
            _ => if quant_type == QuantType::I2S { Activation::SquaredReLU } else { Activation::SiLU },
        };

        // Get vocab_size from embedding tensor dimensions
        let embed_idx = *gguf
            .tensor_map
            .get("token_embd.weight")
            .ok_or("missing tensor: token_embd.weight")?;
        let embed_dims = &gguf.tensors[embed_idx].dims;
        let vocab_size = if embed_dims.len() == 2 {
            embed_dims[1] as usize
        } else {
            return Err(format!(
                "token_embd.weight: expected 2 dims, got {}",
                embed_dims.len()
            ));
        };

        if quant_type == QuantType::Q4K {
            let mut q4k_layers = Vec::with_capacity(n_layers);
            for layer in 0..n_layers {
                let prefix = format!("blk.{layer}");
                let attn_norm = tensor_ptr::<f32>(gguf, &format!("{prefix}.attn_norm.weight"))?;
                let (wq, _) = load_q4k_tensor(gguf, &format!("{prefix}.attn_q.weight"))?;
                let (wk, _) = load_q4k_tensor(gguf, &format!("{prefix}.attn_k.weight"))?;
                let (wv, _, wv_bb) = load_qk_tensor(gguf, &format!("{prefix}.attn_v.weight"))?;
                let (wo, _) = load_q4k_tensor(gguf, &format!("{prefix}.attn_output.weight"))?;
                let ffn_norm = tensor_ptr::<f32>(gguf, &format!("{prefix}.ffn_norm.weight"))?;
                let (w_gate, _) = load_q4k_tensor(gguf, &format!("{prefix}.ffn_gate.weight"))?;
                let (w_up, _) = load_q4k_tensor(gguf, &format!("{prefix}.ffn_up.weight"))?;
                let (w_down, _, w_down_bb) = load_qk_tensor(gguf, &format!("{prefix}.ffn_down.weight"))?;
                q4k_layers.push(Q4KLayerWeights {
                    attn_norm, wq, wk,
                    wv, wv_block_bytes: wv_bb,
                    wo,
                    ffn_norm, w_gate, w_up,
                    w_down, w_down_block_bytes: w_down_bb,
                });
            }

            // output.weight may be a separate tensor or tied to token_embd.weight
            let (output_weight, _, output_bb) = if gguf.tensor_map.contains_key("output.weight") {
                load_qk_tensor(gguf, "output.weight")?
            } else {
                // Tied embeddings: reuse token_embd.weight for output projection
                load_qk_tensor(gguf, "token_embd.weight")?
            };

            let embed_ptr = gguf.tensor_data("token_embd.weight")
                .ok_or("missing tensor: token_embd.weight")?;
            let embed_weight_ptr = embed_ptr.as_ptr();

            let embed_idx = gguf.tensor_map["token_embd.weight"];
            let embed_dtype = gguf.tensors[embed_idx].dtype;

            eprintln!("  Q4K_M: {} layers, embed dtype={}, output={} bytes/blk",
                n_layers, embed_dtype, output_bb);

            return Ok(BitNetModel {
                quant_type, activation,
                n_layers, hidden_dim, n_heads, n_kv_heads, head_dim, kv_dim, ffn_dim,
                vocab_size, rope_theta, rms_eps,
                layers: Vec::new(),
                embed_weight_f16: embed_weight_ptr,
                embed_weight_i8: Vec::new(),
                embed_row_scales: Vec::new(),
                #[cfg(target_arch = "aarch64")]
                embed_sketch: Vec::new(),
                #[cfg(target_arch = "aarch64")]
                embed_sketch_dim: 0,
                norm_weight: tensor_ptr::<f32>(gguf, "output_norm.weight")?,
                _weight_data: Vec::new(),
                q4k_layers,
                output_weight,
                embed_dtype,
                output_block_bytes: output_bb,
            });
        }

        // --- I2_S weight loading path ---

        // Tied embeddings: token_embd.weight is F16, used for both embed and output
        let embed_weight_f16: *const u8 = tensor_ptr(gguf, "token_embd.weight")?;
        let norm_weight: *const f32 = tensor_ptr(gguf, "output_norm.weight")?;

        // Quantize embedding rows to i8 (per-row absmax) + u8 bias for output projection
        let embed_data = gguf.tensor_data("token_embd.weight")
            .ok_or("missing tensor: token_embd.weight")?;
        let n_f16 = vocab_size * hidden_dim;
        let embed_f16 = unsafe { std::slice::from_raw_parts(embed_data.as_ptr() as *const u16, n_f16) };
        let f16_to_f32 = |h: u16| -> f32 {
            let sign = ((h >> 15) & 1) as u32;
            let exp = ((h >> 10) & 0x1f) as u32;
            let frac = (h & 0x3ff) as u32;
            if exp == 0 {
                if frac == 0 { return f32::from_bits(sign << 31); }
                let mut e = 0i32;
                let mut fr = frac;
                while fr & 0x400 == 0 { fr <<= 1; e -= 1; }
                fr &= 0x3ff;
                return f32::from_bits((sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | (fr << 13));
            }
            if exp == 31 {
                return f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13));
            }
            f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
        };
        let mut embed_weight_i8 = Vec::with_capacity(n_f16);
        let mut embed_row_scales = Vec::with_capacity(vocab_size);
        for row in 0..vocab_size {
            let base = row * hidden_dim;
            let mut amax = 0.0f32;
            for d in 0..hidden_dim {
                let v = f16_to_f32(embed_f16[base + d]).abs();
                if v > amax { amax = v; }
            }
            embed_row_scales.push(amax);
            if amax < 1e-10 {
                #[cfg(target_arch = "aarch64")]
                for _ in 0..hidden_dim { embed_weight_i8.push(0u8); } // zero (no bias on ARM)
                #[cfg(not(target_arch = "aarch64"))]
                for _ in 0..hidden_dim { embed_weight_i8.push(128u8); } // zero → 128 (bias)
            } else {
                let inv = 127.0 / amax;
                for d in 0..hidden_dim {
                    let q = (f16_to_f32(embed_f16[base + d]) * inv).round().clamp(-127.0, 127.0) as i8;
                    #[cfg(target_arch = "aarch64")]
                    embed_weight_i8.push(q as u8); // store raw i8 (no bias on ARM)
                    #[cfg(not(target_arch = "aarch64"))]
                    embed_weight_i8.push((q as i16 + 128) as u8);
                }
            }
        }
        // Build sketch table for speculative top-k (ARM only)
        #[cfg(target_arch = "aarch64")]
        let (embed_sketch, sketch_dim) = {
            const SKETCH_STRIDE: usize = 4;
            let sd = hidden_dim / SKETCH_STRIDE;
            let mut sketch = Vec::with_capacity(vocab_size * sd);
            for row in 0..vocab_size {
                let base = row * hidden_dim;
                for s in 0..sd {
                    sketch.push(embed_weight_i8[base + s * SKETCH_STRIDE]);
                }
            }
            eprintln!("  Embedding: {} vocab × {} dim, i8 ({:.0} MB), sketch {}d ({:.1} MB)",
                vocab_size, hidden_dim,
                embed_weight_i8.len() as f64 / 1e6,
                sd, sketch.len() as f64 / 1e6);
            (sketch, sd)
        };
        #[cfg(not(target_arch = "aarch64"))]
        eprintln!("  Embedding: {} vocab × {} dim, i8 ({:.0} MB)",
            vocab_size, hidden_dim,
            embed_weight_i8.len() as f64 / 1e6);

        let mut layers = Vec::with_capacity(n_layers);
        let mut weight_data: Vec<Vec<u8>> = Vec::new();
        for i in 0..n_layers {
            let (wq, wq_s) = load_i2s_tensor(gguf, &format!("blk.{i}.attn_q.weight"), &mut weight_data)?;
            let (wk, wk_s) = load_i2s_tensor(gguf, &format!("blk.{i}.attn_k.weight"), &mut weight_data)?;
            let (wv, wv_s) = load_i2s_tensor(gguf, &format!("blk.{i}.attn_v.weight"), &mut weight_data)?;
            let (wo, wo_s) = load_i2s_tensor(gguf, &format!("blk.{i}.attn_output.weight"), &mut weight_data)?;
            let (wg, wg_s) = load_i2s_tensor(gguf, &format!("blk.{i}.ffn_gate.weight"), &mut weight_data)?;
            let (wu, wu_s) = load_i2s_tensor(gguf, &format!("blk.{i}.ffn_up.weight"), &mut weight_data)?;
            let (wd, wd_s) = load_i2s_tensor(gguf, &format!("blk.{i}.ffn_down.weight"), &mut weight_data)?;
            layers.push(LayerWeights {
                attn_norm: tensor_ptr(gguf, &format!("blk.{i}.attn_norm.weight"))?,
                attn_sub_norm: tensor_ptr(gguf, &format!("blk.{i}.attn_sub_norm.weight"))?,
                wq, wk, wv, wo,
                ffn_norm: tensor_ptr(gguf, &format!("blk.{i}.ffn_norm.weight"))?,
                ffn_sub_norm: tensor_ptr(gguf, &format!("blk.{i}.ffn_sub_norm.weight"))?,
                w_gate: wg, w_up: wu, w_down: wd,
                wq_scale: wq_s, wk_scale: wk_s, wv_scale: wv_s, wo_scale: wo_s,
                w_gate_scale: wg_s, w_up_scale: wu_s, w_down_scale: wd_s,
            });
        }

        Ok(BitNetModel {
            quant_type,
            activation,
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            ffn_dim,
            vocab_size,
            rope_theta,
            rms_eps,
            layers,
            embed_weight_f16,
            embed_weight_i8,
            embed_row_scales,
            #[cfg(target_arch = "aarch64")]
            embed_sketch,
            #[cfg(target_arch = "aarch64")]
            embed_sketch_dim: sketch_dim,
            norm_weight,
            _weight_data: weight_data,
            q4k_layers: Vec::new(),
            output_weight: std::ptr::null(),
            embed_dtype: 1, // F16
            output_block_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert!(std::mem::size_of::<LayerWeights>() > 0);
        assert!(std::mem::size_of::<BitNetModel>() > 0);
    }
}
