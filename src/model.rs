//! BitNet b1.58 2B-4T model structure and weight loading from GGUF.

use crate::gguf::GgufFile;

pub struct BitNetModel {
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
    pub embed_weight: *const u8,
    pub norm_weight: *const f32,
    /// Owned repacked I2_S weight data (keeps pointers valid)
    _weight_data: Vec<Vec<u8>>,
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
    static LUT: std::sync::LazyLock<[u8; 256]> = std::sync::LazyLock::new(|| {
        let mut tbl = [0u8; 256];
        for b in 0..=255u8 {
            let g0 = (b >> 6) & 3;
            let g1 = (b >> 4) & 3;
            let g2 = (b >> 2) & 3;
            let g3 = b & 3;
            tbl[b as usize] = (((g0 + 1) % 3) << 6)
                | (((g1 + 1) % 3) << 4)
                | (((g2 + 1) % 3) << 2)
                | ((g3 + 1) % 3);
        }
        tbl
    });

    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    if data.len() < 32 {
        return Err(format!("{name}: tensor too small for I2_S scale"));
    }
    let scale_off = data.len() - 32;
    let sb = &data[scale_off..scale_off + 4];
    let scale = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);

    let weight_bytes = &data[..scale_off];
    let mut repacked = Vec::with_capacity(weight_bytes.len());
    let lut = &*LUT;
    for &b in weight_bytes {
        repacked.push(lut[b as usize]);
    }
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

        // Tied embeddings: token_embd.weight is F16, used for both embed and output
        let embed_weight: *const u8 = tensor_ptr(gguf, "token_embd.weight")?;
        let norm_weight: *const f32 = tensor_ptr(gguf, "output_norm.weight")?;

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
            embed_weight,
            norm_weight,
            _weight_data: weight_data,
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
