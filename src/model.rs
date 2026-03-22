//! BitNet b1.58 2B-4T model structure and weight loading from GGUF.

use crate::gguf::GgufFile;

pub struct BitNetModel {
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,

    pub layers: Vec<LayerWeights>,
    pub embed_weight: *const f32,
    pub norm_weight: *const f32,
    pub output_weight: *const u8,
}

pub struct LayerWeights {
    pub attn_norm: *const f32,
    pub wq: *const u8,
    pub wk: *const u8,
    pub wv: *const u8,
    pub wo: *const u8,
    pub ffn_norm: *const f32,
    pub w_gate: *const u8,
    pub w_up: *const u8,
    pub w_down: *const u8,
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

impl BitNetModel {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let n_layers = get_meta_u32(gguf, "llama.block_count")
            .or_else(|| get_meta_u32(gguf, "general.block_count"))
            .ok_or("missing metadata: block_count")? as usize;

        let hidden_dim = get_meta_u32(gguf, "llama.embedding_length")
            .ok_or("missing metadata: embedding_length")? as usize;

        let n_heads = get_meta_u32(gguf, "llama.attention.head_count")
            .ok_or("missing metadata: head_count")? as usize;

        let n_kv_heads = get_meta_u32(gguf, "llama.attention.head_count_kv")
            .unwrap_or(n_heads as u32) as usize;

        let head_dim = hidden_dim / n_heads;

        let ffn_dim = get_meta_u32(gguf, "llama.feed_forward_length")
            .ok_or("missing metadata: feed_forward_length")? as usize;

        let rope_theta = get_meta_f32(gguf, "llama.rope.freq_base")
            .unwrap_or(10000.0);

        let rms_eps = get_meta_f32(gguf, "llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        // Get vocab_size from embedding tensor dimensions
        let embed_idx = *gguf
            .tensor_map
            .get("token_embd.weight")
            .ok_or("missing tensor: token_embd.weight")?;
        let embed_dims = &gguf.tensors[embed_idx].dims;
        let vocab_size = if embed_dims.len() == 2 {
            embed_dims[1] as usize // [hidden_dim, vocab_size] — GGUF is row-major
        } else {
            return Err(format!(
                "token_embd.weight: expected 2 dims, got {}",
                embed_dims.len()
            ));
        };

        let embed_weight: *const f32 = tensor_ptr(gguf, "token_embd.weight")?;
        let norm_weight: *const f32 = tensor_ptr(gguf, "output_norm.weight")?;
        let output_weight: *const u8 = tensor_ptr(gguf, "output.weight")?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(LayerWeights {
                attn_norm: tensor_ptr(gguf, &format!("blk.{i}.attn_norm.weight"))?,
                wq: tensor_ptr(gguf, &format!("blk.{i}.attn_q.weight"))?,
                wk: tensor_ptr(gguf, &format!("blk.{i}.attn_k.weight"))?,
                wv: tensor_ptr(gguf, &format!("blk.{i}.attn_v.weight"))?,
                wo: tensor_ptr(gguf, &format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: tensor_ptr(gguf, &format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: tensor_ptr(gguf, &format!("blk.{i}.ffn_gate.weight"))?,
                w_up: tensor_ptr(gguf, &format!("blk.{i}.ffn_up.weight"))?,
                w_down: tensor_ptr(gguf, &format!("blk.{i}.ffn_down.weight"))?,
            });
        }

        Ok(BitNetModel {
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            rope_theta,
            rms_eps,
            layers,
            embed_weight,
            norm_weight,
            output_weight,
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
