use crate::gemm_i2s::{BatchI8, i2s_gemm_mt};
use crate::matmul::ternary_matmul_mt;
use crate::threadpool::ThreadPool;

/// Helper: pack a slice of ternary weights {-1, 0, +1} into the BitNet i2 format.
/// in_dim must be a multiple of 128.
fn pack_ternary_weights(ternary: &[i8], in_dim: usize) -> Vec<u8> {
    let n_blocks = in_dim / 128;
    let mut packed = vec![0u8; n_blocks * 32];
    for blk in 0..n_blocks {
        for i in 0..32 {
            let base = blk * 128;
            let g0 = (ternary[base + i] + 1) as u8;
            let g1 = (ternary[base + i + 32] + 1) as u8;
            let g2 = (ternary[base + i + 64] + 1) as u8;
            let g3 = (ternary[base + i + 96] + 1) as u8;
            packed[blk * 32 + i] = (g0 << 6) | (g1 << 4) | (g2 << 2) | g3;
        }
    }
    packed
}

fn pack_weight_matrix(weights_raw: &[i8], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let row_bytes = in_dim / 4;
    let mut packed = Vec::with_capacity(out_dim * row_bytes);
    for r in 0..out_dim {
        packed.extend_from_slice(
            &pack_ternary_weights(&weights_raw[r * in_dim..(r + 1) * in_dim], in_dim),
        );
    }
    packed
}

#[test]
fn batch_i8_quantize_roundtrip() {
    let dim = 256;
    let mut batch = BatchI8::new(2, dim);
    let x0: Vec<f32> = (0..dim).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let x1: Vec<f32> = (0..dim).map(|i| (i as f32) / 256.0).collect();
    batch.quantize(0, &x0);
    batch.quantize(1, &x1);
    assert!(batch.scale(0).abs() > 1e-10);
    assert!(batch.scale(1).abs() > 1e-10);
    assert_ne!(batch.sum(0), 0);
}

#[test]
fn i2s_gemm_matches_sequential() {
    let pool = ThreadPool::new();
    let in_dim = 256;
    let out_dim = 64;
    let n_tokens = 4;
    let weight_scale = 1.5f32;

    let weights_raw: Vec<i8> = (0..out_dim * in_dim)
        .map(|i| (i % 3) as i8 - 1)
        .collect();
    let packed = pack_weight_matrix(&weights_raw, out_dim, in_dim);

    let xs: Vec<Vec<f32>> = (0..n_tokens)
        .map(|t| (0..in_dim).map(|i| ((t * in_dim + i) as f32 - 500.0) / 500.0).collect())
        .collect();

    let mut batch = BatchI8::new(n_tokens, in_dim);
    for t in 0..n_tokens {
        batch.quantize(t, &xs[t]);
    }

    let mut gemm_out = vec![0.0f32; n_tokens * out_dim];
    i2s_gemm_mt(packed.as_ptr(), weight_scale, &batch, &mut gemm_out, out_dim, in_dim, &pool);

    for t in 0..n_tokens {
        let mut seq_out = vec![0.0f32; out_dim];
        ternary_matmul_mt(
            packed.as_ptr(), batch.qs_ptr(t),
            batch.scale(t), batch.sum(t), weight_scale,
            &mut seq_out, out_dim, in_dim, &pool,
        );
        for r in 0..out_dim {
            assert!(
                (gemm_out[t * out_dim + r] - seq_out[r]).abs() < 1e-4,
                "mismatch at token {t} row {r}: gemm={} seq={}",
                gemm_out[t * out_dim + r], seq_out[r],
            );
        }
    }
}
