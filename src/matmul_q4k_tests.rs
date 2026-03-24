use super::*;

// ── unpack_q4k_scales ────────────────────────────────────────────────

#[test]
fn test_unpack_q4k_scales_zeros() {
    let packed = [0u8; 12];
    let mut scales = [0xFFu8; 8];
    let mut mins = [0xFFu8; 8];
    unpack_q4k_scales(&packed, &mut scales, &mut mins);
    assert_eq!(scales, [0u8; 8]);
    assert_eq!(mins, [0u8; 8]);
}

#[test]
fn test_unpack_q4k_scales_low6() {
    // bytes 0..3 = 0x3F (all low 6 bits set), bytes 4..7 = 0x3F, bytes 8..11 = 0
    // scales[0..4] = 0x3F = 63, mins[0..4] = 0x3F = 63
    // scales[4..8] = (0 & 0x0F) | ((0x3F >> 6) << 4) = 0 | 0 = 0  -- wait, 0x3F >> 6 = 0
    // Actually 0x3F = 0b00111111, >> 6 = 0. So scales[4..8] = 0, mins[4..8] = 0.
    let mut packed = [0u8; 12];
    packed[0] = 0x3F;
    packed[1] = 0x3F;
    packed[2] = 0x3F;
    packed[3] = 0x3F;
    packed[4] = 0x3F;
    packed[5] = 0x3F;
    packed[6] = 0x3F;
    packed[7] = 0x3F;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    unpack_q4k_scales(&packed, &mut scales, &mut mins);

    assert_eq!(scales[0..4], [63, 63, 63, 63]);
    assert_eq!(mins[0..4], [63, 63, 63, 63]);
    assert_eq!(scales[4..8], [0, 0, 0, 0]);
    assert_eq!(mins[4..8], [0, 0, 0, 0]);
}

#[test]
fn test_unpack_q4k_scales_high_bits() {
    // Test that bits 6-7 of bytes 0..3 contribute to scales[4..8] bits 4-5
    // and bits 6-7 of bytes 4..7 contribute to mins[4..8] bits 4-5
    let mut packed = [0u8; 12];
    // byte 0: 0xC0 = 0b11000000 -> scales[0] = 0, scales[4] gets (3 << 4) = 48
    packed[0] = 0xC0;
    // byte 4: 0x80 = 0b10000000 -> mins[0] = 0, mins[4] gets (2 << 4) = 32
    packed[4] = 0x80;
    // byte 8: 0x57 -> scales[4] low4 = 7, mins[4] low4 = 5
    packed[8] = 0x57;

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    unpack_q4k_scales(&packed, &mut scales, &mut mins);

    assert_eq!(scales[0], 0);     // 0xC0 & 0x3F = 0
    assert_eq!(mins[0], 0);       // 0x80 & 0x3F = 0
    // scales[4] = (0x57 & 0x0F) | ((0xC0 >> 6) << 4) = 7 | (3 << 4) = 7 | 48 = 55
    assert_eq!(scales[4], 55);
    // mins[4] = (0x57 >> 4) | ((0x80 >> 6) << 4) = 5 | (2 << 4) = 5 | 32 = 37
    assert_eq!(mins[4], 37);
}

#[test]
fn test_unpack_q4k_scales_reference() {
    // Reference values matching ggml dequantize_row_q4_K pattern
    // Pack known 6-bit scale/min values and verify round-trip
    let s_low: [u8; 4] = [10, 20, 30, 40];    // scales[0..4] (6-bit, < 64)
    let m_low: [u8; 4] = [5, 15, 25, 35];     // mins[0..4] (6-bit, < 64)
    let s_high: [u8; 4] = [50, 55, 60, 63];   // scales[4..8] (6-bit)
    let m_high: [u8; 4] = [45, 48, 52, 58];   // mins[4..8] (6-bit)

    let mut packed = [0u8; 12];
    for i in 0..4 {
        // bytes 0..3: scales[0..4] low 6 bits | (scales[4+i] >> 4) << 6
        packed[i] = s_low[i] | ((s_high[i] >> 4) << 6);
        // bytes 4..7: mins[0..4] low 6 bits | (mins[4+i] >> 4) << 6
        packed[4 + i] = m_low[i] | ((m_high[i] >> 4) << 6);
        // bytes 8..11: scales[4+i] low 4 bits | (mins[4+i] low 4 bits) << 4
        packed[8 + i] = (s_high[i] & 0x0F) | ((m_high[i] & 0x0F) << 4);
    }

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    unpack_q4k_scales(&packed, &mut scales, &mut mins);

    for i in 0..4 {
        assert_eq!(scales[i], s_low[i], "scales[{i}]");
        assert_eq!(mins[i], m_low[i], "mins[{i}]");
        assert_eq!(scales[4 + i], s_high[i], "scales[{}]", 4 + i);
        assert_eq!(mins[4 + i], m_high[i], "mins[{}]", 4 + i);
    }
}

// ── Q4_K matmul (requires kernel .so) ────────────────────────────────

/// Build a minimal Q4_K block (144 bytes) with known values.
/// All nibbles set to `nibble_val` (0..15), with given d, dmin, and uniform scales/mins.
fn make_q4k_block(d_f16: u16, dmin_f16: u16, scale: u8, min: u8, nibble_val: u8) -> [u8; 144] {
    let mut block = [0u8; 144];

    // d at offset 0 (f16, little-endian)
    block[0..2].copy_from_slice(&d_f16.to_le_bytes());
    // dmin at offset 2 (f16, little-endian)
    block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

    // Pack scales: all 8 scales = `scale`, all 8 mins = `min`
    // Both must be <= 63 to fit in 6 bits for this simple packing.
    assert!(scale <= 63 && min <= 63, "simple packing requires scale/min <= 63");
    for i in 0..4 {
        block[4 + i] = scale;       // scales[0..4] low 6 bits
        block[8 + i] = min;         // mins[0..4] low 6 bits
        // bytes 8..11: scales[4..8] low 4 = scale & 0x0F, mins[4..8] low 4 = min & 0x0F
        block[12 + i] = (scale & 0x0F) | ((min & 0x0F) << 4);
    }

    // Pack nibbles: both low and high nibble = nibble_val
    let byte = (nibble_val & 0x0F) | ((nibble_val & 0x0F) << 4);
    for i in 0..128 {
        block[16 + i] = byte;
    }

    block
}

/// Compute expected Q4_K dot product for uniform blocks (reference implementation).
fn reference_q4k_dot(
    n_blocks: usize,
    d: f32,
    dmin: f32,
    scale: u8,
    min_val: u8,
    nibble_val: u8,
    q8_vals: &[i8],
    q8_d: &[f32],
) -> f32 {
    let mut result = 0.0f32;
    for blk in 0..n_blocks {
        let blk_d = d * q8_d[blk];
        let blk_dmin = dmin * q8_d[blk];

        // sumi: for each of 8 sub-blocks (4 chunks * 2 halves),
        // accumulate nibble * q8 * sub-block scale.
        // With uniform nibbles and scales, this simplifies.
        let mut sumi: i32 = 0;
        for j in 0..4 {
            // Low nibbles: 32 elements, each = nibble_val
            let mut dot_lo: i32 = 0;
            for k in 0..32 {
                dot_lo += nibble_val as i32 * q8_vals[blk * 256 + j * 64 + k] as i32;
            }
            sumi += dot_lo * scale as i32;

            // High nibbles: 32 elements, each = nibble_val
            let mut dot_hi: i32 = 0;
            for k in 0..32 {
                dot_hi += nibble_val as i32 * q8_vals[blk * 256 + j * 64 + 32 + k] as i32;
            }
            sumi += dot_hi * scale as i32;
        }

        // summs: sum of min * (bsums pairs)
        // bsums[i] = sum of 16 consecutive q8 values
        let mut summs: i32 = 0;
        for j in 0..8 {
            let bs0: i32 = q8_vals[blk * 256 + j * 32..blk * 256 + j * 32 + 16]
                .iter().map(|&v| v as i32).sum();
            let bs1: i32 = q8_vals[blk * 256 + j * 32 + 16..blk * 256 + j * 32 + 32]
                .iter().map(|&v| v as i32).sum();
            summs += min_val as i32 * (bs0 + bs1);
        }

        result += blk_d * sumi as f32 - blk_dmin * summs as f32;
    }
    result
}

#[test]
fn test_q4k_matmul_basic() {
    let n_blocks: usize = 1;
    let out_dim: usize = 4;
    let n_elem = n_blocks * 256;

    // d = 1.0 (f16 = 0x3C00), dmin = 0.5 (f16 = 0x3800)
    let d_f16: u16 = 0x3C00;
    let dmin_f16: u16 = 0x3800;
    let scale: u8 = 2;
    let min_val: u8 = 1;
    let nibble_val: u8 = 3;

    let block = make_q4k_block(d_f16, dmin_f16, scale, min_val, nibble_val);

    // Build weight: 4 identical rows
    let row_stride = n_blocks * Q4K_BLOCK_BYTES;
    let mut weight = Vec::with_capacity(out_dim * row_stride);
    for _ in 0..out_dim {
        weight.extend_from_slice(&block);
    }

    // Q8_K activations: all ones
    let q8_qs: Vec<i8> = vec![1i8; n_elem];

    // Q8_K scales: all 1.0
    let q8_d: Vec<f32> = vec![1.0f32; n_blocks];

    // Q8_K bsums: each group of 16 = sum of 16 ones = 16
    let q8_bsums: Vec<i32> = vec![16i32; n_blocks * 16];

    let mut out = vec![0.0f32; out_dim];
    let pool = crate::threadpool::ThreadPool::new();

    q4k_matmul_mt(
        weight.as_ptr(),
        row_stride,
        n_blocks,
        q8_qs.as_ptr(),
        q8_d.as_ptr(),
        q8_bsums.as_ptr(),
        &mut out,
        out_dim,
        &pool,
    );

    let d = f16_to_f32(d_f16);
    let dmin = f16_to_f32(dmin_f16);
    let expected = reference_q4k_dot(
        n_blocks, d, dmin, scale, min_val, nibble_val,
        &q8_qs, &q8_d,
    );

    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] is not finite: {v}");
        assert!(
            (v - expected).abs() < 1e-3,
            "row {i}: expected {expected}, got {v}"
        );
    }
}

#[test]
fn test_q4k_matmul_multi_block() {
    let n_blocks: usize = 2;
    let out_dim: usize = 8;
    let n_elem = n_blocks * 256;

    let d_f16: u16 = 0x3C00; // 1.0
    let dmin_f16: u16 = 0x0000; // 0.0 (no min correction)
    let scale: u8 = 1;
    let min_val: u8 = 0;
    let nibble_val: u8 = 1;

    let block = make_q4k_block(d_f16, dmin_f16, scale, min_val, nibble_val);

    let row_stride = n_blocks * Q4K_BLOCK_BYTES;
    let mut weight = Vec::with_capacity(out_dim * row_stride);
    for _ in 0..out_dim {
        for _ in 0..n_blocks {
            weight.extend_from_slice(&block);
        }
    }

    let q8_qs: Vec<i8> = vec![1i8; n_elem];
    let q8_d: Vec<f32> = vec![1.0f32; n_blocks];
    let q8_bsums: Vec<i32> = vec![16i32; n_blocks * 16];

    let mut out = vec![0.0f32; out_dim];
    let pool = crate::threadpool::ThreadPool::new();

    q4k_matmul_mt(
        weight.as_ptr(),
        row_stride,
        n_blocks,
        q8_qs.as_ptr(),
        q8_d.as_ptr(),
        q8_bsums.as_ptr(),
        &mut out,
        out_dim,
        &pool,
    );

    let d = f16_to_f32(d_f16);
    let dmin = f16_to_f32(dmin_f16);
    let expected = reference_q4k_dot(
        n_blocks, d, dmin, scale, min_val, nibble_val,
        &q8_qs, &q8_d,
    );

    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] is not finite: {v}");
        assert!(
            (v - expected).abs() < 1e-3,
            "row {i}: expected {expected}, got {v}"
        );
    }
}
