use super::*;
use crate::model::BitNetModel;

/// Minimal model for testing InferenceState without real weights.
fn mock_model(hidden_dim: usize, n_layers: usize, n_heads: usize, vocab_size: usize) -> BitNetModel {
    let head_dim = if n_heads > 0 { hidden_dim / n_heads } else { hidden_dim };
    let n_kv_heads = n_heads.max(1);
    BitNetModel {
        quant_type: crate::model::QuantType::I2S,
        activation: crate::model::Activation::SquaredReLU,
        n_layers,
        hidden_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_dim: n_kv_heads * head_dim,
        ffn_dim: hidden_dim * 4,
        vocab_size,
        rope_theta: 500000.0,
        rms_eps: 1e-5,
        layers: Vec::new(),
        embed_weight_f16: std::ptr::null(),
        embed_weight_i8: Vec::new(),
        embed_row_scales: Vec::new(),
        #[cfg(target_arch = "aarch64")]
        embed_sketch: Vec::new(),
        #[cfg(target_arch = "aarch64")]
        embed_sketch_dim: 0,
        norm_weight: std::ptr::null(),
        _weight_data: Vec::new(),
        q4k_layers: Vec::new(),
        output_weight: std::ptr::null(),
        embed_dtype: 1,
        output_block_bytes: 0,
    }
}

// ── argmax ─────────────────────────────────────────────────────────────

#[test]
fn argmax_basic() {
    assert_eq!(argmax(&[1.0f32, 3.0, 2.0]), 1);
}

#[test]
fn argmax_single() {
    assert_eq!(argmax(&[42.0f32]), 0);
}

#[test]
fn argmax_all_equal() {
    let v = [5.0f32, 5.0, 5.0];
    let idx = argmax(&v) as usize;
    assert_eq!(v[idx], 5.0f32);
}

#[test]
fn argmax_first_is_max() {
    assert_eq!(argmax(&[9.0f32, 1.0, 2.0]), 0);
}

#[test]
fn argmax_last_is_max() {
    assert_eq!(argmax(&[1.0f32, 2.0, 8.0]), 2);
}

// ── sample ─────────────────────────────────────────────────────────────

#[test]
fn sample_temperature_zero_matches_argmax() {
    let logits = [0.5f32, 3.0, 1.0, 2.0];
    assert_eq!(sample(&logits, 0.0), argmax(&logits));
}

#[test]
fn sample_temperature_negative_matches_argmax() {
    let logits = [1.0f32, 5.0, 2.0];
    assert_eq!(sample(&logits, -1.0), argmax(&logits));
}

#[test]
fn sample_positive_temperature_returns_valid_index() {
    let logits = [1.0f32, 2.0, 3.0, 0.5];
    let idx = sample(&logits, 1.0);
    assert!((idx as usize) < logits.len(), "index {idx} out of range");
}

#[test]
fn sample_high_temperature_still_valid() {
    let logits = [0.1f32, 0.2, 0.9, 0.05];
    let idx = sample(&logits, 100.0);
    assert!((idx as usize) < logits.len());
}

// ── xorshift_f32 ─────────────────────────────────────────────────────

#[test]
fn xorshift_range_0_to_1() {
    for _ in 0..1000 {
        let v = xorshift_f32();
        assert!(v >= 0.0 && v < 1.0, "xorshift out of range: {v}");
    }
}

#[test]
fn xorshift_not_constant() {
    let a = xorshift_f32();
    let b = xorshift_f32();
    let c = xorshift_f32();
    assert!(a != b || b != c, "xorshift returned same value 3 times");
}

// ── sample (softmax + random) ────────────────────────────────────────

#[test]
fn sample_temperature_distribution_favors_max() {
    // With low temperature, sampling should almost always pick the max logit
    let logits = [0.0f32, 0.0, 10.0, 0.0];
    let mut counts = [0u32; 4];
    for _ in 0..200 {
        let idx = sample(&logits, 0.1);
        counts[idx as usize] += 1;
    }
    assert!(counts[2] > 150, "max logit should dominate at low temp, got {}", counts[2]);
}

#[test]
fn sample_high_temperature_spreads_distribution() {
    // With very high temperature, all tokens should get some hits
    let logits = [1.0f32, 1.0, 1.0, 1.0];
    let mut counts = [0u32; 4];
    for _ in 0..400 {
        let idx = sample(&logits, 10.0);
        counts[idx as usize] += 1;
    }
    for (i, &c) in counts.iter().enumerate() {
        assert!(c > 20, "token {i} should appear with uniform logits, got {c}");
    }
}

// ── repetition penalty ───────────────────────────────────────────────

#[test]
fn repetition_penalty_reduces_positive_logits() {
    let model = mock_model(4, 1, 1, 4);
    let mut state = InferenceState::new(&model, 16);
    state.logits = vec![2.0, 4.0, 6.0, 8.0];
    state.apply_repetition_penalty(&[1, 3], 2.0);
    assert_eq!(state.logits[0], 2.0); // untouched
    assert_eq!(state.logits[1], 2.0); // 4.0 / 2.0
    assert_eq!(state.logits[2], 6.0); // untouched
    assert_eq!(state.logits[3], 4.0); // 8.0 / 2.0
}

#[test]
fn repetition_penalty_amplifies_negative_logits() {
    let model = mock_model(4, 1, 1, 4);
    let mut state = InferenceState::new(&model, 16);
    state.logits = vec![-2.0, -4.0, 1.0, -6.0];
    state.apply_repetition_penalty(&[0, 1], 2.0);
    assert_eq!(state.logits[0], -4.0);  // -2.0 * 2.0
    assert_eq!(state.logits[1], -8.0);  // -4.0 * 2.0
    assert_eq!(state.logits[2], 1.0);   // untouched
    assert_eq!(state.logits[3], -6.0);  // untouched
}

#[test]
fn repetition_penalty_1_0_is_noop() {
    let model = mock_model(4, 1, 1, 4);
    let mut state = InferenceState::new(&model, 16);
    state.logits = vec![1.0, 2.0, 3.0, 4.0];
    let before = state.logits.clone();
    state.apply_repetition_penalty(&[0, 1, 2, 3], 1.0);
    assert_eq!(state.logits, before);
}

#[test]
fn repetition_penalty_out_of_range_token_ignored() {
    let model = mock_model(4, 1, 1, 4);
    let mut state = InferenceState::new(&model, 16);
    state.logits = vec![1.0, 2.0, 3.0, 4.0];
    let before = state.logits.clone();
    state.apply_repetition_penalty(&[999], 2.0); // token 999 > vocab size
    assert_eq!(state.logits, before);
}

// ── build_rope_freqs ───────────────────────────────────────────────────

#[test]
fn rope_freqs_pos0_cos1_sin0() {
    let head_dim = 8usize;
    let mut freqs = vec![0.0f32; head_dim];
    build_rope_freqs(&mut freqs, head_dim, 0, 10000.0);

    for i in 0..head_dim / 2 {
        let cos = freqs[2 * i];
        let sin = freqs[2 * i + 1];
        assert!((cos - 1.0f32).abs() < 1e-6, "cos[{i}] = {cos}, expected 1.0");
        assert!(sin.abs() < 1e-6,            "sin[{i}] = {sin}, expected 0.0");
    }
}

#[test]
fn rope_freqs_output_length() {
    let head_dim = 16usize;
    let mut freqs = vec![0.0f32; head_dim];
    build_rope_freqs(&mut freqs, head_dim, 3, 10000.0);
    assert_eq!(freqs.len(), head_dim);
}

#[test]
fn rope_freqs_pos1_first_pair() {
    let head_dim = 4usize;
    let theta = 10000.0f32;
    let mut freqs = vec![0.0f32; head_dim];
    build_rope_freqs(&mut freqs, head_dim, 1, theta);

    let angle0 = 1.0f32 / theta.powf(0.0);
    assert!((freqs[0] - angle0.cos()).abs() < 1e-6, "cos mismatch: {}", freqs[0]);
    assert!((freqs[1] - angle0.sin()).abs() < 1e-6, "sin mismatch: {}", freqs[1]);
}

// ── apply_rope ─────────────────────────────────────────────────────────

#[test]
fn apply_rope_pos0_identity() {
    let head_dim = 4usize;
    let n_heads = 2usize;
    let mut freqs = vec![0.0f32; head_dim];
    build_rope_freqs(&mut freqs, head_dim, 0, 10000.0);

    let original: Vec<f32> = (0..(head_dim * n_heads)).map(|i| i as f32 * 0.1 + 1.0).collect();
    let mut data = original.clone();
    apply_rope(&mut data, &freqs, head_dim, n_heads);

    for (i, (&orig, &got)) in original.iter().zip(data.iter()).enumerate() {
        assert!((got - orig).abs() < 1e-6, "element {i}: got {got}, expected {orig}");
    }
}

#[test]
fn apply_rope_known_rotation() {
    let head_dim = 2usize;
    let n_heads = 1usize;
    let angle = std::f32::consts::FRAC_PI_2;
    let freqs = vec![angle.cos(), angle.sin()];
    let mut data = vec![1.0f32, 0.0f32];
    apply_rope(&mut data, &freqs, head_dim, n_heads);

    assert!((data[0] - 0.0f32).abs() < 1e-6, "real part: {}", data[0]);
    assert!((data[1] - 1.0f32).abs() < 1e-6, "imag part: {}", data[1]);
}

#[test]
fn apply_rope_multi_head_consistent() {
    let head_dim = 4usize;
    let n_heads = 2usize;
    let mut freqs = vec![0.0f32; head_dim];
    build_rope_freqs(&mut freqs, head_dim, 5, 10000.0);

    let mut data: Vec<f32> = (0..head_dim * n_heads).map(|i| (i % head_dim) as f32).collect();
    apply_rope(&mut data, &freqs, head_dim, n_heads);

    for i in 0..head_dim {
        assert!(
            (data[i] - data[head_dim + i]).abs() < 1e-6,
            "head mismatch at dim {i}: {} vs {}",
            data[i], data[head_dim + i]
        );
    }
}
