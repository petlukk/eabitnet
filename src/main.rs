mod ffi;
mod forward;
mod gguf;
mod model;
mod tokenizer;

use forward::InferenceState;
use model::BitNetModel;
use tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = None;
    let mut prompt = None;
    let mut max_tokens: usize = 128;
    let mut temperature: f32 = 0.0;
    let mut max_seq_len: usize = 2048;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args[i].as_str());
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].as_str());
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("invalid --max-tokens");
            }
            "--temperature" => {
                i += 1;
                temperature = args[i].parse().expect("invalid --temperature");
            }
            "--max-seq-len" => {
                i += 1;
                max_seq_len = args[i].parse().expect("invalid --max-seq-len");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text> [--max-tokens N] [--temperature T]");
        std::process::exit(1);
    });
    let prompt_text = prompt.unwrap_or_else(|| {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text>");
        std::process::exit(1);
    });

    // Open GGUF
    let gguf = match gguf::GgufFile::open(model_path) {
        Ok(gf) => {
            eprintln!(
                "GGUF v{}: {} tensors, {} metadata keys",
                gf.version,
                gf.tensors.len(),
                gf.metadata.len(),
            );
            gf
        }
        Err(e) => {
            eprintln!("Failed to open GGUF: {e}");
            std::process::exit(1);
        }
    };

    // Build tokenizer
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to build tokenizer: {e}");
        std::process::exit(1);
    });

    // Load model
    let model = BitNetModel::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Model: {} layers, {} hidden, {} heads, {} head_dim, {} ffn, {} vocab",
        model.n_layers, model.hidden_dim, model.n_heads, model.head_dim,
        model.ffn_dim, model.vocab_size,
    );

    // Encode prompt (prepend BOS)
    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt_text));
    eprintln!("Prompt: {} tokens", tokens.len());

    // Generate
    let output = InferenceState::generate(
        &model,
        &tokens,
        max_tokens,
        temperature,
        tokenizer.eos_id,
        max_seq_len,
    );

    // Decode generated tokens (skip prompt)
    let generated = &output[tokens.len()..];
    let text = tokenizer.decode(generated);
    println!("{text}");
}
