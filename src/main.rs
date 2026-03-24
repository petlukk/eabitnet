mod embed;
mod ffi;
mod forward;
mod gguf;
mod matmul;
mod model;
mod repl;
mod server;
mod threadpool;
mod tokenizer;

use forward::InferenceState;
use model::BitNetModel;
use tokenizer::Tokenizer;

fn main() {
    embed::init().unwrap_or_else(|e| {
        eprintln!("Failed to load kernels: {e}");
        std::process::exit(1);
    });

    let args: Vec<String> = std::env::args().collect();

    let mut model_path = None;
    let mut prompt = None;
    let mut interactive = false;
    let mut serve = false;
    let mut port: u16 = 8080;
    let mut max_tokens: usize = 128;
    let mut temperature: f32 = 0.0;
    let mut repetition_penalty: f32 = 1.1;
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
            "--repetition-penalty" => {
                i += 1;
                repetition_penalty = args[i].parse().expect("invalid --repetition-penalty");
            }
            "--max-seq-len" => {
                i += 1;
                max_seq_len = args[i].parse().expect("invalid --max-seq-len");
            }
            "--interactive" => {
                interactive = true;
            }
            "--serve" => {
                serve = true;
            }
            "--port" => {
                i += 1;
                port = args[i].parse().expect("invalid --port");
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Usage: cougar --model <path.gguf> [--prompt <text> | --interactive | --serve] [--port P] [--max-tokens N] [--temperature T] [--repetition-penalty F]");
        std::process::exit(1);
    });
    if !interactive && !serve && prompt.is_none() {
        eprintln!("Usage: cougar --model <path.gguf> [--prompt <text> | --interactive | --serve]");
        std::process::exit(1);
    }

    let gguf = match gguf::GgufFile::open(model_path) {
        Ok(gf) => gf,
        Err(e) => {
            eprintln!("Failed to open GGUF: {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to build tokenizer: {e}");
        std::process::exit(1);
    });

    let model = BitNetModel::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "cougar> {} layers, {}d, {} heads, {} vocab",
        model.n_layers, model.hidden_dim, model.n_heads, model.vocab_size,
    );

    if serve {
        server::run(&model, &tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len, port);
        return;
    }

    if interactive {
        repl::run(&model, &tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len);
        return;
    }

    let prompt_text = prompt.unwrap();
    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(prompt_text));
    eprintln!("cougar> prompt: {} tokens", tokens.len());

    let (output, _prefill_ms, _decode_ms) = InferenceState::generate(
        &model, &tokens, max_tokens, temperature, repetition_penalty,
        tokenizer.eos_id, max_seq_len, |_| {},
    );

    let generated = &output[tokens.len()..];
    let text = tokenizer.decode(generated);
    println!("{text}");
}
