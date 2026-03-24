mod embed;
mod ffi;
mod forward;
mod gguf;
mod matmul;
mod matmul_q4k;
mod model;
mod repl;
mod server;
mod threadpool;
mod tokenizer;

use forward::InferenceState;
use model::BitNetModel;
use tokenizer::Tokenizer;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "\
Usage: cougar --model <path.gguf> [--prompt <text> | --interactive | --serve]

Modes:
  --prompt <text>         Generate from a single prompt
  --interactive           Interactive REPL (stdin/stdout)
  --serve                 Web chat UI with SSE streaming

Options:
  --max-tokens N          Maximum tokens to generate (default: 128)
  --temperature T         Sampling temperature, 0 = greedy (default: 0)
  --repetition-penalty F  Penalize repeated tokens (default: 1.1)
  --max-seq-len N         Maximum sequence length (default: 2048)
  --port N                Server port for --serve (default: 8080)
  --help, -h              Show this help message
  --version               Show version";

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn parse_arg<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str) -> T {
    *i += 1;
    if *i >= args.len() {
        die(&format!("{flag} requires a value"));
    }
    args[*i].parse().unwrap_or_else(|_| die(&format!("{flag}: invalid value '{}'", args[*i])))
}

fn main() {
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
            "--help" | "-h" => {
                println!("{USAGE}");
                return;
            }
            "--version" => {
                println!("cougar {VERSION}");
                return;
            }
            "--model" => { i += 1; model_path = Some(args[i].clone()); }
            "--prompt" => { i += 1; prompt = Some(args[i].clone()); }
            "--max-tokens" => { max_tokens = parse_arg(&args, &mut i, "--max-tokens"); }
            "--temperature" => { temperature = parse_arg(&args, &mut i, "--temperature"); }
            "--repetition-penalty" => { repetition_penalty = parse_arg(&args, &mut i, "--repetition-penalty"); }
            "--max-seq-len" => { max_seq_len = parse_arg(&args, &mut i, "--max-seq-len"); }
            "--port" => { port = parse_arg(&args, &mut i, "--port"); }
            "--interactive" => { interactive = true; }
            "--serve" => { serve = true; }
            other => die(&format!("unknown argument: {other}\n\nRun 'cougar --help' for usage.")),
        }
        i += 1;
    }

    // Default model path: ~/.cougar/models/ggml-model-i2_s.gguf
    let model_path = model_path.unwrap_or_else(|| {
        let default = format!(
            "{}/.cougar/models/ggml-model-i2_s.gguf",
            std::env::var("HOME").unwrap_or_default()
        );
        if std::path::Path::new(&default).exists() {
            eprintln!("cougar> using default model: {default}");
            default
        } else {
            eprintln!("{USAGE}");
            die("--model is required (no default model found at ~/.cougar/models/ggml-model-i2_s.gguf)");
        }
    });

    if !interactive && !serve && prompt.is_none() {
        eprintln!("{USAGE}");
        die("one of --prompt, --interactive, or --serve is required");
    }

    embed::init().unwrap_or_else(|e| die(&format!("failed to load kernels: {e}")));

    let gguf = match gguf::GgufFile::open(&model_path) {
        Ok(gf) => gf,
        Err(e) => die(&format!("failed to open GGUF '{}': {e}", model_path)),
    };

    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
    let model = BitNetModel::from_gguf(&gguf).unwrap_or_else(|e| die(&format!("model: {e}")));

    eprintln!(
        "cougar> {} layers, {}d, {} heads, {} vocab",
        model.n_layers, model.hidden_dim, model.n_heads, model.vocab_size,
    );
    eprintln!("cougar> quant: {:?}, activation: {:?}", model.quant_type, model.activation);

    if model.quant_type == model::QuantType::Q4K {
        die("Q4_K_M inference not yet implemented — coming in v0.3.0");
    }

    if serve {
        server::run(&model, &tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len, port);
        return;
    }

    if interactive {
        repl::run(&model, &tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len);
        return;
    }

    let prompt_text = prompt.unwrap();
    if prompt_text.is_empty() {
        die("--prompt cannot be empty");
    }

    let mut tokens = vec![tokenizer.bos_id];
    tokens.extend(tokenizer.encode(&prompt_text));
    eprintln!("cougar> prompt: {} tokens", tokens.len());

    let (output, _prefill_ms, _decode_ms) = InferenceState::generate(
        &model, &tokens, max_tokens, temperature, repetition_penalty,
        tokenizer.eos_id, max_seq_len, |_| {},
    );

    let generated = &output[tokens.len()..];
    let text = tokenizer.decode(generated);
    println!("{text}");
}
