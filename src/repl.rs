//! Interactive REPL: read prompts from stdin, stream tokens to stdout.

use crate::forward::InferenceState;
use crate::forward_llama;
use crate::model::BitNetModel;
use crate::tokenizer::Tokenizer;
use std::io::{self, BufRead, Write};

pub fn run(
    model: &BitNetModel,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    max_seq_len: usize,
) {
    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        eprint!("\nYou: ");
        let _ = io::stderr().flush();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF (Ctrl-D)
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" {
            break;
        }

        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(line));

        let handle = stdout.lock();
        let mut handle = io::BufWriter::new(handle);

        let (_output, _prefill_ms, _decode_ms) = InferenceState::generate(
            model,
            &tokens,
            max_tokens,
            temperature,
            repetition_penalty,
            tokenizer.eos_id,
            max_seq_len,
            |tok_id| {
                let text = tokenizer.decode(&[tok_id]);
                let _ = handle.write_all(text.as_bytes());
                let _ = handle.flush();
            },
        );
        let _ = writeln!(handle);
    }
}

pub fn run_q4k(
    model: &BitNetModel,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    max_seq_len: usize,
) {
    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        eprint!("\nYou: ");
        let _ = io::stderr().flush();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "/quit" { break; }

        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend(tokenizer.encode(line));

        let handle = stdout.lock();
        let mut handle = io::BufWriter::new(handle);

        let _ = forward_llama::generate(
            model, &tokens, max_tokens, temperature, repetition_penalty,
            tokenizer.eos_id, max_seq_len,
            |tok_id| {
                let text = tokenizer.decode(&[tok_id]);
                let _ = handle.write_all(text.as_bytes());
                let _ = handle.flush();
            },
        );
        let _ = writeln!(handle);
    }
}
