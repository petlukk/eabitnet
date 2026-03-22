mod ffi;
mod gguf;
mod model;
mod tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] != "--model" {
        eprintln!("Usage: eabitnet --model <path.gguf> --prompt <text>");
        std::process::exit(1);
    }
    let model_path = &args[2];
    println!("eabitnet: model={model_path}");

    match gguf::GgufFile::open(model_path) {
        Ok(gf) => {
            println!(
                "GGUF v{}: {} tensors, {} metadata keys",
                gf.version,
                gf.tensors.len(),
                gf.metadata.len(),
            );
            if let Some(arch) = gf.get_str("general.architecture") {
                println!("  architecture: {arch}");
            }
            if let Some(name) = gf.get_str("general.name") {
                println!("  name: {name}");
            }
        }
        Err(e) => {
            eprintln!("Failed to open GGUF: {e}");
            std::process::exit(1);
        }
    }

    println!("Kernels linked successfully.");
}
