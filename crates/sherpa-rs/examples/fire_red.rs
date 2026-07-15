/// Smoke test for FireRedASR AED models (v1 and v2 exports).
///
/// cargo run --example fire_red -- <encoder.onnx> <decoder.onnx> <tokens.txt> <wav>
use sherpa_rs::{fire_red, read_audio_file};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: fire_red <encoder> <decoder> <tokens> <wav>");
        std::process::exit(1);
    }

    let (samples, sample_rate) = read_audio_file(&args[4]).expect("failed to read wav");

    let config = fire_red::FireRedConfig {
        encoder: args[1].clone(),
        decoder: args[2].clone(),
        tokens: args[3].clone(),
        num_threads: Some(4),
        debug: false,
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let mut recognizer = fire_red::FireRedRecognizer::new(config).expect("failed to load model");
    eprintln!("load: {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let result = recognizer.transcribe(sample_rate, &samples);
    eprintln!("decode: {:?}", start.elapsed());
    println!("TEXT: {}", result.text);
}
