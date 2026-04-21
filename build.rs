fn main() {
    tonic_build::configure()
        .compile(&["proto/embodied_fl.proto"], &["proto"])
        .unwrap_or_else(|e| {
            eprintln!("Failed to compile proto: {}", e);
            std::process::exit(1);
        });
}
