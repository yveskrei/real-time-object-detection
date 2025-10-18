fn main() {
    let ffmpeg_dir = std::env::var("FFMPEG_DIR").expect("FFMPEG_DIR not set");
    let deps_dir = std::env::var("DEPS_DIR")
        .unwrap_or_else(|_| format!("{}/dependencies", std::env::var("PROJECT_ROOT").unwrap_or_else(|_| ".".to_string())));

    println!("cargo:rustc-link-search=native={}/lib", ffmpeg_dir);
    println!("cargo:rustc-link-search=native={}/lib", deps_dir);
    
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-env-changed=DEPS_DIR");
}