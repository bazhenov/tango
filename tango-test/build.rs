fn main() {
    // Using `rustc-link-arg` instead of `rustc-link-arg-benches`,
    // becase we build benchmarks from the main executable
    println!("cargo:rustc-link-arg=-rdynamic");
    println!("cargo:rerun-if-changed=build.rs");
}
