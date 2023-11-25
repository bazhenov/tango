fn main() {
    println!("cargo:rustc-link-arg-benches=-rdynamic");
}
