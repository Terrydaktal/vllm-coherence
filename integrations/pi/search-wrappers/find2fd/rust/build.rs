#[path = "../build-support/app_version.rs"]
mod app_version;

fn main() {
    app_version::emit("..", "rust:rust");
}
