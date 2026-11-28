use std::path::Path;
use std::process::Command;

pub fn emit(root: &str, component: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(root);
    // npm packages and pinned runtime snapshots ship source without their
    // original Git history. Use the crate release version for those builds.
    if !root.join(".git").exists() {
        println!("cargo:rerun-if-changed=Cargo.toml");
        println!("cargo:rerun-if-changed=../APP-VERSION");
        println!("cargo:rustc-env=APP_VERSION={}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let python = std::env::var_os("APP_VERSION_PYTHON").unwrap_or_else(|| "python3".into());
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let output = Command::new(python)
        .arg(root.join("build-support/app_version.py"))
        .arg("--root")
        .arg(&root)
        .args(["--component", component, "--allocate", "--cargo"])
        .arg(&manifest)
        .arg("--output")
        .arg(
            Path::new(&std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR"))
                .join("app-version.json"),
        )
        .output()
        .expect("run Python 3.11+ to generate the application version");
    assert!(
        output.status.success(),
        "application version generation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    print!(
        "{}",
        String::from_utf8(output.stdout).expect("UTF-8 Cargo version instructions")
    );
}
