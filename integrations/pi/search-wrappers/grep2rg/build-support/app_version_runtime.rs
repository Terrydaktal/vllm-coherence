// This dependency-free producer follows program-metadata's exact v1 wire format.
// Keep vendored copies in sync with the version repository's canonical template.
const fn metadata_product(product: &str) -> bool {
    let bytes = product.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !byte.is_ascii_alphanumeric()
            && byte != b'.'
            && byte != b'_'
            && byte != b'+'
            && byte != b'-'
        {
            return false;
        }
        index += 1;
    }
    true
}

const fn metadata_version(version: &str) -> bool {
    let bytes = version.as_bytes();
    if bytes.len() < 3 || bytes.len() > 128 || !bytes[0].is_ascii_digit() {
        return false;
    }
    // The allocator's explicit unborn-source baseline is a real build release,
    // unlike an unqualified placeholder 0.0.0.
    if bytes.len() > 6
        && bytes[0] == b'0'
        && bytes[1] == b'.'
        && bytes[2] == b'0'
        && bytes[3] == b'.'
        && bytes[4] == b'0'
        && bytes[5] == b'-'
    {
        if bytes.len() > 7 && bytes[6] == b'0' {
            return false;
        }
        let mut at = 6;
        while at < bytes.len() {
            if !bytes[at].is_ascii_digit() {
                return false;
            }
            at += 1;
        }
        return true;
    }
    let mut index = 0;
    let mut core = true;
    let mut dots = 0;
    let mut nonzero = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'.' {
            if index == bytes.len() - 1 || (index > 0 && bytes[index - 1] == b'.') {
                return false;
            }
            if core {
                dots += 1;
            }
        } else if byte.is_ascii_digit() {
            if core && byte != b'0' {
                nonzero = true;
            }
        } else {
            if !byte.is_ascii_alphanumeric()
                && byte != b'-'
                && byte != b'_'
                && byte != b'+'
                && byte != b'~'
                && byte != b'('
                && byte != b')'
            {
                return false;
            }
            if core && index > 0 && bytes[index - 1] == b'.' {
                return false;
            }
            core = false;
        }
        index += 1;
    }
    dots >= 1 && nonzero
}

const PRODUCT_METADATA_BYTES: &[u8] = concat!(
    "PROGRAM-METADATA-V1\n",
    env!("CARGO_BIN_NAME"),
    "\n",
    env!("APP_VERSION"),
    "\nnone\n"
)
.as_bytes();

const fn metadata_record() -> [u8; PRODUCT_METADATA_BYTES.len()] {
    assert!(
        metadata_product(env!("CARGO_BIN_NAME")),
        "invalid metadata product basename"
    );
    assert!(
        metadata_version(env!("APP_VERSION")),
        "invalid canonical metadata version"
    );
    let mut record = [0; PRODUCT_METADATA_BYTES.len()];
    let mut index = 0;
    while index < record.len() {
        record[index] = PRODUCT_METADATA_BYTES[index];
        index += 1;
    }
    record
}

#[used]
#[unsafe(link_section = ".program_metadata")]
static PRODUCT_METADATA: [u8; PRODUCT_METADATA_BYTES.len()] = metadata_record();

pub fn handle_cli(args: &[std::ffi::OsString]) {
    // #[used] alone does not stop the linker from discarding an entire section.
    std::hint::black_box(&PRODUCT_METADATA);
    if option_env!("APP_VERSION_COMMIT").is_some()
        && args.len() == 1
        && (args[0] == "--version" || args[0] == "-V")
    {
        println!("{} {}", env!("CARGO_BIN_NAME"), env!("APP_VERSION"));
        std::process::exit(0);
    }
}
