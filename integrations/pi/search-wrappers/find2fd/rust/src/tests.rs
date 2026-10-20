//! Fast independent contracts. Real GNU CLI qualification lives in the opt-in
//! integration executable and always tests the optimized production release.
use super::*;
use std::os::unix::ffi::OsStringExt;
use std::sync::{Mutex, MutexGuard};

static ENVIRONMENT: Mutex<()> = Mutex::new(());
struct Environment {
    _lock: MutexGuard<'static, ()>,
    old: Vec<(&'static str, Option<OsString>)>,
    sigpipe: bool,
}
fn environment() -> Environment {
    let lock = ENVIRONMENT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let old = ["LC_ALL", "POSIXLY_CORRECT", "FIND2FD_ACCELERATION"]
        .into_iter()
        .map(|key| (key, env::var_os(key)))
        .collect();
    env::set_var("LC_ALL", "C");
    env::remove_var("POSIXLY_CORRECT");
    env::remove_var("FIND2FD_ACCELERATION");
    let sigpipe = INHERITED_SIGPIPE_IGNORED.swap(false, Ordering::Relaxed);
    Environment {
        _lock: lock,
        old,
        sigpipe,
    }
}
impl Drop for Environment {
    fn drop(&mut self) {
        for (key, value) in &self.old {
            if let Some(value) = value {
                env::set_var(key, value);
            } else {
                env::remove_var(key);
            }
        }
        INHERITED_SIGPIPE_IGNORED.store(self.sigpipe, Ordering::Relaxed);
    }
}
fn argv(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}
fn entry() -> Entry {
    Entry {
        path: b"dir/foo".to_vec(),
        name_range: (4, 7),
        kind: b'f',
        depth: 0,
        parent_directory: None,
    }
}
fn evaluate(expression: &[&str]) -> (bool, Vec<u8>) {
    let mut args = argv(&["root"]);
    args.extend(argv(expression));
    let options = parse(&args).expect("qualified expression");
    let budget = Budget::new();
    let mut output = Output::new();
    let value = options
        .expression
        .evaluate(&entry(), &mut output, &budget)
        .unwrap();
    (value, output.into_bytes(&budget))
}
#[test]
fn parser_preserves_raw_roots_and_leading_depth_overrides() {
    let _environment = environment();
    let mut args = vec![
        OsString::from("-P"),
        OsString::from_vec(b"./raw-\xff///".to_vec()),
        OsString::from("other"),
    ];
    args.extend(argv(&[
        "-maxdepth",
        "3",
        "-maxdepth",
        "0001",
        "-mindepth",
        "2",
        "-print0",
    ]));
    let options = parse(&args).unwrap();
    assert_eq!(
        options.roots,
        vec![b"./raw-\xff///".to_vec(), b"other".to_vec()]
    );
    assert_eq!((options.min_depth, options.max_depth), (2, 1));
    assert!(!options.newline_output);
    let default = parse(&[]).unwrap();
    assert_eq!(default.roots, vec![b".".to_vec()]);
    assert!(default.newline_output);
}
#[test]
fn parser_rejects_malformed_metadata_and_position_sensitive_depth() {
    let _environment = environment();
    for expression in [
        &["-name"][..],
        &["("][..],
        &[")"][..],
        &["-o"][..],
        &["!"][..],
        &[","][..],
        &["-type", "x"][..],
        &["-type", "f,d"][..],
        &["-name", "*/*"][..],
        &["-path", "tree/"][..],
        &["-size", "1c"][..],
        &["-true", "-maxdepth", "1"][..],
    ] {
        let mut args = argv(&["root"]);
        args.extend(argv(expression));
        assert!(parse(&args).is_err(), "{expression:?}");
    }
    for depth in ["", "-1", "bad", "1000001", "18446744073709551616"] {
        assert!(parse(&argv(&["root", "-maxdepth", depth])).is_err());
    }
    assert!(parse(&[OsString::new()]).is_err());
    assert!(parse(&[OsString::from_vec(vec![b'x'; PATH_LIMIT])]).is_err());
}
#[test]
fn parser_bounds_and_explicit_gnu_policy_fail_before_work() {
    let _environment = environment();
    let mut nesting = argv(&["root"]);
    nesting.extend((0..66).map(|_| OsString::from("!")));
    nesting.push("-true".into());
    assert!(parse(&nesting).is_err());
    let mut nodes = argv(&["root"]);
    nodes.extend((0..300).map(|_| OsString::from("-true")));
    assert!(parse(&nodes).is_err());
    assert!(parse(
        &(0..4097)
            .map(|_| OsString::from("root"))
            .collect::<Vec<_>>()
    )
    .is_err());
    for value in ["off", "0"] {
        env::set_var("FIND2FD_ACCELERATION", value);
        assert!(parse(&argv(&["root", "-print0"])).is_err());
    }
    env::remove_var("FIND2FD_ACCELERATION");
    env::set_var("POSIXLY_CORRECT", "");
    assert!(parse(&argv(&["root", "-print0"])).is_err());
    env::remove_var("POSIXLY_CORRECT");
    INHERITED_SIGPIPE_IGNORED.store(true, Ordering::Relaxed);
    assert!(parse(&argv(&["root", "-print0"])).is_err());
}
#[test]
fn implicit_print_and_explicit_actions_follow_independent_ordered_examples() {
    let _environment = environment();
    for (expression, truth, bytes) in [
        (&["-false"][..], false, &b""[..]),
        (&["-true"][..], true, &b"dir/foo\n"[..]),
        (&["-false", "-print0"][..], false, &b""[..]),
        (&["-print0", "-a", "-false"][..], false, &b"dir/foo\0"[..]),
        (&["-true", "-o", "-print0"][..], true, &b""[..]),
        (&["-false", "-o", "-print0"][..], true, &b"dir/foo\0"[..]),
        (&["-print0", ",", "-false"][..], false, &b"dir/foo\0"[..]),
        (
            &["!", "(", "-print0", "-a", "-false", ")"][..],
            true,
            &b"dir/foo\0"[..],
        ),
        (
            &["-print", "-print0", "-print"][..],
            true,
            &b"dir/foo\ndir/foo\0dir/foo\n"[..],
        ),
    ] {
        assert_eq!(
            evaluate(expression),
            (truth, bytes.to_vec()),
            "{expression:?}"
        );
    }
}
#[test]
fn boolean_precedence_exhausts_independent_truth_tables() {
    let _environment = environment();
    for a in [false, true] {
        for b in [false, true] {
            for c in [false, true] {
                let atom = |value| if value { "-true" } else { "-false" };
                for (expression, expected) in [
                    (vec![atom(a), "-o", atom(b), "-a", atom(c)], a || (b && c)),
                    (
                        vec!["(", atom(a), "-o", atom(b), ")", "-a", atom(c)],
                        (a || b) && c,
                    ),
                    (vec![atom(a), ",", atom(b), "-o", atom(c)], b || c),
                    (
                        vec!["!", "(", atom(a), "-o", atom(b), ")", "-a", atom(c)],
                        !(a || b) && c,
                    ),
                ] {
                    let (actual, bytes) = evaluate(&expression);
                    assert_eq!(actual, expected);
                    assert_eq!(
                        bytes,
                        if expected {
                            b"dir/foo\n".to_vec()
                        } else {
                            Vec::new()
                        }
                    );
                }
            }
        }
    }
}
fn words(alphabet: &[u8], depth: usize) -> Vec<Vec<u8>> {
    let mut all = vec![Vec::new()];
    let mut level = vec![Vec::new()];
    for _ in 0..depth {
        let mut next = Vec::new();
        for prefix in &level {
            for byte in alphabet {
                let mut word = prefix.clone();
                word.push(*byte);
                next.push(word);
            }
        }
        all.extend(next.iter().cloned());
        level = next;
    }
    all
}
#[test]
fn specialized_globs_exhaust_short_raw_strings_against_libc_oracle() {
    let _environment = environment();
    let patterns = words(b"aB*", 4);
    let names = words(&[b'a', b'b', b'B', 0xff, b'\n'], 3);
    for pattern in &patterns {
        for fold in [false, true] {
            let glob = Glob::new(pattern, fold).unwrap();
            let reference = CString::new(pattern.clone()).unwrap();
            for name in &names {
                let name_c = CString::new(name.clone()).unwrap();
                let expected = unsafe {
                    libc::fnmatch(
                        reference.as_ptr(),
                        name_c.as_ptr(),
                        if fold { libc::FNM_CASEFOLD } else { 0 },
                    )
                } == 0;
                assert_eq!(
                    glob.matches(name),
                    expected,
                    "{pattern:?} on {name:?}, fold={fold}"
                );
            }
        }
    }
}
#[test]
fn glob_edge_examples_complex_patterns_and_locale_rejection() {
    let _environment = environment();
    for (pattern, value, fold, expected) in [
        (&b""[..], &b""[..], false, true),
        (&b"*"[..], &b"\xff\n"[..], false, true),
        (&b"pre*"[..], &b"prefix"[..], false, true),
        (&b"*fix"[..], &b"prefix"[..], false, true),
        (&b"*ref*"[..], &b"prefix"[..], false, true),
        (&b"*ref*"[..], &b"preFIX"[..], true, true),
        (&b"literal\\*"[..], &b"literal*"[..], false, true),
        (&b"[a-c]?"[..], &b"bZ"[..], false, true),
        (&b"[!a-z]*"[..], &b"foo"[..], false, false),
        (&b"a**B"[..], &b"a\xff\nB"[..], false, true),
    ] {
        assert_eq!(Glob::new(pattern, fold).unwrap().matches(value), expected);
    }
    assert!(Glob::new(&vec![b'*'; 8193], false).is_err());
    assert!(Glob::new(&vec![b'*'; 8192], false).is_ok());
    env::set_var("LC_ALL", "C.UTF-8");
    assert!(Glob::new(b"foo*", false).is_ok());
    for (pattern, fold) in [
        (&b"foo*"[..], true),
        (&b"[a-z]"[..], false),
        ("π*".as_bytes(), false),
    ] {
        assert!(Glob::new(pattern, fold).is_err());
    }
}
#[test]
fn shared_budgets_accept_exact_limits_and_reject_one_more_without_large_allocations() {
    let _environment = environment();
    let budget = Budget::new();
    budget.entries(ENTRY_LIMIT).unwrap();
    assert_eq!(budget.entries(1), Err("entry count bound"));
    let budget = Budget::new();
    budget.reserve_output(OUTPUT_LIMIT).unwrap();
    assert_eq!(budget.reserve_output(1), Err("rendered output byte bound"));
    let budget = Budget::new();
    budget.reserve_discovery(DISCOVERY_BYTES_LIMIT).unwrap();
    assert_eq!(
        budget.reserve_discovery(1),
        Err("discovery path byte bound")
    );
    let budget = Budget {
        started: Instant::now() - Duration::from_secs(11),
        ..Budget::new()
    };
    assert_eq!(budget.entries(0), Err("traversal time bound"));
    let budget = Budget::new();
    budget.bytes.store(usize::MAX, Ordering::Relaxed);
    assert_eq!(budget.reserve_output(1), Err("rendered output byte bound"));
}
#[test]
fn output_credit_settlement_preserves_exact_payload_and_shared_remaining_capacity() {
    let _environment = environment();
    let budget = Budget::new();
    let mut first = Output::new();
    let mut second = Output::new();
    first.print(b"raw-\xff", 0, &budget).unwrap();
    second.print(b"x", b'\n', &budget).unwrap();
    assert_eq!(budget.bytes.load(Ordering::Relaxed), 2048);
    assert_eq!(first.into_bytes(&budget), b"raw-\xff\0");
    assert_eq!(budget.bytes.load(Ordering::Relaxed), 1030);
    assert_eq!(second.into_bytes(&budget), b"x\n");
    assert_eq!(budget.bytes.load(Ordering::Relaxed), 8);
    let mut output = Output::new();
    budget.bytes.store(OUTPUT_LIMIT - 1023, Ordering::Relaxed);
    assert!(output.print(b"x", 0, &budget).is_err());
    assert!(
        output.bytes.is_empty(),
        "budget rejection precedes visible bytes"
    );
}
fn record(inode: u64, name: &[u8], dtype: u8) -> Vec<u8> {
    let length = (19 + name.len() + 1 + 7) & !7;
    let mut bytes = vec![0; length];
    bytes[..8].copy_from_slice(&inode.to_ne_bytes());
    bytes[16..18].copy_from_slice(&(length as u16).to_ne_bytes());
    bytes[18] = dtype;
    bytes[19..19 + name.len()].copy_from_slice(name);
    bytes
}
#[test]
fn kernel_record_parser_preserves_raw_fields_and_exact_batch_boundaries() {
    let _environment = environment();
    for inode in [0, 1, u64::MAX] {
        for name in [&b"a"[..], &b"\xff\n"[..], &b"."[..], &b".."[..]] {
            let mut bytes = record(inode, name, libc::DT_REG);
            let length = bytes.len();
            bytes.extend(record(7, b"next", libc::DT_DIR));
            let parsed = directory_record(&bytes).unwrap();
            assert_eq!(parsed.inode, inode);
            assert_eq!(parsed.name, name);
            assert_eq!(parsed.dtype, libc::DT_REG);
            assert_eq!(parsed.length, length);
            let next = directory_record(&bytes[length..]).unwrap();
            assert_eq!(next.name, b"next");
        }
    }
}
#[test]
fn kernel_record_parser_rejects_truncation_invalid_lengths_empty_and_missing_nul() {
    let _environment = environment();
    let valid = record(1, b"filename", libc::DT_REG);
    for length in 0..20 {
        assert!(directory_record(&valid[..length]).is_err());
    }
    for length in [0, 1, 19, valid.len() + 1, 65535] {
        let mut bytes = valid.clone();
        bytes[16..18].copy_from_slice(&(length as u16).to_ne_bytes());
        assert!(directory_record(&bytes).is_err());
    }
    let mut bytes = valid.clone();
    bytes[19..].fill(b'x');
    assert!(directory_record(&bytes).is_err());
    let bytes = record(1, b"", libc::DT_REG);
    assert!(directory_record(&bytes).is_err());
}
