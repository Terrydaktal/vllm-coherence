//! Independent small contracts and seeded properties. External GNU comparisons
//! live in tests/gnu_oracle.rs and qualify the uninstrumented release artifact.
use super::*;
use std::os::unix::ffi::OsStringExt;

fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}
fn literal(patterns: &[&[u8]], ignore_case: bool) -> Options {
    Options {
        fixed: true,
        text: true,
        ignore_case,
        patterns: patterns.iter().map(|p| p.to_vec()).collect(),
        ..Options::default()
    }
}

#[test]
fn parser_preserves_raw_patterns_and_operand_order() {
    let pattern = OsString::from_vec(vec![0xff, b'a']);
    let path = OsString::from_vec(vec![b'p', 0xfe]);
    let o = parse(&[
        "-aF".into(),
        "-e".into(),
        pattern,
        "--".into(),
        path.clone(),
        "-looks-like-an-option".into(),
        path.clone(),
    ])
    .unwrap();
    assert_eq!(o.patterns, vec![vec![0xff, b'a']]);
    assert_eq!(
        o.operands,
        vec![path.clone(), "-looks-like-an-option".into(), path]
    );
}

#[test]
fn parser_distinguishes_empty_missing_and_multiline_patterns() {
    assert!(parse(&[]).is_err());
    assert!(parse(&args(&["-e"])).is_err());
    let empty = parse(&args(&["-aF", "-e", "", "file"])).unwrap();
    assert_eq!(empty.patterns, vec![Vec::<u8>::new()]);
    let multiple = parse(&args(&["-aF", "-e", "a\nb", "-e", "c", "file", "file"])).unwrap();
    assert_eq!(
        multiple.patterns,
        vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
    );
    assert_eq!(multiple.operands, args(&["file", "file"]));
}

#[test]
fn pattern_files_distinguish_empty_file_blank_record_and_final_lf() {
    let directory = tempfile::tempdir().unwrap();
    for (name, data, expected) in [
        ("empty", b"".as_slice(), vec![]),
        ("blank", b"\n".as_slice(), vec![vec![]]),
        (
            "unterminated",
            b"a\nb".as_slice(),
            vec![b"a".to_vec(), b"b".to_vec()],
        ),
        (
            "terminated",
            b"a\nb\n".as_slice(),
            vec![b"a".to_vec(), b"b".to_vec()],
        ),
    ] {
        let path = directory.path().join(name);
        fs::write(&path, data).unwrap();
        let mut o = Options::default();
        pattern_file(&mut o, path.as_os_str().as_bytes()).unwrap();
        assert_eq!(o.patterns, expected);
    }
    assert!(pattern_file(&mut Options::default(), b"-").is_err());
}

#[test]
fn parser_retains_ordered_option_precedence_and_rejects_unqualified_forms() {
    let o = parse(&args(&["-aF", "-l", "-L", "-H", "-h", "needle", "file"])).unwrap();
    assert_eq!(o.listing, Some(false));
    assert_eq!(o.prefix, Some(false));
    let reverse = parse(&args(&["-aF", "-L", "-l", "-h", "-H", "needle", "file"])).unwrap();
    assert_eq!(reverse.listing, Some(true));
    assert_eq!(reverse.prefix, Some(true));
    for argv in [
        vec!["--count=1", "x"],
        vec!["-FE", "x"],
        vec!["-aEo", "a|aa"],
        vec!["-aFowx", "x"],
        vec!["--unknown", "x"],
        vec!["-Fv", ""],
        vec!["-arF", "x", "tree//sub"],
    ] {
        assert!(parse(&args(&argv)).is_err(), "{argv:?}");
    }
}

#[test]
fn literal_extraction_selects_earliest_then_longest_and_skips_empty_matches() {
    let mut o = literal(&[b"", b"a", b"aa", b"aaa"], false);
    o.only = true;
    let matcher = Matcher::compile(&o).unwrap();
    let mut spans = Vec::new();
    matcher
        .extraction(b"baa aaa a", &o, |s, e| {
            spans.push((s, e));
            Ok(())
        })
        .unwrap();
    assert_eq!(spans, vec![(1, 3), (4, 7), (8, 9)]);
    let empty = literal(&[b""], false);
    let matcher = Matcher::compile(&empty).unwrap();
    let mut count = 0;
    matcher
        .extraction(b"abc", &empty, |_, _| {
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn whole_word_and_whole_record_qualification_uses_actual_record_boundaries() {
    let mut o = literal(&[b"needle"], false);
    o.word = true;
    let matcher = Matcher::compile(&o).unwrap();
    for (record, yes) in [
        (b"needle".as_slice(), true),
        (b"(needle)", true),
        (b"xneedle", false),
        (b"needlex", false),
        (b"needle_", false),
        (b"\xffneedle\xfe", true),
    ] {
        assert_eq!(matcher.selected(record, &o), yes);
    }
    o.word = false;
    o.whole = true;
    for (record, yes) in [
        (b"needle".as_slice(), true),
        (b"needle ", false),
        (b" needle", false),
        (b"needle\0", false),
    ] {
        assert_eq!(matcher.selected(record, &o), yes);
    }
}

#[test]
fn independent_literal_selection_property_over_raw_bytes() {
    let mut state = 0x5011_7e55u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let ascii_fold = |byte: u8| {
        if byte.is_ascii_uppercase() {
            byte + 32
        } else {
            byte
        }
    };
    let alphabet = [0xff, b'a', b'A', b'b', b'B', b'_', 0, b' ', b'\n'];
    for case in 0..2048 {
        let record: Vec<u8> = (0..next() % 48)
            .map(|_| alphabet[next() as usize % alphabet.len()])
            .collect();
        let pattern: Vec<u8> = (0..next() % 7)
            .map(|_| alphabet[next() as usize % alphabet.len()])
            .collect();
        let folded = next() & 1 != 0;
        let inverted = next() & 1 != 0;
        let mut o = literal(&[&pattern], folded);
        o.invert = inverted;
        let matcher = Matcher::compile(&o).unwrap();
        let expected = if pattern.is_empty() {
            true
        } else {
            record.windows(pattern.len()).any(|candidate| {
                candidate.iter().zip(&pattern).all(|(a, b)| {
                    if folded {
                        ascii_fold(*a) == ascii_fold(*b)
                    } else {
                        a == b
                    }
                })
            })
        };
        assert_eq!(
            matcher.selected(&record, &o),
            expected != inverted,
            "case {case}, record={record:?}, pattern={pattern:?}"
        );
    }
}

#[test]
fn scan_record_property_preserves_numbering_delimiters_and_original_bytes() {
    let mut state = 0x51ca_11adu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for case in 0..1024 {
        let nul = next() & 1 != 0;
        let delimiter = if nul { 0 } else { b'\n' };
        let invert = next() & 1 != 0;
        let counting = next() & 1 != 0;
        let mut data = Vec::new();
        for _ in 0..next() % 14 {
            let record = if next().is_multiple_of(3) {
                b"needle\xff".as_slice()
            } else {
                b"other\xfe".as_slice()
            };
            data.extend_from_slice(record);
            data.push(delimiter);
        }
        if !data.is_empty() && next() & 1 != 0 {
            data.pop();
        }
        let mut o = literal(&[b"needle"], false);
        o.number = true;
        o.null_records = nul;
        o.invert = invert;
        o.count = counting;
        let matcher = Matcher::compile(&o).unwrap();
        let mut actual = Vec::new();
        let selected = scan(&data, b"raw-name\xff", &o, &matcher, true, &mut actual).unwrap();
        let mut records: Vec<&[u8]> = if data.is_empty() {
            vec![]
        } else {
            data.split(|b| *b == delimiter).collect()
        };
        if data.last() == Some(&delimiter) {
            records.pop();
        }
        let mut expected = Vec::new();
        let mut count = 0;
        for (index, record) in records.iter().enumerate() {
            if record.windows(6).any(|p| p == b"needle") != invert {
                count += 1;
                if !counting {
                    expected.extend_from_slice(b"raw-name\xff:");
                    expected.extend_from_slice(format!("{}:", index + 1).as_bytes());
                    expected.extend_from_slice(record);
                    expected.push(delimiter);
                }
            }
        }
        if counting {
            expected.extend_from_slice(b"raw-name\xff:");
            expected.extend_from_slice(format!("{count}\n").as_bytes());
        }
        assert_eq!(actual, expected, "case {case}");
        assert_eq!(selected, count > 0, "case {case}");
    }
}

#[test]
fn no_phantom_records_for_empty_or_terminated_inputs() {
    let mut o = literal(&[b""], false);
    o.count = true;
    let matcher = Matcher::compile(&o).unwrap();
    for (data, expected) in [
        (b"".as_slice(), b"0\n".as_slice()),
        (b"a\n", b"1\n"),
        (b"\n", b"1\n"),
        (b"a", b"1\n"),
        (b"a\n\n", b"2\n"),
    ] {
        let mut out = Vec::new();
        scan(data, b"input", &o, &matcher, false, &mut out).unwrap();
        assert_eq!(out, expected);
    }
}

#[test]
fn output_spool_crosses_the_memory_boundary_without_reordering_bytes() {
    let mut spool = Spool::new();
    let first = vec![b'a'; OUTPUT_MEMORY];
    spool.write_all(&first).unwrap();
    assert!(matches!(spool, Spool::Memory(_)));
    spool.write_all(b"\0\xfftail").unwrap();
    assert!(matches!(spool, Spool::Disk(_)));
    let mut rendered = Vec::new();
    spool.render(&mut rendered).unwrap();
    let mut expected = first;
    expected.extend_from_slice(b"\0\xfftail");
    assert_eq!(rendered, expected);
    let mut repeated = Vec::new();
    spool.render(&mut repeated).unwrap();
    assert_eq!(repeated, expected);
}

#[test]
fn output_spool_propagates_publication_failure_without_reporting_success() {
    struct Failed;
    impl Write for Failed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    for size in [1, OUTPUT_MEMORY + 1] {
        let mut spool = Spool::new();
        spool.write_all(&vec![b'x'; size]).unwrap();
        assert_eq!(
            spool.render(&mut Failed).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}

#[test]
fn output_prefix_retains_raw_name_bytes_and_nul_separator() {
    let mut o = literal(&[b"needle"], false);
    o.number = true;
    o.null_names = true;
    o.null_records = true;
    let matcher = Matcher::compile(&o).unwrap();
    let mut out = Vec::new();
    emit_record(
        &mut out,
        b"name\xff\n",
        b"needle\xfe",
        17,
        &o,
        &matcher,
        true,
    )
    .unwrap();
    assert_eq!(out, b"name\xff\n\x0017:needle\xfe\0");
}
