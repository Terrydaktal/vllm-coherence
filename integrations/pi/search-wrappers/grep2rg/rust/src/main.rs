//! GNU-facing byte search. Unsupported syntax reaches GNU before search output.
//! Directory discovery uses OS enumeration order and ignores no hidden files.
use aho_corasick::AhoCorasick;
use memchr::{memchr, memchr_iter, memmem, memrchr};
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{meta::Regex, util::syntax, Input as RegexInput};
use std::collections::HashSet;
use std::env;
use std::ffi::{CString, OsStr, OsString};
use std::fs::{self, File, FileType};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Command;

const OUTPUT_MEMORY: usize = 256 * 1024;
const BATCH: usize = 256;

#[derive(Default)]
struct Options {
    fixed: bool,
    extended: bool,
    basic: bool,
    text: bool,
    recursive: bool,
    follow: bool,
    count: bool,
    listing: Option<bool>,
    invert: bool,
    number: bool,
    prefix: Option<bool>,
    quiet: bool,
    word: bool,
    whole: bool,
    only: bool,
    null_records: bool,
    null_names: bool,
    ignore_case: bool,
    patterns: Vec<Vec<u8>>,
    operands: Vec<OsString>,
    include: Vec<CString>,
    exclude_dirs: Vec<OsString>,
    max_count: Option<usize>,
    validate_ascii: bool,
}

fn option_value(
    args: &[OsString],
    index: &mut usize,
    rest: &[u8],
) -> Result<Vec<u8>, &'static str> {
    if !rest.is_empty() {
        return Ok(rest.to_vec());
    }
    *index += 1;
    args.get(*index)
        .map(|x| x.as_bytes().to_vec())
        .ok_or("missing option argument")
}

fn max_count(value: &[u8]) -> Result<usize, &'static str> {
    std::str::from_utf8(value)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0 && n <= isize::MAX as usize)
        .ok_or("GNU max-count boundary")
}

fn add_pattern(options: &mut Options, value: &[u8]) {
    // argv -e empty is one empty pattern; an empty pattern file is zero patterns.
    options
        .patterns
        .extend(value.split(|&b| b == b'\n').map(Vec::from));
}

fn pattern_file(options: &mut Options, value: &[u8]) -> Result<(), &'static str> {
    if value == b"-" {
        return Err("stdin pattern file");
    }
    let path = Path::new(OsStr::from_bytes(value));
    if !fs::metadata(path)
        .map_err(|_| "pattern file metadata")?
        .is_file()
    {
        return Err("special pattern file");
    }
    let data = fs::read(path).map_err(|_| "pattern file read")?;
    if data.is_empty() {
        return Ok(());
    }
    let data = data.strip_suffix(b"\n").unwrap_or(&data);
    add_pattern(options, data);
    Ok(())
}

fn parse(args: &[OsString]) -> Result<Options, &'static str> {
    let mut o = Options::default();
    let mut operands = Vec::new();
    let mut explicit_pattern = false;
    let mut end_options = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_bytes();
        if !end_options && arg == b"--" {
            end_options = true;
            i += 1;
            continue;
        }
        if !end_options && arg.starts_with(b"--") {
            let (name, value) = match arg.iter().position(|&x| x == b'=') {
                Some(p) => (&arg[2..p], Some(&arg[p + 1..])),
                None => (&arg[2..], None),
            };
            let short = match name {
                b"fixed-strings" => Some(b'F'),
                b"extended-regexp" => Some(b'E'),
                b"basic-regexp" => Some(b'G'),
                b"text" => Some(b'a'),
                b"recursive" => Some(b'r'),
                b"dereference-recursive" => Some(b'R'),
                b"count" => Some(b'c'),
                b"files-with-matches" => Some(b'l'),
                b"files-without-match" => Some(b'L'),
                b"invert-match" => Some(b'v'),
                b"line-number" => Some(b'n'),
                b"with-filename" => Some(b'H'),
                b"no-filename" => Some(b'h'),
                b"quiet" | b"silent" => Some(b'q'),
                b"word-regexp" => Some(b'w'),
                b"line-regexp" => Some(b'x'),
                b"only-matching" => Some(b'o'),
                b"null-data" => Some(b'z'),
                b"null" => Some(b'Z'),
                b"ignore-case" => Some(b'i'),
                _ => None,
            };
            if let Some(flag) = short {
                if value.is_some() {
                    return Err("unexpected long-option value");
                }
                set_flag(&mut o, flag)?;
            } else {
                match name {
                    b"regexp" | b"file" | b"include" => {
                        let val = match value {
                            Some(v) => v.to_vec(),
                            None => option_value(args, &mut i, b"")?,
                        };
                        if name == b"regexp" {
                            explicit_pattern = true;
                            add_pattern(&mut o, &val);
                        } else if name == b"file" {
                            explicit_pattern = true;
                            pattern_file(&mut o, &val)?;
                        } else {
                            o.include
                                .push(CString::new(val).map_err(|_| "NUL include pattern")?);
                        }
                    }
                    b"exclude-dir" | b"max-count" => {
                        let val = match value {
                            Some(v) => v.to_vec(),
                            None => option_value(args, &mut i, b"")?,
                        };
                        if name == b"max-count" {
                            o.max_count = Some(max_count(&val)?);
                        } else {
                            // GNU matches path globs differently for command-line roots and children.
                            // Only literal directory basenames use this byte-comparison shortcut.
                            if val.is_empty()
                                || matches!(val.as_slice(), b"." | b"..")
                                || !val
                                    .iter()
                                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(b))
                            {
                                return Err("GNU directory exclusion pattern");
                            }
                            o.exclude_dirs.push(OsString::from_vec(val));
                        }
                    }
                    b"binary-files" if value == Some(b"text".as_slice()) => o.text = true,
                    b"binary-files" if value == Some(b"binary".as_slice()) => o.text = false,
                    b"color" | b"colour" if value == Some(b"never".as_slice()) => {}
                    b"no-ignore-case" if value.is_none() => o.ignore_case = false,
                    _ => return Err("unsupported long option"),
                }
            }
        } else if !end_options && arg.len() > 1 && arg[0] == b'-' {
            let mut j = 1;
            while j < arg.len() {
                if arg[j] == b'm' {
                    let value = option_value(args, &mut i, &arg[j + 1..])?;
                    o.max_count = Some(max_count(&value)?);
                    break;
                }
                if arg[j] == b'e' || arg[j] == b'f' {
                    explicit_pattern = true;
                    let val = option_value(args, &mut i, &arg[j + 1..])?;
                    if arg[j] == b'e' {
                        add_pattern(&mut o, &val);
                    } else {
                        pattern_file(&mut o, &val)?;
                    }
                    break;
                }
                set_flag(&mut o, arg[j])?;
                j += 1;
            }
        } else {
            operands.push(args[i].clone());
        }
        i += 1;
    }
    if !explicit_pattern {
        if operands.is_empty() {
            return Err("missing pattern");
        }
        add_pattern(&mut o, operands.remove(0).as_bytes());
    }
    o.operands = operands;
    if o.recursive
        && o.operands
            .iter()
            .any(|x| x.as_bytes().windows(2).any(|p| p == b"//"))
    {
        return Err("GNU recursive pathname normalization");
    }
    if !o.exclude_dirs.is_empty()
        && (!o.recursive
            || !o.include.is_empty()
            || o.operands.iter().any(|p| {
                p.as_bytes().ends_with(b"/")
                    || p.as_bytes().ends_with(b"/.")
                    || p.as_bytes().ends_with(b"/..")
            }))
    {
        return Err("GNU directory exclusion operand");
    }
    if o.fixed as usize + o.extended as usize + o.basic as usize > 1 {
        return Err("conflicting matchers");
    }
    if o.only && o.extended {
        return Err("POSIX longest regex extraction");
    }
    if o.only && o.whole && o.word {
        return Err("GNU whole-word extraction interaction");
    }
    if o.extended && o.word {
        return Err("POSIX word regex alternatives");
    }
    if o.patterns.iter().any(|x| x.contains(&0)) {
        return Err("NUL pattern");
    }
    if o.invert && o.patterns.iter().any(Vec::is_empty) {
        return Err("GNU inverted empty-pattern shortcut");
    }
    let locale = env::var_os("LC_ALL")
        .filter(|x| !x.is_empty())
        .or_else(|| env::var_os("LC_CTYPE").filter(|x| !x.is_empty()))
        .or_else(|| env::var_os("LANG").filter(|x| !x.is_empty()));
    let c_locale = locale.as_ref().is_none_or(|x| x == "C" || x == "POSIX");
    if !c_locale && (o.ignore_case || o.word || o.extended || !o.include.is_empty()) {
        return Err("locale-sensitive semantics");
    }
    if !c_locale && !o.text && o.listing.is_none() && !o.quiet {
        // In these UTF-8 locales a nonempty ASCII literal has byte-identical
        // matching on NUL-free ASCII files. Validate owned inputs before output.
        let utf8 = locale.as_deref().is_some_and(|name| {
            [
                b"C.UTF-8".as_slice(),
                b"C.utf8",
                b"en_GB.UTF-8",
                b"en_GB.utf8",
            ]
            .iter()
            .any(|valid| name.as_bytes().eq_ignore_ascii_case(valid))
        });
        if !utf8
            || o.extended
            || o.only
            || o.invert
            || o.whole
            || o.null_records
            || o.patterns.len() != 1
            || o.patterns[0].is_empty()
            || !o.patterns[0].is_ascii()
            || (o.operands.is_empty() && !o.recursive)
            || o.operands.iter().any(|x| x == "-")
        {
            return Err("locale-sensitive binary detection");
        }
        o.validate_ascii = true;
    }
    if o.extended {
        if o.patterns.iter().any(|p| {
            p.iter().any(|b| !b.is_ascii() || b"[\\{}".contains(b))
                || p.windows(2)
                    .any(|x| x == b"(?" || x.iter().all(|b| b"*+?".contains(b)))
        }) {
            return Err("unsupported POSIX ERE grammar");
        }
    } else if !o.fixed
        && o.patterns
            .iter()
            .any(|p| p.iter().any(|b| b".*^$[\\".contains(b)))
    {
        return Err("basic regular expression");
    }
    Ok(o)
}

fn set_flag(o: &mut Options, flag: u8) -> Result<(), &'static str> {
    match flag {
        b'F' => o.fixed = true,
        b'E' => o.extended = true,
        b'G' => o.basic = true,
        b'a' => o.text = true,
        b'r' => o.recursive = true,
        b'R' => {
            o.recursive = true;
            o.follow = true;
        }
        b'c' => o.count = true,
        b'l' => o.listing = Some(true),
        b'L' => o.listing = Some(false),
        b'v' => o.invert = true,
        b'n' => o.number = true,
        b'H' => o.prefix = Some(true),
        b'h' => o.prefix = Some(false),
        b'q' => o.quiet = true,
        b'w' => o.word = true,
        b'x' => o.whole = true,
        b'o' => o.only = true,
        b'z' => o.null_records = true,
        b'Z' => o.null_names = true,
        b'i' | b'y' => o.ignore_case = true,
        b'U' | b's' => {}
        _ => return Err("unsupported short option"),
    }
    Ok(())
}

#[derive(Clone)]
enum Matcher {
    Never,
    Any,
    Literal {
        bytes: Vec<u8>,
        finder: Box<memmem::Finder<'static>>,
    },
    Literals {
        patterns: Vec<Vec<u8>>,
        ac: AhoCorasick,
        finders: Option<Vec<memmem::Finder<'static>>>,
    },
    Regex(Regex),
}

impl Matcher {
    fn compile(o: &Options) -> Result<Self, &'static str> {
        if o.patterns.is_empty() {
            return Ok(Self::Never);
        }
        if o.extended {
            // Literal ERE alternatives form a literal language.
            if o.patterns
                .iter()
                .all(|p| p.iter().all(|b| !b".*+?()^$".contains(b)))
            {
                let alternatives: Vec<Vec<u8>> = o
                    .patterns
                    .iter()
                    .flat_map(|p| p.split(|&b| b == b'|').map(Vec::from))
                    .collect();
                return Self::literals(&alternatives, o);
            }
            let patterns: Vec<String> = o
                .patterns
                .iter()
                .map(|p| String::from_utf8(p.clone()).unwrap())
                .collect();
            let mut pattern = patterns
                .iter()
                .map(|p| format!("(?:{p})"))
                .collect::<Vec<_>>()
                .join("|");
            if o.whole {
                pattern = format!("^(?:{pattern})$");
            }
            let regex = Regex::builder()
                .configure(
                    Regex::config()
                        .utf8_empty(false)
                        .nfa_size_limit(Some(16 * 1024 * 1024)),
                )
                .syntax(
                    syntax::Config::new()
                        .unicode(false)
                        .utf8(false)
                        .multi_line(!o.null_records)
                        .dot_matches_new_line(o.null_records)
                        .case_insensitive(o.ignore_case),
                )
                .build(&pattern)
                .map_err(|_| "regex compilation")?;
            return Ok(Self::Regex(regex));
        }
        Self::literals(&o.patterns, o)
    }
    fn literals(patterns: &[Vec<u8>], o: &Options) -> Result<Self, &'static str> {
        if patterns.iter().any(Vec::is_empty) && !o.whole && !o.word && !o.only {
            return Ok(Self::Any);
        }
        if patterns.len() == 1 && !o.ignore_case {
            let bytes = patterns[0].clone();
            let finder = Box::new(memmem::Finder::new(&bytes).into_owned());
            Ok(Self::Literal { bytes, finder })
        } else {
            let extract = o.only && !o.word && !o.whole && patterns.iter().all(|p| !p.is_empty());
            let ac = AhoCorasick::builder()
                .match_kind(if extract {
                    aho_corasick::MatchKind::LeftmostLongest
                } else {
                    aho_corasick::MatchKind::Standard
                })
                .ascii_case_insensitive(o.ignore_case)
                .build(patterns)
                .map_err(|_| "literal compilation")?;
            Ok(Self::Literals {
                patterns: patterns.to_vec(),
                ac,
                finders: (!o.only && !o.ignore_case && patterns.len() <= 8).then(|| {
                    patterns
                        .iter()
                        .map(|p| memmem::Finder::new(p).into_owned())
                        .collect()
                }),
            })
        }
    }
    fn selected(&self, record: &[u8], o: &Options) -> bool {
        let yes = match self {
            Self::Never => false,
            Self::Any => true,
            Self::Regex(r) => r.is_match(record),
            Self::Literal { bytes, finder } => {
                if o.whole {
                    record == bytes
                } else if o.word {
                    let mut cursor = 0;
                    let mut found = false;
                    while cursor <= record.len() {
                        let Some(offset) = finder.find(&record[cursor..]) else {
                            break;
                        };
                        let s = cursor + offset;
                        if word_span(record, s, s + bytes.len()) {
                            found = true;
                            break;
                        }
                        cursor = s + 1;
                    }
                    found
                } else {
                    finder.find(record).is_some()
                }
            }
            Self::Literals { patterns, ac, .. } => {
                if o.whole {
                    patterns
                        .iter()
                        .any(|p| equal_literal(p, record, o.ignore_case))
                } else if o.word {
                    ac.find_overlapping_iter(record)
                        .any(|m| word_span(record, m.start(), m.end()))
                } else {
                    ac.is_match(record)
                }
            }
        };
        yes != o.invert
    }
    fn extraction(
        &self,
        record: &[u8],
        o: &Options,
        mut emit: impl FnMut(usize, usize) -> io::Result<()>,
    ) -> io::Result<()> {
        // POSIX literal extraction chooses the earliest, then longest accepted span.
        if let Self::Literals { ac, .. } = self {
            if ac.match_kind() == aho_corasick::MatchKind::LeftmostLongest {
                for m in ac.find_iter(record) {
                    emit(m.start(), m.end())?;
                }
                return Ok(());
            }
        }
        if let Self::Literal { bytes, finder } = self {
            if !bytes.is_empty() && !o.word && !o.whole {
                for s in finder.find_iter(record) {
                    emit(s, s + bytes.len())?;
                }
                return Ok(());
            }
        }
        let patterns: Vec<&[u8]> = match self {
            Self::Literal { bytes, .. } => vec![bytes],
            Self::Literals { patterns, .. } => patterns.iter().map(Vec::as_slice).collect(),
            _ => return Ok(()),
        };
        let mut start = 0;
        while start < record.len() {
            let mut best: Option<(usize, usize)> = None;
            for p in &patterns {
                if p.is_empty() {
                    continue;
                }
                let mut cursor = start;
                while cursor + p.len() <= record.len() {
                    let found = if o.ignore_case {
                        record[cursor..]
                            .windows(p.len())
                            .position(|x| equal_literal(p, x, true))
                    } else {
                        memmem::find(&record[cursor..], p)
                    };
                    let Some(offset) = found else {
                        break;
                    };
                    let s = cursor + offset;
                    let e = s + p.len();
                    if (!o.whole || (s == 0 && e == record.len()))
                        && (!o.word || word_span(&record[start..], s - start, e - start))
                    {
                        if best.is_none_or(|(bs, be)| s < bs || (s == bs && e > be)) {
                            best = Some((s, e));
                        }
                        break;
                    }
                    cursor = s + 1;
                }
            }
            let Some((s, e)) = best else {
                break;
            };
            emit(s, e)?;
            start = e;
        }
        Ok(())
    }
}
fn equal_literal(a: &[u8], b: &[u8], folded: bool) -> bool {
    a.len() == b.len()
        && if folded {
            a.eq_ignore_ascii_case(b)
        } else {
            a == b
        }
}
fn word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
fn word_span(record: &[u8], start: usize, end: usize) -> bool {
    (start == 0 || !word_byte(record[start - 1]))
        && (end == record.len() || !word_byte(record[end]))
}

enum Data {
    Bytes(Vec<u8>),
    Map(Mmap),
    Reader,
    Selected,
    Summary { matched: bool, output: Vec<u8> },
}
impl Data {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(b) => b,
            Self::Map(m) => m,
            Self::Reader | Self::Selected | Self::Summary { .. } => {
                panic!("streamed file requires worker buffer")
            }
        }
    }
}
struct Input {
    path: OsString,
    data: Option<Data>,
    len: u64,
}
impl Input {
    fn bytes(&self) -> &[u8] {
        self.data.as_ref().expect("preflighted input").bytes()
    }
}
struct Plan {
    files: Vec<Input>,
    stdin: bool,
    prefix: bool,
    bytes: u64,
    directory_operand: bool,
    quiet_match: bool,
    quiet_files: usize,
    pool: Option<rayon::ThreadPool>,
    workers: usize,
}

fn descriptors() -> Result<(u64, u64), &'static str> {
    unsafe {
        for fd in 0..=2 {
            if libc::fcntl(fd, libc::F_GETFD) == -1 {
                return Err("closed standard descriptor");
            }
            if libc::fcntl(fd, libc::F_GETFL) & libc::O_NONBLOCK != 0 {
                return Err("nonblocking standard descriptor");
            }
        }
        let mut out: libc::stat = std::mem::zeroed();
        let mut err: libc::stat = std::mem::zeroed();
        if libc::fstat(1, &mut out) != 0 || libc::fstat(2, &mut err) != 0 {
            return Err("descriptor metadata");
        }
        if out.st_mode & libc::S_IFMT == libc::S_IFDIR
            || err.st_mode & libc::S_IFMT == libc::S_IFDIR
        {
            return Err("directory output descriptor");
        }
        if out.st_mode & libc::S_IFMT == libc::S_IFCHR
            && libc::isatty(1) == 0
            && out.st_rdev != libc::makedev(1, 3)
        {
            return Err("special output descriptor");
        }
        Ok((out.st_dev, out.st_ino))
    }
}

fn plan(o: &Options, matcher: &Matcher) -> Result<Plan, &'static str> {
    let output = descriptors()?;
    let mut p = Plan {
        files: Vec::new(),
        stdin: false,
        prefix: false,
        bytes: 0,
        directory_operand: false,
        quiet_match: false,
        quiet_files: 0,
        pool: None,
        workers: 1,
    };
    if o.operands.is_empty() && !o.recursive || o.operands.len() == 1 && o.operands[0] == "-" {
        if o.max_count.is_some() {
            return Err("GNU max-count stdin positioning");
        }
        p.stdin = true;
        p.prefix = o.prefix.unwrap_or(false);
        let metadata = fs::metadata("/proc/self/fd/0").map_err(|_| "stdin metadata")?;
        if metadata.is_dir() || metadata.file_type().is_block_device() {
            return Err("special stdin descriptor");
        }
        if metadata.is_file() {
            return Err("seekable stdin descriptor");
        }
        if (o.quiet || o.listing.is_some())
            && (o.extended
                || o.ignore_case
                || o.word
                || o.whole
                || o.patterns.len() != 1
                || o.patterns[0].is_empty())
        {
            return Err("complex early-exit stdin matcher");
        }
        if metadata.dev() == output.0 && metadata.ino() == output.1 {
            return Err("input is output");
        }
        return Ok(p);
    }
    if o.operands.iter().any(|x| x == "-") {
        return Err("mixed stdin operands");
    }
    let operands = if o.operands.is_empty() {
        vec![OsString::from(".")]
    } else {
        o.operands.clone()
    };
    let mut ancestors = HashSet::new();
    let mut visit = |path: &Path, plan: &mut Plan| {
        // GNU strips the implicit "./" only when recursion supplied the root.
        let path = if o.operands.is_empty() {
            path.strip_prefix(".").unwrap_or(path)
        } else {
            path
        };
        if o.validate_ascii && (plan.files.len() >= 256 || !path.as_os_str().as_bytes().is_ascii())
        {
            return Err("UTF-8 input validation budget or pathname");
        }
        let mut input = Input {
            path: path.as_os_str().to_owned(),
            data: None,
            len: 0,
        };
        if !o.quiet {
            plan.files.push(input);
            return Ok(false);
        }
        // Before a quiet match no native output has escaped. A diagnostic
        // can still delegate the original invocation to GNU in operand order.
        load_input(&mut input, o, output, None)?;
        plan.bytes = plan.bytes.saturating_add(input.len);
        plan.quiet_files += 1;
        scan(
            input.bytes(),
            path.as_os_str().as_bytes(),
            o,
            matcher,
            false,
            &mut io::sink(),
        )
        .map_err(|_| "quiet file scan")
    };
    for name in &operands {
        if walk(
            Path::new(name),
            true,
            None,
            o,
            &mut p,
            &mut ancestors,
            &mut visit,
        )? {
            p.quiet_match = true;
            break;
        }
    }
    if o.quiet {
        return Ok(p);
    }
    p.prefix = o
        .prefix
        .unwrap_or(operands.len() > 1 || p.directory_operand && o.recursive);
    // Open a bounded sample first. It decides whether thread startup is useful
    // for the actual file sizes and whether filename matches finish immediately.
    let sampled = p.files.len().min(8);
    for input in &mut p.files[..sampled] {
        load_input(input, o, output, None)?;
    }
    let sample_bytes: u64 = p.files[..sampled].iter().map(|input| input.len).sum();
    let estimated_bytes = if sampled == 0 {
        0
    } else {
        sample_bytes.saturating_mul(p.files.len() as u64) / sampled as u64
    };
    let early_dense = o.listing.is_some()
        && !o.invert
        && sampled >= 6
        && p.files[..sampled]
            .iter()
            .filter(|input| {
                let data = input.bytes();
                let probe = &data[..data.len().min(4096)];
                memchr(if o.null_records { 0 } else { b'\n' }, probe)
                    .is_some_and(|end| matcher.selected(&probe[..end], o))
            })
            .count()
            >= 6;
    if p.files.len() >= 64 && estimated_bytes >= 8 * 1024 * 1024 {
        p.workers = std::thread::available_parallelism()
            .map_or(4, usize::from)
            .min(
                if matches!(matcher, Matcher::Regex(_)) && p.files.len() >= 512 {
                    16
                } else {
                    8
                },
            );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(p.workers)
            .build()
            .map_err(|_| "worker pool initialization")?;
        pool.install(|| {
            p.files[sampled..].par_iter_mut().try_for_each_init(
                SearchBuffer::default,
                |buffer, input| {
                    load_input(
                        input,
                        o,
                        output,
                        Some(WorkerInput {
                            matcher,
                            prefix: p.prefix,
                            early_dense,
                            buffer,
                        }),
                    )
                },
            )
        })?;
        p.pool = Some(pool);
    } else {
        for input in &mut p.files[sampled..] {
            load_input(input, o, output, None)?;
        }
    }
    p.bytes = p.files.iter().map(|input| input.len).sum();
    Ok(p)
}

#[derive(Default)]
struct SearchBuffer {
    bytes: Vec<u8>,
    regex: Option<regex_automata::meta::Cache>,
}

struct WorkerInput<'a> {
    matcher: &'a Matcher,
    prefix: bool,
    early_dense: bool,
    buffer: &'a mut SearchBuffer,
}

fn load_input(
    input: &mut Input,
    o: &Options,
    output: (u64, u64),
    worker: Option<WorkerInput<'_>>,
) -> Result<(), &'static str> {
    // All access, identity and binary checks finish before any native output.
    let file = File::open(&input.path).map_err(|_| "input open")?;
    let metadata = file.metadata().map_err(|_| "input metadata")?;
    if !metadata.is_file() {
        return Err("special input file");
    }
    if metadata.dev() == output.0 && metadata.ino() == output.1 {
        return Err("input is output");
    }
    if let Some(worker) = worker.as_ref().filter(|w| w.early_dense) {
        // A complete first record can establish a filename/quiet match without
        // mapping or copying the remainder of a dense input. No guessed match
        // or truncated-record test is used here.
        let mut first = [0u8; 4096];
        let read = loop {
            let n =
                unsafe { libc::pread(file.as_raw_fd(), first.as_mut_ptr().cast(), first.len(), 0) };
            if n >= 0 {
                break n as usize;
            }
            if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                return Err("input read");
            }
        };
        if let Some(end) = memchr(if o.null_records { 0 } else { b'\n' }, &first[..read]) {
            if worker.matcher.selected(&first[..end], o) {
                input.len = metadata.len();
                input.data = Some(Data::Selected);
                return Ok(());
            }
        }
    }
    let data = if o.validate_ascii {
        // Both limits cap speculative preflight at 64 MiB. Keep precisely
        // these bytes for scanning so a pathname change cannot bypass validation.
        const FILE_LIMIT: u64 = 256 * 1024;
        if metadata.len() > FILE_LIMIT {
            return Err("UTF-8 input validation budget");
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize + 1);
        file.take(FILE_LIMIT + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "input read")?;
        if bytes.len() as u64 > FILE_LIMIT || !bytes.is_ascii() {
            return Err("GNU UTF-8 binary detection");
        }
        Data::Bytes(bytes)
    } else if worker.is_some()
        && metadata.len() >= 4096
        && metadata.len() < 1024 * 1024
        && (o.text || o.listing.is_some() || o.quiet || o.null_records)
    {
        if o.count || o.listing.is_some() {
            // A count or filename is bounded output. Compute it while the
            // verified descriptor is open; publish in GNU order after preflight.
            let worker = worker.expect("parallel input worker");
            read_buffer(&file, metadata.len(), &mut worker.buffer.bytes)
                .map_err(|_| "input read")?;
            let mut output = Vec::new();
            let matched = scan_cached(
                &worker.buffer.bytes,
                input.path.as_bytes(),
                o,
                worker.matcher,
                worker.prefix,
                &mut output,
                &mut worker.buffer.regex,
            )
            .map_err(|_| "file summary")?;
            Data::Summary { matched, output }
        } else {
            Data::Reader
        }
    } else if metadata.len() < 4096 {
        let mut bytes = Vec::<u8>::with_capacity(metadata.len() as usize + 1);
        loop {
            if bytes.len() == bytes.capacity() {
                bytes.reserve(64 * 1024);
            }
            let spare = bytes.spare_capacity_mut();
            // read initializes exactly its positive return count; only those
            // bytes become visible through the vector's length.
            let n = unsafe { libc::read(file.as_raw_fd(), spare.as_mut_ptr().cast(), spare.len()) };
            if n == -1 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err("input read");
            }
            if n == 0 {
                break;
            }
            unsafe {
                bytes.set_len(bytes.len() + n as usize);
            }
        }
        Data::Bytes(bytes)
    } else {
        Data::Map(unsafe {
            memmap2::MmapOptions::new()
                .len(metadata.len() as usize)
                .map(&file)
                .map_err(|_| "input mmap")?
        })
    };
    if !o.text
        && o.listing.is_none()
        && !o.quiet
        && !o.null_records
        && memchr(0, data.bytes()).is_some()
    {
        return Err("GNU binary output semantics");
    }
    input.len = metadata.len();
    input.data = Some(data);
    Ok(())
}

fn walk(
    path: &Path,
    command_line: bool,
    known_type: Option<FileType>,
    o: &Options,
    p: &mut Plan,
    ancestors: &mut HashSet<(u64, u64)>,
    visit: &mut impl FnMut(&Path, &mut Plan) -> Result<bool, &'static str>,
) -> Result<bool, &'static str> {
    // Directory entries already carry d_type on common filesystems. Avoid a
    // serial lstat for regular files; opened descriptors are checked in preflight.
    let link = if known_type.is_some() {
        None
    } else {
        Some(fs::symlink_metadata(path).map_err(|_| "input metadata")?)
    };
    let kind = known_type.unwrap_or_else(|| link.as_ref().unwrap().file_type());
    if kind.is_symlink() && !command_line && !o.follow {
        return Ok(false);
    }
    let metadata = if kind.is_symlink() {
        Some(fs::metadata(path).map_err(|_| "input symlink target")?)
    } else if kind.is_dir() {
        Some(match link {
            Some(metadata) => metadata,
            None => fs::metadata(path).map_err(|_| "input metadata")?,
        })
    } else {
        None
    };
    let is_dir = metadata.as_ref().map_or(kind.is_dir(), |m| m.is_dir());
    let is_file = metadata.as_ref().map_or(kind.is_file(), |m| m.is_file());
    if is_dir {
        p.directory_operand |= command_line;
        if !o.recursive {
            return Err("directory operand");
        }
        let metadata = metadata.unwrap();
        if path
            .file_name()
            .is_some_and(|name| o.exclude_dirs.iter().any(|pattern| pattern == name))
        {
            return Ok(false);
        }
        let key = (metadata.dev(), metadata.ino());
        if !ancestors.insert(key) {
            return Err("recursive directory loop");
        }
        for entry in fs::read_dir(path).map_err(|_| "directory read")? {
            let entry = entry.map_err(|_| "directory entry read")?;
            let kind = entry.file_type().map_err(|_| "directory entry type")?;
            if walk(&entry.path(), false, Some(kind), o, p, ancestors, visit)? {
                return Ok(true);
            }
        }
        ancestors.remove(&key);
    } else {
        if !is_file {
            if !command_line && !o.follow {
                return Ok(false);
            }
            return Err("special input file");
        }
        if !o.include.is_empty() {
            let name = CString::new(path.file_name().unwrap_or(path.as_os_str()).as_bytes())
                .map_err(|_| "NUL input filename")?;
            if !o
                .include
                .iter()
                .any(|pattern| unsafe { libc::fnmatch(pattern.as_ptr(), name.as_ptr(), 0) == 0 })
            {
                return Ok(false);
            }
        }
        return visit(path, p);
    }
    Ok(false)
}

/// Per-file output rolls to an anonymous temporary file, keeping ordered parallel
/// rendering bounded even for a file consisting entirely of selected records.
enum Spool {
    Memory(Vec<u8>),
    Disk(BufWriter<File>),
}
impl Spool {
    fn new() -> Self {
        Self::Memory(Vec::new())
    }
    fn with_bytes<T>(&mut self, consume: impl FnOnce(&[u8]) -> io::Result<T>) -> io::Result<T> {
        match self {
            Self::Memory(data) => consume(data),
            Self::Disk(file) => {
                file.flush()?;
                // Only this process can write the anonymous spill.
                let map = unsafe { Mmap::map(file.get_ref())? };
                consume(&map)
            }
        }
    }
    fn render(&mut self, out: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Memory(data) => out.write_all(data),
            Self::Disk(file) => {
                file.seek(SeekFrom::Start(0))?;
                io::copy(file.get_mut(), out)?;
                Ok(())
            }
        }
    }
}
impl Write for Spool {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Self::Memory(data) = self {
            if data.len().saturating_add(buf.len()) <= OUTPUT_MEMORY {
                data.extend_from_slice(buf);
                return Ok(buf.len());
            }
            let mut file = tempfile::tempfile()?;
            file.write_all(data)?;
            *self = Self::Disk(BufWriter::with_capacity(64 * 1024, file));
        }
        if let Self::Disk(file) = self {
            file.write(buf)
        } else {
            unreachable!()
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Memory(_) => Ok(()),
            Self::Disk(file) => file.flush(),
        }
    }
}

fn label(
    out: &mut impl Write,
    name: &[u8],
    number: usize,
    o: &Options,
    prefix: bool,
) -> io::Result<()> {
    if prefix {
        out.write_all(name)?;
        out.write_all(if o.null_names { b"\0" } else { b":" })?;
    }
    if o.number {
        write!(out, "{number}:")?;
    }
    Ok(())
}

fn emit_record(
    out: &mut impl Write,
    name: &[u8],
    record: &[u8],
    number: usize,
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
) -> io::Result<()> {
    if o.only {
        if !o.invert {
            matcher.extraction(record, o, |s, e| {
                label(out, name, number, o, prefix)?;
                out.write_all(&record[s..e])?;
                out.write_all(if o.null_records { b"\0" } else { b"\n" })
            })?;
        }
    } else {
        label(out, name, number, o, prefix)?;
        out.write_all(record)?;
        out.write_all(if o.null_records { b"\0" } else { b"\n" })?;
    }
    Ok(())
}

fn finish(
    out: &mut impl Write,
    name: &[u8],
    count: usize,
    o: &Options,
    prefix: bool,
) -> io::Result<()> {
    if o.quiet {
        return Ok(());
    }
    if let Some(with_matches) = o.listing {
        if (count > 0) == with_matches {
            out.write_all(name)?;
            out.write_all(if o.null_names { b"\0" } else { b"\n" })?;
        }
    } else if o.count {
        if prefix {
            out.write_all(name)?;
            out.write_all(if o.null_names { b"\0" } else { b":" })?;
        }
        writeln!(out, "{count}")?;
    }
    Ok(())
}

fn scan(
    data: &[u8],
    name: &[u8],
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
) -> io::Result<bool> {
    scan_cached(data, name, o, matcher, prefix, out, &mut None)
}

fn scan_cached(
    data: &[u8],
    name: &[u8],
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
    cache: &mut Option<regex_automata::meta::Cache>,
) -> io::Result<bool> {
    let delimiter = if o.null_records { 0 } else { b'\n' };
    let early = o.quiet || o.listing.is_some();
    let mut count = 0;
    if o.invert && (!o.count || early || o.max_count.is_some()) {
        return scan_records(data, name, o, matcher, prefix, out);
    }
    if !o.only && !o.word {
        match matcher {
            Matcher::Regex(regex) if !o.null_records && !regex.is_match(b"".as_slice()) => {
                // An explicit local cache avoids lock contention per record.
                // Only the end of a match is needed to identify its record;
                // earliest half searches avoid computing POSIX-irrelevant spans.
                let cache = cache.get_or_insert_with(|| regex.create_cache());
                if early && !o.invert {
                    let found = regex
                        .search_half_with(cache, &RegexInput::new(data).earliest(true))
                        .is_some();
                    finish(out, name, usize::from(found), o, prefix)?;
                    return Ok(found);
                }
                let mut search_at = 0;
                let matches = std::iter::from_fn(|| {
                    if search_at >= data.len() {
                        return None;
                    }
                    let found = regex.search_half_with(
                        cache,
                        &RegexInput::new(data)
                            .range(search_at..data.len())
                            .earliest(true),
                    )?;
                    let hit = found.offset().checked_sub(1)?;
                    let end = memchr(delimiter, &data[hit..]).map_or(data.len(), |x| hit + x);
                    search_at = end.saturating_add(1);
                    Some((hit, Some(end)))
                });
                return scan_matches(data, name, o, matcher, prefix, out, matches);
            }
            Matcher::Literals {
                patterns,
                ac,
                finders,
            } if !o.whole
                && patterns
                    .iter()
                    .all(|p| !p.is_empty() && !p.contains(&delimiter)) =>
            {
                if let Some(finders) = finders {
                    let mut next: Vec<Option<usize>> =
                        finders.iter().map(|finder| finder.find(data)).collect();
                    let matches = std::iter::from_fn(|| {
                        let (index, start) = next
                            .iter()
                            .enumerate()
                            .filter_map(|(i, s)| s.map(|s| (i, s)))
                            .min_by_key(|(_, s)| *s)?;
                        let end = start + patterns[index].len();
                        next[index] = finders[index].find(&data[end..]).map(|s| end + s);
                        Some((start, None))
                    });
                    return scan_matches(data, name, o, matcher, prefix, out, matches);
                }
                return scan_matches(
                    data,
                    name,
                    o,
                    matcher,
                    prefix,
                    out,
                    ac.find_iter(data).map(|m| (m.start(), None)),
                );
            }
            _ => {}
        }
    }
    // Skip whole nonmatching records in the literal lane. A huge record never
    // needs allocating, and sparse files avoid invoking a matcher per line.
    if let Matcher::Literal { bytes, finder } = matcher {
        if !bytes.is_empty() && !bytes.contains(&delimiter) && !o.word && !o.whole && !o.only {
            let mut cursor = 0;
            let mut numbered_at = 0;
            let mut number = 1;
            while cursor < data.len() {
                let Some(offset) = finder.find(&data[cursor..]) else {
                    break;
                };
                let hit = cursor + offset;
                count += 1;
                if early && !o.invert {
                    break;
                }
                let end = memchr(delimiter, &data[hit..]).map_or(data.len(), |x| hit + x);
                if !o.invert && !o.count && o.listing.is_none() && !o.quiet {
                    let start =
                        memrchr(delimiter, &data[cursor..hit]).map_or(cursor, |x| cursor + x + 1);
                    if o.number {
                        number += memchr_iter(delimiter, &data[numbered_at..start]).count();
                        numbered_at = start;
                    }
                    emit_record(out, name, &data[start..end], number, o, matcher, prefix)?;
                }
                if o.max_count == Some(count) {
                    break;
                }
                cursor = end.saturating_add(1);
            }
            if o.invert {
                let records = memchr_iter(delimiter, data).count()
                    + usize::from(!data.is_empty() && data.last() != Some(&delimiter));
                count = records - count;
            }
            finish(out, name, count, o, prefix)?;
            return Ok(count > 0);
        }
    }
    scan_records(data, name, o, matcher, prefix, out)
}

fn scan_matches(
    data: &[u8],
    name: &[u8],
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
    matches: impl Iterator<Item = (usize, Option<usize>)>,
) -> io::Result<bool> {
    let delimiter = if o.null_records { 0 } else { b'\n' };
    let mut cursor = 0;
    let mut numbered_at = 0;
    let mut number = 1;
    let mut count = 0;
    for (hit, record_end) in matches {
        if hit < cursor {
            continue;
        }
        if hit == data.len() {
            break;
        }
        count += 1;
        if (o.quiet || o.listing.is_some()) && !o.invert {
            break;
        }
        let end = record_end
            .unwrap_or_else(|| memchr(delimiter, &data[hit..]).map_or(data.len(), |x| hit + x));
        if !o.invert && !o.count && o.listing.is_none() && !o.quiet {
            let start = memrchr(delimiter, &data[cursor..hit]).map_or(cursor, |x| cursor + x + 1);
            if o.number {
                number += memchr_iter(delimiter, &data[numbered_at..start]).count();
                numbered_at = start;
            }
            emit_record(out, name, &data[start..end], number, o, matcher, prefix)?;
        }
        if o.max_count == Some(count) {
            break;
        }
        cursor = end.saturating_add(1);
    }
    if o.invert {
        let records = memchr_iter(delimiter, data).count()
            + usize::from(!data.is_empty() && data.last() != Some(&delimiter));
        count = records - count;
    }
    finish(out, name, count, o, prefix)?;
    Ok(count > 0)
}

fn scan_records(
    data: &[u8],
    name: &[u8],
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
) -> io::Result<bool> {
    let delimiter = if o.null_records { 0 } else { b'\n' };
    let mut count = 0;
    let mut start = 0;
    let mut number = 1;
    while start < data.len() {
        let end = memchr(delimiter, &data[start..]).map_or(data.len(), |x| start + x);
        let record = &data[start..end];
        if matcher.selected(record, o) {
            count += 1;
            if o.quiet || o.listing.is_some() {
                break;
            }
            if !o.count {
                emit_record(out, name, record, number, o, matcher, prefix)?;
            }
            if o.max_count == Some(count) {
                break;
            }
        }
        start = end.saturating_add(1);
        number += 1;
    }
    finish(out, name, count, o, prefix)?;
    Ok(count > 0)
}

fn scan_stdin(
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
) -> io::Result<bool> {
    if o.quiet || o.listing.is_some() {
        if let Matcher::Literal { bytes, finder } = matcher {
            return scan_stdin_literal(o, bytes, finder, prefix, out);
        }
    }
    let stdin = io::stdin();
    let mut input = BufReader::with_capacity(64 * 1024, stdin.lock());
    let delimiter = if o.null_records { 0 } else { b'\n' };
    let mut pending: Option<Spool> = None;
    let mut count = 0;
    let mut number = 1;
    let mut record = |bytes: &[u8]| -> io::Result<bool> {
        if matcher.selected(bytes, o) {
            count += 1;
            if o.quiet || o.listing.is_some() {
                return Ok(true);
            }
            if !o.count {
                emit_record(out, b"(standard input)", bytes, number, o, matcher, prefix)?;
            }
        }
        number += 1;
        Ok(false)
    };
    loop {
        let data = input.fill_buf()?;
        if data.is_empty() {
            if let Some(mut partial) = pending {
                partial.with_bytes(&mut record)?;
            }
            break;
        }
        let length = if let Some(end) = memchr(delimiter, data) {
            let stop = if let Some(mut partial) = pending.take() {
                partial.write_all(&data[..end])?;
                partial.with_bytes(&mut record)?
            } else {
                record(&data[..end])?
            };
            if stop {
                break;
            }
            end + 1
        } else {
            // Most records borrow one input block. Bound heap use even for
            // an unbounded pipe that supplies no newline or NUL separator.
            pending.get_or_insert_with(Spool::new).write_all(data)?;
            data.len()
        };
        input.consume(length);
    }
    finish(out, b"(standard input)", count, o, prefix)?;
    Ok(count > 0)
}

fn scan_stdin_literal(
    o: &Options,
    bytes: &[u8],
    finder: &memmem::Finder<'_>,
    prefix: bool,
    out: &mut impl Write,
) -> io::Result<bool> {
    let stdin = io::stdin();
    let mut input = BufReader::with_capacity(64 * 1024, stdin.lock());
    let delimiter = if o.null_records { 0 } else { b'\n' };
    let mut tail = Vec::new();
    let keep = bytes.len().saturating_sub(1);
    let mut hit = false;
    let mut seen = false;
    let mut count = 0;
    loop {
        let data = input.fill_buf()?;
        if data.is_empty() {
            if seen && (hit != o.invert) {
                count = 1;
            }
            break;
        }
        let end = memchr(delimiter, data).unwrap_or(data.len());
        let ended = end < data.len();
        let part = &data[..end];
        seen |= !part.is_empty();
        if !hit {
            hit = finder.find(part).is_some();
            if !hit && !tail.is_empty() {
                let mut bridge = tail.clone();
                bridge.extend_from_slice(&part[..part.len().min(keep)]);
                hit = finder.find(&bridge).is_some();
            }
        }
        if hit && !o.invert {
            count = 1;
            break;
        }
        if ended {
            if hit != o.invert {
                count = 1;
                break;
            }
            tail.clear();
            hit = false;
            seen = false;
        } else if !hit {
            if part.len() >= keep {
                tail.clear();
                tail.extend_from_slice(&part[part.len() - keep..]);
            } else {
                tail.extend_from_slice(part);
                if tail.len() > keep {
                    tail.drain(..tail.len() - keep);
                }
            }
        }
        input.consume(end + usize::from(ended));
    }
    finish(out, b"(standard input)", count, o, prefix)?;
    Ok(count > 0)
}

fn report(engine: &str, strategy: &str, reason: &str, workers: usize, files: usize, bytes: u64) {
    if let Some(path) = env::var_os("GREP2RG_ENGINE_REPORT") {
        let value = serde_json::json!({"engine":engine,"strategy":strategy,"reason":reason,"workers":workers,"files":files,"bytes":bytes});
        if let Ok(mut file) = File::create(path) {
            let _ = writeln!(file, "{value}");
        }
    }
}

fn gnu(args: &[OsString], reason: &str) -> ! {
    report("gnu", "exec", reason, 1, 0, 0);
    let err = Command::new("/usr/bin/grep").arg0("grep").args(args).exec();
    eprintln!("grep2rg-rust: GNU grep: {err}");
    std::process::exit(2)
}

fn prepare_stdin(args: &[OsString], o: &Options) -> io::Result<Option<Spool>> {
    if o.listing.is_none() && !o.quiet && !o.text && !o.null_records {
        // A pipe is replayable after bounded spooling. Inspect before producing
        // output so GNU owns all binary diagnostics, including mixed text/binary.
        let mut spool = Spool::new();
        let mut binary = false;
        let stdin = io::stdin();
        let mut input = stdin.lock();
        let mut buffer = [0; 64 * 1024];
        loop {
            let n = input.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            binary |= memchr(0, &buffer[..n]).is_some();
            spool.write_all(&buffer[..n])?;
        }
        if binary && !o.text && !o.null_records {
            let mut replay = match spool {
                Spool::Disk(file) => file.into_inner().map_err(|e| e.into_error())?,
                Spool::Memory(data) => {
                    let mut file = tempfile::tempfile()?;
                    file.write_all(&data)?;
                    file
                }
            };
            replay.seek(SeekFrom::Start(0))?;
            if unsafe { libc::dup2(replay.as_raw_fd(), 0) } < 0 {
                return Err(io::Error::last_os_error());
            }
            gnu(args, "GNU binary stdin output semantics");
        }
        return Ok(Some(spool));
    }
    Ok(None)
}

fn read_buffer(file: &File, length: u64, buffer: &mut Vec<u8>) -> io::Result<()> {
    buffer.clear();
    let required = length as usize + 1;
    if buffer.capacity() < required {
        buffer.reserve_exact(required);
    }
    loop {
        if buffer.len() == buffer.capacity() {
            buffer.reserve(64 * 1024);
        }
        let spare = buffer.spare_capacity_mut();
        let n = unsafe { libc::read(file.as_raw_fd(), spare.as_mut_ptr().cast(), spare.len()) };
        if n == -1 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if n == 0 {
            break;
        }
        unsafe {
            buffer.set_len(buffer.len() + n as usize);
        }
    }
    Ok(())
}

fn scan_input(
    input: &Input,
    o: &Options,
    matcher: &Matcher,
    prefix: bool,
    out: &mut impl Write,
    buffer: &mut SearchBuffer,
) -> io::Result<bool> {
    if let Some(Data::Summary { matched, output }) = &input.data {
        out.write_all(output)?;
        return Ok(*matched);
    }
    if let Some(Data::Selected) = &input.data {
        finish(out, input.path.as_bytes(), 1, o, prefix)?;
        return Ok(true);
    }
    let data = if let Some(Data::Reader) = &input.data {
        let file = File::open(&input.path)?;
        read_buffer(&file, input.len, &mut buffer.bytes)?;
        &buffer.bytes
    } else {
        input.bytes()
    };
    scan_cached(
        data,
        input.path.as_bytes(),
        o,
        matcher,
        prefix,
        out,
        &mut buffer.regex,
    )
}

fn execute(
    o: &Options,
    matcher: &Matcher,
    plan: &Plan,
    workers: usize,
    replay: Option<Spool>,
) -> io::Result<bool> {
    if o.quiet && !plan.stdin {
        return Ok(plan.quiet_match);
    }
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(64 * 1024, stdout.lock());
    let mut selected = false;
    if plan.stdin {
        selected = match replay {
            Some(mut spool) => spool.with_bytes(|data| {
                scan(data, b"(standard input)", o, matcher, plan.prefix, &mut out)
            })?,
            None => scan_stdin(o, matcher, plan.prefix, &mut out)?,
        };
    } else if workers == 1
        || plan
            .files
            .iter()
            .skip(8)
            .all(|f| matches!(f.data, Some(Data::Summary { .. } | Data::Selected)))
    {
        // Precomputed summaries are already indexed in GNU order.
        let mut buffer = SearchBuffer::default();
        for input in &plan.files {
            selected |= scan_input(input, o, matcher, plan.prefix, &mut out, &mut buffer)?;
            if o.quiet && selected {
                break;
            }
        }
    } else {
        let pool = plan.pool.as_ref().expect("preflight worker pool");
        for batch in plan.files.chunks(BATCH) {
            let results = pool.install(|| {
                batch
                    .par_iter()
                    .with_min_len(16)
                    .map_init(SearchBuffer::default, |buffer, input| {
                        let mut output = Spool::new();
                        let matched =
                            scan_input(input, o, matcher, plan.prefix, &mut output, buffer)?;
                        Ok::<_, io::Error>((matched, output))
                    })
                    .collect::<Vec<_>>()
            });
            for result in results {
                let (matched, mut output) = result?;
                selected |= matched;
                output.render(&mut out)?;
            }
        }
    }
    out.flush()?;
    Ok(selected)
}

fn main() {
    mod app_version_runtime {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../build-support/app_version_runtime.rs"
        ));
    }
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    app_version_runtime::handle_cli(&args);
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    if env::var_os("POSIXLY_CORRECT").is_some() {
        gnu(&args, "POSIXLY_CORRECT option parsing");
    }
    if env::var_os("GREP2RG_ACCELERATION").is_some_and(|x| x == "gnu") {
        gnu(&args, "explicit GNU policy");
    }
    let o = parse(&args).unwrap_or_else(|reason| gnu(&args, reason));
    let matcher = Matcher::compile(&o).unwrap_or_else(|reason| gnu(&args, reason));
    let plan = plan(&o, &matcher).unwrap_or_else(|reason| gnu(&args, reason));
    let workers = plan.workers;
    let strategy = if workers > 1 {
        "parallel-buffered"
    } else if plan.stdin {
        if o.quiet || o.listing.is_some() || o.text || o.null_records {
            "stream"
        } else {
            "spooled-stdin"
        }
    } else {
        "sequential-mmap"
    };
    let replay = if plan.stdin {
        match prepare_stdin(&args, &o) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("grep: {e}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    report(
        "rust",
        strategy,
        "supported GNU byte semantics",
        workers,
        plan.files.len() + plan.quiet_files,
        plan.bytes,
    );
    match execute(&o, &matcher, &plan, workers, replay) {
        Ok(selected) => std::process::exit(if selected { 0 } else { 1 }),
        Err(error) => {
            eprintln!("grep: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests;
