//! GNU-facing physical filename discovery with ordered, bounded parallel walks.
//! Unsupported semantics and preparation failures exec GNU before any output.
use rayon::prelude::*;
use std::env;
use std::ffi::{CStr, CString, OsStr, OsString};
use std::fs::{self, File, FileType};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const OUTPUT_LIMIT: usize = 64 * 1024 * 1024;
const ENTRY_LIMIT: usize = 1_000_000;
const DIRECTORY_LIMIT: usize = 100_000;
const CHILD_LIMIT: usize = 10_000;
const DEPTH_LIMIT: usize = 256;
const PATH_LIMIT: usize = 4096;
const DISCOVERY_BYTES_LIMIT: usize = 128 * 1024 * 1024;
type Failure = &'static str;

// Rust's runtime ignores SIGPIPE before main. Capture its exec-inherited state
// first: silently replacing a caller's SIG_IGN with SIG_DFL changes GNU status.
static INHERITED_SIGPIPE_IGNORED: AtomicBool = AtomicBool::new(false);
unsafe extern "C" fn capture_sigpipe() {
    let mut action: libc::sigaction = std::mem::zeroed();
    if libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut action) == 0 {
        INHERITED_SIGPIPE_IGNORED.store(action.sa_sigaction == libc::SIG_IGN, Ordering::Relaxed);
    }
}
#[used]
#[link_section = ".preinit_array"]
static CAPTURE_SIGPIPE: unsafe extern "C" fn() = capture_sigpipe;

#[derive(Clone)]
enum Glob {
    All,
    Literal { pattern: Vec<u8>, fold: bool },
    Prefix { pattern: Vec<u8>, fold: bool },
    Suffix { pattern: Vec<u8>, fold: bool },
    Contains(Vec<u8>),
    Simple { pattern: Vec<u8>, fold: bool },
    Posix { pattern: CString, fold: bool },
}

impl Glob {
    fn new(pattern: &[u8], fold: bool) -> Result<Self, Failure> {
        if pattern.len() > 8192 {
            return Err("pattern byte bound");
        }
        let byte_locale = byte_locale();
        let complex = pattern.iter().any(|b| matches!(b, b'?' | b'[' | b'\\'));
        // GNU find uses the locale's fnmatch folding, including for non-ASCII
        // filenames matched against an ASCII pattern. Keep that behavior.
        let locale_fold =
            !byte_locale && fold && pattern.is_ascii() && !complex && utf8_case_locale();
        if !byte_locale && !locale_fold && (fold || complex || !pattern.is_ascii()) {
            return Err("GNU locale-sensitive pattern");
        }
        if complex || locale_fold {
            Ok(Self::Posix {
                pattern: CString::new(pattern).map_err(|_| "NUL pattern")?,
                fold,
            })
        } else {
            let stars = pattern.iter().filter(|b| **b == b'*').count();
            Ok(if stars == 0 {
                Self::Literal {
                    pattern: pattern.to_vec(),
                    fold,
                }
            } else if stars == pattern.len() {
                Self::All
            } else if stars == 1 && pattern.ends_with(b"*") {
                Self::Prefix {
                    pattern: pattern[..pattern.len() - 1].to_vec(),
                    fold,
                }
            } else if stars == 1 && pattern.starts_with(b"*") {
                Self::Suffix {
                    pattern: pattern[1..].to_vec(),
                    fold,
                }
            } else if stars == 2 && !fold && pattern.starts_with(b"*") && pattern.ends_with(b"*") {
                Self::Contains(pattern[1..pattern.len() - 1].to_vec())
            } else {
                Self::Simple {
                    pattern: pattern.to_vec(),
                    fold,
                }
            })
        }
    }

    fn matches(&self, value: &[u8]) -> bool {
        match self {
            Self::All => true,
            Self::Literal { pattern, fold } => equal_bytes(pattern, value, *fold),
            Self::Prefix { pattern, fold } => value
                .get(..pattern.len())
                .is_some_and(|prefix| equal_bytes(pattern, prefix, *fold)),
            Self::Suffix { pattern, fold } => value
                .len()
                .checked_sub(pattern.len())
                .is_some_and(|start| equal_bytes(pattern, &value[start..], *fold)),
            Self::Contains(pattern) => unsafe {
                !libc::memmem(
                    value.as_ptr().cast(),
                    value.len(),
                    pattern.as_ptr().cast(),
                    pattern.len(),
                )
                .is_null()
            },
            Self::Simple { pattern, fold } => simple_glob(pattern, value, *fold),
            Self::Posix { pattern, fold } => {
                let value = CString::new(value).expect("OS path has no NUL");
                unsafe {
                    libc::fnmatch(
                        pattern.as_ptr(),
                        value.as_ptr(),
                        if *fold { libc::FNM_CASEFOLD } else { 0 },
                    ) == 0
                }
            }
        }
    }
}

fn equal_bytes(a: &[u8], b: &[u8], fold: bool) -> bool {
    if fold {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

fn simple_glob(pattern: &[u8], value: &[u8], fold: bool) -> bool {
    let equal = |a: u8, b: u8| {
        if fold {
            a.eq_ignore_ascii_case(&b)
        } else {
            a == b
        }
    };
    let (mut p, mut v, mut star, mut retry) = (0, 0, None, 0);
    while v < value.len() {
        if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            retry = v;
        } else if p < pattern.len() && equal(pattern[p], value[v]) {
            p += 1;
            v += 1;
        } else if let Some(previous) = star {
            retry += 1;
            v = retry;
            p = previous + 1;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

fn utf8_case_locale() -> bool {
    let locale = unsafe { libc::setlocale(libc::LC_CTYPE, std::ptr::null()) };
    !locale.is_null()
        && matches!(
            unsafe { CStr::from_ptr(locale) }.to_bytes(),
            b"C.UTF-8" | b"C.utf8" | b"en_GB.UTF-8" | b"en_GB.utf8"
        )
}

fn byte_locale() -> bool {
    env::var_os("LC_ALL")
        .filter(|s| !s.is_empty())
        .or_else(|| env::var_os("LC_CTYPE").filter(|s| !s.is_empty()))
        .or_else(|| env::var_os("LANG").filter(|s| !s.is_empty()))
        .is_none_or(|s| s == "C" || s == "POSIX")
}

enum Expr {
    True,
    False,
    Type(u8),
    Name(Glob),
    Path(Glob),
    Print(u8),
    Prune,
    Quit,
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Comma(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn evaluate(
        &self,
        entry: &Entry,
        output: &mut Output,
        budget: &Budget,
    ) -> Result<bool, Failure> {
        if output.quit {
            return Ok(false);
        }
        Ok(match self {
            Self::Prune => {
                output.prune = true;
                true
            }
            Self::Quit => {
                output.quit = true;
                false
            }
            Self::True => true,
            Self::False => false,
            Self::Type(kind) => *kind == entry.kind,
            Self::Name(glob) => glob.matches(entry.name()),
            Self::Path(glob) => glob.matches(&entry.path),
            Self::Print(separator) => {
                output.print(&entry.path, *separator, budget)?;
                true
            }
            Self::Not(child) => !child.evaluate(entry, output, budget)?,
            Self::And(left, right) => {
                left.evaluate(entry, output, budget)? && right.evaluate(entry, output, budget)?
            }
            Self::Or(left, right) => {
                left.evaluate(entry, output, budget)? || right.evaluate(entry, output, budget)?
            }
            Self::Comma(left, right) => {
                left.evaluate(entry, output, budget)?;
                right.evaluate(entry, output, budget)?
            }
        })
    }
}

struct Options {
    roots: Vec<Vec<u8>>,
    min_depth: usize,
    max_depth: usize,
    expression: Expr,
    leaf_filter: Option<LeafFilter>,
    can_quit: bool,
    newline_output: bool,
}

// For pure basename/type queries, a rejected leaf never needs an owned path.
struct LeafFilter {
    name: Glob,
    kind: Option<u8>,
    separator: u8,
}

impl LeafFilter {
    fn from_expression(expression: &Expr) -> Option<Self> {
        let Expr::And(test, action) = expression else {
            return None;
        };
        let Expr::Print(separator) = **action else {
            return None;
        };
        let (name, kind) = match &**test {
            Expr::Name(name) => (name, None),
            Expr::And(left, right) => match (&**left, &**right) {
                (Expr::Name(name), Expr::Type(kind)) | (Expr::Type(kind), Expr::Name(name)) => {
                    (name, Some(*kind))
                }
                _ => return None,
            },
            _ => return None,
        };
        Some(Self {
            name: name.clone(),
            kind,
            separator,
        })
    }
    fn matches(&self, record: &DirectoryRecord, kind: u8) -> bool {
        if self.kind.is_some_and(|wanted| kind != wanted) {
            return false;
        }
        if let Glob::Posix { pattern, fold } = &self.name {
            // directory_record validates the NUL immediately after this name.
            return unsafe {
                libc::fnmatch(
                    pattern.as_ptr(),
                    record.name.as_ptr().cast(),
                    if *fold { libc::FNM_CASEFOLD } else { 0 },
                ) == 0
            };
        }
        self.name.matches(record.name)
    }
}

struct Parser<'a> {
    args: &'a [OsString],
    index: usize,
    nodes: usize,
    explicit_print: bool,
    can_quit: bool,
    newline_output: bool,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&[u8]> {
        self.args.get(self.index).map(|s| s.as_bytes())
    }
    fn consume(&mut self) -> Result<&[u8], Failure> {
        let value = self
            .args
            .get(self.index)
            .ok_or("missing GNU expression argument")?;
        self.index += 1;
        Ok(value.as_bytes())
    }
    fn node(&mut self, expression: Expr) -> Result<Expr, Failure> {
        self.nodes += 1;
        if self.nodes > 256 {
            Err("expression node bound")
        } else {
            Ok(expression)
        }
    }
    fn primary(&mut self, depth: usize) -> Result<Expr, Failure> {
        if depth > 64 {
            return Err("expression nesting bound");
        }
        let flag = self.consume()?.to_vec();
        let expression = match flag.as_slice() {
            b"!" | b"-not" => Expr::Not(Box::new(self.primary(depth + 1)?)),
            b"(" => {
                let expression = self.comma(depth + 1)?;
                if self.consume()? != b")" {
                    return Err("unmatched GNU expression group");
                }
                expression
            }
            b"-true" => Expr::True,
            b"-false" => Expr::False,
            b"-type" => {
                let value = self.consume()?;
                if value.len() != 1 || !b"fdlpsbc".contains(&value[0]) {
                    return Err("GNU type argument");
                }
                Expr::Type(value[0])
            }
            b"-name" | b"-iname" | b"-path" | b"-ipath" | b"-wholename" | b"-iwholename" => {
                let path = !matches!(flag.as_slice(), b"-name" | b"-iname");
                let fold = matches!(flag.as_slice(), b"-iname" | b"-ipath" | b"-iwholename");
                let value = self.consume()?;
                if (!path && value.contains(&b'/')) || (path && value.ends_with(b"/")) {
                    return Err("GNU pathname pattern warnings");
                }
                let glob = Glob::new(value, fold)?;
                if path {
                    Expr::Path(glob)
                } else {
                    Expr::Name(glob)
                }
            }
            b"-prune" => Expr::Prune,
            b"-quit" => {
                self.can_quit = true;
                Expr::Quit
            }
            b"-print" | b"-print0" => {
                self.explicit_print = true;
                let separator = if flag == b"-print" {
                    self.newline_output = true;
                    b'\n'
                } else {
                    0
                };
                Expr::Print(separator)
            }
            _ => return Err("unsupported GNU predicate, option, action or control"),
        };
        self.node(expression)
    }
    fn and(&mut self, depth: usize) -> Result<Expr, Failure> {
        let mut expression = self.primary(depth)?;
        while self
            .peek()
            .is_some_and(|flag| !matches!(flag, b"-o" | b"-or" | b")" | b","))
        {
            if self
                .peek()
                .is_some_and(|flag| matches!(flag, b"-a" | b"-and"))
            {
                self.index += 1;
            }
            let right = self.primary(depth)?;
            expression = self.node(Expr::And(Box::new(expression), Box::new(right)))?;
        }
        Ok(expression)
    }
    fn or(&mut self, depth: usize) -> Result<Expr, Failure> {
        let mut expression = self.and(depth)?;
        while self
            .peek()
            .is_some_and(|flag| matches!(flag, b"-o" | b"-or"))
        {
            self.index += 1;
            let right = self.and(depth)?;
            expression = self.node(Expr::Or(Box::new(expression), Box::new(right)))?;
        }
        Ok(expression)
    }
    fn comma(&mut self, depth: usize) -> Result<Expr, Failure> {
        let mut expression = self.or(depth)?;
        while self.peek() == Some(b",") {
            self.index += 1;
            let right = self.or(depth)?;
            expression = self.node(Expr::Comma(Box::new(expression), Box::new(right)))?;
        }
        Ok(expression)
    }
}

fn parse(args: &[OsString]) -> Result<Options, Failure> {
    if INHERITED_SIGPIPE_IGNORED.load(Ordering::Relaxed) {
        return Err("inherited ignored SIGPIPE GNU I/O policy");
    }
    if env::var_os("POSIXLY_CORRECT").is_some() {
        return Err("POSIXLY_CORRECT GNU policy");
    }
    if env::var_os("FIND2FD_ACCELERATION").is_some_and(|s| s == "off" || s == "0") {
        return Err("explicit GNU routing");
    }
    let mut parser = Parser {
        args,
        index: 0,
        nodes: 0,
        explicit_print: false,
        can_quit: false,
        newline_output: false,
    };
    if parser.peek() == Some(b"-P") {
        parser.index += 1;
    }
    let mut roots = Vec::new();
    while parser.peek().is_some_and(|value| {
        !value.starts_with(b"-") && !matches!(value, b"!" | b"(" | b")" | b",")
    }) {
        let root = parser.consume()?.to_vec();
        if root.is_empty() || root.len() >= PATH_LIMIT {
            return Err("empty or excessive starting path");
        }
        roots.push(root);
        if roots.len() > 4096 {
            return Err("starting path count bound");
        }
    }
    if roots.is_empty() {
        roots.push(b".".to_vec());
    }
    let (mut min_depth, mut max_depth) = (0, usize::MAX);
    while parser
        .peek()
        .is_some_and(|s| matches!(s, b"-mindepth" | b"-maxdepth"))
    {
        let minimum = parser.consume()? == b"-mindepth";
        let text = parser.consume()?;
        if text.is_empty() || !text.iter().all(u8::is_ascii_digit) || text.len() > 10 {
            return Err("GNU depth validation");
        }
        let depth = std::str::from_utf8(text)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n <= 1_000_000)
            .ok_or("GNU depth value bound")?;
        if minimum {
            min_depth = depth;
        } else {
            max_depth = depth;
        }
    }
    let mut expression = if parser.peek().is_none() {
        Expr::True
    } else {
        parser.comma(0)?
    };
    if parser.index != args.len() {
        return Err("GNU expression syntax");
    }
    if !parser.explicit_print {
        expression = Expr::And(Box::new(expression), Box::new(Expr::Print(b'\n')));
        parser.newline_output = true;
    }
    Ok(Options {
        roots,
        min_depth,
        max_depth,
        leaf_filter: LeafFilter::from_expression(&expression),
        expression,
        can_quit: parser.can_quit,
        newline_output: parser.newline_output,
    })
}

struct Entry {
    path: Vec<u8>,
    name_range: (usize, usize),
    kind: u8,
    depth: usize,
    parent_directory: Option<Arc<File>>,
    leaf_output: Option<u8>,
}
impl Entry {
    fn name(&self) -> &[u8] {
        &self.path[self.name_range.0..self.name_range.1]
    }
}
fn kind(file_type: FileType) -> u8 {
    if file_type.is_dir() {
        b'd'
    } else if file_type.is_file() {
        b'f'
    } else if file_type.is_symlink() {
        b'l'
    } else if file_type.is_fifo() {
        b'p'
    } else if file_type.is_socket() {
        b's'
    } else if file_type.is_block_device() {
        b'b'
    } else if file_type.is_char_device() {
        b'c'
    } else {
        b'?'
    }
}

fn root_entry(path: &[u8]) -> Result<Entry, Failure> {
    let metadata = fs::symlink_metadata(Path::new(OsStr::from_bytes(path)))
        .map_err(|_| "starting path metadata or access error")?;
    // GNU -name strips terminal slashes, even though -printf %f can display one.
    let end = path.iter().rposition(|b| *b != b'/').map_or(0, |p| p + 1);
    let name_range = if end == 0 {
        (0, 1)
    } else {
        let start = path[..end]
            .iter()
            .rposition(|b| *b == b'/')
            .map_or(0, |p| p + 1);
        (start, end)
    };
    Ok(Entry {
        path: path.to_vec(),
        name_range,
        kind: kind(metadata.file_type()),
        depth: 0,
        parent_directory: None,
        leaf_output: None,
    })
}

struct Budget {
    entries: AtomicUsize,
    directories: AtomicUsize,
    bytes: AtomicUsize,
    discovery_bytes: AtomicUsize,
    started: Instant,
    cancelled: AtomicBool,
}
impl Budget {
    fn new() -> Self {
        Self {
            entries: AtomicUsize::new(0),
            directories: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            discovery_bytes: AtomicUsize::new(0),
            started: Instant::now(),
            cancelled: AtomicBool::new(false),
        }
    }
    fn entries(&self, count: usize) -> Result<(), Failure> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err("worker traversal cancelled");
        }
        if self
            .entries
            .fetch_add(count, Ordering::Relaxed)
            .saturating_add(count)
            > ENTRY_LIMIT
        {
            return Err("entry count bound");
        }
        if self.started.elapsed() > Duration::from_secs(10) {
            return Err("traversal time bound");
        }
        Ok(())
    }
    fn reserve_output(&self, bytes: usize) -> Result<(), Failure> {
        if self
            .bytes
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes)
            > OUTPUT_LIMIT
        {
            Err("rendered output byte bound")
        } else {
            Ok(())
        }
    }
}

// Each worker charges directory, entry and path budgets in blocks. Keeping
// unused credits private avoids a shared atomic write for every small directory.
struct WalkBudget<'a> {
    shared: &'a Budget,
    directories: usize,
    entries: usize,
    bytes: usize,
    seen: usize,
}

impl<'a> WalkBudget<'a> {
    fn new(shared: &'a Budget) -> Self {
        Self {
            shared,
            directories: 0,
            entries: 0,
            bytes: 0,
            seen: 0,
        }
    }
    fn charge(
        credit: &mut usize,
        shared: &AtomicUsize,
        limit: usize,
        wanted: usize,
        batch: usize,
    ) -> bool {
        if *credit < wanted {
            let needed = wanted - *credit;
            let previous = shared.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                let remaining = limit.checked_sub(used)?;
                (remaining >= needed).then(|| used + needed.max(batch.min(remaining)))
            });
            let Ok(used) = previous else { return false };
            *credit += needed.max(batch.min(limit - used));
        }
        *credit -= wanted;
        true
    }
}

impl Drop for WalkBudget<'_> {
    fn drop(&mut self) {
        for (counter, unused) in [
            (&self.shared.directories, self.directories),
            (&self.shared.entries, self.entries),
            (&self.shared.discovery_bytes, self.bytes),
        ] {
            if unused != 0 {
                counter.fetch_sub(unused, Ordering::Relaxed);
            }
        }
    }
}

struct Output {
    bytes: Vec<u8>,
    credited: usize,
    prune: bool,
    quit: bool,
}
impl Output {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            credited: 0,
            prune: false,
            quit: false,
        }
    }
    fn print(&mut self, path: &[u8], separator: u8, budget: &Budget) -> Result<(), Failure> {
        let needed = self.bytes.len() + path.len() + 1;
        if needed > self.credited {
            let additional = (needed - self.credited).max(1024);
            budget.reserve_output(additional)?;
            self.credited += additional;
        }
        self.bytes.extend_from_slice(path);
        self.bytes.push(separator);
        Ok(())
    }
    fn settle(&mut self, budget: &Budget) {
        budget
            .bytes
            .fetch_sub(self.credited - self.bytes.len(), Ordering::Relaxed);
        self.credited = self.bytes.len();
    }
    fn into_bytes(mut self, budget: &Budget) -> Vec<u8> {
        self.settle(budget);
        self.bytes
    }
}

fn child_entry(
    parent: &Entry,
    parent_directory: Option<Arc<File>>,
    name: &[u8],
    kind: u8,
) -> Result<Entry, Failure> {
    let mut path = Vec::with_capacity(parent.path.len() + 1 + name.len());
    path.extend_from_slice(&parent.path);
    if !path.ends_with(b"/") {
        path.push(b'/');
    }
    let name_start = path.len();
    path.extend_from_slice(name);
    if path.len() >= PATH_LIMIT {
        return Err("GNU anchored long-path traversal");
    }
    let name_end = path.len();
    Ok(Entry {
        path,
        name_range: (name_start, name_end),
        kind,
        depth: parent.depth + 1,
        leaf_output: None,
        parent_directory,
    })
}

fn open_directory(entry: &Entry) -> Result<File, Failure> {
    let (parent, name) = match &entry.parent_directory {
        Some(parent) => (parent.as_raw_fd(), entry.name()),
        None => (libc::AT_FDCWD, entry.path.as_slice()),
    };
    let name = CString::new(name).map_err(|_| "NUL directory path")?;
    // Preserve the resolved ancestor when its displayed name is replaced.
    // Reject child symlinks introduced between enumeration and descent.
    let descriptor = unsafe {
        libc::openat(
            parent,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if descriptor < 0 {
        return Err("physical directory anchor or namespace error");
    }
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

struct DirectoryRecord<'a> {
    inode: u64,
    name: &'a [u8],
    dtype: u8,
    length: usize,
}
fn directory_record(buffer: &[u8]) -> Result<DirectoryRecord<'_>, Failure> {
    if buffer.len() < 20 {
        return Err("short kernel directory record");
    }
    let length = u16::from_ne_bytes([buffer[16], buffer[17]]) as usize;
    if length < 20 || length > buffer.len() {
        return Err("invalid kernel directory record");
    }
    let name = &buffer[19..length];
    let end = name
        .iter()
        .position(|b| *b == 0)
        .ok_or("unterminated kernel directory name")?;
    if end == 0 {
        return Err("empty kernel directory name");
    }
    Ok(DirectoryRecord {
        inode: u64::from_ne_bytes(buffer[..8].try_into().unwrap()),
        name: &name[..end],
        dtype: buffer[18],
        length,
    })
}
fn record_kind(record: &DirectoryRecord<'_>, directory: std::os::fd::RawFd) -> Result<u8, Failure> {
    Ok(match record.dtype {
        libc::DT_DIR => b'd',
        libc::DT_REG => b'f',
        libc::DT_LNK => b'l',
        libc::DT_FIFO => b'p',
        libc::DT_SOCK => b's',
        libc::DT_BLK => b'b',
        libc::DT_CHR => b'c',
        libc::DT_UNKNOWN => {
            let mut stat: libc::stat = unsafe { std::mem::zeroed() };
            // directory_record proved the kernel's terminating NUL immediately
            // after this borrowed name; no CString allocation is necessary.
            if unsafe {
                libc::fstatat(
                    directory,
                    record.name.as_ptr().cast(),
                    &mut stat,
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } != 0
            {
                return Err("unknown directory-entry type metadata error");
            }
            match stat.st_mode & libc::S_IFMT {
                libc::S_IFDIR => b'd',
                libc::S_IFREG => b'f',
                libc::S_IFLNK => b'l',
                libc::S_IFIFO => b'p',
                libc::S_IFSOCK => b's',
                libc::S_IFBLK => b'b',
                libc::S_IFCHR => b'c',
                _ => return Err("unknown filesystem entry type"),
            }
        }
        _ => return Err("unknown kernel directory-entry type"),
    })
}

#[inline(never)]
fn children(
    parent: &Entry,
    options: &Options,
    work: &mut WalkBudget,
) -> Result<Vec<Entry>, Failure> {
    let budget = work.shared;
    if parent.kind != b'd' || parent.depth >= options.max_depth {
        return Ok(Vec::new());
    }
    if parent.depth >= DEPTH_LIMIT {
        return Err("traversal depth bound");
    }
    if budget.cancelled.load(Ordering::Relaxed) {
        return Err("worker traversal cancelled");
    }
    if work.directories == 0 && budget.started.elapsed() > Duration::from_secs(10) {
        return Err("traversal time bound");
    }
    if !WalkBudget::charge(
        &mut work.directories,
        &budget.directories,
        DIRECTORY_LIMIT,
        1,
        32,
    ) {
        return Err("directory count bound");
    }
    let directory = open_directory(parent)?;
    let fd = directory.as_raw_fd();
    // Keep leaf-directory descriptors on the stack. Only a descending child
    // needs shared ownership of this anchor after enumeration returns.
    let mut local = Some(directory);
    let mut shared = None;
    let mut entries = Vec::new();
    let mut total = 0;
    let mut raw = std::mem::MaybeUninit::<[u8; 32768]>::uninit();
    loop {
        let size = unsafe {
            libc::syscall(
                libc::SYS_getdents64,
                fd,
                raw.as_mut_ptr().cast::<u8>(),
                32768,
            )
        };
        if size < 0 {
            if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err("directory enumeration or access error");
        }
        if size > 32768 {
            return Err("oversized kernel directory block");
        }
        let buffer =
            unsafe { std::slice::from_raw_parts(raw.as_ptr().cast::<u8>(), size as usize) };
        if buffer.is_empty() {
            break;
        }
        // Charge an upper bound for this kernel block before allocating its
        // paths. Only a walk near its global byte limit needs an exact prepass.
        let per_entry = parent.path.len() + 1 + std::mem::size_of::<Entry>();
        let mut reserved = (buffer.len() / 20) * per_entry + buffer.len();
        if !WalkBudget::charge(
            &mut work.bytes,
            &budget.discovery_bytes,
            DISCOVERY_BYTES_LIMIT,
            reserved,
            32768,
        ) {
            let mut cursor = 0;
            reserved = 0;
            while cursor < buffer.len() {
                let record = directory_record(&buffer[cursor..])?;
                cursor += record.length;
                if record.inode != 0 && !matches!(record.name, b"." | b"..") {
                    reserved += per_entry + record.name.len();
                }
            }
            if !WalkBudget::charge(
                &mut work.bytes,
                &budget.discovery_bytes,
                DISCOVERY_BYTES_LIMIT,
                reserved,
                0,
            ) {
                return Err("discovery path byte bound");
            }
        }
        let (mut offset, mut count, mut bytes) = (0, 0, 0);
        while offset < buffer.len() {
            let record = directory_record(&buffer[offset..])?;
            offset += record.length;
            if record.inode == 0 || matches!(record.name, b"." | b"..") {
                continue;
            }
            count += 1;
            bytes += per_entry + record.name.len();
            let kind = record_kind(&record, fd)?;
            let mut leaf_output = None;
            if kind != b'd' {
                if let Some(filter) = &options.leaf_filter {
                    if parent.depth + 1 < options.min_depth || !filter.matches(&record, kind) {
                        let length = parent.path.len()
                            + usize::from(!parent.path.ends_with(b"/"))
                            + record.name.len();
                        if length >= PATH_LIMIT {
                            return Err("GNU anchored long-path traversal");
                        }
                        continue;
                    }
                    leaf_output = Some(filter.separator);
                }
            }
            let anchor = if kind == b'd' && parent.depth + 1 < options.max_depth {
                Some(Arc::clone(
                    shared.get_or_insert_with(|| Arc::new(local.take().unwrap())),
                ))
            } else {
                None
            };
            let mut child = child_entry(parent, anchor, record.name, kind)?;
            child.leaf_output = leaf_output;
            entries.push(child);
        }
        work.bytes += reserved - bytes;
        if !WalkBudget::charge(&mut work.entries, &budget.entries, ENTRY_LIMIT, count, 256) {
            return Err("entry count bound");
        }
        work.seen += count;
        total += count;
        // FTS may sort wide directories by inode. Zero or one retained leaf
        // has the same output in either order. An unopened directory can
        // still produce matches or errors, so it cannot use this exception.
        if total > CHILD_LIMIT
            && (options.leaf_filter.is_none()
                || entries.len() > 1
                || entries.first().is_some_and(|e| e.kind == b'd'))
        {
            return Err("GNU FTS large-directory ordering");
        }
    }
    Ok(entries)
}

fn evaluate(
    entry: &Entry,
    options: &Options,
    budget: &Budget,
    output: &mut Output,
) -> Result<(), Failure> {
    output.prune = false;
    if let Some(separator) = entry.leaf_output {
        output.print(&entry.path, separator, budget)?;
    } else if entry.depth >= options.min_depth {
        options.expression.evaluate(entry, output, budget)?;
    }
    Ok(())
}

fn walk(
    entry: &Entry,
    options: &Options,
    work: &mut WalkBudget,
    output: &mut Output,
) -> Result<(), Failure> {
    evaluate(entry, options, work.shared, output)?;
    if output.prune || output.quit || entry.kind != b'd' {
        return Ok(());
    }
    for child in children(entry, options, work)? {
        walk(&child, options, work, output)?;
        if output.quit {
            break;
        }
    }
    Ok(())
}

struct ResultSet {
    output: Vec<Vec<u8>>,
    workers: usize,
    entries: usize,
    directories: usize,
}

#[derive(Default)]
struct Traversal {
    chunks: Vec<Vec<u8>>,
    pool: Option<rayon::ThreadPool>,
    workers: usize,
}

impl Traversal {
    fn flush_chunk(&mut self, output: &mut Output, budget: &Budget) {
        if !output.bytes.is_empty() {
            let chunk = std::mem::replace(output, Output::new()).into_bytes(budget);
            self.chunks.push(chunk);
        }
    }

    fn start_pool(&mut self, directories: usize) -> Result<(), Failure> {
        if self.pool.is_some() {
            return Ok(());
        }
        let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
        let descriptor_workers = if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0
        {
            (limit.rlim_cur.saturating_sub(16) / 4).clamp(1, 16) as usize
        } else {
            1
        };
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(descriptor_workers)
            .min(directories);
        if threads > 1 {
            self.pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|_| "worker pool creation")?,
            );
            self.workers = threads;
        }
        Ok(())
    }

    fn visit(
        &mut self,
        entry: &Entry,
        options: &Options,
        work: &mut WalkBudget,
        output: &mut Output,
    ) -> Result<(), Failure> {
        let budget = work.shared;
        if options.can_quit {
            return walk(entry, options, work, output);
        }
        evaluate(entry, options, budget, output)?;
        if output.prune || entry.kind != b'd' || entry.depth >= options.max_depth {
            return Ok(());
        }
        let list = children(entry, options, work)?;
        let mut directories = list
            .iter()
            .filter(|e| e.kind == b'd' && e.depth < options.max_depth)
            .count();
        let mut siblings = list.into_iter();
        let before = work.seen;
        // Observe real descendant work before paying for threads. Looking at
        // each depth also admits a wide subtree below a narrow starting path.
        if directories >= 64 {
            self.start_pool(directories)?;
        }
        if directories < 64 || self.pool.is_none() {
            for child in siblings.by_ref() {
                let directory = child.kind == b'd' && child.depth < options.max_depth;
                self.visit(&child, options, work, output)?;
                if directory {
                    directories -= 1;
                    if directories >= 8 && (work.seen - before >= 8 || self.pool.is_some()) {
                        self.start_pool(directories)?;
                        if self.pool.is_some() {
                            break;
                        }
                    }
                }
            }
        }
        if siblings.len() == 0 {
            return Ok(());
        }

        self.flush_chunk(output, budget);
        let pool = self.pool.as_ref().unwrap();
        let first_error = std::sync::OnceLock::new();
        let chunk_size = siblings.len().div_ceil(self.workers * 16);
        let parts: Vec<Vec<u8>> = pool.install(|| {
            siblings
                .as_slice()
                .par_chunks(chunk_size)
                .map(|group| {
                    let mut part = Output::new();
                    let mut local = WalkBudget::new(budget);
                    for child in group {
                        if budget.cancelled.load(Ordering::Relaxed) {
                            break;
                        }
                        if let Err(reason) = walk(child, options, &mut local, &mut part) {
                            first_error.get_or_init(|| reason);
                            budget.cancelled.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                    part.into_bytes(budget)
                })
                .collect()
        });
        if let Some(reason) = first_error.into_inner() {
            return Err(reason);
        }
        self.chunks
            .extend(parts.into_iter().filter(|part| !part.is_empty()));
        Ok(())
    }
}

fn prepare(options: &Options) -> Result<ResultSet, Failure> {
    check_stdout(options)?;
    let budget = Budget::new();
    let mut output = Output::new();
    let mut traversal = Traversal::default();
    let mut work = WalkBudget::new(&budget);
    for root in &options.roots {
        let entry = root_entry(root)?;
        budget.entries(1)?;
        traversal.visit(&entry, options, &mut work, &mut output)?;
        if output.quit {
            break;
        }
    }
    drop(work);
    budget.entries(0)?;
    traversal.flush_chunk(&mut output, &budget);
    Ok(ResultSet {
        output: traversal.chunks,
        workers: traversal.workers.max(1),
        entries: budget.entries.load(Ordering::Relaxed),
        directories: budget.directories.load(Ordering::Relaxed),
    })
}

fn check_stdout(options: &Options) -> Result<(), Failure> {
    unsafe {
        let mut stat: libc::stat = std::mem::zeroed();
        if libc::fstat(1, &mut stat) != 0 {
            return Err("closed output descriptor");
        }
        let flags = libc::fcntl(1, libc::F_GETFL);
        if flags < 0 || flags & libc::O_ACCMODE == libc::O_RDONLY {
            return Err("unwritable output descriptor");
        }
        if flags & libc::O_NONBLOCK != 0 {
            return Err("GNU nonblocking output policy");
        }
        if libc::isatty(1) != 0 && options.newline_output {
            return Err("GNU terminal pathname quoting");
        }
        if stat.st_mode & libc::S_IFMT == libc::S_IFCHR
            && libc::isatty(1) == 0
            && stat.st_rdev != libc::makedev(1, 3)
        {
            return Err("special output descriptor");
        }
    }
    Ok(())
}

fn gnu(args: &[OsString]) -> ! {
    let ignored = INHERITED_SIGPIPE_IGNORED.load(Ordering::Relaxed);
    let mut command = Command::new("/usr/bin/find");
    command.arg0("find").args(args);
    // Command's Unix exec preparation also resets SIGPIPE. Restore the caller's
    // ignored disposition after that preparation, immediately before execve.
    unsafe {
        command.pre_exec(move || {
            libc::signal(
                libc::SIGPIPE,
                if ignored {
                    libc::SIG_IGN
                } else {
                    libc::SIG_DFL
                },
            );
            Ok(())
        });
    }
    let error = command.exec();
    eprintln!("find2fd-rust: GNU find: {error}");
    std::process::exit(127);
}

fn main() {
    mod app_version_runtime {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../build-support/app_version_runtime.rs"
        ));
    }
    let mut args: Vec<OsString> = env::args_os().skip(1).collect();
    app_version_runtime::handle_cli(&args);
    unsafe {
        libc::signal(
            libc::SIGPIPE,
            if INHERITED_SIGPIPE_IGNORED.load(Ordering::Relaxed) {
                libc::SIG_IGN
            } else {
                libc::SIG_DFL
            },
        );
        libc::setlocale(libc::LC_ALL, c"".as_ptr());
    }
    let diagnostic = args.first().is_some_and(|s| s == "--find2fd-route-json");
    if diagnostic {
        args.remove(0);
    }
    let result = parse(&args).and_then(|options| prepare(&options));
    if diagnostic {
        let report = match result {
            Ok(result) => {
                serde_json::json!({"route":"native","strategy":if result.workers > 1 { "parallel" } else { "sequential" },"reason":"qualified GNU physical filename semantics","workers":result.workers,"entries":result.entries,"directories":result.directories})
            }
            Err(reason) => {
                serde_json::json!({"route":"gnu","strategy":"exec","reason":reason,"workers":1,"entries":0,"directories":0})
            }
        };
        println!("{report}");
        return;
    }
    let result = result.unwrap_or_else(|_| gnu(&args));
    // std::io::stdout has a process-global line buffer. A successful write_all
    // can therefore defer NUL-record failures until an ignored shutdown flush.
    if let Err(error) = write_output(&result.output) {
        let detail =
            unsafe { CStr::from_ptr(libc::strerror(error.raw_os_error().unwrap_or(libc::EIO))) };
        if byte_locale() {
            eprintln!(
                "find: 'standard output': {}\nfind: write error",
                detail.to_string_lossy()
            );
        } else {
            eprintln!(
                "find: ‘standard output’: {}\nfind: write error",
                detail.to_string_lossy()
            );
        }
        std::process::exit(1);
    }
}

// Write in order without copying every parallel result into a second buffer.
// Direct file writes retain GNU's observable flush/error behavior.
fn write_output(chunks: &[Vec<u8>]) -> io::Result<()> {
    use std::io::{IoSlice, Write};
    let mut stdout = std::mem::ManuallyDrop::new(unsafe { File::from_raw_fd(1) });
    for batch in chunks.chunks(64) {
        let mut slices = [IoSlice::new(&[]); 64];
        for (slice, chunk) in slices.iter_mut().zip(batch) {
            *slice = IoSlice::new(chunk);
        }
        let mut pending = &mut slices[..batch.len()];
        while !pending.is_empty() {
            match stdout.write_vectored(pending) {
                Ok(0) => return Err(io::Error::from_raw_os_error(libc::EIO)),
                Ok(n) => IoSlice::advance_slices(&mut pending, n),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
