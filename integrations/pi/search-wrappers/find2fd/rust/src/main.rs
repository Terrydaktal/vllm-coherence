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
        if !byte_locale && (fold || complex || !pattern.is_ascii()) {
            return Err("GNU locale-sensitive pattern");
        }
        if complex {
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
        Ok(match self {
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
    newline_output: bool,
}

struct Parser<'a> {
    args: &'a [OsString],
    index: usize,
    nodes: usize,
    explicit_print: bool,
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
        expression,
        newline_output: parser.newline_output,
    })
}

struct Entry {
    path: Vec<u8>,
    name_range: (usize, usize),
    kind: u8,
    depth: usize,
    parent_directory: Option<Arc<File>>,
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
    })
}

struct Budget {
    entries: AtomicUsize,
    directories: AtomicUsize,
    bytes: AtomicUsize,
    discovery_bytes: AtomicUsize,
    started: Instant,
}
impl Budget {
    fn new() -> Self {
        Self {
            entries: AtomicUsize::new(0),
            directories: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            discovery_bytes: AtomicUsize::new(0),
            started: Instant::now(),
        }
    }
    fn entries(&self, count: usize) -> Result<(), Failure> {
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
    fn reserve_discovery(&self, bytes: usize) -> Result<(), Failure> {
        if self
            .discovery_bytes
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes)
            > DISCOVERY_BYTES_LIMIT
        {
            Err("discovery path byte bound")
        } else {
            Ok(())
        }
    }
}

struct Output {
    bytes: Vec<u8>,
    credited: usize,
}
impl Output {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            credited: 0,
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
    directory: &Arc<File>,
    name: &[u8],
    kind: u8,
    options: &Options,
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
        parent_directory: if kind == b'd' && parent.depth + 1 < options.max_depth {
            Some(Arc::clone(directory))
        } else {
            None
        },
    })
}

fn open_directory(entry: &Entry) -> Result<Arc<File>, Failure> {
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
    Ok(Arc::new(unsafe { File::from_raw_fd(descriptor) }))
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
fn record_kind(record: &DirectoryRecord<'_>, directory: &File) -> Result<u8, Failure> {
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
                    directory.as_raw_fd(),
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
fn children(parent: &Entry, options: &Options, budget: &Budget) -> Result<Vec<Entry>, Failure> {
    if parent.kind != b'd' || parent.depth >= options.max_depth {
        return Ok(Vec::new());
    }
    if parent.depth >= DEPTH_LIMIT {
        return Err("traversal depth bound");
    }
    if budget.directories.fetch_add(1, Ordering::Relaxed) >= DIRECTORY_LIMIT {
        return Err("directory count bound");
    }
    let anchor = open_directory(parent)?;
    let mut entries = Vec::new();
    let mut raw = std::mem::MaybeUninit::<[u8; 32768]>::uninit();
    loop {
        let size = unsafe {
            libc::syscall(
                libc::SYS_getdents64,
                anchor.as_raw_fd(),
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
        let (mut offset, mut count, mut bytes) = (0, 0, 0);
        while offset < buffer.len() {
            let record = directory_record(&buffer[offset..])?;
            if record.inode != 0 && !matches!(record.name, b"." | b"..") {
                count += 1;
                bytes += parent.path.len() + 1 + record.name.len() + std::mem::size_of::<Entry>();
            }
            offset += record.length;
        }
        if entries.len() + count > CHILD_LIMIT {
            return Err("GNU FTS large-directory ordering");
        }
        budget.entries(count)?;
        budget.reserve_discovery(bytes)?;
        if size == 0 {
            break;
        }
        entries.reserve(count);
        offset = 0;
        while offset < buffer.len() {
            let record = directory_record(&buffer[offset..])?;
            if record.inode != 0 && !matches!(record.name, b"." | b"..") {
                let kind = record_kind(&record, &anchor)?;
                entries.push(child_entry(parent, &anchor, record.name, kind, options)?);
            }
            offset += record.length;
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
    if entry.depth >= options.min_depth {
        options.expression.evaluate(entry, output, budget)?;
    }
    Ok(())
}

fn walk(
    entry: &Entry,
    options: &Options,
    budget: &Budget,
    output: &mut Output,
) -> Result<(), Failure> {
    evaluate(entry, options, budget, output)?;
    for child in children(entry, options, budget)? {
        walk(&child, options, budget, output)?;
    }
    Ok(())
}

struct ResultSet {
    output: Vec<u8>,
    workers: usize,
    entries: usize,
    directories: usize,
}

fn prepare(options: &Options) -> Result<ResultSet, Failure> {
    check_stdout(options)?;
    let budget = Budget::new();
    let mut output = Output::new();
    let mut workers = 1;
    let mut pool = None;
    for root in &options.roots {
        let entry = root_entry(root)?;
        budget.entries(1)?;
        evaluate(&entry, options, &budget, &mut output)?;
        let children = children(&entry, options, &budget)?;
        let child_directories = children
            .iter()
            .filter(|e| e.kind == b'd' && e.depth < options.max_depth)
            .count();
        let parallel = children.len() >= 32 && child_directories >= 4;
        if parallel {
            if pool.is_none() {
                let mut descriptor_limit: libc::rlimit = unsafe { std::mem::zeroed() };
                let descriptor_workers = if unsafe {
                    libc::getrlimit(libc::RLIMIT_NOFILE, &mut descriptor_limit)
                } == 0
                {
                    (descriptor_limit.rlim_cur.saturating_sub(16) / 4).clamp(1, 16) as usize
                } else {
                    1
                };
                let threads = std::thread::available_parallelism()
                    .map_or(1, |n| n.get())
                    .min(descriptor_workers);
                pool = Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(|_| "worker pool creation")?,
                );
                workers = threads;
            }
            output.settle(&budget);
            let parts: Vec<Result<Vec<u8>, Failure>> = pool.as_ref().unwrap().install(|| {
                children
                    .par_iter()
                    .map(|child| {
                        let mut part = Output::new();
                        walk(child, options, &budget, &mut part)?;
                        Ok(part.into_bytes(&budget))
                    })
                    .collect()
            });
            for part in parts {
                output.bytes.extend_from_slice(&part?);
            }
            output.credited = output.bytes.len();
        } else {
            for child in children {
                walk(&child, options, &budget, &mut output)?;
            }
        }
    }
    budget.entries(0)?;
    Ok(ResultSet {
        output: output.into_bytes(&budget),
        workers,
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
    let mut args: Vec<OsString> = env::args_os().skip(1).collect();
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
    let mut remaining = result.output.as_slice();
    while !remaining.is_empty() {
        let written = unsafe { libc::write(1, remaining.as_ptr().cast(), remaining.len()) };
        if written > 0 {
            remaining = &remaining[written as usize..];
            continue;
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
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

#[cfg(test)]
mod tests;
