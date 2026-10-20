#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <regex.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>

/* The native lane handles aggregate and sparse full-line byte searches. It does not consume
 * stdin, evaluate locale regexes, or replace GNU diagnostics. A bounded
 * first-page hint avoids dense output/early filename cases; it never supplies
 * a matching result.
 * All other argv reaches GNU unchanged. Results are reordered to GNU operand /
 * readdir order; parallel rg completion order is never presented as GNU order. */
typedef struct { char *name; unsigned long long count; bool matched; char *rows; size_t row_bytes, row_capacity; } File;
static size_t output_bytes;
static File *files; static size_t nfiles, capacity; static uint64_t bytes;
static char **original; static pid_t child; static volatile sig_atomic_t interrupted;
static bool recursive, text, ignore_case, invert, null_names, counts, filenames, without, extended, whole_line, whole_word, content, line_number;
static int prefix = -1;
static char **patterns; static size_t npatterns;
static unsigned early_probes, early_matches, shape_probes;
static void native(void) { original[0] = (char *)"grep"; execvp("grep", original); perror("grep2rg: GNU grep"); exit(2); }
static void signal_child(int sig) { interrupted = sig; if (child > 0) kill(child, sig); }
static bool add_file(const char *name, const struct stat *st) {
  /* First-line selection makes GNU filename probes cheaper. This bounded hint
   * only changes engine choice; it never supplies a matching result. */
  bool dense_hint = (filenames || content) && !counts && !ignore_case && !invert && !whole_word && !whole_line && early_probes < 8;
  if (shape_probes < 8 || dense_hint) {
    int fd = open(name, O_RDONLY | O_NONBLOCK);
    struct stat current; char sample[4097]; ssize_t n = 0;
    if (fd >= 0 && !fstat(fd, &current) && S_ISREG(current.st_mode)) n = read(fd, sample, 4096);
    if (fd >= 0) close(fd);
    if (n > 0) {
      shape_probes++;
      /* GNU's literal scanner wins on large sparse records; parallel
       * dispatch is not qualified for a first record longer than this page. */
      if (n == 4096 && !memchr(sample, '\n', (size_t)n)) native();
      if (dense_hint) {
      early_probes++; sample[n] = 0; bool hit = false;
      if (extended) {
        char *eol = memchr(sample, '\n', (size_t)n); if (eol) *eol = 0;
        for (size_t i = 0; i < npatterns; i++) { regex_t match; if (!regcomp(&match, patterns[i], REG_EXTENDED | REG_NOSUB)) { hit |= regexec(&match, sample, 0, NULL, 0) == 0; regfree(&match); } }
      } else for (size_t i = 0; i < npatterns; i++) hit |= memmem(sample, (size_t)n, patterns[i], strlen(patterns[i])) != NULL;
      early_matches += hit ? 1 : 0;
      if (early_probes == 8 && early_matches >= 6) native();
      }
    }
  }
  if (!S_ISREG(st->st_mode) || access(name, R_OK)) return false;
  if (nfiles >= 100000) return false;
  if (nfiles == capacity) { capacity = capacity ? capacity * 2 : 256; File *p = realloc(files, capacity * sizeof(*p)); if (!p) return false; files = p; }
  files[nfiles++] = (File){ .name = strdup(name) }; if (!files[nfiles - 1].name) return false;
  if (st->st_size > 0 && UINT64_MAX - bytes > (uint64_t)st->st_size) bytes += st->st_size;
  /* Many tiny files are dominated by metadata/dispatch, where GNU wins.
   * Reject from 32 descriptor sizes instead of walking the entire small tree. */
  if (nfiles == 32 && bytes < 32ULL * 4096) native();
  return true;
}
static bool walk(const char *name, bool command_line, unsigned depth) {
  if (depth > 128) return false;
  struct stat st; if (lstat(name, &st)) return false;
  /* A descriptor symlink can resolve to a caller regular file, then to rg
   * capture pipes after stdout/stderr are redirected. Never reinterpret it.
   * Ordinary explicit symlinks also retain GNU rather than guessing chains. */
  if (command_line && S_ISLNK(st.st_mode)) return false;
  if (!command_line && S_ISLNK(st.st_mode)) return true;
  if (S_ISREG(st.st_mode)) return add_file(name, &st);
  if (!S_ISDIR(st.st_mode)) return !command_line;
  if (!recursive || access(name, R_OK | X_OK)) return false;
  DIR *dir = opendir(name); if (!dir) return false;
  bool ok = true; struct dirent *entry;
  while (true) {
    errno = 0; entry = readdir(dir); if (!entry) { if (errno) ok = false; break; }
    if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
    size_t len = strlen(name); while (len > 1 && name[len - 1] == '/') len--;
    char *path = NULL;
    if (asprintf(&path, "%.*s%s%s", (int)len, name, len == 1 && name[0] == '/' ? "" : "/", entry->d_name) < 0) { ok = false; break; }
    ok = walk(path, false, depth + 1); free(path); if (!ok) break;
  }
  closedir(dir); return ok;
}
static unsigned long hash(const char *s) { unsigned long h = 1469598103934665603UL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; } return h; }
static File **table; static size_t table_size;
static File *lookup(const char *s) { size_t slot = hash(s) & (table_size - 1); while (table[slot]) { if (!strcmp(table[slot]->name, s)) return table[slot]; slot = (slot + 1) & (table_size - 1); } return NULL; }
static bool index_files(void) {
  table_size = 1; while (table_size < nfiles * 2) table_size *= 2;
  table = calloc(table_size, sizeof(*table)); if (!table) return false;
  for (size_t i = 0; i < nfiles; i++) { size_t slot = hash(files[i].name) & (table_size - 1); while (table[slot]) { if (!strcmp(table[slot]->name, files[i].name)) return false; slot = (slot + 1) & (table_size - 1); } table[slot] = &files[i]; }
  return true;
}
/* Unlike getline, this cannot allocate beyond the unpublished output budget
 * for a single enormous matching line. Failure discards all rg output. */
static ssize_t bounded_row(FILE *stream, char **row, size_t *capacity, size_t maximum) {
  size_t n = 0; int byte;
  while ((byte = getc_unlocked(stream)) != EOF) {
    if (n == maximum) return -2;
    if (n == *capacity) {
      size_t next = *capacity ? *capacity * 2 : 256;
      if (next > maximum) next = maximum;
      char *buffer = realloc(*row, next); if (!buffer) return -2;
      *row = buffer; *capacity = next;
    }
    (*row)[n++] = (char)byte;
    if (byte == '\n') return (ssize_t)n;
  }
  return -1;
}
static bool rg_batch(size_t begin, size_t end) {
  char **args = calloc(end - begin + 2 * npatterns + 32, sizeof(*args)); if (!args) return false;
  size_t a = 0; args[a++] = (char *)"rg"; args[a++] = (char *)"--no-config";
  args[a++] = (char *)"--no-ignore"; args[a++] = (char *)"--hidden";
  args[a++] = (char *)"--color=never"; args[a++] = (char *)"--encoding=none";
  args[a++] = (char *)"--no-unicode"; args[a++] = (char *)"--threads=8";
  args[a++] = (char *)(text ? "--text" : "--binary");
  if (!extended) args[a++] = (char *)"--fixed-strings";
  args[a++] = (char *)"--null";
  if (ignore_case) args[a++] = (char *)"--ignore-case";
  if (whole_line) args[a++] = (char *)"--line-regexp";
  if (whole_word) args[a++] = (char *)"--word-regexp";
  if (invert) args[a++] = (char *)"--invert-match";
  if (counts) { args[a++] = (char *)"--count"; args[a++] = (char *)"--include-zero"; args[a++] = (char *)"--with-filename"; }
  else if (filenames) args[a++] = (char *)"--files-with-matches";
  else { args[a++] = (char *)"--with-filename"; args[a++] = (char *)"--no-heading"; args[a++] = (char *)(line_number ? "--line-number" : "--no-line-number"); }
  for (size_t i = 0; i < npatterns; i++) { args[a++] = (char *)"-e"; args[a++] = patterns[i]; }
  args[a++] = (char *)"--"; for (size_t i = begin; i < end; i++) args[a++] = files[i].name;
  int pipefd[2]; if (pipe(pipefd)) { free(args); return false; }
  child = fork();
  if (child == 0) {
    close(pipefd[0]); if (dup2(pipefd[1], STDOUT_FILENO) < 0) _exit(127); close(pipefd[1]);
    int devnull = open("/dev/null", O_WRONLY); if (devnull < 0 || dup2(devnull, STDERR_FILENO) < 0) _exit(127); if (devnull > 2) close(devnull);
    execvp("rg", args); _exit(127);
  }
  close(pipefd[1]); free(args);
  if (child < 0) { close(pipefd[0]); child = 0; return false; }
  FILE *stream = fdopen(pipefd[0], "r"); bool ok = stream != NULL; char *name = NULL, *number = NULL; size_t cap = 0, numcap = 0;
  if (stream) while (true) {
    ssize_t len = getdelim(&name, &cap, '\0', stream); if (len < 0) { if (ferror(stream)) ok = false; break; }
    if (len == 0 || name[len - 1] != '\0') { ok = false; break; }
    File *file = lookup(name); if (!file) { ok = false; break; }
    if (counts) {
      ssize_t n = getline(&number, &numcap, stream); char *tail = NULL;
      if (n < 2 || number[n - 1] != '\n' || number[0] < '0' || number[0] > '9') { ok = false; break; }
      errno = 0; unsigned long long count = strtoull(number, &tail, 10);
      if (errno || tail != number + n - 1) { ok = false; break; }
      file->count = count; file->matched = count > 0;
    } else if (filenames) file->matched = true;
    else {
      ssize_t n = bounded_row(stream, &number, &numcap, 32ULL * 1024 * 1024 - output_bytes);
      if (n <= 0 || number[n - 1] != '\n' || output_bytes + (size_t)n > 32ULL * 1024 * 1024) { ok = false; break; }
      size_t required = file->row_bytes + (size_t)n;
      if (required > file->row_capacity) {
        size_t capacity = required * 2; char *rows = realloc(file->rows, capacity);
        if (!rows) { ok = false; break; } file->rows = rows; file->row_capacity = capacity;
      }
      memcpy(file->rows + file->row_bytes, number, (size_t)n); file->row_bytes += (size_t)n; output_bytes += (size_t)n; file->matched = true;
    }
  }
  free(name); free(number); if (stream) fclose(stream); else close(pipefd[0]);
  if (!ok && !interrupted) kill(child, SIGTERM);
  int status; while (waitpid(child, &status, 0) < 0) { if (errno != EINTR) { ok = false; break; } }
  child = 0;
  if (interrupted) exit(128 + interrupted);
  return ok && WIFEXITED(status) && WEXITSTATUS(status) <= 1;
}
int main(int argc, char **argv) {
  original = calloc((size_t)argc + 1, sizeof(*original)); if (!original) return 2;
  memcpy(original, argv, (size_t)argc * sizeof(*original));
  if (getenv("POSIXLY_CORRECT") || (getenv("GREP2RG_ACCELERATION") && strcmp(getenv("GREP2RG_ACCELERATION"), "gnu") == 0)) native();
  for (int fd = 0; fd <= 2; fd++) if (fcntl(fd, F_GETFD) == -1) native();
  struct stat output; if (fstat(STDOUT_FILENO, &output)) native();
  if (S_ISDIR(output.st_mode) || (S_ISCHR(output.st_mode) && !isatty(STDOUT_FILENO) && output.st_rdev != makedev(1, 3))) native();
  struct stat diagnostic; if (fstat(STDERR_FILENO, &diagnostic) || S_ISDIR(diagnostic.st_mode)) native();
  patterns = calloc((size_t)argc, sizeof(*patterns)); if (!patterns) native();
  bool fixed = false, basic = false; int opt;
  static struct option options[] = {
    {"fixed-strings",0,0,'F'},{"basic-regexp",0,0,'G'},{"extended-regexp",0,0,'E'},{"regexp",1,0,'e'},
    {"text",0,0,'a'},{"files-with-matches",0,0,'l'},{"files-without-match",0,0,'L'},{"count",0,0,'c'},
    {"line-number",0,0,'n'},{"with-filename",0,0,'H'},{"no-filename",0,0,'h'},
    {"null",0,0,'Z'},{"recursive",0,0,'r'},{"ignore-case",0,0,'i'},
    {"no-ignore-case",0,0,1000},{"binary-files",1,0,1001},{"color",2,0,1002},{"colour",2,0,1002},
    {"line-regexp",0,0,'x'},{"word-regexp",0,0,'w'},{"binary",0,0,'U'},
    {"invert-match",0,0,'v'},{"no-messages",0,0,'s'}, {0,0,0,0}
  };
  /* A reduced getopt table must not resolve an abbreviation which is
   * ambiguous against GNU's complete option table. Exact names only here;
   * GNU itself still accepts/diagnoses every abbreviation through fallback. */
  for (int i = 1; i < argc; i++) {
    /* Raw -- may be the value of -e. Scan conservatively rather than treating
     * it as a parsed delimiter and exposing abbreviated later options. */
    if (!strcmp(argv[i], "--")) continue;
    if (!strncmp(argv[i], "--", 2)) {
      const char *eq = strchr(argv[i] + 2, '=');
      size_t len = eq ? (size_t)(eq - argv[i] - 2) : strlen(argv[i] + 2);
      bool known = false;
      for (size_t j = 0; options[j].name; j++) known |= strlen(options[j].name) == len && !strncmp(options[j].name, argv[i] + 2, len);
      if (!known) native();
    }
  }
  opterr = 0;
  while ((opt = getopt_long(argc, argv, "FEGae:lLcniyHhZrvsxwU", options, NULL)) != -1) {
    switch (opt) {
      case 'F': fixed = true; break; case 'G': basic = true; break; case 'E': extended = true; break;
      case 'a': text = true; break; case 'e': patterns[npatterns++] = optarg; break;
      case 'l': filenames = true; without = false; break; case 'L': filenames = true; without = true; break; case 'c': counts = true; break;
      case 'i': case 'y': ignore_case = true; break; case 'v': invert = true; break;
      case 'x': whole_line = true; break; case 'w': whole_word = true; break;
      case 1000: ignore_case = false; break;
      case 1001: if (!strcmp(optarg, "text")) text = true; else if (!strcmp(optarg, "binary")) text = false; else native(); break;
      case 1002: if (!optarg || strcmp(optarg, "never")) native(); break;
      case 'H': prefix = 1; break; case 'h': prefix = 0; break;
      case 'Z': null_names = true; break; case 'r': recursive = true; break;
      case 'n': line_number = true; break; case 's': case 'U': break; default: native();
    }
  }
  if ((fixed ? 1 : 0) + (basic ? 1 : 0) + (extended ? 1 : 0) > 1) native();
  if (!npatterns) { if (optind >= argc) native(); patterns[npatterns++] = argv[optind++]; }
  if (filenames) counts = false;
  content = !filenames && !counts;
  if (content && (!text || invert)) native();
  if (!text && (counts || invert)) native();
  if (filenames && invert) native();
  const char *locale = getenv("LC_ALL"); if (!locale || !*locale) locale = getenv("LC_CTYPE"); if (!locale || !*locale) locale = getenv("LANG");
  if ((ignore_case || extended || whole_word) && locale && strcmp(locale,"C") && strcmp(locale,"POSIX")) native();
  for (size_t i = 0; i < npatterns; i++) {
    /* Default BRE without any operator is a byte literal. */
    if (!*patterns[i] || strchr(patterns[i], '\n')) native();
    if (extended) {
      /* Aggregate outputs depend on line selection, not POSIX match spans.
       * Restrict the direct ERE grammar to byte anchors/grouping/alternation/
       * repetition; libc brackets, escapes and locale classes stay GNU. */
      if (strpbrk(patterns[i], "[\\{}")) native();
      for (const unsigned char *p = (const unsigned char *)patterns[i]; *p; p++) if (*p > 127) native();
      regex_t check; if (regcomp(&check, patterns[i], REG_EXTENDED | REG_NOSUB | (ignore_case ? REG_ICASE : 0))) native(); regfree(&check);
    } else if (!fixed && strpbrk(patterns[i], ".*^$[\\")) native();
  }
  if (optind == argc) native();
  int operands = argc - optind; bool directory_operand = false;
  if (!recursive && operands < 256) native();
  for (int i = optind; i < argc; i++) {
    struct stat st; if (!strcmp(argv[i], "-") || stat(argv[i], &st)) native();
    directory_operand |= S_ISDIR(st.st_mode);
    if (!walk(argv[i], true, 0)) native();
  }
  /* Full-file counters can amortize metadata and parallel dispatch. Filename
   * probes that select early are deliberately left to GNU below this scale. */
  if (nfiles < 256 || bytes < 64ULL * 1024 * 1024) native();
  if (!index_files()) native();
  const char *trace = getenv("GREP2RG_ROUTE_LOG");
  if (trace && *trace) { FILE *f = fopen(trace, "a"); if (f) { fprintf(f,"rg-native files=%zu bytes=%llu mode=%s\n", nfiles,(unsigned long long)bytes,counts?"count":filenames?"filenames":"content"); fclose(f); } }
  signal(SIGINT, signal_child); signal(SIGTERM, signal_child); signal(SIGHUP, signal_child);
  for (size_t begin = 0; begin < nfiles;) {
    size_t end = begin, args_bytes = 0;
    while (end < nfiles && args_bytes + strlen(files[end].name) + 1 < 48000) args_bytes += strlen(files[end++].name) + 1;
    if (end == begin || !rg_batch(begin, end)) native();
    begin = end;
  }
  bool selected = false;
  for (size_t i = 0; i < nfiles; i++) {
    selected |= files[i].matched;
    if (counts) {
      if (prefix == 1 || (prefix == -1 && (operands > 1 || directory_operand))) { fputs(files[i].name, stdout); fputc(null_names ? '\0' : ':', stdout); }
      fprintf(stdout, "%llu\n", files[i].count);
    } else if (filenames && files[i].matched != without) { fputs(files[i].name, stdout); fputc(null_names ? '\0' : '\n', stdout); }
    else if (content && files[i].matched) {
      size_t begin = 0;
      while (begin < files[i].row_bytes) {
        char *end = memchr(files[i].rows + begin, '\n', files[i].row_bytes - begin);
        if (!end) return 2; /* parser already requires complete records; never rerun GNU after publication */
        if (prefix == 1 || (prefix == -1 && (operands > 1 || directory_operand))) { fputs(files[i].name, stdout); fputc(null_names ? '\0' : ':', stdout); }
        size_t length = (size_t)(end - files[i].rows) - begin + 1;
        fwrite(files[i].rows + begin, 1, length, stdout); begin += length;
      }
    }
  }
  if (fflush(stdout) || ferror(stdout)) { int error = errno; fprintf(stderr, "grep: write error: %s\n", strerror(error)); return 2; }
  return selected ? 0 : 1;
}
