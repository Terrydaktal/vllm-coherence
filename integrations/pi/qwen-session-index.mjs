import { createHash } from "node:crypto";
import { constants, closeSync, existsSync, fstatSync, lstatSync, mkdirSync, openSync, readSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { DatabaseSync } from "node:sqlite";

const APPLICATION_ID = 0x51545349;
const MAX_CHARS = 16000;
const ROLES = new Set(["user", "assistant", "toolResult", "bashExecution", "custom", "summary"]);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const own = (stat) => Number(stat.uid) === process.getuid();
const words = (text) => text.match(/[\p{L}\p{N}][\p{L}\p{N}\p{M}]*/gu) ?? [];

function directory(path, privateMode = false) {
  const stat = lstatSync(path);
  if (!stat.isDirectory() || stat.isSymbolicLink() || !own(stat) ||
      (Number(stat.mode) & (privateMode ? 0o077 : 0o022))) {
    throw new Error("index paths require owned, nonsymlink directories with safe permissions");
  }
  return stat;
}

function snapshot(stat) {
  const size = Number(stat.size);
  if (!Number.isSafeInteger(size)) throw new Error("source file is too large");
  return { dev: String(stat.dev), ino: String(stat.ino), size,
    mtime: String(stat.mtimeNs), ctime: String(stat.ctimeNs) };
}

function sameFile(left, right) {
  return ["dev", "ino", "size", "mtime", "ctime"].every((key) => left[key] === right[key]);
}

function hashRange(fd, length) {
  const hash = createHash("sha256"), buffer = Buffer.allocUnsafe(65536);
  for (let offset = 0; offset < length;) {
    const read = readSync(fd, buffer, 0, Math.min(buffer.length, length - offset), offset);
    if (!read) throw new Error("source changed during indexing");
    hash.update(buffer.subarray(0, read)); offset += read;
  }
  return hash.digest("hex");
}

function readRange(fd, offset, length) {
  const bytes = Buffer.allocUnsafe(length);
  for (let read = 0; read < length;) {
    const count = readSync(fd, bytes, read, length - read, offset + read);
    if (!count) throw new Error("source changed during reading");
    read += count;
  }
  return bytes;
}

function searchable(entry) {
  let role, content;
  const parts = [], kinds = new Set();
  const add = (kind, text) => {
    if (typeof text === "string" && text.length) { parts.push(text); kinds.add(kind); }
  };
  if (entry.type === "compaction" || entry.type === "branch_summary") {
    role = "summary"; add("summary", entry.summary);
  } else if (entry.type === "custom_message") {
    role = "custom"; content = entry.content;
  } else if (entry.type === "message") {
    const message = entry.message ?? {};
    role = ROLES.has(message.role) ? message.role : undefined;
    content = message.content;
    if (message.role === "bashExecution") {
      add("command", message.command); add("output", message.output);
    }
  }
  if (typeof content === "string") add("text", content);
  else if (Array.isArray(content)) for (const block of content) {
    if (block?.type === "text") add("text", block.text);
    else if (block?.type === "thinking" && !block.redacted) add("thinking", block.thinking);
    else if (block?.type === "toolCall") {
      add("toolCall", block.name);
      if (block.arguments !== undefined) add("toolCall", JSON.stringify(block.arguments));
    }
  }
  return { role, text: parts.join("\n\n"), sourceKinds: [...kinds] };
}

function matchPosition(text, query, mode) {
  if (mode === "literal") return text.indexOf(query);
  // Keep a map back to the original UTF-16 positions while folding accents.
  let folded = "";
  const positions = [];
  for (let index = 0; index < text.length;) {
    const character = String.fromCodePoint(text.codePointAt(index));
    const normalized = character.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
    for (let offset = 0; offset < normalized.length; offset++) positions.push(index);
    folded += normalized; index += character.length;
  }
  const sourceWords = [...folded.matchAll(/[\p{L}\p{N}][\p{L}\p{N}\p{M}]*/gu)];
  const normalize = (value) => value.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
  for (const candidate of query.matchAll(/[\p{L}\p{N}][\p{L}\p{N}\p{M}]*/gu)) {
    if (mode === "fts" && (["AND", "OR", "NOT", "NEAR"].includes(candidate[0]) ||
        /^\s*:/.test(query.slice(candidate.index + candidate[0].length)))) continue;
    const term = normalize(candidate[0]);
    const prefix = mode === "fts" && /^(?:"\s*)?\*/.test(query.slice(candidate.index + candidate[0].length).trimStart());
    const matched = sourceWords.find((word) => prefix ? word[0].startsWith(term) : word[0] === term);
    if (matched) return positions[matched.index];
  }
  return 0;
}

function excerpt(text, position, budget) {
  let start = text.length <= budget ? 0 : Math.max(0, position - Math.floor(budget / 3));
  start = Math.min(start, Math.max(0, text.length - budget));
  let end = Math.min(text.length, start + budget);
  // Never return half a UTF-16 surrogate pair.
  if (start && /[\uDC00-\uDFFF]/.test(text[start])) start++;
  if (end < text.length && /[\uD800-\uDBFF]/.test(text[end - 1])) end--;
  return { text: text.slice(start, end), charStart: start, charEnd: end, truncated: start > 0 || end < text.length };
}

/** Disposable search postings and graph metadata; JSONL remains authoritative. */
export class TranscriptIndex {
  constructor({ databasePath, sourceRoot } = {}) {
    if (typeof sourceRoot !== "string" || !isAbsolute(sourceRoot) || typeof databasePath !== "string" ||
        !isAbsolute(databasePath) || sourceRoot.includes("\0") || databasePath.includes("\0")) {
      throw new Error("databasePath and sourceRoot must be absolute paths");
    }
    this.sourceRoot = resolve(sourceRoot);
    this.databasePath = resolve(databasePath);
    directory(this.sourceRoot);
    mkdirSync(dirname(this.databasePath), { recursive: true, mode: 0o700 });
    directory(dirname(this.databasePath), true);
    if (!existsSync(this.databasePath)) {
      const fd = openSync(this.databasePath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW, 0o600);
      closeSync(fd);
    }
    const stat = lstatSync(this.databasePath);
    if (!stat.isFile() || stat.isSymbolicLink() || !own(stat) || stat.nlink !== 1 || (stat.mode & 0o077)) {
      throw new Error("index database must be a private owned regular file");
    }
    this.db = new DatabaseSync(this.databasePath);
    try {
      this.db.exec("PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; BEGIN IMMEDIATE");
      const application = this.db.prepare("PRAGMA application_id").get().application_id;
      if (application !== 0 && application !== APPLICATION_ID) throw new Error("unrecognized transcript index database");
      if (application === 0 && this.db.prepare("SELECT name FROM sqlite_master WHERE type='table'").get()) {
        throw new Error("database is not an empty transcript index");
      }
      this.db.exec(`PRAGMA application_id=${APPLICATION_ID};
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, sessionId TEXT,
          dev TEXT NOT NULL, ino TEXT NOT NULL, size INTEGER NOT NULL, mtime TEXT NOT NULL, ctime TEXT NOT NULL,
          parsedBytes INTEGER NOT NULL, sha256 TEXT NOT NULL, leafId TEXT);
        CREATE TABLE IF NOT EXISTS entries(id INTEGER PRIMARY KEY, fileId INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
          entryId TEXT NOT NULL, parentId TEXT, role TEXT, entryType TEXT NOT NULL, timestamp TEXT,
          byteOffset INTEGER NOT NULL, byteLength INTEGER NOT NULL, sha256 TEXT NOT NULL, searchable INTEGER NOT NULL,
          UNIQUE(fileId,entryId));
        CREATE INDEX IF NOT EXISTS entries_file ON entries(fileId,byteOffset);
        CREATE VIRTUAL TABLE IF NOT EXISTS postings USING fts5(body,content='',contentless_delete=1,detail=full,columnsize=1);`);
      const configured = this.db.prepare("SELECT value FROM settings WHERE key='sourceRoot'").get();
      if (configured && configured.value !== this.sourceRoot) throw new Error("index belongs to a different sourceRoot");
      this.db.prepare("INSERT OR IGNORE INTO settings VALUES('sourceRoot',?)").run(this.sourceRoot);
      this.db.exec("COMMIT; CREATE TEMP TABLE scope(id INTEGER PRIMARY KEY, alternate INTEGER NOT NULL)");
    } catch (error) {
      try { this.db.exec("ROLLBACK"); } catch { /* No transaction may have started. */ }
      this.db.close(); this.db = undefined; throw error;
    }
  }

  _open() { if (!this.db) throw new Error("transcript index is closed"); }

  _path(file, allowMissing = false) {
    if (typeof file !== "string" || !isAbsolute(file) || file.includes("\0") || !file.endsWith(".jsonl")) {
      throw new Error("source paths must be absolute .jsonl files");
    }
    const path = resolve(file), within = relative(this.sourceRoot, path);
    if (!within || within === ".." || within.startsWith(`..${sep}`) || isAbsolute(within)) throw new Error("source path is outside sourceRoot");
    const rootStat = directory(this.sourceRoot);
    let parent = this.sourceRoot;
    for (const component of within.split(sep).slice(0, -1)) { parent = resolve(parent, component); directory(parent); }
    let stat;
    try { stat = lstatSync(path, { bigint: true }); }
    catch (error) { if (allowMissing && error.code === "ENOENT") return { path }; throw error; }
    const mode = Number(stat.mode);
    // Pi inherits the user's umask for JSONL files. A private session root
    // prevents group access even when a new source file has mode 0660.
    const unsafeWrite = (mode & 0o002) || ((mode & 0o020) && (rootStat.mode & 0o077));
    if (!stat.isFile() || stat.isSymbolicLink() || !own(stat) || unsafeWrite) {
      throw new Error("source must be an owned nonsymlink regular file with safe permissions");
    }
    return { path, stat: snapshot(stat) };
  }

  _remove(fileId) {
    this.db.prepare("DELETE FROM postings WHERE rowid IN (SELECT id FROM entries WHERE fileId=?)").run(fileId);
    this.db.prepare("DELETE FROM entries WHERE fileId=?").run(fileId);
    this.db.prepare("DELETE FROM files WHERE id=?").run(fileId);
  }

  sync(files) {
    this._open();
    if (!Array.isArray(files)) throw new Error("sync files must be an array");
    const selected = [...new Set(files.map((file) => this._path(file, true).path))];
    const report = { files: 0, indexedEntries: 0, parsedEntries: 0, removedFiles: 0, warnings: [] };
    this.db.exec("BEGIN IMMEDIATE");
    try {
      // A scoped sync preserves other sessions, but never retains missing/unsafe sources.
      for (const file of this.db.prepare("SELECT * FROM files").all()) {
        try { if (this._path(file.path, true).stat) continue; }
        catch { /* Unsafe sources are removed from the derived index. */ }
        this._remove(file.id); report.removedFiles++;
      }
      for (const path of selected) {
        const source = this._path(path, true);
        if (!source.stat) continue;
        const previous = this.db.prepare("SELECT * FROM files WHERE path=?").get(path) ??
          this.db.prepare("SELECT * FROM files WHERE dev=? AND ino=?").get(source.stat.dev, source.stat.ino);
        if (previous && previous.path !== path) this.db.prepare("UPDATE files SET path=? WHERE id=?").run(path, previous.id);
        if (previous && sameFile(previous, source.stat)) continue;
        const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
        try {
          const stat = snapshot(fstatSync(fd, { bigint: true }));
          if (!sameFile(stat, source.stat)) throw new Error("source changed before indexing");
          const append = previous && previous.dev === stat.dev && previous.ino === stat.ino && stat.size >= previous.size &&
            hashRange(fd, previous.size) === previous.sha256;
          if (previous && !append) this._remove(previous.id);
          let file = append ? previous : undefined;
          if (!file) {
            const inserted = this.db.prepare("INSERT INTO files(path,dev,ino,size,mtime,ctime,parsedBytes,sha256) VALUES(?,?,?,?,?,?,0,?)")
              .run(path, stat.dev, stat.ino, stat.size, stat.mtime, stat.ctime, digest(Buffer.alloc(0)));
            file = { id: Number(inserted.lastInsertRowid), parsedBytes: 0, sessionId: null, leafId: null };
          }
          const bytes = readRange(fd, file.parsedBytes, stat.size - file.parsedBytes);
          let offset = 0, sessionId = file.sessionId, leafId = file.leafId;
          while (true) {
            const end = bytes.indexOf(10, offset);
            if (end < 0) break; // A partial trailing JSONL record is deferred.
            const line = bytes.subarray(offset, end);
            const byteOffset = file.parsedBytes + offset;
            offset = end + 1;
            if (!line.toString("utf8").trim()) continue;
            let entry;
            try { entry = JSON.parse(line.toString("utf8")); }
            catch { throw new Error(`invalid JSONL record at byte ${byteOffset} in ${path}`); }
            if (!entry || typeof entry !== "object") throw new Error(`invalid session record at byte ${byteOffset} in ${path}`);
            if (entry.type === "session") {
              if (byteOffset !== 0 || typeof entry.id !== "string" || !entry.id) throw new Error("invalid session header");
              sessionId = entry.id; continue;
            }
            if (!sessionId || typeof entry.id !== "string" || !entry.id || typeof entry.type !== "string" ||
                (entry.parentId != null && typeof entry.parentId !== "string") || entry.parentId === entry.id) {
              throw new Error(`invalid session graph record at byte ${byteOffset} in ${path}`);
            }
            const body = searchable(entry);
            const inserted = this.db.prepare("INSERT INTO entries(fileId,entryId,parentId,role,entryType,timestamp,byteOffset,byteLength,sha256,searchable) VALUES(?,?,?,?,?,?,?,?,?,?)")
              .run(file.id, entry.id, entry.parentId ?? null, body.role ?? null, entry.type,
                typeof entry.timestamp === "string" ? entry.timestamp : null, byteOffset, line.length, digest(line), body.role && body.text.trim() ? 1 : 0);
            if (body.role && body.text.trim()) {
              this.db.prepare("INSERT INTO postings(rowid,body) VALUES(?,?)").run(Number(inserted.lastInsertRowid), body.text);
              report.indexedEntries++;
            }
            report.parsedEntries++; leafId = entry.id;
          }
          const sha256 = hashRange(fd, stat.size);
          if (!sameFile(stat, snapshot(fstatSync(fd, { bigint: true }))) || !sameFile(stat, this._path(path).stat)) {
            throw new Error("source changed during indexing; retry sync");
          }
          this.db.prepare("UPDATE files SET sessionId=?,dev=?,ino=?,size=?,mtime=?,ctime=?,parsedBytes=?,sha256=?,leafId=? WHERE id=?")
            .run(sessionId, stat.dev, stat.ino, stat.size, stat.mtime, stat.ctime, file.parsedBytes + offset, sha256, leafId, file.id);
          report.files++;
        } finally { closeSync(fd); }
      }
      this.db.exec("COMMIT"); return report;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }

  _hydrate(row, state) {
    try {
      const source = this._path(row.sessionFile);
      if (!sameFile(row, source.stat)) throw new Error("source changed");
      const fd = openSync(row.sessionFile, constants.O_RDONLY | constants.O_NOFOLLOW);
      try {
        if (!sameFile(row, snapshot(fstatSync(fd, { bigint: true })))) throw new Error("source changed");
        const bytes = readRange(fd, row.byteOffset, row.byteLength);
        if (digest(bytes) !== row.sha256) throw new Error("source digest changed");
        const entry = JSON.parse(bytes.toString("utf8"));
        if (entry.id !== row.entryId || !sameFile(row, this._path(row.sessionFile).stat)) throw new Error("source changed");
        return searchable(entry);
      } finally { closeSync(fd); }
    } catch {
      state.stale++;
      if (!state.warnings.length) state.warnings.push("Source changed or became unavailable; sync and retry.");
      return undefined;
    }
  }

  search(options = {}) {
    this._open();
    this.db.exec("BEGIN");
    try {
      const result = this._search(options);
      this.db.exec("COMMIT"); return result;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }

  _search({ query, mode = "words", limit = 3, sessionFile, leafId, includeBranches = false, roles,
    maxChars = 6000, aroundEntryId, window = 2 } = {}) {
    this._open();
    if (!["words", "fts", "literal"].includes(mode) || !Number.isInteger(limit) || limit < 1 || limit > 10 ||
        !Number.isInteger(maxChars) || maxChars < 1 || maxChars > MAX_CHARS ||
        !Number.isInteger(window) || window < 0 || window > 3 || typeof includeBranches !== "boolean") throw new Error("invalid search options");
    if (roles !== undefined && (!Array.isArray(roles) || !roles.length || roles.some((role) => !ROLES.has(role)))) throw new Error("invalid search roles");
    if (leafId !== undefined && (typeof leafId !== "string" || !leafId || !sessionFile)) throw new Error("leafId requires a sessionFile");
    if (aroundEntryId !== undefined && (typeof aroundEntryId !== "string" || !aroundEntryId || !sessionFile)) throw new Error("aroundEntryId requires a sessionFile");
    if (!aroundEntryId && (typeof query !== "string" || !query.trim() || query.length > 2048)) throw new Error("query must contain 1–2048 characters");
    if (aroundEntryId && query !== undefined) throw new Error("use query or aroundEntryId, not both");
    const source = sessionFile === undefined ? undefined : this._path(sessionFile, true);
    let scopeFiles = source ? this.db.prepare("SELECT * FROM files WHERE path=?").all(source.path) : this.db.prepare("SELECT * FROM files ORDER BY path").all();
    if (source?.stat && !scopeFiles.length) scopeFiles = this.db.prepare("SELECT * FROM files WHERE dev=? AND ino=?").all(source.stat.dev, source.stat.ino);
    const state = { matches: [], truncated: false, stale: 0, warnings: [] };
    this.db.exec("DELETE FROM scope");
    const around = [];
    for (const file of scopeFiles) {
      const entries = this.db.prepare("SELECT id,entryId,parentId,role,searchable,byteOffset FROM entries WHERE fileId=? ORDER BY byteOffset").all(file.id);
      const byId = new Map(entries.map((entry) => [entry.entryId, entry]));
      const branch = new Set();
      if (leafId && !byId.has(leafId) && !state.warnings.length) state.warnings.push("Requested leaf is not indexed; sync the session and retry.");
      for (let current = byId.get(leafId ?? file.leafId); current && !branch.has(current.entryId); current = byId.get(current.parentId)) branch.add(current.entryId);
      for (const entry of entries) if (entry.searchable && (includeBranches || branch.has(entry.entryId)) && (!roles || roles.includes(entry.role))) {
        this.db.prepare("INSERT INTO scope VALUES(?,?)").run(entry.id, branch.has(entry.entryId) ? 0 : 1);
      }
      if (aroundEntryId) {
        const target = byId.get(aroundEntryId);
        if (!target || (!includeBranches && !branch.has(target.entryId))) continue;
        const context = new Set();
        for (let current = target; current && !context.has(current.entryId); current = byId.get(current.parentId)) context.add(current.entryId);
        // Follow the selected branch after the target, without crossing to an alternate sibling.
        const tail = branch.has(target.entryId) ? entries.filter((entry) => branch.has(entry.entryId) && entry.byteOffset > target.byteOffset) : [];
        const before = window ? entries.filter((entry) => context.has(entry.entryId) && entry.searchable && entry.entryId !== aroundEntryId).slice(-window) : [];
        const after = tail.filter((entry) => entry.searchable).slice(0, window);
        around.push(...before, ...(target.searchable ? [target] : []), ...after);
      }
    }
    const columns = "e.*,f.path AS sessionFile,f.sessionId,f.dev,f.ino,f.size,f.mtime,f.ctime,s.alternate AS alternateBranch";
    let candidates;
    if (aroundEntryId) {
      candidates = around.map((entry) => this.db.prepare(`SELECT ${columns},NULL AS rank FROM entries e JOIN files f ON f.id=e.fileId JOIN scope s ON s.id=e.id WHERE e.id=?`).get(entry.id)).filter(Boolean);
    } else if (mode === "literal") {
      candidates = this.db.prepare(`SELECT ${columns},NULL AS rank FROM entries e JOIN files f ON f.id=e.fileId JOIN scope s ON s.id=e.id ORDER BY f.path,e.byteOffset DESC`).iterate();
    } else {
      const tokens = words(query);
      if (mode === "words" && !tokens.length) throw new Error("words query has no searchable words; use literal mode");
      const expression = mode === "words" ? tokens.map((token) => `"${token}"`).join(" AND ") : query;
      try {
        candidates = this.db.prepare(`SELECT ${columns},postings.rank AS rank FROM postings JOIN entries e ON e.id=postings.rowid JOIN files f ON f.id=e.fileId JOIN scope s ON s.id=e.id WHERE postings MATCH ? ORDER BY postings.rank,e.id LIMIT ?`).all(expression, limit + 1);
      } catch { throw new Error("invalid FTS query"); }
    }
    const hydrated = [];
    for (const row of candidates) {
      const body = this._hydrate(row, state);
      if (!body || (mode === "literal" && !aroundEntryId && !body.text.includes(query))) continue;
      hydrated.push({ row, body });
      if (!aroundEntryId && hydrated.length > limit) { state.truncated = true; break; }
    }
    if (hydrated.length > limit) state.truncated = true;
    let remaining = maxChars;
    const selected = hydrated.slice(0, limit);
    for (let index = 0; index < selected.length && remaining > 0; index++) {
      const { row, body } = selected[index];
      const budget = Math.max(1, Math.floor(remaining / (selected.length - index)));
      const snippet = excerpt(body.text, aroundEntryId ? 0 : matchPosition(body.text, query, mode), budget);
      remaining -= snippet.text.length;
      state.matches.push({ sessionFile: row.sessionFile, sessionId: row.sessionId, entryId: row.entryId, parentId: row.parentId,
        role: row.role, entryType: row.entryType, timestamp: row.timestamp, sourceKinds: body.sourceKinds,
        alternateBranch: Boolean(row.alternateBranch), rank: row.rank, byteOffset: row.byteOffset, byteLength: row.byteLength,
        sha256: row.sha256, ...snippet });
      state.truncated ||= snippet.truncated;
    }
    state.truncated ||= state.matches.length < selected.length;
    return state;
  }

  close() { if (this.db) { this.db.close(); this.db = undefined; } }
}
