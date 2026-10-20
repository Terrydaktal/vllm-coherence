// GNU grep's presentation layer, fed by ripgrep JSON search events. This module
// preserves bytes: JSON's base64 form is used for non-UTF-8 file names/content.
const LF = Buffer.from("\n");
const NUL = Buffer.from([0]);
const bytes = (value) => Buffer.isBuffer(value) ? value : Buffer.from(String(value ?? ""));
const fieldBytes = (value) => value?.bytes !== undefined ? Buffer.from(value.bytes, "base64") : bytes(value?.text ?? "");
const key = (value) => bytes(value).toString("hex");

// rg selects matching records. GNU's literal extraction/highlighting additionally
// chooses the longest needle at the earliest byte position, independently of
// pattern ordering. Search only those selected/context records, preserving raw
// offsets; absent needles are not rescanned for every emitted fragment.
function literalLongestMatches(record, needles) {
  const searches = needles.map((needle) => ({ needle, position: record.indexOf(needle) }));
  const matches = [];
  for (;;) {
    let best;
    for (const search of searches) {
      if (search.position < 0) continue;
      if (!best || search.position < best.position ||
          (search.position === best.position && search.needle.length > best.needle.length)) best = search;
    }
    if (!best) return matches;
    const start = best.position, end = start + best.needle.length;
    matches.push({ start, end });
    for (const search of searches) if (search.position >= 0 && search.position < end)
      search.position = record.indexOf(search.needle, end);
  }
}

function colorConfiguration(env, enabled) {
  const legacy = /^[0-9;]+$/.test(env.GREP_COLOR || "") ? env.GREP_COLOR : "";
  const caps = { ms: legacy || "01;31", mc: legacy || "01;31", sl: "", cx: "", fn: "35", ln: "32", bn: "32", se: "36", ne: false, rv: false };
  let legacyOverridden = false;
  for (const entry of (env.GREP_COLORS || "").split(":")) {
    const split = entry.indexOf("="), name = split < 0 ? entry : entry.slice(0, split);
    if (name === "ne" || name === "rv") caps[name] = true;
    else if (split >= 0) {
      const value = entry.slice(split + 1);
      // GNU accepts numeric SGR capabilities; arbitrary strings aren't emitted.
      if (!/^[0-9;]*$/.test(value)) continue;
      if (name === "mt") { caps.ms = caps.mc = value; legacyOverridden = true; }
      else if (Object.hasOwn(caps, name)) caps[name] = value;
    }
  }
  const erase = caps.ne ? "" : "\x1b[K";
  const start = (cap) => enabled && cap ? Buffer.from(`\x1b[${cap}m${erase}`) : Buffer.alloc(0);
  const reset = enabled ? Buffer.from(`\x1b[m${erase}`) : Buffer.alloc(0);
  const paint = (value, cap) => enabled && cap ? Buffer.concat([start(cap), bytes(value), reset]) : bytes(value);
  return { caps, start, reset, paint, enabled, legacyWarning: enabled && legacy && !legacyOverridden ? `grep: warning: GREP_COLOR='${legacy}' is deprecated; use GREP_COLORS='mt=${legacy}'\n` : null };
}

/**
 * Create a streaming presentation adapter. Register enumerated inputs before
 * their rg batch: rg JSON omits files with no selected lines. finishFile or a
 * later begin event completes preceding zero-match files in operand order.
 * File metadata accepts {size, stdin, binary}; classification should be supplied
 * before the first content event. No complete file or corpus is buffered here.
 */
export function createGrepFormatter(options, { stdout = (data) => process.stdout.write(data), stderr = (data) => process.stderr.write(data), env = process.env, tty = false } = {}) {
  const p = { beforeContext: 0, afterContext: 0, binary: "binary", color: "never", ...options };
  const files = new Map(), active = new Map(), ordered = [];
  const enabled = p.color === "always" || (p.color === "auto" && tty && env.TERM !== "dumb");
  const colors = colorConfiguration(env, enabled);
  if (colors.legacyWarning) stderr(Buffer.from(colors.legacyWarning));
  const terminator = p.nullData ? NUL : LF;
  const longestNeedles = p.fixedLongestPatterns?.map(bytes).filter((needle) => needle.length > 0);
  const contextEnabled = Boolean(p.contextRequested || p.beforeContext || p.afterContext);
  let selected = false, totalCount = 0, errors = 0, outputGroup = false;

  function registerFile(path, metadata = {}) {
    const raw = bytes(path), id = key(raw);
    // Each explicit operand is an occurrence: grep file file searches twice.
    const file = { path: raw, count: 0, done: false, begun: false, error: false, lastLine: null, finalContextLine: null, grouped: false, binaryNotice: false, binary: false, ...metadata };
    if (!files.has(id)) files.set(id, []);
    files.get(id).push(file); ordered.push(file);
    return file;
  }
  function inputFile(path, begin = false) {
    const id = key(path), current = active.get(id);
    if (!begin && current && !current.done) return current;
    const file = files.get(id)?.find((candidate) => !candidate.done && (!begin || !candidate.begun)) ?? registerFile(path);
    if (begin) file.begun = true;
    active.set(id, file);
    return file;
  }
  function displayPath(file) {
    return file.stdin ? bytes(p.label ?? "(standard input)") : file.path;
  }
  function showFilename(file) { return p.filename ?? file.filenameRequired ?? false; }
  function namedPrefix(file, separator) {
    if (!showFilename(file)) return [];
    return [colors.paint(displayPath(file), colors.caps.fn), p.nullNames ? NUL : colors.paint(separator, colors.caps.se)];
  }
  function prefix(file, line, offset, context, hasBody = true) {
    const separator = context ? "-" : ":", result = namedPrefix(file, separator);
    // GNU reserves space for the possible EOF offset, including the byte after
    // the last input byte (a nine-byte file therefore needs two digits).
    const width = p.initialTab ? (file.stdin || !Number.isFinite(file.size) ? 19 : String(Math.max(0, file.size) + 1).length) : 0;
    if (p.lineNumber) result.push(colors.paint(String(line ?? 0).padStart(width), colors.caps.ln), colors.paint(separator, colors.caps.se));
    if (p.byteOffset) result.push(colors.paint(String(offset ?? 0).padStart(width), colors.caps.bn), colors.paint(separator, colors.caps.se));
    if (p.initialTab && result.length && hasBody) result.push(Buffer.from("\t"));
    return result;
  }
  function outputFilename(file) {
    stdout(Buffer.concat([colors.paint(displayPath(file), colors.caps.fn), p.nullNames ? NUL : LF]));
  }
  function complete(file) {
    if (file.done) return;
    file.done = true;
    if (file.error || p.quiet) return;
    if (p.filesMode === "with") { if (file.count > 0) outputFilename(file); }
    else if (p.filesMode === "without") { if (file.count === 0) outputFilename(file); }
    else if (p.count && p.maxCount !== 0) stdout(Buffer.concat([...namedPrefix(file, ":"), bytes(file.count), LF]));
  }
  function finishFile(path) {
    const id = key(path), current = active.get(id);
    const file = current && !current.done ? current : files.get(id)?.find((candidate) => !candidate.done);
    if (file) complete(file);
  }
  function markError(path) { const file = inputFile(path); if (!file.error) { file.error = true; errors++; } }
  function markBinary(path) { inputFile(path).binary = true; }
  function precedingFiles(file) {
    // rg searches the explicitly enumerated batch on one worker, so a new
    // begin proves that earlier operands with no JSON events are complete.
    for (const predecessor of ordered) {
      if (predecessor === file) break;
      complete(predecessor);
    }
  }
  function group(file, line) {
    if (!contextEnabled) return;
    const discontinuous = file.lastLine !== null && Number.isFinite(line) && line > file.lastLine + 1;
    const newFile = !file.grouped;
    if ((newFile && outputGroup) || discontinuous) {
      if (p.groupSeparator !== null) stdout(Buffer.concat([colors.paint(p.groupSeparator ?? "--", colors.caps.se), LF]));
    }
    file.grouped = true; outputGroup = true; file.lastLine = line;
  }
  function content(raw, matches, context) {
    const cap = (p.invertMatch && colors.caps.rv) ? (context ? colors.caps.sl : colors.caps.cx) : (context ? colors.caps.cx : colors.caps.sl);
    if (!enabled) return raw;
    // GNU leaves an unmatched trailing CR outside the line color. A match
    // containing that byte still receives its match color; no byte is removed.
    const trailingCR = raw.length > 0 && raw[raw.length - 1] === 13 && !matches.some((match) => match.end > match.start && match.end > raw.length - 1);
    const tail = trailingCR ? raw.subarray(raw.length - 1) : Buffer.alloc(0);
    if (trailingCR) raw = raw.subarray(0, raw.length - 1);
    if (!raw.length) return tail;
    const result = [], highlight = context ? colors.caps.mc : colors.caps.ms;
    if (cap) result.push(colors.start(cap));
    let lineColorActive = Boolean(cap);
    let cursor = 0;
    for (const match of matches) {
      if (match.end <= match.start || match.start < cursor) continue;
      result.push(raw.subarray(cursor, match.start));
      if (highlight) {
        result.push(colors.start(highlight), raw.subarray(match.start, match.end), colors.reset);
        lineColorActive = false;
        if (cap && match.end < raw.length) { result.push(colors.start(cap)); lineColorActive = true; }
      } else result.push(raw.subarray(match.start, match.end));
      cursor = match.end;
    }
    result.push(raw.subarray(cursor));
    if (lineColorActive) result.push(colors.reset);
    result.push(tail);
    return Buffer.concat(result);
  }
  function event(message) {
    if (!message?.data?.path) return;
    const file = inputFile(fieldBytes(message.data.path), message.type === "begin");
    if (message.type === "begin") { precedingFiles(file); return; }
    if (message.type === "end") { complete(file); return; }
    if (message.type !== "match" && message.type !== "context") return;
    let context = message.type === "context";
    const raw = fieldBytes(message.data.lines);
    const line = message.data.line_number;
    // Some rg inverse/regex searches continue beyond -m's trailing context.
    // GNU ends this operand at the final selected line plus its -A allowance.
    if (file.finalContextLine !== null && Number.isFinite(line) && line > file.finalContextLine) return;
    if (p.binary === "without-match" && file.binary) return;
    if (!context) {
      // rg can mark a matching trailing-context record as a match after -m's
      // selected-record limit. GNU prints that record with context separators.
      if (p.maxCount !== null && p.maxCount !== undefined && file.count >= p.maxCount) context = true;
      else {
        file.count++; totalCount++; selected = true;
        if (p.maxCount !== null && p.maxCount !== undefined && file.count >= p.maxCount && Number.isFinite(line)) file.finalContextLine = line + p.afterContext;
      }
    }
    if (p.quiet || p.filesMode || p.count || p.maxCount === 0) return;
    if (file.binary && p.binary !== "text") {
      if (!context && !file.binaryNotice) { stderr(Buffer.concat([Buffer.from("grep: "), displayPath(file), Buffer.from(": binary file matches\n")])); file.binaryNotice = true; }
      return;
    }
    const record = raw.length && raw[raw.length - 1] === terminator[0] ? raw.subarray(0, raw.length - 1) : raw;
    // GNU stops ordinary matching after -m and prints its remaining -A records
    // as unsearched context. Inverse matching still identifies context spans.
    const unsearchedTail = !p.invertMatch && file.finalContextLine !== null && Number.isFinite(line) &&
      line > file.finalContextLine - p.afterContext;
    // rg selects NUL records correctly with absolute whole-record anchors, but
    // some releases omit their spans after the first selected record. A positive
    // -x selection already proves that its complete body is the matched span.
    const wholeRecord = p.nullData && p.lineRegexp && !p.posixSpanArgs && !p.invertMatch && !context;
    const matches = unsearchedTail ? [] : wholeRecord ? [{ start: 0, end: record.length }] :
      longestNeedles ? literalLongestMatches(record, longestNeedles) : message.data.submatches || [];
    const absoluteOffset = message.data.absolute_offset ?? 0;
    if (p.onlyMatching) {
      // Selected empty/negative lines still establish context groups. Inverse
      // -o has no selected fragments but can print matches in context records.
      group(file, line);
      if (context && !p.invertMatch) return;
      const fragments = matches.filter((match) => match.end > match.start || match.extraTerminators);
      for (const match of fragments) stdout(Buffer.concat([...prefix(file, line, absoluteOffset + match.start, context),
        colors.paint(Buffer.concat([raw.subarray(match.start, match.end),
          ...(match.extraTerminators ? [Buffer.alloc(match.extraTerminators, terminator[0])] : [])]),
        context ? colors.caps.mc : colors.caps.ms), terminator]));
      return;
    }
    group(file, line);
    // GNU always supplies the record terminator, including an unterminated
    // final input record. Keep CR bytes; grep does not apply rg's CRLF policy.
    const body = record;
    const matchedTerminator = matches.some((match) => match.extraTerminators && match.end === body.length);
    const coloredBody = matchedTerminator ? Buffer.concat([body, terminator]) : body;
    const coloredMatches = matchedTerminator ? matches.map((match) => ({ ...match,
      end: match.end + (match.extraTerminators && match.end === body.length ? 1 : 0) })) : matches;
    stdout(Buffer.concat([...prefix(file, line, absoluteOffset, context, body.length > 0),
      content(coloredBody, coloredMatches, context), ...(matchedTerminator ? [] : [terminator])]));
  }
  function finish() {
    for (const file of ordered) complete(file);
    return { selected, count: totalCount, errors, status: p.quiet && selected ? 0 : errors ? 2 : selected ? 0 : 1 };
  }
  return { event, registerFile, finishFile, markBinary, markError, finish, get selected() { return selected; } };
}
