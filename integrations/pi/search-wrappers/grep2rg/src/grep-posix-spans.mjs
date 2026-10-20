// GNU is used only to resolve spans in records selected by rg. The original
// files are never passed to this helper: their selection, context and metadata
// come from rg. Disk spools bound memory to the current record, not the corpus.
import { spawn } from "node:child_process";
import { createReadStream, closeSync, mkdtempSync, openSync, rmSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { inheritedGrepStdio } from "./grep-descriptors.mjs";

const dataBytes = (field) => field?.bytes !== undefined ? Buffer.from(field.bytes, "base64") : Buffer.from(field?.text ?? "");

function writeAll(descriptor, value) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  let written = 0;
  while (written < bytes.length) {
    const length = writeSync(descriptor, bytes, written, bytes.length - written);
    if (!length) throw new Error("GNU span spool write made no progress");
    written += length;
  }
}

export function gnuSpanArguments(options, patterns) {
  return ["--text", "--color=never", "--no-filename", "--line-number", "--byte-offset", "--only-matching",
    { basic: "--basic-regexp", extended: "--extended-regexp", fixed: "--fixed-strings", perl: "--perl-regexp" }[options.matcher],
    ...(options.ignoreCase ? ["--ignore-case"] : []), ...(options.wordRegexp ? ["--word-regexp"] : []),
    ...(options.lineRegexp ? ["--line-regexp"] : []), ...(options.nullData ? ["--null-data"] : []),
    ...patterns.flatMap((pattern) => ["--regexp", pattern])];
}

async function* delimitedRecords(path, delimiter) {
  let pending = Buffer.alloc(0);
  for await (const chunk of createReadStream(path)) {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
    let position;
    while ((position = pending.indexOf(delimiter)) >= 0) {
      yield pending.subarray(0, position);
      pending = pending.subarray(position + 1);
    }
  }
  if (pending.length) throw new Error("unterminated GNU span record");
}

function parseSpan(record) {
  const first = record.indexOf(58), second = record.indexOf(58, first + 1);
  if (first < 1 || second < first + 2) throw new Error("invalid GNU span prefix");
  const ordinal = record.subarray(0, first).toString("ascii"), offset = record.subarray(first + 1, second).toString("ascii");
  if (!/^\d+$/.test(ordinal) || !/^\d+$/.test(offset)) throw new Error("invalid GNU span offset");
  return { ordinal: BigInt(ordinal), offset: BigInt(offset), length: record.length - second - 1 };
}

async function* spanRecords(path, delimiter) {
  let previous;
  for await (const record of delimitedRecords(path, delimiter)) {
    // Some GNU releases include the input terminator in a -w -x match. GNU -o
    // adds its own delimiter too; retain the matched delimiter for color spans.
    if (!record.length && previous) { previous.extraTerminators++; continue; }
    const next = { ...parseSpan(record), extraTerminators: 0 };
    if (previous) yield previous;
    previous = next;
  }
  if (previous) yield previous;
}

export function createPosixSpanBroker(args, { nullData = false } = {}) {
  const directory = mkdtempSync(join(tmpdir(), "grep2rg-posix-"));
  const eventsPath = join(directory, "events.jsonl"), inputPath = join(directory, "candidate-records"), spansPath = join(directory, "spans");
  const terminator = Buffer.from([nullData ? 0 : 10]);
  let events, input, closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    for (const descriptor of [events, input]) if (descriptor !== undefined) { try { closeSync(descriptor); } catch {} }
    process.removeListener("exit", close);
    rmSync(directory, { recursive: true, force: true });
  };
  try {
    events = openSync(eventsPath, "wx", 0o600); input = openSync(inputPath, "wx", 0o600);
  } catch (error) { close(); throw error; }
  process.once("exit", close);
  return {
    capture(message, rawJson) {
      writeAll(events, rawJson + "\n");
      if (message.type !== "match" && message.type !== "context") return;
      const raw = dataBytes(message.data.lines);
      writeAll(input, raw);
      if (!raw.length || raw[raw.length - 1] !== terminator[0]) writeAll(input, terminator);
    },
    async replay(formatter, { setChild = () => {}, stopped = () => false, waitForOutput = async () => {} } = {}) {
      closeSync(events); events = undefined; closeSync(input); input = undefined;
      if (stopped()) return;
      const spans = openSync(spansPath, "wx", 0o600);
      let diagnostic = "";
      let status;
      try {
        const child = spawn("grep", [...args, "--", inputPath], { stdio: inheritedGrepStdio(["ignore", spans, "pipe"]) });
        setChild(child);
        status = await new Promise((resolve) => {
          child.once("error", (error) => { diagnostic = `grep: GNU span matcher: ${error.code ?? "spawn failure"}\n`; resolve(2); });
          child.stderr.on("data", (data) => { diagnostic += data; if (diagnostic.length > 65536) diagnostic = diagnostic.slice(0, 65536); });
          child.once("close", (code) => resolve(code ?? 2));
        });
      } finally { closeSync(spans); setChild(null); }
      if (stopped()) return;
      if (status > 1 || diagnostic) throw new Error(diagnostic || "GNU span matcher failed");
      const iterator = spanRecords(spansPath, terminator[0])[Symbol.asyncIterator]();
      let next = await iterator.next(), ordinal = 0n, payloadOffset = 0n;
      let parsed = next.done ? undefined : next.value;
      const lines = createInterface({ input: createReadStream(eventsPath), crlfDelay: Infinity });
      try {
        for await (const json of lines) {
          if (stopped()) break;
          const message = JSON.parse(json);
          if (message.type === "match" || message.type === "context") {
            ordinal++;
            const raw = dataBytes(message.data.lines), terminated = raw.length && raw[raw.length - 1] === terminator[0];
            const bodyLength = raw.length - (terminated ? 1 : 0), matches = [];
            if (parsed && parsed.ordinal < ordinal) throw new Error("out-of-order GNU span record");
            while (parsed?.ordinal === ordinal) {
              const start = parsed.offset - payloadOffset, end = start + BigInt(parsed.length);
              if (start < 0n || end > BigInt(bodyLength)) throw new Error("GNU span outside candidate record");
              matches.push({ start: Number(start), end: Number(end), extraTerminators: parsed.extraTerminators });
              next = await iterator.next(); parsed = next.done ? undefined : next.value;
            }
            message.data.submatches = matches;
            payloadOffset += BigInt(raw.length + (terminated ? 0 : 1));
          }
          formatter.event(message);
          await waitForOutput();
        }
        if (!stopped() && parsed) throw new Error("unconsumed GNU span records");
      } finally { lines.close(); lines.input.destroy(); await iterator.return?.(); }
    },
    close,
  };
}
