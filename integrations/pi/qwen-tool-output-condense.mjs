import { createHash, randomUUID } from "node:crypto";
import {
    chmodSync,
    closeSync,
    fsyncSync,
    linkSync,
    lstatSync,
    mkdirSync,
    openSync,
    readFileSync,
    readdirSync,
    unlinkSync,
    writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const TARGET_PROVIDER = "qwen-r9700";
const MAX_CONTEXT_BYTES_ENV = "QWEN_PI_TOOL_RESULT_MAX_BYTES";
const OUTPUT_DIR_ENV = "QWEN_PI_TOOL_RESULT_DIR";
const FIXED_SLOT_RESUME_ENV = "QWEN_PI_FIXED_SLOT_RESUME";
const LIVE_ARCHIVE_SCHEMA = "qwen-pi-live-tool-result-v1";
const REHYDRATE_TOOL_NAME = "qwen_rehydrate_tool_turn";
const DEFAULT_OUTPUT_DIR = join(
    process.env.XDG_STATE_HOME ?? join(homedir(), ".local", "state"),
    "qwen-r9700",
    "pi-tool-results",
);
const ERROR_PATTERN =
    /\b(error|errors|warning|warnings|warn|failed|failure|fatal|exception|traceback|panic|segmentation fault|denied|timeout|timed out)\b/i;
const MULTILINE_ERROR_START_PATTERN =
    /\b(?:traceback \(most recent call last\):?|exception in thread\b|fatal python error\b|panic:|stack trace:)/i;
const TEST_PATTERN =
    /\b(pytest|unittest|passed|failed|failure|skipped|xfailed|xpassed|ruff|shellcheck|shfmt|cargo test|ctest)\b/i;
const COMPILER_PATTERN =
    /(?:^|\s)(?:[^\s:]+:\d+(?::\d+)?:\s*)?(?:fatal\s+)?(?:error|warning|note):|undefined reference|linker command failed/i;
const BENCHMARK_PATTERN =
    /\b(?:\d+(?:\.\d+)?\s*(?:ms|s|us|µs|ns|t\/s|tok(?:en)?s?\/s|MB\/s|GB\/s)|latency|throughput|elapsed|median|p\d{2}|speedup)\b/i;
const DIFF_PATTERN = /^(?:diff --git|index [0-9a-f]+\.\.[0-9a-f]+|@@ |--- |\+\+\+ |[+-](?![+-]))/;
const TABLE_PATTERN = /(?:^\s*\|.*\|\s*$)|(?:\S+\s{2,}\S+\s{2,}\S+)/;
const JSON_PATTERN = /^\s*(?:[{}\[\]]|"[^"\n]+"\s*:)/;
const MAX_LINE_CHARACTERS = 180;
const MAX_SIGNAL_SCAN_CHARACTERS = 16 * 1024;
const GUIDANCE_MARKER = "Qwen bounded tool-output discipline:";

// Keep ordinary, decision-relevant tool results inline. The former 1.5 KiB
// budget archived small source excerpts and immediately forced extra model
// turns to retrieve them again. Results beyond this budget are still published
// to the authenticated content-addressed archive before the bounded view is
// returned to the model, so condensation never discards evidence.
export const DEFAULT_MAX_CONTEXT_BYTES = 8 * 1024;
export const TOOL_OUTPUT_GUIDANCE = `${GUIDANCE_MARKER}
- Keep command output small at its source. Prefer rg with a narrow pattern and path, jq projections or selected keys, sed -n with a focused line range, and focused tests.
- Do not dump whole files, recursive trees, unbounded searches, complete logs, or full test suites when a targeted query answers the question.
- For potentially large commands, obtain counts first and request bounded sections or matched diagnostic blocks.
- Oversized tool results are retained as immutable SHA-256-addressed archives. Use qwen_rehydrate_tool_turn with a literal pattern or tight start_line/end_line range for exact follow-up evidence; a digest-only call returns only a small preview.`;

const UNBOUNDED_ROOT_FIND_PATTERN =
    /(?:^|[\s;&|()])(?:\/(?:usr\/)?bin\/)?find\s+\/(?=\s|$)/;
const UNBOUNDED_ROOT_FIND_REASON =
    "Refusing an unbounded find / scan. Search a known bounded root such as the working directory or the relevant package/state directory, and narrow by name or pattern.";

export function isUnboundedRootFind(toolName, input) {
    return (
        toolName === "bash" &&
        typeof input?.command === "string" &&
        UNBOUNDED_ROOT_FIND_PATTERN.test(input.command)
    );
}

function monotonicNow() {
    return typeof globalThis.performance?.now === "function" ? globalThis.performance.now() : Date.now();
}

function sha256(data) {
    return createHash("sha256").update(data).digest("hex");
}

function configuredMaxBytes() {
    const parsed = Number(process.env[MAX_CONTEXT_BYTES_ENV] ?? DEFAULT_MAX_CONTEXT_BYTES);
    return Number.isSafeInteger(parsed) && parsed >= 1024 && parsed <= 16_384
        ? parsed
        : DEFAULT_MAX_CONTEXT_BYTES;
}

function outputDirectory() {
    const configured = process.env[OUTPUT_DIR_ENV];
    return typeof configured === "string" && configured.startsWith("/") ? configured : DEFAULT_OUTPUT_DIR;
}

function ensureDirectory(path) {
    mkdirSync(path, { mode: 0o700, recursive: true });
    const stat = lstatSync(path);
    if (
        !stat.isDirectory() ||
        stat.isSymbolicLink() ||
        (typeof process.getuid === "function" && stat.uid !== process.getuid()) ||
        (stat.mode & 0o777) !== 0o700
    ) {
        throw new Error(`tool-result archive directory is not an owned mode-0700 directory: ${path}`);
    }
}

function fsyncDirectory(path) {
    const descriptor = openSync(path, "r");
    try {
        fsyncSync(descriptor);
    } finally {
        closeSync(descriptor);
    }
}

function authenticateArchive(path, digest, bytes) {
    const stat = lstatSync(path);
    if (
        !stat.isFile() ||
        stat.isSymbolicLink() ||
        stat.nlink !== 1 ||
        (typeof process.getuid === "function" && stat.uid !== process.getuid()) ||
        (stat.mode & 0o777) !== 0o400 ||
        stat.size !== bytes
    ) {
        throw new Error(`tool-result archive protected identity differs: ${path}`);
    }
    const observed = sha256(readFileSync(path));
    if (observed !== digest) throw new Error(`tool-result archive SHA-256 mismatch: ${path}`);
}

function recoverInterruptedArchive(path, digest, bytes, prefixDirectory) {
    const before = lstatSync(path);
    if (
        !before.isFile() ||
        before.isSymbolicLink() ||
        before.nlink !== 2 ||
        (typeof process.getuid === "function" && before.uid !== process.getuid()) ||
        (before.mode & 0o777) !== 0o400 ||
        before.size !== bytes
    ) {
        throw new Error(`tool-result archive protected identity differs: ${path}`);
    }
    const data = readFileSync(path);
    const afterRead = lstatSync(path);
    if (afterRead.nlink === 1) {
        authenticateArchive(path, digest, bytes);
        return;
    }
    if (
        before.dev !== afterRead.dev ||
        before.ino !== afterRead.ino ||
        before.size !== afterRead.size ||
        afterRead.nlink !== 2 ||
        sha256(data) !== digest
    ) {
        throw new Error(`interrupted tool-result archive changed while it was authenticated: ${path}`);
    }

    const temporaryPattern = new RegExp(
        `^\\.${digest}\\.[1-9][0-9]*\\.[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\\.tmp$`,
    );
    const candidates = [];
    for (const entry of readdirSync(prefixDirectory, { withFileTypes: true })) {
        if (!entry.isFile() || !temporaryPattern.test(entry.name)) continue;
        const candidate = join(prefixDirectory, entry.name);
        const metadata = lstatSync(candidate);
        if (metadata.dev === before.dev && metadata.ino === before.ino) candidates.push(candidate);
    }
    if (candidates.length === 0) {
        // The original publisher may have removed its temporary link between
        // our directory scan and this check.
        authenticateArchive(path, digest, bytes);
        return;
    }
    if (candidates.length !== 1) {
        throw new Error(`interrupted tool-result archive has ambiguous temporary links: ${path}`);
    }
    try {
        unlinkSync(candidates[0]);
    } catch (error) {
        if (error?.code !== "ENOENT") throw error;
    }
    fsyncDirectory(prefixDirectory);
    authenticateArchive(path, digest, bytes);
}

function retainText(text) {
    const data = Buffer.from(text, "utf8");
    const digest = sha256(data);
    const root = outputDirectory();
    const algorithmDirectory = join(root, "sha256");
    const prefixDirectory = join(algorithmDirectory, digest.slice(0, 2));
    ensureDirectory(root);
    ensureDirectory(algorithmDirectory);
    ensureDirectory(prefixDirectory);
    const path = join(prefixDirectory, `${digest}.txt`);
    try {
        authenticateArchive(path, digest, data.length);
        return { bytes: data.length, path, sha256: digest };
    } catch (error) {
        if (error?.code !== "ENOENT") {
            recoverInterruptedArchive(path, digest, data.length, prefixDirectory);
            return { bytes: data.length, path, sha256: digest };
        }
    }

    const temporary = join(prefixDirectory, `.${digest}.${process.pid}.${randomUUID()}.tmp`);
    const descriptor = openSync(temporary, "wx", 0o400);
    try {
        writeFileSync(descriptor, data);
        fsyncSync(descriptor);
    } finally {
        closeSync(descriptor);
    }
    try {
        linkSync(temporary, path);
        fsyncDirectory(prefixDirectory);
    } catch (error) {
        if (error?.code !== "EEXIST") throw error;
    } finally {
        try {
            unlinkSync(temporary);
        } catch (error) {
            if (error?.code !== "ENOENT") throw error;
        }
        fsyncDirectory(prefixDirectory);
    }
    chmodSync(path, 0o400);
    authenticateArchive(path, digest, data.length);
    return { bytes: data.length, path, sha256: digest };
}

function clampLine(line) {
    const source = String(line);
    const wasTrimmed = source.length > MAX_LINE_CHARACTERS;
    const clean = cleanLine(wasTrimmed ? source.slice(0, MAX_LINE_CHARACTERS - 14) : source);
    return wasTrimmed ? `${clean}... [trimmed]` : clean;
}

function cleanLine(line) {
    return String(line).replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "");
}

function textLines(text) {
    const lines = text.length === 0 ? [] : text.split("\n");
    if (text.endsWith("\n")) lines.pop();
    return lines;
}

function entriesMatching(lines, pattern, maximum = 24) {
    const entries = [];
    for (let index = 0; index < lines.length && entries.length < maximum; index += 1) {
        const line = lines[index];
        const matches =
            pattern === TABLE_PATTERN
                ? isTableLine(line)
                : pattern.test(line.slice(0, MAX_SIGNAL_SCAN_CHARACTERS));
        if (matches) entries.push({ line: index + 1, text: clampLine(line) });
    }
    return entries;
}

function isTableLine(line) {
    const bounded = line.slice(0, MAX_SIGNAL_SCAN_CHARACTERS);
    const trimmed = bounded.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) return true;

    let separatedFields = 0;
    let sawNonWhitespace = false;
    for (let index = 0; index < bounded.length; ) {
        if (!/\s/.test(bounded[index])) {
            sawNonWhitespace = true;
            index += 1;
            continue;
        }
        const start = index;
        while (index < bounded.length && /\s/.test(bounded[index])) index += 1;
        if (sawNonWhitespace && index - start >= 2 && index < bounded.length) {
            separatedFields += 1;
            if (separatedFields >= 2) return true;
        }
    }
    return false;
}

function errorBlocks(lines) {
    const blocks = [];
    let index = 0;
    while (index < lines.length) {
        if (!ERROR_PATTERN.test(lines[index])) {
            index += 1;
            continue;
        }
        const multiline = MULTILINE_ERROR_START_PATTERN.test(lines[index]);
        let start = index;
        while (start > 0 && lines[start - 1].trim() !== "" && index - start < 2) start -= 1;
        let end = index;
        // Tracebacks are indivisible through the next paragraph boundary.  A
        // standalone warning/error keeps bounded surrounding context instead of
        // swallowing an entire newline-delimited command result with no blanks.
        while (
            end + 1 < lines.length &&
            lines[end + 1].trim() !== "" &&
            (multiline || end - index < 2)
        ) {
            end += 1;
        }
        if (blocks.length > 0 && start <= blocks.at(-1).end + 1) {
            blocks.at(-1).end = Math.max(blocks.at(-1).end, end);
        } else {
            blocks.push({ start, end });
        }
        index = end + 1;
    }
    return blocks.slice(0, 8).map(({ start, end }) => ({
        end: end + 1,
        lines: lines.slice(start, end + 1).map((line, offset) => ({
            line: start + offset + 1,
            // Error blocks are atomic evidence.  Keep every line byte-for-byte
            // apart from unsafe controls; fitErrorBlocks either includes the
            // complete block or emits only its authenticated archive locator.
            text: cleanLine(line),
        })),
        start: start + 1,
    }));
}

export function inspectText(text) {
    const lines = textLines(text);
    const entries = lines.map((line, index) => ({ line: index + 1, text: clampLine(line) }));
    return {
        bytes: Buffer.byteLength(text, "utf8"),
        errorBlocks: errorBlocks(lines),
        head: entries.slice(0, 12),
        lines: lines.length,
        signals: {
            benchmarks: entriesMatching(lines, BENCHMARK_PATTERN),
            compiler: entriesMatching(lines, COMPILER_PATTERN),
            diffs: entriesMatching(lines, DIFF_PATTERN),
            json: entriesMatching(lines, JSON_PATTERN),
            tables: entriesMatching(lines, TABLE_PATTERN),
            tests: entriesMatching(lines, TEST_PATTERN),
        },
        tail: entries.slice(-16),
    };
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function statusFor(event, text) {
    if (event.toolName !== "bash") return event.isError ? "error" : "success";
    const exited = text.match(/Command exited with code\s+(-?\d+)/i);
    if (exited) return exited[1];
    if (/Command (aborted|timed out)/i.test(text)) return "cancelled";
    return event.isError ? "error" : "0";
}

function encodedBytes(value) {
    return Buffer.byteLength(value, "utf8");
}

function fitEntries(title, entries, budget, fromEnd = false) {
    if (budget < encodedBytes(`${title}:\n(none)`)) return "";
    const chosen = [];
    const candidates = fromEnd ? [...entries].reverse() : entries;
    for (const entry of candidates) {
        const candidate = fromEnd ? [entry, ...chosen] : [...chosen, entry];
        const rendered = `${title}:\n${candidate.map((item) => `${item.line}: ${item.text}`).join("\n")}`;
        if (encodedBytes(rendered) > budget) break;
        chosen.splice(0, chosen.length, ...candidate);
    }
    return chosen.length === 0 ? `${title}:\n(none)` : `${title}:\n${chosen.map((entry) => `${entry.line}: ${entry.text}`).join("\n")}`;
}

function fitErrorBlocks(blocks, budget) {
    const title = "Complete error/traceback blocks";
    if (blocks.length === 0 || budget <= 0) return "";
    const renderedBlocks = [];
    for (const block of blocks) {
        const rendered = block.lines.map((entry) => `${entry.line}: ${entry.text}`).join("\n");
        const candidate = `${title}:\n${[...renderedBlocks, rendered].join("\n---\n")}`;
        if (encodedBytes(candidate) > budget) {
            if (renderedBlocks.length === 0) {
                const locator = `${title}:\n(lines ${block.start}-${block.end} retained whole in authenticated archive)`;
                return encodedBytes(locator) <= budget ? locator : "";
            }
            break;
        }
        renderedBlocks.push(rendered);
    }
    return `${title}:\n${renderedBlocks.join("\n---\n")}`;
}

function prioritizedSignals(signals, blocks) {
    const result = [];
    const seen = new Set();
    for (const [label, entries] of Object.entries(signals)) {
        for (const entry of entries) {
            if (blocks.some((block) => entry.line >= block.start && entry.line <= block.end)) continue;
            const key = `${entry.line}:${entry.text}`;
            if (seen.has(key)) continue;
            seen.add(key);
            result.push({ ...entry, text: `[${label}] ${entry.text}` });
        }
    }
    return result.sort((left, right) => left.line - right.line);
}

export function buildCondensedSummary(
    { archive, durationMs, inspection, status, toolName },
    maxBytes,
) {
    const duration = Number.isFinite(durationMs) ? `${(durationMs / 1000).toFixed(2)} s` : "unavailable";
    const mandatoryMetadata = [
        `[${toolName} output condensed for model context]`,
        `Record schema: ${LIVE_ARCHIVE_SCHEMA}`,
        `Exit status: ${status}`,
        `Original size: ${inspection.lines} lines / ${inspection.bytes} bytes (${formatBytes(inspection.bytes)})`,
        `Archive SHA-256: ${archive.sha256}`,
        `Content-addressed archive: ${archive.path}`,
        "Semantic summary: absent; the exact archive is authoritative.",
        "Do not infer omitted content. Before depending on it, call qwen_rehydrate_tool_turn with this SHA-256 and a literal pattern or tight start_line/end_line range.",
    ].join("\n");
    const durationLine = `Duration: ${duration}`;
    const metadata =
        encodedBytes(`${mandatoryMetadata}\n${durationLine}`) + 230 + 8 <= maxBytes
            ? `${mandatoryMetadata}\n${durationLine}`
            : mandatoryMetadata;
    const separatorBytes = encodedBytes("\n\n") * 4;
    const remaining = maxBytes - encodedBytes(metadata) - separatorBytes;
    if (remaining < 220) {
        throw new Error(`configured condensation budget is too small for authenticated metadata and tail: ${maxBytes}`);
    }

    // Budgets are explicit and sum exactly to the available bytes.  Tail owns a
    // hard floor large enough for the final, clamped line; every other section
    // divides only what remains and therefore cannot crowd the tail out.
    const tailBudget = Math.max(220, Math.floor(remaining * 0.24));
    const nonTailBudget = remaining - tailBudget;
    const errorBudget = Math.floor(nonTailBudget * 0.5);
    const signalBudget = Math.floor(nonTailBudget * 0.3);
    const headBudget = nonTailBudget - errorBudget - signalBudget;
    const sections = [
        fitErrorBlocks(inspection.errorBlocks, errorBudget),
        fitEntries(
            "Tool-aware signals",
            prioritizedSignals(inspection.signals, inspection.errorBlocks),
            signalBudget,
        ),
        fitEntries("Head", inspection.head, headBudget),
        fitEntries("Tail (guaranteed)", inspection.tail, tailBudget, true),
    ].filter(Boolean);
    const summary = [metadata, ...sections].join("\n\n");
    if (encodedBytes(summary) > maxBytes) {
        throw new Error(`sectional condensation exceeded its exact byte budget: ${encodedBytes(summary)} > ${maxBytes}`);
    }
    if (!summary.includes("Tail (guaranteed):\n") || inspection.tail.length > 0 && !summary.includes(`${inspection.tail.at(-1).line}:`)) {
        throw new Error("sectional condensation failed to preserve the final output line");
    }
    return summary;
}

function textContent(content) {
    if (!Array.isArray(content) || content.some((block) => block?.type !== "text")) return undefined;
    return content.map((block) => String(block.text ?? "")).join("\n");
}

export default function qwenToolOutputCondense(pi) {
    const started = new Map();

    pi.on("before_agent_start", (event, ctx) => {
        if (
            process.env[FIXED_SLOT_RESUME_ENV] === "1" ||
            ctx?.model?.provider !== TARGET_PROVIDER ||
            event.systemPrompt.includes(GUIDANCE_MARKER)
        ) {
            return undefined;
        }
        return { systemPrompt: `${event.systemPrompt}\n\n${TOOL_OUTPUT_GUIDANCE}` };
    });

    pi.on("tool_execution_start", (event) => {
        started.set(event.toolCallId, monotonicNow());
    });

    pi.on("tool_call", (event) => {
        if (!isUnboundedRootFind(event.toolName, event.input)) return undefined;
        return { block: true, reason: UNBOUNDED_ROOT_FIND_REASON };
    });

    pi.on("tool_result", async (event) => {
        const startedAt = started.get(event.toolCallId);
        started.delete(event.toolCallId);
        // Rehydration has already authenticated and bounded the authoritative
        // archive. Re-condensing its output creates a second digest, obscures
        // the original archive identity, and can send the model into a
        // recursive rehydrate/condense loop.
        if (event.toolName === REHYDRATE_TOOL_NAME) return undefined;
        const contentText = textContent(event.content);
        if (contentText === undefined) return undefined;

        try {
            const maxBytes = configuredMaxBytes();
            const existingPath = event.details?.fullOutputPath;
            let exactText = contentText;
            if (typeof existingPath === "string" && existingPath.startsWith("/")) {
                try {
                    const stat = lstatSync(existingPath);
                    if (
                        stat.isFile() &&
                        !stat.isSymbolicLink() &&
                        (typeof process.getuid !== "function" || stat.uid === process.getuid())
                    ) {
                        exactText = readFileSync(existingPath, "utf8");
                    }
                } catch {
                    // The inline text remains the exact available result.
                }
            }
            const inspection = inspectText(exactText);
            if (inspection.bytes <= maxBytes) return undefined;
            const archive = retainText(exactText);
            const summary = buildCondensedSummary(
                {
                    archive,
                    durationMs: startedAt === undefined ? undefined : Math.max(0, monotonicNow() - startedAt),
                    inspection,
                    status: statusFor(event, contentText),
                    toolName: event.toolName,
                },
                maxBytes,
            );
            const details = {
                ...(event.details ?? {}),
                fullOutputPath: archive.path,
                qwenToolResultArchive: {
                    bytes: archive.bytes,
                    path: archive.path,
                    schema: LIVE_ARCHIVE_SCHEMA,
                    semanticSummary: null,
                    sha256: archive.sha256,
                },
            };
            return { content: [{ type: "text", text: summary }], details };
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            process.stderr.write(`pi-remote-qwen: tool-result condensation failed open: ${message}\n`);
            return undefined;
        }
    });

    pi.on("agent_settled", () => started.clear());
    pi.on("session_shutdown", () => started.clear());
}
