import { createHash } from "node:crypto";
import { closeSync, constants, lstatSync, openSync, readFileSync, realpathSync } from "node:fs";
import { dirname } from "node:path";

const SUMMARY_PATH_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY";
const SUMMARY_SHA_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY_SHA256";
const SUMMARY_BYTES_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_SUMMARY_BYTES";
const FILE_OPERATIONS_PATH_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS";
const FILE_OPERATIONS_SHA_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS_SHA256";
const FILE_OPERATIONS_BYTES_ENV = "QWEN_PI_TRANSACTIONAL_COMPACTION_FILE_OPERATIONS_BYTES";
const DIGEST_RE = /^[a-f0-9]{64}$/;

function stableSummary() {
	const path = process.env[SUMMARY_PATH_ENV];
	const expectedSha = process.env[SUMMARY_SHA_ENV];
	const expectedBytes = Number(process.env[SUMMARY_BYTES_ENV]);
	if (typeof path !== "string" || !path.startsWith("/") || path.includes("\0")) {
		throw new Error(`${SUMMARY_PATH_ENV} is missing or invalid`);
	}
	if (!DIGEST_RE.test(expectedSha ?? "")) {
		throw new Error(`${SUMMARY_SHA_ENV} is missing or invalid`);
	}
	if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 1 || expectedBytes > 4 * 1024 * 1024) {
		throw new Error(`${SUMMARY_BYTES_ENV} is missing or invalid`);
	}
	const parent = dirname(path);
	const parentMetadata = lstatSync(parent);
	if (
		!parentMetadata.isDirectory() ||
		parentMetadata.isSymbolicLink() ||
		parentMetadata.uid !== process.getuid() ||
		(parentMetadata.mode & 0o777) !== 0o700 ||
		realpathSync(parent) !== parent
	) {
		throw new Error("transactional-compaction summary parent is not protected");
	}
    const before = lstatSync(path, { bigint: true });
    if (
        !before.isFile() ||
        before.isSymbolicLink() ||
        before.uid !== BigInt(process.getuid()) ||
        before.nlink !== 1n ||
        (before.mode & 0o777n) !== 0o400n ||
        before.size !== BigInt(expectedBytes) ||
        realpathSync(path) !== path
	) {
		throw new Error("transactional-compaction summary has an unsafe identity");
	}
    const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
	let payload;
	try {
		payload = readFileSync(descriptor);
	} finally {
		closeSync(descriptor);
	}
    const after = lstatSync(path, { bigint: true });
	if (
		before.dev !== after.dev ||
		before.ino !== after.ino ||
		before.size !== after.size ||
		before.mtimeNs !== after.mtimeNs ||
		before.ctimeNs !== after.ctimeNs
	) {
		throw new Error("transactional-compaction summary changed while being authenticated");
	}
	const observedSha = createHash("sha256").update(payload).digest("hex");
	if (observedSha !== expectedSha) {
		throw new Error("transactional-compaction summary digest differs");
	}
    let summary;
    try {
        summary = new TextDecoder("utf-8", { fatal: true }).decode(payload);
    } catch (error) {
        throw new Error("transactional-compaction summary is not valid UTF-8", { cause: error });
    }
	if (!summary.trim()) throw new Error("transactional-compaction summary is empty");
	return Object.freeze({ path, sha256: observedSha, summary });
}

function stableFileOperations() {
    const path = process.env[FILE_OPERATIONS_PATH_ENV];
    const expectedSha = process.env[FILE_OPERATIONS_SHA_ENV];
    const expectedBytes = Number(process.env[FILE_OPERATIONS_BYTES_ENV]);
    if (typeof path !== "string" || !path.startsWith("/") || path.includes("\0")) {
        throw new Error(`${FILE_OPERATIONS_PATH_ENV} is missing or invalid`);
    }
    if (!DIGEST_RE.test(expectedSha ?? "")) {
        throw new Error(`${FILE_OPERATIONS_SHA_ENV} is missing or invalid`);
    }
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes < 1 || expectedBytes > 4 * 1024 * 1024) {
        throw new Error(`${FILE_OPERATIONS_BYTES_ENV} is missing or invalid`);
    }
    const parent = dirname(path);
    const parentMetadata = lstatSync(parent);
    const before = lstatSync(path, { bigint: true });
    if (
        !parentMetadata.isDirectory() ||
        parentMetadata.isSymbolicLink() ||
        parentMetadata.uid !== process.getuid() ||
        (parentMetadata.mode & 0o777) !== 0o700 ||
        realpathSync(parent) !== parent ||
        !before.isFile() ||
        before.isSymbolicLink() ||
        before.uid !== BigInt(process.getuid()) ||
        before.nlink !== 1n ||
        (before.mode & 0o777n) !== 0o400n ||
        before.size !== BigInt(expectedBytes) ||
        realpathSync(path) !== path
    ) {
        throw new Error("transactional-compaction file operations have an unsafe identity");
    }
    const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    let payload;
    try {
        payload = readFileSync(descriptor);
    } finally {
        closeSync(descriptor);
    }
    const after = lstatSync(path, { bigint: true });
    if (
        before.dev !== after.dev ||
        before.ino !== after.ino ||
        before.size !== after.size ||
        before.mtimeNs !== after.mtimeNs ||
        before.ctimeNs !== after.ctimeNs
    ) {
        throw new Error("transactional-compaction file operations changed while being authenticated");
    }
    if (createHash("sha256").update(payload).digest("hex") !== expectedSha) {
        throw new Error("transactional-compaction file-operations digest differs");
    }
    let document;
    try {
        document = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(payload));
    } catch (error) {
        throw new Error("transactional-compaction file operations are not canonical JSON", { cause: error });
    }
    const keys = Object.keys(document ?? {}).sort();
    if (
        JSON.stringify(keys) !== JSON.stringify(["modifiedFiles", "readFiles", "schema"]) ||
        document.schema !== "urn:qwen-r9700:deterministic-file-operations:v1"
    ) {
        throw new Error("transactional-compaction file operations have an invalid schema");
    }
    for (const name of ["readFiles", "modifiedFiles"]) {
        const values = document[name];
        if (
            !Array.isArray(values) ||
            values.some((value) => typeof value !== "string" || !value || value.includes("\0")) ||
            JSON.stringify(values) !== JSON.stringify([...new Set(values)].sort())
        ) {
            throw new Error(`transactional-compaction ${name} are not sorted unique paths`);
        }
    }
    if (document.readFiles.some((path) => document.modifiedFiles.includes(path))) {
        throw new Error("transactional-compaction read and modified files overlap");
    }
    return Object.freeze({
        modifiedFiles: Object.freeze([...document.modifiedFiles]),
        readFiles: Object.freeze([...document.readFiles]),
    });
}

export default function qwenTransactionalCompaction(pi) {
    const sealed = stableSummary();
    const fileOperations = stableFileOperations();
	let used = false;
	pi.on("session_before_compact", (event) => {
		if (used) throw new Error("transactional-compaction summary was requested more than once");
		if (event.reason !== "manual" || event.willRetry !== false) {
			throw new Error("transactional-compaction hook is valid only for one manual compaction");
		}
		const preparation = event.preparation;
		if (
			!preparation ||
			typeof preparation.firstKeptEntryId !== "string" ||
			!preparation.firstKeptEntryId ||
			!Number.isSafeInteger(preparation.tokensBefore) ||
			preparation.tokensBefore <= 0
		) {
			throw new Error("Pi supplied an invalid compaction preparation");
		}
		used = true;
		return {
			compaction: {
				summary: sealed.summary,
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				details: {
					schema: "urn:qwen-r9700:prefix-compatible-semantic-summary:v1",
                    strategy: "snapshot-prefix-continuation",
                    summary_sha256: sealed.sha256,
                    readFiles: fileOperations.readFiles,
                    modifiedFiles: fileOperations.modifiedFiles,
                },
			},
		};
	});
}
