import { createHash, randomBytes } from "node:crypto";
import {
    closeSync,
    constants,
    fsyncSync,
    linkSync,
    lstatSync,
    mkdirSync,
    openSync,
    readFileSync,
    realpathSync,
    unlinkSync,
    writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

const REQUEST_ROOT_ENV = "QWEN_PI_COMPACTION_HANDOFF_REQUEST_ROOT";
const SOURCE_ENV = "QWEN_PI_COMPACTION_HANDOFF_SOURCE";
const TRANSACTION_ROOT_ENV = "QWEN_PI_COMPACTION_HANDOFF_TRANSACTION_ROOT";
const LOGICAL_SESSION_ENV = "QWEN_PI_COMPACTION_HANDOFF_LOGICAL_SESSION_ID";
const COMPACTION_ABI_ENV = "QWEN_PI_COMPACTION_HANDOFF_ABI_SHA256";
const PROMPT_ABI_ENV = "QWEN_PI_COMPACTION_HANDOFF_PROMPT_ABI";
const RESUME_REQUEST_ENV = "QWEN_PI_COMPACTION_RESUME_REQUEST";
const RESUME_REQUEST_SHA_ENV = "QWEN_PI_COMPACTION_RESUME_REQUEST_SHA256";
const RESUME_SUBMITTED_ENV = "QWEN_PI_COMPACTION_RESUME_SUBMITTED";
const SAFE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA256_RE = /^[a-f0-9]{64}$/;
const TRANSACTION_KEY_DOMAIN = "qwen-r9700-transactional-compaction-key-v3\0";
const ALLOWED_PROMPT_ABIS = new Set([
	"searchtool-toolarchive-v1",
	"searchtool-toolarchive-outcome-v1",
]);

function sha256(payload) {
    return createHash("sha256").update(payload).digest("hex");
}

function transactionKey(sourceSha256, compactionAbiSha256, promptAbi) {
	if (!SHA256_RE.test(sourceSha256) || !SHA256_RE.test(compactionAbiSha256)) {
		throw new Error("transactional compaction identity contains an invalid digest");
	}
	if (!ALLOWED_PROMPT_ABIS.has(promptAbi)) {
		throw new Error("transactional compaction identity contains an unsupported prompt ABI");
	}
	return sha256(
		Buffer.from(
			`${TRANSACTION_KEY_DOMAIN}${sourceSha256}${compactionAbiSha256}${promptAbi}`,
			"utf8",
		),
	);
}

function protectedDirectory(path) {
    const metadata = lstatSync(path);
    if (
        !metadata.isDirectory() ||
        metadata.isSymbolicLink() ||
        metadata.uid !== process.getuid() ||
        (metadata.mode & 0o777) !== 0o700 ||
        realpathSync(path) !== path
    ) {
        throw new Error(`compaction handoff directory is unsafe: ${path}`);
    }
    return path;
}

function stableSource(path) {
    if (typeof path !== "string" || !path.startsWith("/") || path.includes("\0")) {
        throw new Error(`${SOURCE_ENV} is missing or invalid`);
    }
    protectedDirectory(dirname(path));
    const before = lstatSync(path, { bigint: true });
    if (
        !before.isFile() ||
        before.isSymbolicLink() ||
        before.uid !== BigInt(process.getuid()) ||
        before.nlink !== 1n ||
        (before.mode & 0o022n) !== 0n ||
        realpathSync(path) !== path
    ) {
        throw new Error("compaction handoff source has an unsafe identity");
    }
    const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    let payload;
    try {
        fsyncSync(descriptor);
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
        throw new Error("compaction handoff source changed while being authenticated");
    }
    return Object.freeze({ bytes: payload.length, sha256: sha256(payload) });
}

function fsyncDirectory(path) {
    const descriptor = openSync(
        path,
        constants.O_RDONLY | constants.O_DIRECTORY | constants.O_CLOEXEC | constants.O_NOFOLLOW,
    );
    try {
        fsyncSync(descriptor);
    } finally {
        closeSync(descriptor);
    }
}

function publishCreateOnly(path, payload) {
    const directory = protectedDirectory(dirname(path));
    const temporary = `${path}.publish-${process.pid}-${randomBytes(16).toString("hex")}.tmp`;
    let descriptor;
    try {
        descriptor = openSync(
            temporary,
            constants.O_WRONLY |
                constants.O_CREAT |
                constants.O_EXCL |
                constants.O_CLOEXEC |
                constants.O_NOFOLLOW,
            0o600,
        );
        writeFileSync(descriptor, payload);
        fsyncSync(descriptor);
        closeSync(descriptor);
        descriptor = undefined;
        try {
            linkSync(temporary, path);
        } catch (error) {
            if (error?.code !== "EEXIST") throw error;
            const metadata = lstatSync(path);
            if (
                !metadata.isFile() ||
                metadata.isSymbolicLink() ||
                metadata.uid !== process.getuid() ||
                metadata.nlink !== 1 ||
                (metadata.mode & 0o777) !== 0o600 ||
                !readFileSync(path).equals(payload)
            ) {
                throw new Error("existing compaction handoff request differs");
            }
        }
        fsyncDirectory(directory);
    } finally {
        if (descriptor !== undefined) closeSync(descriptor);
        try {
            unlinkSync(temporary);
            fsyncDirectory(directory);
        } catch (error) {
            if (error?.code !== "ENOENT") throw error;
        }
    }
}

function contract() {
    const requestRoot = process.env[REQUEST_ROOT_ENV];
    const source = process.env[SOURCE_ENV];
    const transactionRoot = process.env[TRANSACTION_ROOT_ENV];
	const logicalSessionId = process.env[LOGICAL_SESSION_ENV];
	const compactionAbiSha256 = process.env[COMPACTION_ABI_ENV];
	const promptAbi = process.env[PROMPT_ABI_ENV];
    for (const [name, value] of [
        [REQUEST_ROOT_ENV, requestRoot],
        [TRANSACTION_ROOT_ENV, transactionRoot],
    ]) {
        if (typeof value !== "string" || !value.startsWith("/") || value.includes("\0")) {
            throw new Error(`${name} is missing or invalid`);
        }
    }
    if (!SAFE_NAME_RE.test(logicalSessionId ?? "")) {
        throw new Error(`${LOGICAL_SESSION_ENV} is missing or invalid`);
    }
	if (!SHA256_RE.test(compactionAbiSha256 ?? "")) {
		throw new Error(`${COMPACTION_ABI_ENV} is missing or invalid`);
	}
	if (!ALLOWED_PROMPT_ABIS.has(promptAbi)) {
		throw new Error(`${PROMPT_ABI_ENV} is missing or invalid`);
	}
    protectedDirectory(requestRoot);
    protectedDirectory(transactionRoot);
    const resumeRequest = process.env[RESUME_REQUEST_ENV];
    const resumeRequestSha256 = process.env[RESUME_REQUEST_SHA_ENV];
    const resumeSubmitted = process.env[RESUME_SUBMITTED_ENV];
    const resumeValues = [resumeRequest, resumeRequestSha256, resumeSubmitted];
    if (resumeValues.some((value) => value !== undefined)) {
        if (
            resumeValues.some((value) => typeof value !== "string" || !value) ||
            !resumeRequest.startsWith("/") ||
            resumeRequest.includes("\0") ||
            !resumeSubmitted.startsWith("/") ||
            resumeSubmitted.includes("\0") ||
            !/^[a-f0-9]{64}$/.test(resumeRequestSha256)
        ) {
            throw new Error("transactional compaction resume contract is incomplete or invalid");
        }
        protectedDirectory(dirname(resumeRequest));
        if (dirname(resumeSubmitted) !== dirname(resumeRequest)) {
            throw new Error("transactional compaction resume receipt leaves its request directory");
        }
        const requestMetadata = lstatSync(resumeRequest);
        if (
            !requestMetadata.isFile() ||
            requestMetadata.isSymbolicLink() ||
            requestMetadata.uid !== process.getuid() ||
            requestMetadata.nlink !== 1 ||
            (requestMetadata.mode & 0o777) !== 0o600 ||
            sha256(readFileSync(resumeRequest)) !== resumeRequestSha256
        ) {
            throw new Error("transactional compaction resume request is unsafe or changed");
        }
    }
    return Object.freeze({
        compactionAbiSha256,
		logicalSessionId,
		promptAbi,
        requestRoot,
        resumeRequest,
        resumeRequestSha256,
        resumeSubmitted,
        source,
        transactionRoot,
    });
}

export default function qwenTransactionalCompactionHandoff(pi) {
    const configured = contract();
    let requested = false;
    if (configured.resumeRequest !== undefined) {
        const controlType = "qwen-transactional-compaction-resume-v1";
        let triggered = false;
        let submitted = false;
        pi.on("context", (event) => {
            const messages = Array.isArray(event?.messages) ? event.messages : [];
            const matching = messages.filter((message) => message?.customType === controlType);
            if (triggered && matching.length !== 1) {
                throw new Error("transactional compaction resume control cardinality differs");
            }
            return { messages: messages.filter((message) => message?.customType !== controlType) };
        });
        pi.on("session_start", () => {
            if (triggered) throw new Error("transactional compaction resume fired more than once");
            triggered = true;
            pi.sendMessage(
                {
                    customType: controlType,
                    content: "transactional compaction continuation control",
                    display: false,
                },
                { triggerTurn: true },
            );
        });
        pi.on("before_provider_request", () => {
            if (!triggered || submitted) return undefined;
            publishCreateOnly(
                configured.resumeSubmitted,
                Buffer.from(
                    `${JSON.stringify(
                        {
                            request_sha256: configured.resumeRequestSha256,
                            schema: "urn:qwen-r9700:transactional-compaction-resume-submitted:v1",
                        },
                        null,
                        2,
                    )}\n`,
                    "utf8",
                ),
            );
            submitted = true;
            return undefined;
        });
    }
    pi.on("session_before_compact", (event, ctx) => {
        if (requested) throw new Error("compaction handoff was requested more than once");
        const sessionFile = ctx.sessionManager.getSessionFile();
        if (sessionFile !== configured.source) {
            throw new Error("compaction handoff source differs from Pi's active session");
        }
        if (!event?.preparation || !["manual", "threshold", "overflow"].includes(event.reason)) {
            throw new Error("Pi supplied an invalid compaction handoff event");
        }
        if (
            typeof event.preparation.firstKeptEntryId !== "string" ||
            !event.preparation.firstKeptEntryId ||
            !Number.isSafeInteger(event.preparation.tokensBefore) ||
            event.preparation.tokensBefore <= 0 ||
            typeof event.willRetry !== "boolean"
        ) {
            throw new Error("Pi supplied invalid compaction preparation metadata");
        }
        if (
            event.customInstructions !== undefined &&
            (typeof event.customInstructions !== "string" || event.customInstructions.length > 65536)
        ) {
            throw new Error("Pi supplied invalid custom compaction instructions");
        }
        const branchEntries = Array.isArray(event.branchEntries) ? event.branchEntries : [];
        const leafId = branchEntries.at(-1)?.id;
        if (typeof leafId !== "string" || !leafId) {
            throw new Error("Pi supplied an invalid active compaction leaf");
        }
        const source = stableSource(configured.source);
		const transactionKeyValue = transactionKey(
			source.sha256,
			configured.compactionAbiSha256,
			configured.promptAbi,
		);
        const requestRoot = `${configured.requestRoot}/${transactionKeyValue}`;
        try {
            mkdirSync(requestRoot, { mode: 0o700 });
            fsyncDirectory(configured.requestRoot);
        } catch (error) {
            if (error?.code !== "EEXIST") throw error;
        }
        protectedDirectory(requestRoot);
        const request = `${requestRoot}/request.json`;
        const outputRoot = `${configured.transactionRoot}/${transactionKeyValue}`;
        const snapshotSessionId = `pi-${configured.logicalSessionId}-semantic-${transactionKeyValue.slice(0, 16)}`;
        const document = {
            compaction_abi_sha256: configured.compactionAbiSha256,
            custom_instructions: event.customInstructions ?? null,
            output_root: outputRoot,
            preparation: {
                first_kept_entry_id: event.preparation.firstKeptEntryId,
                tokens_before: event.preparation.tokensBefore,
            },
            reason: event.reason,
			prompt_abi: configured.promptAbi,
			schema: "urn:qwen-r9700:transactional-compaction-handoff:v3",
            snapshot_session_id: snapshotSessionId,
            source: {
                bytes: source.bytes,
                leaf_id: leafId,
                path: configured.source,
                sha256: source.sha256,
            },
            will_retry: event.willRetry,
        };
        const payload = Buffer.from(`${JSON.stringify(document, null, 2)}\n`, "utf8");
        publishCreateOnly(request, payload);
        requested = true;
        process.stderr.write(
            `pi-remote-qwen: durable ${event.reason} compaction handoff recorded; shutting down Pi\n`,
        );
        ctx.shutdown();
        return { cancel: true };
    });
}
