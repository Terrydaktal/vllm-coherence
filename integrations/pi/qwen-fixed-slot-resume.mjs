import { createHash, randomBytes } from "node:crypto";
import { basename, dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import {
	closeSync,
	constants,
	fstatSync,
	fsyncSync,
	linkSync,
	lstatSync,
	openSync,
	readdirSync,
	readFileSync,
	realpathSync,
	renameSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";

const TARGET_PROVIDER = "qwen-r9700";
const TARGET_API = "openai-completions";
const DEFAULT_TARGET_MODEL = "qwen3.8-27b-frozenlock";
const EPHEMERAL_RECOVERY_REQUEST = Symbol.for("qwen-r9700:ephemeral-recovery-request:v1");
const ENABLE_ENV = "QWEN_PI_FIXED_SLOT_RESUME";
const TARGET_MODEL_ENV = "QWEN_PI_FIXED_SLOT_TARGET_MODEL";
const SESSION_ENV = "QWEN_PI_FIXED_SLOT_SESSION_ID";
const BRANCH_ENV = "QWEN_PI_FIXED_SLOT_BRANCH";
const MANIFEST_ENV = "QWEN_PI_FIXED_SLOT_MANIFEST_SHA256";
const MODE_ENV = "QWEN_PI_FIXED_SLOT_MODE";
const SNAPSHOT_TOKENS_ENV = "QWEN_PI_FIXED_SLOT_SNAPSHOT_TOKENS";
const BOOTSTRAP_BASE_URL_ENV = "QWEN_PI_FIXED_SLOT_BOOTSTRAP_BASE_URL";
const BOOTSTRAP_OUTPUT_ENV = "QWEN_PI_FIXED_SLOT_BOOTSTRAP_OUTPUT";
const BOOTSTRAP_TOKEN_FILE_ENV = "QWEN_PI_FIXED_SLOT_BOOTSTRAP_TOKEN_FILE";
const BOOTSTRAP_AUTOSTART_ENV = "QWEN_PI_FIXED_SLOT_BOOTSTRAP_AUTOSTART";
const PREFIX_SHA256_ENV = "QWEN_PI_FIXED_SLOT_PREFIX_SHA256";
const RESUME_PREFIX_TOKENS_ENV = "QWEN_PI_FIXED_SLOT_RESUME_PREFIX_TOKENS";
const LATEST_TOKEN_ROOT_ENV = "QWEN_PI_FIXED_SLOT_LATEST_TOKEN_ROOT";
const BACKGROUND_RUN_ID_ENV = "QWEN_PI_FIXED_SLOT_BACKGROUND_RUN_ID";
const SETTLED_SCHEMA = "urn:qwen-r9700:pi-fixed-slot-settled-token-vector:v1";
const DURABLE_HEAD_SCHEMA = "urn:qwen-r9700:authenticated-snapshot-head:v1";
const BACKGROUND_STATUS_KEY = "qwen-snapshot-background-publication";
const BACKGROUND_STATUS_POLL_MS = 100;
const FOOTER_PATCH_SYMBOL = Symbol.for("qwen-r9700.snapshot-status-footer.v1");
const TOKEN_DIGEST_DOMAIN = Buffer.from("qwen-r9700-token-ids-u32be-v1\0", "utf8");
const STAGED_TOKEN_MAGIC = Buffer.from("QWENSTG1", "ascii");

function isRecord(value) {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isIdentifier(value) {
	return typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value);
}

function isSha256(value) {
	return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function fitFooterLine(value, width) {
	const safe = value
		.replace(/[\u0000-\u001f\u007f]/g, " ")
		.replace(/ +/g, " ")
		.trim();
	if (!Number.isSafeInteger(width) || width < 1 || safe.length <= width) return safe;
	if (width <= 3) return ".".repeat(width);
	return `${safe.slice(0, width - 3)}...`;
}

async function installStackedSnapshotFooter(ctx) {
	if (
		ctx?.mode !== "tui" ||
		typeof ctx.ui?.setStatus !== "function" ||
		typeof process.argv[1] !== "string"
	) {
		return;
	}
	try {
		const cli = realpathSync(process.argv[1]);
		if (basename(cli) !== "cli.js") return;
		const module = await import(pathToFileURL(join(dirname(cli), "index.js")).href);
		const FooterComponent = module.FooterComponent;
		if (typeof FooterComponent?.prototype?.render !== "function") return;
		const prototype = FooterComponent.prototype;
		if (prototype[FOOTER_PATCH_SYMBOL] === true) return;
		const originalRender = prototype.render;
		prototype.render = function renderWithSnapshotStatus(width) {
			const statuses = this.footerData?.getExtensionStatuses?.();
			const snapshotStatus = statuses?.get?.(BACKGROUND_STATUS_KEY);
			if (typeof snapshotStatus !== "string" || snapshotStatus.length === 0) {
				return originalRender.call(this, width);
			}
			statuses.delete(BACKGROUND_STATUS_KEY);
			let lines;
			try {
				lines = originalRender.call(this, width);
			} finally {
				statuses.set(BACKGROUND_STATUS_KEY, snapshotStatus);
			}
			return [...lines, fitFooterLine(snapshotStatus, width)];
		};
		Object.defineProperty(prototype, FOOTER_PATCH_SYMBOL, { value: true });
	} catch {
		// The authenticated snapshot event still uses Pi's ordinary footer status
		// if this pinned-version presentation enhancement is unavailable.
	}
}

function loadContract() {
	if (process.env[ENABLE_ENV] !== "1") return undefined;
	const targetModel = process.env[TARGET_MODEL_ENV] ?? DEFAULT_TARGET_MODEL;
	const sessionId = process.env[SESSION_ENV];
	const branch = process.env[BRANCH_ENV];
	const manifest = process.env[MANIFEST_ENV];
	const mode = process.env[MODE_ENV] ?? "resume";
	const prefixSha256 = process.env[PREFIX_SHA256_ENV];
	const snapshotTokens = Number(process.env[SNAPSHOT_TOKENS_ENV] ?? "60298");
	const resumePrefixTokensValue = process.env[RESUME_PREFIX_TOKENS_ENV];
	const bootstrapAutostartValue = process.env[BOOTSTRAP_AUTOSTART_ENV];
	const latestTokenRoot = process.env[LATEST_TOKEN_ROOT_ENV];
	const backgroundRunId = process.env[BACKGROUND_RUN_ID_ENV];
	if (!isIdentifier(targetModel)) throw new Error(`${TARGET_MODEL_ENV} is invalid`);
	if (!isIdentifier(sessionId)) throw new Error(`${SESSION_ENV} is missing or invalid`);
	if (!isIdentifier(branch)) throw new Error(`${BRANCH_ENV} is missing or invalid`);
	if (
		mode !== "resume" &&
		mode !== "capture" &&
		mode !== "capture_checkpoint" &&
		mode !== "resume_checkpoint"
	) {
		throw new Error(
			`${MODE_ENV} must be exactly resume, resume_checkpoint, capture, or capture_checkpoint`,
		);
	}
	const authenticatedRollingResume =
		mode === "resume" &&
		isSha256(manifest) &&
		resumePrefixTokensValue !== undefined &&
		resumePrefixTokensValue !== "";
	const minimumSnapshotTokens =
		mode === "capture_checkpoint" || mode === "resume_checkpoint" || authenticatedRollingResume
			? 8_258
			: 60_298;
	if (
		!Number.isSafeInteger(snapshotTokens) ||
		snapshotTokens < minimumSnapshotTokens ||
		snapshotTokens > 253_792
	) {
		throw new Error(`${SNAPSHOT_TOKENS_ENV} is missing or invalid`);
	}
	const bootstrapAutostart = bootstrapAutostartValue === "1";
	if (
		bootstrapAutostartValue !== undefined &&
		bootstrapAutostartValue !== "" &&
		bootstrapAutostartValue !== "1"
	) {
		throw new Error(`${BOOTSTRAP_AUTOSTART_ENV} must be exactly 1 when present`);
	}
	if (bootstrapAutostart && mode !== "capture_checkpoint") {
		throw new Error(`${BOOTSTRAP_AUTOSTART_ENV} is valid only in capture_checkpoint mode`);
	}
	if (
		latestTokenRoot !== undefined &&
		(latestTokenRoot === "" || !latestTokenRoot.startsWith("/") || latestTokenRoot.includes("\0"))
	) {
		throw new Error(`${LATEST_TOKEN_ROOT_ENV} is invalid`);
	}
	if ((latestTokenRoot === undefined) !== (backgroundRunId === undefined)) {
		throw new Error(`${LATEST_TOKEN_ROOT_ENV} and ${BACKGROUND_RUN_ID_ENV} must be supplied together`);
	}
	if (backgroundRunId !== undefined && !isIdentifier(backgroundRunId)) {
		throw new Error(`${BACKGROUND_RUN_ID_ENV} is invalid`);
	}
	if (mode === "resume" && !isSha256(manifest)) {
		throw new Error(`${MANIFEST_ENV} is missing or invalid`);
	}
	if ((mode === "resume" || mode === "resume_checkpoint") && !isSha256(prefixSha256)) {
		throw new Error(`${PREFIX_SHA256_ENV} is missing or invalid`);
	}
	let resumePrefixTokens = snapshotTokens;
	if (
		mode === "resume_checkpoint" ||
		(mode === "resume" && resumePrefixTokensValue !== undefined && resumePrefixTokensValue !== "")
	) {
		resumePrefixTokens = Number(resumePrefixTokensValue);
		if (
			!Number.isSafeInteger(resumePrefixTokens) ||
			resumePrefixTokens < 1 ||
			resumePrefixTokens > snapshotTokens
		) {
			throw new Error(`${RESUME_PREFIX_TOKENS_ENV} is missing or invalid`);
		}
	} else if (resumePrefixTokensValue !== undefined && resumePrefixTokensValue !== "") {
		throw new Error(`${RESUME_PREFIX_TOKENS_ENV} is valid only in resume modes`);
	}
	if (
		(mode === "capture" || mode === "capture_checkpoint" || mode === "resume_checkpoint") &&
		manifest !== undefined &&
		manifest !== ""
	) {
		throw new Error(`${MANIFEST_ENV} must be absent in capture mode`);
	}
	if (
		(mode === "capture" || mode === "capture_checkpoint") &&
		prefixSha256 !== undefined &&
		prefixSha256 !== ""
	) {
		throw new Error(`${PREFIX_SHA256_ENV} must be absent in capture mode`);
	}
	return Object.freeze({
		backgroundRunId,
		bootstrapAutostart,
		latestTokenRoot,
		mode,
		prefixSha256,
		resumePrefixTokens,
		snapshotTokens,
		targetModel,
		qwen_250k_cache: Object.freeze({
			branch,
			checkpoint_tokens: null,
			manifest_sha256: mode === "resume" ? manifest : null,
			mode,
			session_id: sessionId,
		}),
	});
}

function isTarget(ctx, payload, contract) {
	return (
		ctx?.model?.provider === TARGET_PROVIDER &&
		ctx.model.api === TARGET_API &&
		isRecord(payload) &&
		payload.model === contract.targetModel
	);
}

function sameContract(value, expected) {
	const cache = value?.qwen_250k_cache;
	return (
		isRecord(value) &&
		Object.keys(value).length === 1 &&
		isRecord(cache) &&
		Object.keys(cache).length === 5 &&
		cache.branch === expected.branch &&
		cache.checkpoint_tokens === expected.checkpoint_tokens &&
		cache.manifest_sha256 === expected.manifest_sha256 &&
		cache.mode === expected.mode &&
		cache.session_id === expected.session_id
	);
}

function tokenDigest(tokens) {
	const hash = createHash("sha256");
	hash.update(TOKEN_DIGEST_DOMAIN);
	const length = Buffer.allocUnsafe(8);
	length.writeBigUInt64BE(BigInt(tokens.length));
	hash.update(length);
	const encoded = Buffer.allocUnsafe(tokens.length * 4);
	for (let index = 0; index < tokens.length; index += 1) {
		const token = tokens[index];
		if (!Number.isInteger(token) || token < 0 || token > 0xffffffff) {
			throw new Error("tokenizer returned an invalid token ID");
		}
		encoded.writeUInt32BE(token, index * 4);
	}
	hash.update(encoded);
	return hash.digest("hex");
}

function protectedParentDirectory(path) {
	const directory = dirname(path);
	const metadata = lstatSync(directory);
	if (
		!metadata.isDirectory() ||
		metadata.isSymbolicLink() ||
		metadata.uid !== process.getuid() ||
		(metadata.mode & 0o777) !== 0o700 ||
		realpathSync(directory) !== directory
	) {
		throw new Error(`bootstrap artifact parent is not a protected real directory: ${directory}`);
	}
	return directory;
}

function protectedExistingFile(path, expectedPayload) {
	protectedParentDirectory(path);
	let metadata = lstatSync(path);
	if (
		metadata.isFile() &&
		!metadata.isSymbolicLink() &&
		metadata.uid === process.getuid() &&
		metadata.nlink === 2 &&
		(metadata.mode & 0o777) === 0o600
	) {
		// A hard-link publish can be killed after the final name appears but
		// before its private temporary is removed. Recover only the one same-inode
		// temporary in the same protected directory; any other link topology is
		// ambiguous and remains a hard failure.
		const directory = dirname(path);
		const prefix = `${basename(path)}.publish-`;
		const candidates = [];
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			if (!entry.isFile() || !entry.name.startsWith(prefix) || !entry.name.endsWith(".tmp")) {
				continue;
			}
			const candidate = join(directory, entry.name);
			try {
				const candidateMetadata = lstatSync(candidate);
				if (
					candidateMetadata.isFile() &&
					!candidateMetadata.isSymbolicLink() &&
					candidateMetadata.uid === metadata.uid &&
					candidateMetadata.dev === metadata.dev &&
					candidateMetadata.ino === metadata.ino &&
					candidateMetadata.nlink === 2 &&
					(candidateMetadata.mode & 0o777) === 0o600
				) {
					candidates.push(candidate);
				}
			} catch (error) {
				if (error?.code !== "ENOENT") throw error;
			}
		}
		if (candidates.length === 0) {
			// The original publisher may have removed its temporary between our
			// lstat and directory scan. Accept only the now-single-link exact file.
			metadata = lstatSync(path);
			if (metadata.nlink === 1 && readFileSync(path).equals(expectedPayload)) {
				return readFileSync(path);
			}
		}
		if (candidates.length !== 1 || !readFileSync(path).equals(expectedPayload)) {
			throw new Error(`existing bootstrap artifact has ambiguous interrupted publication: ${path}`);
		}
		try {
			unlinkSync(candidates[0]);
		} catch (error) {
			if (error?.code !== "ENOENT") throw error;
		}
		fsyncDirectory(directory);
		metadata = lstatSync(path);
	}
	if (
		!metadata.isFile() ||
		metadata.isSymbolicLink() ||
		metadata.uid !== process.getuid() ||
		metadata.nlink !== 1 ||
		(metadata.mode & 0o777) !== 0o600
	) {
		throw new Error(`existing bootstrap artifact has unsafe identity or mode: ${path}`);
	}
	return readFileSync(path);
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

function writeDurableCreateOnlyBuffer(path, payload) {
	const directory = protectedParentDirectory(path);
	const temporary = `${path}.publish-${process.pid}-${randomBytes(16).toString("hex")}.tmp`;
	let fileDescriptor;
	try {
		fileDescriptor = openSync(
			temporary,
			constants.O_WRONLY |
				constants.O_CREAT |
				constants.O_EXCL |
				constants.O_CLOEXEC |
				constants.O_NOFOLLOW,
			0o600,
		);
		writeFileSync(fileDescriptor, payload);
		fsyncSync(fileDescriptor);
		closeSync(fileDescriptor);
		fileDescriptor = undefined;
		try {
			// link(2) gives this create-only publication an atomic no-replace
			// operation. A power loss can leave the private temporary, but can
			// never expose a partial final artifact.
			linkSync(temporary, path);
		} catch (error) {
			if (error?.code !== "EEXIST") throw error;
			const existing = protectedExistingFile(path, payload);
			if (!existing.equals(payload)) {
				throw new Error(`existing bootstrap artifact differs: ${path}`);
			}
		}
		fsyncDirectory(directory);
	} finally {
		if (fileDescriptor !== undefined) closeSync(fileDescriptor);
		try {
			unlinkSync(temporary);
			fsyncDirectory(directory);
		} catch (error) {
			if (error?.code !== "ENOENT") throw error;
		}
	}
}

function writeDurableCreateOnlyJson(path, value) {
	writeDurableCreateOnlyBuffer(path, Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8"));
}

function writeDurableReplaceBuffer(path, payload) {
	const directory = protectedParentDirectory(path);
	try {
		const existing = lstatSync(path);
		if (
			!existing.isFile() ||
			existing.isSymbolicLink() ||
			existing.uid !== process.getuid() ||
			existing.nlink !== 1 ||
			(existing.mode & 0o777) !== 0o600
		) {
			throw new Error(`replaceable snapshot artifact has unsafe identity or mode: ${path}`);
		}
	} catch (error) {
		if (error?.code !== "ENOENT") throw error;
	}
	const temporary = `${path}.replace-${process.pid}-${randomBytes(16).toString("hex")}.tmp`;
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
		renameSync(temporary, path);
		fsyncDirectory(directory);
	} finally {
		if (descriptor !== undefined) closeSync(descriptor);
		try {
			unlinkSync(temporary);
		} catch (error) {
			if (error?.code !== "ENOENT") throw error;
		}
	}
}

function readOptionalProtectedJson(path, expectedMode, maximumBytes) {
	protectedParentDirectory(path);
	let before;
	try {
		before = lstatSync(path);
	} catch (error) {
		if (error?.code === "ENOENT") return undefined;
		throw error;
	}
	if (
		!before.isFile() ||
		before.isSymbolicLink() ||
		before.uid !== process.getuid() ||
		before.nlink !== 1 ||
		(before.mode & 0o777) !== expectedMode ||
		before.size < 1 ||
		before.size > maximumBytes
	) {
		throw new Error(`background snapshot artifact has unsafe identity or mode: ${path}`);
	}
	const descriptor = openSync(path, constants.O_RDONLY | constants.O_CLOEXEC | constants.O_NOFOLLOW);
	let payload;
	try {
		const opened = fstatSync(descriptor);
		if (opened.dev !== before.dev || opened.ino !== before.ino) {
			throw new Error(`background snapshot artifact changed before read: ${path}`);
		}
		payload = readFileSync(descriptor);
	} finally {
		closeSync(descriptor);
	}
	const after = lstatSync(path);
	if (
		after.dev !== before.dev ||
		after.ino !== before.ino ||
		after.size !== before.size ||
		after.mtimeMs !== before.mtimeMs ||
		after.ctimeMs !== before.ctimeMs
	) {
		throw new Error(`background snapshot artifact changed during read: ${path}`);
	}
	try {
		const value = JSON.parse(payload.toString("utf8"));
		if (!isRecord(value)) throw new Error("JSON value is not an object");
		return value;
	} catch (error) {
		throw new Error(`background snapshot artifact is invalid JSON: ${path}: ${error.message}`);
	}
}

function recordLatestTokenVector(tokens, contract) {
	if (contract.latestTokenRoot === undefined) return;
	const root = contract.latestTokenRoot;
	protectedParentDirectory(join(root, "latest.json"));
	const vectors = join(root, "vectors");
	protectedParentDirectory(join(vectors, "placeholder"));
	const encoded = stagedTokenFile(tokens);
	const tokenFileSha256 = createHash("sha256").update(encoded).digest("hex");
	const tokenFile = join(vectors, `${tokenFileSha256}.qwenstg1`);
	writeDurableCreateOnlyBuffer(tokenFile, encoded);
	const receipt = {
		branch: contract.qwen_250k_cache.branch,
		prefix_token_ids_sha256: tokenDigest(tokens),
		prompt_tokens: tokens.length,
		schema: "urn:qwen-r9700:pi-fixed-slot-latest-token-vector:v1",
		session_id: contract.qwen_250k_cache.session_id,
		token_file: tokenFile,
		token_file_sha256: tokenFileSha256,
	};
	writeDurableReplaceBuffer(
		join(root, "latest.json"),
		Buffer.from(`${JSON.stringify(receipt, null, 2)}\n`, "utf8"),
	);
	return receipt;
}

function validateSettledReceipt(value, contract) {
	const expectedKeys = [
		"branch",
		"prefix_token_ids_sha256",
		"prompt_tokens",
		"run_id",
		"schema",
		"session_id",
		"token_file",
		"token_file_sha256",
	];
	return (
		isRecord(value) &&
		JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expectedKeys) &&
		value.schema === SETTLED_SCHEMA &&
		value.session_id === contract.qwen_250k_cache.session_id &&
		value.branch === contract.qwen_250k_cache.branch &&
		value.run_id === contract.backgroundRunId &&
		Number.isSafeInteger(value.prompt_tokens) &&
		value.prompt_tokens >= 8_258 &&
		value.prompt_tokens <= 253_792 &&
		isSha256(value.prefix_token_ids_sha256) &&
		isSha256(value.token_file_sha256) &&
		value.token_file ===
			join(contract.latestTokenRoot, "vectors", `${value.token_file_sha256}.qwenstg1`)
	);
}

function contractFromDurableHead(head, contract) {
	if (
		head.schema !== DURABLE_HEAD_SCHEMA ||
		head.session_id !== contract.qwen_250k_cache.session_id ||
		head.branch !== contract.qwen_250k_cache.branch ||
		head.run_id !== contract.backgroundRunId ||
		!Number.isSafeInteger(head.generation) ||
		head.generation < 1 ||
		!Number.isSafeInteger(head.prompt_tokens) ||
		head.prompt_tokens < 8_258 ||
		head.prompt_tokens > 253_792 ||
		!Number.isSafeInteger(head.replay_boundary_tokens) ||
		head.replay_boundary_tokens < 1 ||
		head.replay_boundary_tokens > head.prompt_tokens ||
		!isSha256(head.manifest_sha256) ||
		!isSha256(head.replay_prefix_token_ids_sha256) ||
		!isSha256(head.settled_token_file_sha256) ||
		head.store_drained !== true ||
		!isRecord(head.deep_evidence) ||
		head.deep_evidence.all_groups_verified !== true ||
		head.deep_evidence.total_groups !== 69
	) {
		throw new Error("background publisher durable head has an invalid all-69 contract");
	}
	return Object.freeze({
		...contract,
		prefixSha256: head.replay_prefix_token_ids_sha256,
		resumePrefixTokens: head.replay_boundary_tokens,
		snapshotTokens: head.prompt_tokens,
		qwen_250k_cache: Object.freeze({
			...contract.qwen_250k_cache,
			// Rolling publication always follows the branch's authenticated current
			// head, so it must retain a null manifest. Ordinary resume remains pinned
			// to the exact manifest it just adopted.
			manifest_sha256: contract.mode === "resume" ? head.manifest_sha256 : null,
		}),
	});
}

async function currentResumeContract(contract, ctx, backgroundPublication) {
	if (contract.latestTokenRoot === undefined) return contract;
	const settledPath = join(contract.latestTokenRoot, "settled.json");
	const durablePath = join(contract.latestTokenRoot, "durable-head.json");
	const errorPath = join(contract.latestTokenRoot, "publisher-error.json");
	const deadline = Date.now() + 15 * 60 * 1000;
	while (true) {
		const observedSettled = readOptionalProtectedJson(settledPath, 0o600, 64 * 1024);
		const settled = observedSettled?.run_id === contract.backgroundRunId ? observedSettled : undefined;
		if (settled !== undefined && !validateSettledReceipt(settled, contract)) {
			throw new Error("background publisher settled receipt has an invalid contract");
		}
		const observedDurable = readOptionalProtectedJson(durablePath, 0o600, 4 * 1024 * 1024);
		const durable = observedDurable?.run_id === contract.backgroundRunId ? observedDurable : undefined;
		const active = durable === undefined ? contract : contractFromDurableHead(durable, contract);
		if (backgroundPublication.failed) return active;
		if (
			settled === undefined ||
			(durable !== undefined &&
				durable.prompt_tokens >= settled.prompt_tokens &&
				durable.settled_token_file_sha256 === settled.token_file_sha256)
		) {
			return active;
		}
		const failure = readOptionalProtectedJson(errorPath, 0o600, 64 * 1024);
		if (
			failure?.run_id === contract.backgroundRunId &&
			failure?.settled_token_file_sha256 === settled.token_file_sha256
		) {
			backgroundPublication.failed = true;
			backgroundPublication.diagnostic = failure.diagnostic ?? "unknown error";
			if (ctx.mode === "tui") {
				ctx.ui.setStatus?.(
					BACKGROUND_STATUS_KEY,
					"background snapshot publication failed safely; continuing from the prior durable head",
				);
			}
			return active;
		}
		if (Date.now() >= deadline) {
			throw new Error("background snapshot publication did not finish before the next provider request");
		}
		if (ctx.mode === "tui") {
			ctx.ui.setWorkingMessage?.(
				`Sealing durable snapshot through ${settled.prompt_tokens.toLocaleString("en-US")} tokens`,
			);
		}
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
}

function stagedTokenFile(tokens) {
	const body = Buffer.allocUnsafe(STAGED_TOKEN_MAGIC.length + 8 + tokens.length * 4);
	STAGED_TOKEN_MAGIC.copy(body, 0);
	body.writeBigUInt64BE(BigInt(tokens.length), STAGED_TOKEN_MAGIC.length);
	for (let index = 0; index < tokens.length; index += 1) {
		const token = tokens[index];
		if (!Number.isInteger(token) || token < 0 || token > 0xffffffff) {
			throw new Error("tokenizer returned an invalid token ID");
		}
		body.writeUInt32BE(token, STAGED_TOKEN_MAGIC.length + 8 + index * 4);
	}
	const checksum = createHash("sha256").update(body).digest();
	return Buffer.concat([body, checksum]);
}

async function postJson(url, body, headers = {}) {
	const response = await fetch(url, {
		body: JSON.stringify(body),
		headers: { "content-type": "application/json", ...headers },
		method: "POST",
	});
	const text = await response.text();
	if (!response.ok) {
		throw new Error(`snapshot bootstrap POST ${url} failed with HTTP ${response.status}`);
	}
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error(`snapshot bootstrap POST ${url} returned invalid JSON`);
	}
	return parsed;
}

function tokenizePayload(payload) {
	const result = { model: payload.model, messages: payload.messages };
	for (const key of [
		"add_generation_prompt",
		"add_special_tokens",
		"chat_template",
		"chat_template_kwargs",
		"continue_final_message",
		"tools",
	]) {
		if (payload[key] !== undefined) result[key] = payload[key];
	}
	return result;
}

function scrubbedAbortPayload(payload) {
	return {
		max_tokens: 1,
		messages: [{ content: ".", role: "user" }],
		model: payload.model,
		stream: false,
		temperature: 0,
		top_p: 1,
	};
}

function stopProviderRequest(ctx, payload, error) {
	if (error !== undefined) {
		const message = error instanceof Error ? error.message : String(error);
		process.stderr.write(`pi-remote-qwen: ${message}\n`);
	}
	ctx.abort();
	ctx.shutdown();
	return scrubbedAbortPayload(payload);
}

async function validateResumePrefix(payload, contract) {
	const baseUrl = process.env[BOOTSTRAP_BASE_URL_ENV];
	if (typeof baseUrl !== "string" || !/^http:\/\/127\.0\.0\.1:\d+$/.test(baseUrl)) {
		throw new Error(`${BOOTSTRAP_BASE_URL_ENV} is missing or invalid`);
	}
	const tokenized = await postJson(`${baseUrl}/tokenize`, tokenizePayload(payload));
	if (!Array.isArray(tokenized.tokens) || tokenized.tokens.length < contract.resumePrefixTokens) {
		throw new Error(
			`snapshot resume prompt has ${tokenized.tokens?.length ?? "unknown"} tokens; ` +
				`need at least ${contract.resumePrefixTokens}`,
		);
	}
	const observed = tokenDigest(tokenized.tokens.slice(0, contract.resumePrefixTokens));
	if (observed !== contract.prefixSha256) {
		throw new Error(
			`snapshot resume prefix digest differs: observed=${observed} ` +
				`expected=${contract.prefixSha256} prefix_tokens=${contract.resumePrefixTokens} ` +
				`prompt_tokens=${tokenized.tokens.length}; ` +
				"refusing a cold fallback",
		);
	}
	// Rolling checkpoints authenticate exactly through their replay boundary.
	// Only the small tail after that boundary may differ when a tool call acquires
	// its result suffix; every earlier token is rejected locally before EngineCore.
	return tokenized.tokens;
}

async function bootstrapSnapshot(payload, contract) {
	const baseUrl = process.env[BOOTSTRAP_BASE_URL_ENV];
	const output = process.env[BOOTSTRAP_OUTPUT_ENV];
	const tokenFile = process.env[BOOTSTRAP_TOKEN_FILE_ENV];
	if (typeof baseUrl !== "string" || !/^http:\/\/127\.0\.0\.1:\d+$/.test(baseUrl)) {
		throw new Error(`${BOOTSTRAP_BASE_URL_ENV} is missing or invalid`);
	}
	if (typeof output !== "string" || !output.startsWith("/")) {
		throw new Error(`${BOOTSTRAP_OUTPUT_ENV} is missing or invalid`);
	}
	if (
		contract.mode === "capture_checkpoint" &&
		(typeof tokenFile !== "string" || !tokenFile.startsWith("/"))
	) {
		throw new Error(`${BOOTSTRAP_TOKEN_FILE_ENV} is missing or invalid`);
	}
	const tokenized = await postJson(`${baseUrl}/tokenize`, tokenizePayload(payload));
	if (!Array.isArray(tokenized.tokens) || tokenized.tokens.length < contract.snapshotTokens) {
		throw new Error(
			`snapshot bootstrap prompt has ${tokenized.tokens?.length ?? "unknown"} tokens; ` +
				`need at least ${contract.snapshotTokens}`,
		);
	}
	const fullHead = contract.mode === "capture_checkpoint";
	if (fullHead && tokenized.tokens.length > 253_792) {
		throw new Error(
			`snapshot full-head prompt has ${tokenized.tokens.length} tokens; maximum is 253792`,
		);
	}
	const prefix = fullHead ? tokenized.tokens : tokenized.tokens.slice(0, contract.snapshotTokens);
	const prefixTokenIdsSha256 = tokenDigest(prefix);
	if (fullHead) {
		const encodedTokens = stagedTokenFile(prefix);
		const tokenFileSha256 = createHash("sha256").update(encodedTokens).digest("hex");
		writeDurableCreateOnlyJson(`${output}.started`, {
			branch: contract.qwen_250k_cache.branch,
			capture_mode: contract.mode,
			prefix_token_ids_sha256: prefixTokenIdsSha256,
			prompt_tokens: prefix.length,
			schema: "urn:qwen-r9700:pi-fixed-slot-token-export-started:v1",
			session_id: contract.qwen_250k_cache.session_id,
			token_file: tokenFile,
			token_file_sha256: tokenFileSha256,
		});
		writeDurableCreateOnlyBuffer(tokenFile, encodedTokens);
		writeDurableCreateOnlyJson(output, {
			branch: contract.qwen_250k_cache.branch,
			capture_mode: contract.mode,
			model: payload.model,
			prefix_token_ids_sha256: prefixTokenIdsSha256,
			prompt_tokens: prefix.length,
			schema: "urn:qwen-r9700:pi-fixed-slot-token-export:v1",
			session_id: contract.qwen_250k_cache.session_id,
			token_file: tokenFile,
			token_file_sha256: tokenFileSha256,
		});
		return;
	}
	// capture_checkpoint returned after durable token export above. This legacy
	// request path is reachable only by the bounded fixed-prefix capture mode.
	const requestContract = contract.qwen_250k_cache;
	const requestId = `pi-snapshot-${contract.qwen_250k_cache.session_id}`;
	writeDurableCreateOnlyJson(`${output}.started`, {
		branch: contract.qwen_250k_cache.branch,
		capture_mode: contract.mode,
		prefix_token_ids_sha256: prefixTokenIdsSha256,
		prompt_tokens: tokenized.tokens.length,
		request_id: requestId,
		schema: "urn:qwen-r9700:pi-fixed-slot-bootstrap-started:v1",
		session_id: contract.qwen_250k_cache.session_id,
		snapshot_tokens: prefix.length,
	});
	const captured = await postJson(
		`${baseUrl}/v1/completions`,
		{
			kv_transfer_params: { qwen_250k_cache: requestContract },
			max_tokens: 1,
			model: payload.model,
			prompt: prefix,
			stream: false,
			temperature: 0,
			top_p: 1,
		},
		{ "x-request-id": requestId },
	);
	if (captured?.usage?.prompt_tokens !== prefix.length) {
		throw new Error("snapshot bootstrap completion did not consume the exact snapshot prefix");
	}
	writeDurableCreateOnlyJson(output, {
		branch: contract.qwen_250k_cache.branch,
		capture_mode: contract.mode,
		prefix_token_ids_sha256: prefixTokenIdsSha256,
		prompt_tokens: tokenized.tokens.length,
		schema: "urn:qwen-r9700:pi-fixed-slot-bootstrap:v2",
		session_id: contract.qwen_250k_cache.session_id,
		snapshot_tokens: prefix.length,
	});
}

export default function qwenFixedSlotResume(pi) {
	const contract = loadContract();
	let bootstrapPromise;
	let backgroundStatusTimer;
	const backgroundPublication = { diagnostic: undefined, failed: false };
	let turnTokenReceipt;
	let turnCompleted = false;
	let turnPublishable = false;

	function stopBackgroundStatusWatch() {
		if (backgroundStatusTimer !== undefined) {
			clearInterval(backgroundStatusTimer);
			backgroundStatusTimer = undefined;
		}
	}

	function watchBackgroundPublication(ctx, settled) {
		if (ctx?.mode !== "tui" || typeof ctx.ui?.setStatus !== "function") return;
		stopBackgroundStatusWatch();
		const durablePath = join(contract.latestTokenRoot, "durable-head.json");
		const errorPath = join(contract.latestTokenRoot, "publisher-error.json");
		const inspect = () => {
			try {
				const observedDurable = readOptionalProtectedJson(durablePath, 0o600, 4 * 1024 * 1024);
				if (observedDurable?.run_id === contract.backgroundRunId) {
					contractFromDurableHead(observedDurable, contract);
					if (
						observedDurable.prompt_tokens >= settled.prompt_tokens &&
						observedDurable.settled_token_file_sha256 === settled.token_file_sha256
					) {
						ctx.ui.setStatus(
							BACKGROUND_STATUS_KEY,
							`background durable snapshot head advanced to ${observedDurable.prompt_tokens} prompt tokens`,
						);
						stopBackgroundStatusWatch();
						return;
					}
				}
				const failure = readOptionalProtectedJson(errorPath, 0o600, 64 * 1024);
				if (
					failure?.run_id === contract.backgroundRunId &&
					failure?.settled_token_file_sha256 === settled.token_file_sha256
				) {
					ctx.ui.setStatus(
						BACKGROUND_STATUS_KEY,
						"background snapshot publication failed safely; prior durable head retained",
					);
					stopBackgroundStatusWatch();
				}
			} catch {
				ctx.ui.setStatus(
					BACKGROUND_STATUS_KEY,
					"background snapshot status authentication failed; prior durable head retained",
				);
				stopBackgroundStatusWatch();
			}
		};
		inspect();
		if (backgroundStatusTimer !== undefined) return;
		backgroundStatusTimer = setInterval(inspect, BACKGROUND_STATUS_POLL_MS);
		backgroundStatusTimer.unref?.();
	}
	function publishSettledTokenVector(ctx) {
		if (
			contract?.latestTokenRoot === undefined ||
			contract.backgroundRunId === undefined ||
			backgroundPublication.failed ||
			!turnCompleted ||
			!turnPublishable ||
			turnTokenReceipt === undefined
		) {
			return;
		}
		const settled = {
			...turnTokenReceipt,
			run_id: contract.backgroundRunId,
			schema: SETTLED_SCHEMA,
		};
		writeDurableReplaceBuffer(
			join(contract.latestTokenRoot, "settled.json"),
			Buffer.from(`${JSON.stringify(settled, null, 2)}\n`, "utf8"),
		);
		watchBackgroundPublication(ctx, settled);
	}
	if (contract?.bootstrapAutostart === true) {
		const controlType = "qwen-snapshot-bootstrap-control-v1";
		let triggered = false;
		pi.on("context", (event) => {
			const messages = Array.isArray(event?.messages) ? event.messages : [];
			const matching = messages.filter((message) => message?.customType === controlType);
			if (triggered && matching.length !== 1) {
				throw new Error("snapshot bootstrap control-message cardinality differs");
			}
			return { messages: messages.filter((message) => message?.customType !== controlType) };
		});
		pi.on("session_start", () => {
			if (triggered) throw new Error("snapshot bootstrap autostart fired more than once");
			triggered = true;
			pi.sendMessage(
				{
					customType: controlType,
					content: "snapshot bootstrap transport control",
					display: false,
				},
				{ triggerTurn: true },
			);
		});
	}
	pi.on("agent_start", async (_event, ctx) => {
		stopBackgroundStatusWatch();
		if (ctx?.mode === "tui") {
			ctx.ui.setStatus?.(BACKGROUND_STATUS_KEY, undefined);
			await installStackedSnapshotFooter(ctx);
		}
		turnTokenReceipt = undefined;
		turnCompleted = false;
		turnPublishable = false;
	});
	pi.on("before_provider_request", async (event, ctx) => {
		const payload = event?.payload;
		if (contract === undefined || !isTarget(ctx, payload, contract)) return undefined;
		if (contract.mode === "capture" || contract.mode === "capture_checkpoint") {
			bootstrapPromise ??= bootstrapSnapshot(payload, contract);
			try {
				await bootstrapPromise;
				return stopProviderRequest(ctx, payload);
			} catch (error) {
				return stopProviderRequest(ctx, payload, error);
			}
		}
		const ephemeralRecovery = payload[EPHEMERAL_RECOVERY_REQUEST] === true;
		if (ephemeralRecovery && payload.kv_transfer_params !== undefined) {
			return stopProviderRequest(
				ctx,
				payload,
				new Error("ephemeral recovery request must not contain kv_transfer_params"),
			);
		}
		let tokenIds;
		let activeContract;
		try {
			// A settled turn may have published a newer branch head while the user was
			// reading the answer. Never race that atomic ref swap with a request still
			// carrying the previous manifest.
			activeContract = await currentResumeContract(contract, ctx, backgroundPublication);
			// Revalidate every outgoing request. A branch edit can change tokens
			// before the fixed boundary after an earlier request has succeeded.
			tokenIds = await validateResumePrefix(payload, activeContract);
			// Corrective retries append request-only steering that is deliberately
			// absent from Pi's durable session. Do not give such a request a fixed-slot
			// transport contract: the backend treats that contract as authority to
			// promote the request prompt, independently of Pi's settled-turn publisher.
			// It must also remain absent from the local latest-token journal so a final
			// drain cannot publish it later. A shared symbol cannot enter the JSON request.
			if (ephemeralRecovery) {
				turnTokenReceipt = undefined;
				turnPublishable = false;
				return payload;
			}
			turnTokenReceipt = recordLatestTokenVector(tokenIds, activeContract);
			turnPublishable = true;
		} catch (error) {
			return stopProviderRequest(ctx, payload, error);
		}
		const requestContract =
			activeContract.mode === "resume_checkpoint"
				? { ...activeContract.qwen_250k_cache, checkpoint_tokens: tokenIds.length }
				: activeContract.qwen_250k_cache;
		if (payload.kv_transfer_params !== undefined) {
			const existing = payload.kv_transfer_params;
			if (!sameContract(existing, requestContract)) {
				return stopProviderRequest(
					ctx,
					payload,
					new Error("fixed-slot resume request already contains a different kv_transfer_params contract"),
				);
			}
			return payload;
		}
		return {
			...payload,
			kv_transfer_params: { qwen_250k_cache: requestContract },
		};
	});
	pi.on("message_end", (event, ctx) => {
		const message = event?.message;
		if (
			message?.role === "assistant" &&
			message.provider === TARGET_PROVIDER &&
			message.api === TARGET_API
		) {
			turnCompleted = message.stopReason !== "error" && message.stopReason !== "aborted";
			publishSettledTokenVector(ctx);
		}
	});
	pi.on("agent_settled", (_event, ctx) => publishSettledTokenVector(ctx));
	pi.on("session_shutdown", () => stopBackgroundStatusWatch());
}
