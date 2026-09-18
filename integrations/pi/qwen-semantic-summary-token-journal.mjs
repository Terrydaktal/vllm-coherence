import { createHash } from "node:crypto";
import {
	closeSync,
	constants,
	fsyncSync,
	linkSync,
	lstatSync,
	openSync,
	realpathSync,
	unlinkSync,
	writeSync,
} from "node:fs";
import { dirname } from "node:path";

const JOURNAL_ENV = "QWEN_PI_SEMANTIC_SUMMARY_TOKEN_JOURNAL";
const TARGET_MODEL = "qwen3.8-27b-frozenlock";
const SCHEMA = "urn:qwen-r9700:semantic-summary-token-journal:v1";
const TOKEN_DIGEST_DOMAIN = Buffer.from("qwen-r9700-token-ids-u32be-v1\0", "utf8");
const BATCH_TOKENS = 64;

function protectedDirectory(path) {
	const metadata = lstatSync(path);
	if (
		!metadata.isDirectory() ||
		metadata.isSymbolicLink() ||
		metadata.uid !== process.getuid() ||
		(metadata.mode & 0o777) !== 0o700 ||
		realpathSync(path) !== path
	) {
		throw new Error(`semantic token-journal directory has an unsafe identity: ${path}`);
	}
}

function fsyncDirectory(path) {
	const descriptor = openSync(path, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
	try {
		fsyncSync(descriptor);
	} finally {
		closeSync(descriptor);
	}
}

function requireAbsent(path, label) {
	try {
		lstatSync(path);
	} catch (error) {
		if (error?.code === "ENOENT") return;
		throw error;
	}
	throw new Error(`${label} already exists: ${path}`);
}

function tokenDigest(tokens) {
	const hash = createHash("sha256");
	hash.update(TOKEN_DIGEST_DOMAIN);
	const encoded = Buffer.allocUnsafe(tokens.length * 4);
	for (let index = 0; index < tokens.length; index += 1) {
		const token = tokens[index];
		if (!Number.isSafeInteger(token) || token < 0 || token > 0xffffffff) {
			throw new Error("semantic summary stream returned an invalid token ID");
		}
		encoded.writeUInt32BE(token, index * 4);
	}
	hash.update(encoded);
	return hash.digest("hex");
}

function textDigest(value) {
	return createHash("sha256").update(Buffer.from(value, "utf8")).digest("hex");
}

async function targetRequest(url, init) {
	const requestObject = typeof Request !== "undefined" && url instanceof Request ? url : undefined;
	if (typeof url !== "string" && !(url instanceof URL) && requestObject === undefined) return undefined;
	const parsed = new URL(requestObject?.url ?? String(url));
	if (parsed.pathname !== "/v1/chat/completions") return undefined;
	const method = (init?.method ?? requestObject?.method ?? "GET").toUpperCase();
	if (method !== "POST") return undefined;
	const bodyText =
		init?.body ?? (requestObject === undefined ? undefined : await requestObject.clone().text());
	if (typeof bodyText !== "string") return undefined;
	let body;
	try {
		body = JSON.parse(bodyText);
	} catch {
		return undefined;
	}
	return body?.model === TARGET_MODEL && body?.stream === true && body?.return_token_ids === true
		? bodyText
		: undefined;
}

class DurableTokenJournal {
	constructor(finalPath, requestBody) {
		if (typeof finalPath !== "string" || !finalPath.startsWith("/") || finalPath.includes("\0")) {
			throw new Error(`${JOURNAL_ENV} is missing or invalid`);
		}
		this.finalPath = finalPath;
		this.partialPath = `${finalPath}.partial`;
		this.directory = dirname(finalPath);
		protectedDirectory(this.directory);
		requireAbsent(this.finalPath, "sealed semantic token journal");
		this.descriptor = openSync(
			this.partialPath,
			constants.O_WRONLY |
				constants.O_CREAT |
				constants.O_EXCL |
				constants.O_CLOEXEC |
				constants.O_NOFOLLOW,
			0o400,
		);
		fsyncSync(this.descriptor);
		fsyncDirectory(this.directory);
		this.decoder = new TextDecoder("utf-8", { fatal: true });
		this.pendingText = "";
		this.pendingEvents = [];
		this.pendingTokens = 0;
		this.tokens = [];
		this.content = "";
		this.reasoning = "";
		this.responseId = undefined;
		this.finishReason = undefined;
		this.usage = undefined;
		this.batchOrdinal = 0;
		this.sawDone = false;
		this.closed = false;
		this.writeRecord({
			model: TARGET_MODEL,
			request_body_sha256: createHash("sha256").update(requestBody).digest("hex"),
			schema: SCHEMA,
			type: "header",
		});
	}

	writeRecord(record) {
		const payload = Buffer.from(`${JSON.stringify(record)}\n`, "utf8");
		let offset = 0;
		while (offset < payload.length) {
			const written = writeSync(
				this.descriptor,
				payload,
				offset,
				payload.length - offset,
				null,
			);
			if (written <= 0) throw new Error("short write in semantic token journal");
			offset += written;
		}
	}

	flushBatch() {
		if (this.pendingEvents.length === 0) return;
		this.writeRecord({
			events: this.pendingEvents,
			ordinal: this.batchOrdinal,
			schema: SCHEMA,
			type: "batch",
		});
		fsyncSync(this.descriptor);
		this.batchOrdinal += 1;
		this.pendingEvents = [];
		this.pendingTokens = 0;
	}

	consumeData(data) {
		if (data === "[DONE]") {
			this.sawDone = true;
			this.finish();
			return;
		}
		let chunk;
		try {
			chunk = JSON.parse(data);
		} catch (error) {
			throw new Error("semantic summary stream emitted invalid SSE JSON", { cause: error });
		}
		if (typeof chunk?.id === "string" && chunk.id.length > 0) {
			if (this.responseId !== undefined && this.responseId !== chunk.id) {
				throw new Error("semantic summary stream changed its response ID");
			}
			this.responseId = chunk.id;
		}
		if (chunk?.usage !== undefined) this.usage = chunk.usage;
		const choice = Array.isArray(chunk?.choices) ? chunk.choices[0] : undefined;
		if (choice === undefined) return;
		const tokenIds = choice.token_ids ?? [];
		if (!Array.isArray(tokenIds)) {
			throw new Error("semantic summary stream returned non-array token IDs");
		}
		for (const token of tokenIds) {
			if (!Number.isSafeInteger(token) || token < 0 || token > 0xffffffff) {
				throw new Error("semantic summary stream returned an invalid token ID");
			}
		}
		const contentDelta = typeof choice?.delta?.content === "string" ? choice.delta.content : "";
		const reasoningDelta = [
			choice?.delta?.reasoning_content,
			choice?.delta?.reasoning,
			choice?.delta?.reasoning_text,
		].find((value) => typeof value === "string" && value.length > 0) ?? "";
		if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
			if (this.finishReason !== undefined && this.finishReason !== choice.finish_reason) {
				throw new Error("semantic summary stream changed its finish reason");
			}
			this.finishReason = choice.finish_reason;
		}
		this.tokens.push(...tokenIds);
		this.content += contentDelta;
		this.reasoning += reasoningDelta;
		this.pendingTokens += tokenIds.length;
		this.pendingEvents.push({
			content_delta: contentDelta,
			finish_reason: choice.finish_reason ?? null,
			reasoning_delta: reasoningDelta,
			response_id: chunk.id ?? null,
			token_ids: tokenIds,
		});
		if (this.pendingTokens >= BATCH_TOKENS || choice.finish_reason) this.flushBatch();
	}

	feed(bytes, final = false) {
		if (this.closed) return;
		this.pendingText += this.decoder.decode(bytes, { stream: !final });
		while (true) {
			const newline = this.pendingText.indexOf("\n");
			if (newline < 0) break;
			const line = this.pendingText.slice(0, newline).replace(/\r$/, "");
			this.pendingText = this.pendingText.slice(newline + 1);
			if (line.startsWith("data: ")) this.consumeData(line.slice(6));
		}
		if (final && this.pendingText.trim().length > 0) {
			throw new Error("semantic summary SSE stream ended with a partial record");
		}
		if (final && !this.sawDone) {
			throw new Error("semantic summary SSE stream ended before its completion marker");
		}
	}

	finish() {
		if (this.closed) return;
		this.flushBatch();
		const outputTokens = this.usage?.completion_tokens;
		if (
			!this.sawDone ||
			typeof this.responseId !== "string" ||
			this.finishReason !== "stop" ||
			!Number.isSafeInteger(outputTokens) ||
			outputTokens !== this.tokens.length
		) {
			throw new Error("semantic summary token stream ended without a complete exact token ledger");
		}
		this.writeRecord({
			batches: this.batchOrdinal,
			content_bytes: Buffer.byteLength(this.content, "utf8"),
			content_sha256: textDigest(this.content),
			finish_reason: this.finishReason,
			output_tokens: outputTokens,
			reasoning_bytes: Buffer.byteLength(this.reasoning, "utf8"),
			reasoning_sha256: textDigest(this.reasoning),
			response_id: this.responseId,
			schema: SCHEMA,
			token_ids_sha256: tokenDigest(this.tokens),
			type: "complete",
		});
		fsyncSync(this.descriptor);
		closeSync(this.descriptor);
		this.descriptor = undefined;
		// link(2) is the portable create-only publication primitive available in
		// Node.  The final name becomes durable before the owned temporary name is
		// removed; a crash in between leaves two names for the same complete inode,
		// never a truncated final file or an overwritten prior receipt.
		linkSync(this.partialPath, this.finalPath);
		fsyncDirectory(this.directory);
		unlinkSync(this.partialPath);
		fsyncDirectory(this.directory);
		this.closed = true;
	}

	abort() {
		if (this.descriptor !== undefined) {
			fsyncSync(this.descriptor);
			closeSync(this.descriptor);
			this.descriptor = undefined;
		}
	}
}

export default function qwenSemanticSummaryTokenJournal() {
	const journalPath = process.env[JOURNAL_ENV];
	const originalFetch = globalThis.fetch.bind(globalThis);
	let intercepted = false;
	globalThis.fetch = async (url, init) => {
		const requestBody = await targetRequest(url, init);
		if (requestBody === undefined) return originalFetch(url, init);
		if (intercepted) throw new Error("semantic token journal observed more than one provider request");
		intercepted = true;
		const journal = new DurableTokenJournal(journalPath, requestBody);
		let response;
		try {
			response = await originalFetch(url, init);
		} catch (error) {
			journal.abort();
			throw error;
		}
		if (!response.body) {
			journal.abort();
			throw new Error("semantic summary response has no streaming body");
		}
		const reader = response.body.getReader();
		const body = new ReadableStream({
			async pull(controller) {
				try {
					const { done, value } = await reader.read();
					if (done) {
						journal.feed(new Uint8Array(), true);
						controller.close();
						return;
					}
					journal.feed(value);
					controller.enqueue(value);
				} catch (error) {
					journal.abort();
					controller.error(error);
				}
			},
			async cancel(reason) {
				journal.abort();
				await reader.cancel(reason);
			},
		});
		return new Response(body, {
			headers: response.headers,
			status: response.status,
			statusText: response.statusText,
		});
	};
}
