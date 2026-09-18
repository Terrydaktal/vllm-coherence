import { createHash } from "node:crypto";

const PHASE_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;
const NONCE_PATTERN = /^[a-f0-9]{16}$/;
const DIGEST_PATTERN = /^[a-f0-9]{64}$/;
const DOMAIN = "qwen-r9700-hauhau-250k-soak-probe-v1\0";

function digestFor(phase, nonce) {
	return createHash("sha256").update(`${DOMAIN}${phase}\0${nonce}`, "utf8").digest("hex");
}

export default function qwenSoakProbe(pi) {
	pi.registerTool({
		name: "qwen_soak_probe",
		label: "Qwen soak probe",
		description:
			"Return one deterministic, side-effect-free qualification receipt for the supplied phase and nonce.",
		promptSnippet: "Emit a deterministic qualification receipt",
		promptGuidelines: [
			"Use this tool when a qualification prompt explicitly asks for a qwen_soak_probe receipt.",
			"Copy phase, nonce, and expectedDigest exactly from the prompt; do not invent values.",
		],
		parameters: {
			type: "object",
			additionalProperties: false,
			required: ["phase", "nonce", "expectedDigest"],
			properties: {
				phase: { type: "string", pattern: PHASE_PATTERN.source },
				nonce: { type: "string", pattern: NONCE_PATTERN.source },
				expectedDigest: { type: "string", pattern: DIGEST_PATTERN.source },
			},
		},
		async execute(_toolCallId, parameters) {
			const { phase, nonce, expectedDigest } = parameters;
			if (!PHASE_PATTERN.test(phase) || !NONCE_PATTERN.test(nonce)) {
				throw new Error("soak probe phase or nonce is outside the qualification contract");
			}
			const digest = digestFor(phase, nonce);
			if (expectedDigest !== digest) {
				throw new Error("soak probe digest does not authenticate the requested phase");
			}
			return {
				content: [
					{
						type: "text",
						text: `SOAK_PROBE_OK phase=${phase} nonce=${nonce} digest=${digest}`,
					},
				],
				details: { schema: "qwen-r9700-hauhau-250k-soak-probe-v1", phase, nonce, digest },
			};
		},
	});
}

export { digestFor };
