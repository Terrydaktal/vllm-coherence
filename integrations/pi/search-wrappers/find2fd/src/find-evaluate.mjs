import { spawnSync } from "node:child_process";
import { inheritedStdio, unusedInputFd } from "./find-descriptors.mjs";

export const MAX_EVALUATION_INPUT_BYTES = 32 * 1024 * 1024;
export const MAX_EVALUATION_RECORDS = 100256;

function argumentsFor(options, source, dereference) {
  return [`-${dereference ?? options.evaluationDereference ?? options.dereference ?? "P"}`, ...(options.optimization ? [options.optimization] : []),
    "-files0-from", source, "-maxdepth", "0", ...options.evaluationArgs];
}

/** Parse GNU's predicates without starting a filesystem traversal or actions. */
export function validateEvaluation(options, { cwd, callerDescriptors = options.callerDescriptors ?? [] } = {}) {
  const result = spawnSync("find", argumentsFor(options, "/dev/null"), {
    cwd, argv0: "find", stdio: inheritedStdio(["ignore", "pipe", "pipe"], callerDescriptors), timeout: 2000, maxBuffer: 1024 * 1024,
  });
  options.evaluationHasWarnings = Boolean(result.stderr?.length);
  return !result.error && !result.signal && result.status === 0;
}

function paddedContext(directive, record) {
  const [, flags, widthText, precisionText, kind] = directive;
  const width = Number(widthText || 0), precision = precisionText === undefined ? null : Number(precisionText);
  if (!Number.isSafeInteger(width) || width > MAX_EVALUATION_INPUT_BYTES ||
      (precision !== null && (!Number.isSafeInteger(precision) || precision > MAX_EVALUATION_INPUT_BYTES)))
    throw new Error("GNU contextual format width or precision bound");
  if (kind !== "d") {
    let value = kind === "H" ? Buffer.from(record.root) : record.relative;
    if (!Buffer.isBuffer(value)) throw new Error("GNU contextual path protocol");
    if (precision !== null) value = value.subarray(0, precision);
    const padding = Buffer.alloc(Math.max(0, width - value.length), 32);
    return Buffer.concat(flags.includes("-") ? [value, padding] : [padding, value]);
  }
  if (!Number.isSafeInteger(record.depth) || record.depth < 0) throw new Error("GNU contextual depth protocol");
  let digits = record.depth === 0 && precision === 0 ? "" : String(record.depth);
  if (precision !== null) digits = digits.padStart(precision, "0");
  const sign = flags.includes("+") ? "+" : flags.includes(" ") ? " " : "";
  let value = sign + digits;
  if (flags.includes("-") ) value = value.padEnd(width, " ");
  else if (flags.includes("0") && precision === null) value = sign + digits.padStart(Math.max(0, width - sign.length), "0");
  else value = value.padStart(width, " ");
  return Buffer.from(value);
}

/** Octal escapes carry raw contextual bytes through Node's Unicode argv safely. */
export function rewritePrintfContext(format, record) {
  let rewritten = "", bytes = 0;
  const append = (part) => {
    bytes += Buffer.byteLength(part);
    if (bytes > MAX_EVALUATION_INPUT_BYTES) throw new Error("GNU contextual format byte bound");
    rewritten += part;
  };
  for (let index = 0; index < format.length;) {
    if (format[index] === "\\") {
      if (format[index + 1] === "c") { append(format.slice(index)); break; }
      append(format.slice(index, index + 2)); index += 2; continue;
    }
    if (format[index] !== "%") { append(format[index++]); continue; }
    if (format[index + 1] === "%") { append("%%"); index += 2; continue; }
    const directive = /^%([#0 +\-]*)(\d*)(?:\.(\d*))?([dHP])/.exec(format.slice(index));
    if (!directive) { append(format[index++]); continue; }
    const value = paddedContext(directive, record);
    if (bytes + value.length * 4 > MAX_EVALUATION_INPUT_BYTES) throw new Error("GNU contextual format byte bound");
    append(Array.from(value, (byte) => `\\${byte.toString(8).padStart(3, "0")}`).join(""));
    index += directive[0].length;
  }
  return rewritten;
}

/** GNU evaluates each fd-ordered path once; its maximum depth is always zero. */
export function createEvaluationRequest(options, records, { dereference, evaluationArgs = options.evaluationArgs } = {}) {
  if (records.length > MAX_EVALUATION_RECORDS) throw new Error("GNU evaluation record bound");
  const chunks = [];
  let bytes = 0;
  for (const record of records) {
    const path = Buffer.isBuffer(record) ? record : record.path;
    if (!Buffer.isBuffer(path) || !path.length || path.includes(0)) throw new Error("GNU evaluation path protocol");
    bytes += path.length + 1;
    if (bytes > MAX_EVALUATION_INPUT_BYTES) throw new Error("GNU evaluation input byte bound");
    chunks.push(path, Buffer.from([0]));
  }
  const inputFd = options.evaluationInputFd ?? unusedInputFd(inheritedStdio(undefined, options.callerDescriptors ?? []));
  return { program: "find", args: argumentsFor({ ...options, evaluationArgs }, `/proc/self/fd/${inputFd}`, dereference),
    input: Buffer.concat(chunks, bytes), inputFd };
}

/** -H follows original starting points, never descendant symlinks fed as roots. */
export function createEvaluationRequests(options, records) {
  if (!options.contextualPrintf && options.dereference !== "H") return [createEvaluationRequest(options, records)];
  if (!records.length) return [createEvaluationRequest(options, records)];
  const requests = [];
  let group = [], dereference, groupArgs, previousKey, bytes = 0;
  const complete = () => {
    const request = createEvaluationRequest(options, group, { dereference, evaluationArgs: groupArgs });
    bytes += request.input.length + request.args.reduce((sum, arg) => sum + Buffer.byteLength(arg) + 1, 0);
    if (bytes > MAX_EVALUATION_INPUT_BYTES) throw new Error("GNU grouped evaluation byte bound");
    requests.push(request);
  };
  for (const record of records) {
    const current = options.dereference === "H" ? record.depth === 0 ? "H" : "P" : options.evaluationDereference ?? options.dereference ?? "P";
    const args = [...options.evaluationArgs];
    for (const index of options.printfContext ?? []) args[index] = rewritePrintfContext(args[index], record);
    const key = JSON.stringify([current, args]);
    if (group.length && key !== previousKey) {
      complete(); group = [];
    }
    dereference = current; groupArgs = args; previousKey = key; group.push(record);
  }
  if (group.length) complete();
  return requests;
}
