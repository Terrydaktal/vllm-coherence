// Qualified -m stdin searches stream positional reads to RG's private input pipe
// without consuming the caller's open file description. Only the proved completed
// cursor is then consumed forward on the original descriptor.
import { constants, fstatSync, read, readFileSync, readSync } from "node:fs";

const CHUNK = 64 * 1024;
const sameVersion = (a, b) => ["dev", "ino", "size", "mtimeNs", "ctimeNs", "mode"].every((key) => a[key] === b[key]);
const readAt = (buffer, length, position) => new Promise((resolve, reject) => {
  read(0, buffer, 0, length, position, (error, bytesRead) => error ? reject(error) : resolve(bytesRead));
});

export function seekableStdin() {
  try {
    const info = fstatSync(0, { bigint: true });
    const descriptor = readFileSync("/proc/self/fdinfo/0", "utf8");
    const offset = /^pos:\s*(\d+)$/m.exec(descriptor), flags = /^flags:\s*([0-7]+)$/m.exec(descriptor);
    const modes = flags ? Number.parseInt(flags[1], 8) : 0;
    // O_PATH cannot be read; O_DIRECT requires native alignment behavior that
    // ordinary Node buffers do not establish. Retain GNU's original diagnostics.
    if (!info.isFile() || !offset || !flags || (modes & 3) === constants.O_WRONLY ||
        (modes & (0o10000000 | (constants.O_DIRECT ?? 0))) ||
        info.size > BigInt(Number.MAX_SAFE_INTEGER)) return undefined;
    const start = BigInt(offset[1]);
    if (start > BigInt(Number.MAX_SAFE_INTEGER)) return undefined;
    // proc/sysfs may present regular descriptors whose size is unrelated to
    // their readable contents. Prove the declared endpoints without changing
    // the caller's cursor; this also rejects unreadable regular descriptors.
    const probe = Buffer.allocUnsafe(1);
    if (readSync(0, probe, 0, 1, info.size) !== 0 ||
        (info.size > 0n && readSync(0, probe, 0, 1, info.size - 1n) !== 1) ||
        !sameVersion(info, fstatSync(0, { bigint: true }))) return undefined;
    return { info, start };
  } catch { return undefined; }
}

export function prepareSeekableStdin() {
  const before = seekableStdin();
  if (!before) throw new Error("stdin is not a qualified seekable regular file");
  const length = before.info.size > before.start ? before.info.size - before.start : 0n;
  let selected = 0, lastEnd = 0n, closed = false;
  const unchanged = () => {
    const current = seekableStdin();
    return current && current.start === before.start && sameVersion(before.info, current.info);
  };
  return {
    close() { closed = true; },
    async feed(stream, stopped = () => false) {
      // -m may close RG's pipe before all remaining bytes have been fed. That
      // expected EPIPE does not consume FD0 or invalidate the selected cursor.
      stream.on("error", () => {});
      const buffer = Buffer.allocUnsafe(CHUNK);
      let sent = 0n;
      try {
        if (!unchanged()) throw new Error("stdin changed before streaming its records");
        while (sent < length && !closed && !stopped() && !stream.destroyed) {
          const count = Number(length - sent < BigInt(CHUNK) ? length - sent : BigInt(CHUNK));
          const bytesRead = await readAt(buffer, count, before.start + sent);
          if (closed || stopped() || stream.destroyed) break;
          if (!bytesRead) throw new Error("stdin ended while streaming its records");
          await new Promise((resolve, reject) => stream.write(buffer.subarray(0, bytesRead),
            (error) => error ? reject(error) : resolve()));
          sent += BigInt(bytesRead);
        }
      } catch (error) {
        if (!["EPIPE", "ERR_STREAM_DESTROYED", "ERR_STREAM_WRITE_AFTER_END"].includes(error.code)) throw error;
      } finally { if (!stream.destroyed && !stream.writableEnded) stream.end(); }
    },
    record(message, maximum) {
      if (message.type !== "match" || selected >= maximum) return;
      const value = message.data;
      if (!Number.isSafeInteger(value.absolute_offset) || value.absolute_offset < 0)
        throw new Error("stdin record offset is not exact");
      const bytes = value.lines?.bytes !== undefined ? Buffer.from(value.lines.bytes, "base64") : Buffer.from(value.lines?.text ?? "");
      const end = BigInt(value.absolute_offset) + BigInt(bytes.length);
      if (end > length || end < lastEnd) throw new Error("stdin record lies outside its prepared input");
      selected++; lastEnd = end;
    },
    async advance(maximum, stopped = () => false) {
      if (stopped()) return;
      const current = seekableStdin();
      if (!current || current.start !== before.start || !sameVersion(before.info, current.info))
        throw new Error("stdin changed before completing its cursor");
      const target = selected >= maximum ? lastEnd : length;
      const buffer = Buffer.allocUnsafe(CHUNK);
      let consumed = 0n;
      while (consumed < target) {
        if (stopped()) return;
        const bytesRead = await readAt(buffer, Number(target - consumed < BigInt(CHUNK) ? target - consumed : BigInt(CHUNK)), null);
        if (!bytesRead) throw new Error("stdin ended before its completed cursor");
        consumed += BigInt(bytesRead);
      }
      if (stopped()) return;
      const after = seekableStdin();
      if (!after || after.start !== before.start + target || !sameVersion(before.info, after.info))
        throw new Error("stdin changed while completing its cursor");
    },
  };
}
