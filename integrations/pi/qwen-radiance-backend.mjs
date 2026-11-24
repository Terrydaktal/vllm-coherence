import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { radianceBridgeRequest, radianceBridgeUrl } from "./qwen-radiance-bridge.mjs";
import { backendStatusDetails, readBackendLocalStatus, statusLabel } from "./qwen-backend-status.mjs";

const exec = promisify(execFile);
export const BACKEND_SCHEMA = "urn:coherence:backend-control:v1";
const ACTIONS = new Set(["status", "start", "stop"]);
const STATES = new Set(["stopped", "starting", "stopping", "idle", "generating", "running", "unavailable"]);
const helper = fileURLToPath(new URL("../../src/qwen_r9700_lab/radiance_backend.py", import.meta.url));
export const BACKEND_PROGRESS_KEY = "qwen-radiance-backend-progress";
const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// The host reports coarse lifecycle stages. Display only the current observed
// stage, without inventing loading/compilation progress or completing skipped work.
function progressLine(action, now) {
  const started = now();
  let entered = started, stage = action === "status" ? "Query backend status" : "Contact backend controller";
  return {
    update(report) {
      const observed = action !== "status" && report.operation?.action === action ? report.operation.stage : backendStatusText(report);
      const next = statusLabel(observed, 512);
      if (next !== stage) {
        stage = next;
        entered = now();
      }
    },
    lines() {
      const time = now(), elapsed = Math.max(0, time - started);
      const duration = Math.max(0, time - entered), spinner = SPINNER[Math.floor(elapsed / 100) % SPINNER.length];
      return [`${spinner} Radiance backend ${action}: ${stage} · ${(duration / 1000).toFixed(1)}s in stage · ${(elapsed / 1000).toFixed(1)}s total`];
    },
  };
}

export function validateBackendReport(value) {
  if (value?.schema !== BACKEND_SCHEMA || !STATES.has(value.state) ||
      (value.error !== undefined && (typeof value.error !== "string" || value.error.length > 512))) {
    throw new Error("invalid backend control response");
  }
  if (value.error) throw new Error(value.error);
  const unknown = value.state === "unavailable";
  if (!["ready", "running", "pinned"].every((key) => typeof value[key] === "boolean" || unknown && value[key] === null) ||
      typeof value.busy !== "boolean") {
    throw new Error("invalid backend status");
  }
  const op = value.operation;
  if (op != null && (!/^[a-f0-9]{32}$/.test(op.id ?? "") || !ACTIONS.has(op.action) ||
      !["pending", "complete", "failed"].includes(op.status) || typeof op.stage !== "string" ||
      op.stage.length > 512 || (op.error != null && (typeof op.error !== "string" || op.error.length > 512)))) {
    throw new Error("invalid backend operation status");
  }
  for (const key of ["active_requests", "queued_requests"]) {
    if (value[key] !== undefined && (!Number.isSafeInteger(value[key]) || value[key] < 0)) {
      throw new Error("invalid backend request counts");
    }
  }
  for (const key of ["release", "resources", "telemetry", "host_control"]) {
    if (value[key] !== undefined && value[key] !== null &&
      (typeof value[key] !== "object" || Array.isArray(value[key]))) throw new Error("invalid backend details");
  }
  if (value.host_control && typeof value.host_control.reachable !== "boolean") throw new Error("invalid host control status");
  for (const side of [value.release?.expected, value.release?.observed]) {
    if (side != null && (typeof side !== "object" || Array.isArray(side) ||
      ["container", "model", "image", "runtime_abi", "runtime_abi_prefix", "data_abi"].some((key) =>
        side[key] != null && (typeof side[key] !== "string" || side[key].length > 512)))) throw new Error("invalid release identity");
  }
  if (JSON.stringify(value).length > 65_536) throw new Error("backend report is too large");
  return value;
}

export async function backendCommand(action) {
  if (!ACTIONS.has(action)) throw new Error("usage: /backend [status|start|stop]");
  if (radianceBridgeUrl()) {
    return validateBackendReport(await radianceBridgeRequest({ operation: "backend", action }, { timeout: 20000 }));
  }
  const args = [helper, action, "--host", process.env.QWEN_RADIANCE_CACHE_HOST || "ai"];
  for (const [flag, key] of [["--container", "QWEN_RADIANCE_CONTAINER"],
    ["--cache-root", "QWEN_RADIANCE_CACHE_ROOT"], ["--abi", "QWEN_RADIANCE_CACHE_ABI"]]) {
    if (process.env[key]) args.push(flag, process.env[key]);
  }
  const { stdout } = await exec("python3", args, { timeout: 20000, maxBuffer: 65536 });
  return validateBackendReport(JSON.parse(stdout));
}

export function backendStatusText(report) {
  const details = [];
  if (report.pinned === false) details.push("container differs from the pinned release");
  if (report.ready) details.push("API ready");
  if (report.active_requests !== undefined) details.push(`${report.active_requests} active, ${report.queued_requests ?? 0} queued request(s)`);
  if (report.busy && report.operation) details.push(report.operation.stage);
  else if (report.operation?.status === "failed") details.push(`Last ${report.operation.action} failed: ${report.operation.error}`);
  return statusLabel(`Backend: ${report.state}${details.length ? ` · ${details.join(" · ")}` : ""}`, 700);
}

export default function installBackend(pi, {
  run = backendCommand,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  now = () => performance.now(),
  setIntervalFn = setInterval,
  clearIntervalFn = clearInterval,
  wall = Date.now,
  readLocal = readBackendLocalStatus,
} = {}) {
  let pending;
  let visible;
  let lastKnown;
  const monitors = new Set();
  const dispose = (monitor) => {
    if (monitor.disposed) return;
    monitor.disposed = true;
    if (monitor.timer !== undefined) clearIntervalFn(monitor.timer);
    monitor.ctx.signal?.removeEventListener?.("abort", monitor.cancel);
    if (visible === monitor) {
      monitor.ctx.ui.setWidget?.(BACKEND_PROGRESS_KEY, undefined);
      visible = undefined;
    }
  };
  for (const event of ["session_switch", "session_tree", "session_shutdown"]) pi.on?.(event, () => {
    for (const monitor of monitors) dispose(monitor);
    pending = undefined;
  });
  pi.registerCommand("backend", {
    description: "Backend, connection, resources and this chat’s cache/request status; start or gracefully stop the pinned model",
    getArgumentCompletions: (prefix) => [...ACTIONS].filter((value) => value.startsWith(prefix))
      .map((value) => ({ value, label: value })),
    handler: async (args, ctx) => {
      const action = args.trim() || "status";
      if (!ACTIONS.has(action)) return ctx.ui.notify("usage: /backend [status|start|stop]", "error");
      if (pending && action !== "status") return ctx.ui.notify("A backend operation is already pending. Use /backend status.", "warning");
      const mutation = action !== "status";
      const monitor = { ctx, disposed: false, timer: undefined };
      monitors.add(monitor);
      if (mutation) pending = monitor;
      const panel = progressLine(action, now);
      const redraw = () => {
        if (!monitor.disposed && visible === monitor) ctx.ui.setWidget?.(BACKEND_PROGRESS_KEY, panel.lines());
      };
      monitor.cancel = () => dispose(monitor);
      ctx.signal?.addEventListener?.("abort", monitor.cancel, { once: true });
      if (ctx.signal?.aborted) dispose(monitor);
      // A concurrent /backend status must not replace a running start/stop panel.
      if (!monitor.disposed && !visible && ctx.ui.setWidget) {
        visible = monitor;
        redraw();
        monitor.timer = setIntervalFn(redraw, 100);
        monitor.timer?.unref?.();
      }
      let local;
      try {
        if (monitor.disposed) return;
        let report;
        if (mutation) report = validateBackendReport(await run(action));
        else {
          const [hostResult, localResult] = await Promise.allSettled([run(action), readLocal(ctx, { now: wall() })]);
          if (localResult.status === "fulfilled") local = localResult.value;
          if (hostResult.status === "rejected") throw hostResult.reason;
          report = validateBackendReport(hostResult.value);
        }
        if (report.host_control?.reachable !== false && typeof report.running === "boolean") lastKnown = { report, observed_at_ms: wall() };
        if (monitor.disposed) return;
        panel.update(report);
        redraw();
        if (!mutation || !report.busy) {
          ctx.ui.notify(mutation ? backendStatusText(report) : backendStatusDetails({ report, local, lastKnown, now: wall(), theme: ctx.ui.theme }),
            report.host_control?.reachable === false ? "error" : report.pinned ? "info" : "warning");
          return;
        }
        const operation = report.operation?.id;
        if (!operation) throw new Error("backend operation identity unavailable");
        const deadline = now() + 16 * 60_000;
        while (now() < deadline) {
          await sleep(1000);
          if (monitor.disposed) return;
          report = validateBackendReport(await run("status"));
          if (monitor.disposed) return;
          if (report.operation?.id !== operation) throw new Error("backend operation changed; use /backend status");
          panel.update(report);
          redraw();
          if (!report.busy) {
            if (report.operation.status === "failed") throw new Error(report.operation.error || "backend operation failed");
            if (report.operation.status !== "complete" || (action === "start" ? !report.ready : report.running)) {
              throw new Error("backend operation interrupted; use /backend status");
            }
            ctx.ui.notify(backendStatusText(report), "info");
            return;
          }
        }
        throw new Error("operation is still pending on the host; use /backend status");
      } catch (error) {
        if (!monitor.disposed) ctx.ui.notify(mutation ?
          statusLabel(`Backend ${action} failed: ${error.message}. Check /backend status before retrying.`, 700) :
          backendStatusDetails({ local, lastKnown, error, now: wall(), theme: ctx.ui.theme }), "error");
      } finally {
        dispose(monitor);
        monitors.delete(monitor);
        if (pending === monitor) pending = undefined;
      }
    },
  });
}
