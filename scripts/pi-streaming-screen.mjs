export const RENDER_CHUNKS = Symbol.for("pi.tui.renderChunks");

export function renderChunks(component, width) {
  return typeof component[RENDER_CHUNKS] === "function"
    ? component[RENDER_CHUNKS](width)
    : [component.render(width)];
}

function commonPrefix(chunks, previousChunks, limit) {
  let rows = 0;
  for (let i = 0; rows < limit && i < chunks.length && i < previousChunks.length; i++) {
    const current = chunks[i];
    const previous = previousChunks[i];
    const count = Math.min(current.length, previous.length, limit - rows);
    if (current !== previous) {
      for (let j = 0; j < count; j++) {
        if (current[j] !== previous[j]) return rows + j;
      }
    }
    rows += count;
    if (current.length !== previous.length) break;
  }
  return rows;
}

export function prepareScreenFrame(tui, width, height) {
  if (tui.hasOverlayEntries) {
    return { lines: tui.render(width), chunks: undefined, reusedPrefix: 0 };
  }

  const rendered = renderChunks(tui, width);
  // Only opt in when a component provides immutable, potentially reusable rows.
  // Snapshot any mutable rows so a custom component may safely reuse its array.
  const chunks = rendered.some(Object.isFrozen)
    ? rendered.map((rows) => Object.isFrozen(rows) ? rows : rows.slice())
    : undefined;
  let reusedPrefix = 0;
  if (chunks && tui.previousRenderChunks && tui.previousWidth === width &&
      tui.previousHeight === height && tui.previousKittyImageIds.size === 0) {
    const rowCount = chunks.reduce((total, rows) => total + rows.length, 0);
    // Cursor extraction removes a marker in the visible viewport. Keep both
    // the previous and current viewport raw, even if no content changed there.
    const limit = Math.max(0, Math.min(tui.previousLines.length, rowCount) - height);
    reusedPrefix = commonPrefix(chunks, tui.previousRenderChunks, limit);
  }

  const lines = reusedPrefix ? tui.previousLines.slice(0, reusedPrefix) : [];
  let row = 0;
  for (const chunk of rendered) {
    for (let i = Math.max(0, reusedPrefix - row); i < chunk.length; i++) {
      lines.push(chunk[i]);
    }
    row += chunk.length;
  }
  return { lines, chunks, reusedPrefix };
}
