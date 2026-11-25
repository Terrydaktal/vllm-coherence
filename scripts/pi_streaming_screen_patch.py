"""Patch Pi 0.84.2's main-screen renderer to reuse unchanged row prefixes."""


def replace(source: bytes, old: bytes, new: bytes, count: int = 1) -> bytes:
    if source.count(old) != count:
        raise ValueError(f"Unexpected Pi TUI patch anchor: {old.splitlines()[0]!r}")
    return source.replace(old, new)


def patch_tui(source: bytes) -> bytes:
    source = replace(
        source,
        b'/** Type guard to check if a component implements Focusable */',
        b'''import { RENDER_CHUNKS, renderChunks } from "./pi-streaming-screen.mjs";
/** Type guard to check if a component implements Focusable */''',
    )
    source = replace(
        source,
        b'''    render(width) {
        const lines = [];
        for (const child of this.children) {''',
        b'''    [RENDER_CHUNKS](width) {
        // A subclass or instance can define its own render semantics.
        if (this.render !== Container.prototype.render)
            return [this.render(width)];
        const chunks = [];
        for (const child of this.children) {
            for (const rows of renderChunks(child, width)) {
                chunks.push(rows);
            }
        }
        return chunks;
    }
    render(width) {
        const lines = [];
        for (const child of this.children) {''',
    )
    return replace(
        source,
        b'''    applyLineResets(lines) {
        const reset = SEGMENT_RESET;
        for (let i = 0; i < lines.length; i++) {''',
        b'''    applyLineResets(lines, fromRow = 0) {
        const reset = SEGMENT_RESET;
        for (let i = fromRow; i < lines.length; i++) {''',
    )


def patch_main_screen(source: bytes) -> bytes:
    source = replace(
        source,
        b'import { TuiBase } from "./tui.js";',
        b'''import { TuiBase } from "./tui.js";
import { prepareScreenFrame } from "./pi-streaming-screen.mjs";''',
    )
    source = replace(
        source,
        b'    previousLines = [];',
        b'''    previousLines = [];
    previousRenderChunks;''',
    )
    for method in (b"restoreRenderState(state)", b"resetRenderState()"):
        source = replace(
            source,
            b"    " + method + b" {",
            b"    " + method + b" {\n        this.previousRenderChunks = undefined;",
        )
    source = replace(
        source,
        b'''    collectKittyImageIds(lines) {
        const ids = new Set();
        for (const line of lines) {
            for (const id of extractKittyImageIds(line)) {''',
        b'''    collectKittyImageIds(lines, fromRow = 0) {
        const ids = new Set();
        for (let i = fromRow; i < lines.length; i++) {
            for (const id of extractKittyImageIds(lines[i])) {''',
    )
    source = replace(
        source,
        b'''        // Render all components to get new lines
        let newLines = this.render(width);''',
        b'''        const { lines, chunks, reusedPrefix } = prepareScreenFrame(this, width, height);
        let newLines = lines;''',
    )
    source = replace(
        source,
        b"        newLines = this.applyLineResets(newLines);",
        b'''        newLines = this.applyLineResets(newLines, reusedPrefix);
        // A reused prefix is admitted only when the old frame had no Kitty IDs.
        const newKittyImageIds = this.collectKittyImageIds(newLines, reusedPrefix);''',
    )
    source = replace(
        source,
        b"this.previousKittyImageIds = this.collectKittyImageIds(newLines);",
        b"this.previousKittyImageIds = newKittyImageIds;",
        count=3,
    )
    for indent, count in ((b" " * 12, 2), (b" " * 8, 1)):
        old = b"\n" + indent + b"this.previousLines = newLines;"
        source = replace(
            source,
            old,
            old + b"\n" + indent + b"this.previousRenderChunks = chunks;",
            count=count,
        )
    source = replace(
        source,
        b"        for (let i = 0; i < maxLines; i++) {",
        b"        for (let i = reusedPrefix; i < maxLines; i++) {",
    )
    source = replace(
        source,
        b'''        if (firstChanged !== -1) {
            const expandedRange = this.expandChangedRangeForKittyImages''',
        b'''        if (firstChanged !== -1 && (this.previousKittyImageIds.size || newKittyImageIds.size)) {
            const expandedRange = this.expandChangedRangeForKittyImages''',
    )
    return replace(
        source,
        b'''        if (firstChanged === -1) {
            this.positionHardwareCursor''',
        b'''        if (firstChanged === -1) {
            this.previousRenderChunks = chunks;
            this.positionHardwareCursor''',
    )
