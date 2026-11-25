"""Patch the pinned Pi TUI Markdown component to retain append-only rows."""


def _replace(source: bytes, old: bytes, new: bytes) -> bytes:
    if source.count(old) != 1:
        raise ValueError("Pi Markdown source does not match the pinned renderer")
    return source.replace(old, new, 1)


def patch_markdown(source: bytes) -> bytes:
    """Keep Pi's original lexer, token styling and line wrapping on both paths."""
    source = _replace(
        source,
        b'const markdownParser = new Marked();',
        b'''const markdownParser = new Marked();
const renderChunks = Symbol.for("pi.tui.renderChunks");''',
    )
    source = b'import { StreamingMarkdownRows } from "./pi-streaming-markdown.mjs";\n' + source
    source = _replace(
        source,
        b'''    cachedLines;
    constructor''',
        b'''    cachedLines;
    streamingRows;
    standardChunks;
    constructor''',
    )
    source = _replace(
        source,
        b'''    invalidate() {
        this.cachedText = undefined;
        this.cachedWidth = undefined;
        this.cachedLines = undefined;
    }
    render(width) {
        // Check cache''',
        b'''    // Appends are checked when rendered, so replacing or truncating the
    // source also works. setText() explicitly invalidates both caches.
    setStreamingText(text) {
        this.text = text;
    }
    invalidate() {
        this.cachedText = undefined;
        this.cachedWidth = undefined;
        this.cachedLines = undefined;
        this.streamingRows = undefined;
        this.standardChunks = undefined;
        this.defaultStylePrefix = undefined;
    }
    canStreamRows() {
        return this.options.streamingAppend === true && !this.options.transform && this.paddingY === 0;
    }
    [renderChunks](width) {
        if (!this.canStreamRows()) {
            const lines = this.render(width);
            if (this.standardChunks?.lines !== lines) {
                this.standardChunks = { lines, chunks: [Object.freeze(lines.slice())] };
            }
            return this.standardChunks.chunks;
        }
        this.streamingRows ??= new StreamingMarkdownRows(this);
        return this.streamingRows.render(this.text, width);
    }
    render(width) {
        if (this.canStreamRows()) {
            return this[renderChunks](width).flat();
        }
        // Check cache''',
    )
    source = _replace(
        source,
        b'''        // Replace tabs with 3 spaces for consistent rendering
        const normalizedText = text.replace(/\\t/g, "   ");
        // Parse markdown to HTML-like tokens
        const tokens = markdownParser.lexer(normalizedText);
        trimPartialClosingFences(tokens);
        // Convert tokens to styled terminal output''',
        b'''        const result = this.renderMarkdownTokens(this.lexMarkdown(text), width);
        this.cachedText = this.text;
        this.cachedWidth = width;
        this.cachedLines = result;
        return result;
    }
    lexMarkdown(text) {
        const tokens = markdownParser.lexer(text.replace(/\\t/g, "   "));
        trimPartialClosingFences(tokens);
        return tokens;
    }
    renderMarkdownTokens(tokens, width) {
        const contentWidth = Math.max(1, width - this.paddingX * 2);
        // Convert tokens to styled terminal output''',
    )
    return _replace(
        source,
        b'''        // Combine top padding, content, and bottom padding
        const result = emptyLines.concat(contentLines, emptyLines);
        // Update cache
        this.cachedText = this.text;
        this.cachedWidth = width;
        this.cachedLines = result;
        return result.length > 0 ? result : [""];''',
        b'''        // Combine top padding, content, and bottom padding
        const result = emptyLines.concat(contentLines, emptyLines);
        return result.length > 0 ? result : [""];''',
    )
