// Incremental row cache for Pi's append-only Markdown display.
const ROWS_PER_CHUNK = 128;
const PROSE_START = /^\p{L}/u;
const PROSE_END = /[\p{L}\p{N}\p{M}.,!?;:'")%…‘’“”。，！？：；、]$/u;
const PROSE_BODY = /^[\p{L}\p{N}\p{M} ,.!?;:'"()/%+=–—…‘’“”。，！？：；、-]+$/u;
const PROSE_IN_PROGRESS = /^[\p{L}\p{N}\p{M} ,.!?;:'"()/%+=–—…‘’“”。，！？：；、*-]+$/u;
const CLOSED_BOLD = /(?<= )\*\*[\p{L}\p{N}\p{M}]+(?: +[\p{L}\p{N}\p{M}]+)*\*\*(?= )/gu;
const BARE_URL = /www\.|:\//i;

function independentLine(line) {
    return PROSE_START.test(line) && PROSE_END.test(line) && !BARE_URL.test(line)
        && PROSE_BODY.test(line.replace(CLOSED_BOLD, "text"));
}

function unfinishedProse(line) {
    return PROSE_START.test(line) && !BARE_URL.test(line) && PROSE_IN_PROGRESS.test(line);
}

class RowChunks {
    chunks = [];
    pending = [];

    append(lines) {
        for (const line of lines) {
            this.pending.push(line);
            if (this.pending.length === ROWS_PER_CHUNK) {
                this.chunks.push(Object.freeze(this.pending));
                this.pending = [];
            }
        }
    }

    appendChunks(other) {
        if (this.pending.length && other.chunks.length) {
            this.chunks.push(Object.freeze(this.pending));
            this.pending = [];
        }
        for (const rows of other.chunks) {
            this.chunks.push(rows);
        }
        this.append(other.pending);
    }

    view() {
        return this.pending.length
            ? [...this.chunks, Object.freeze(this.pending.slice())]
            : this.chunks.slice();
    }
}

export class StreamingMarkdownRows {
    constructor(markdown) {
        this.markdown = markdown;
    }

    reset(width) {
        this.width = width;
        this.text = undefined;
        this.sealed = new RowChunks();
        this.offset = 0;
        this.canonical = false;
        this.resetProse();
    }

    resetProse() {
        this.prose = new RowChunks();
        this.proseOffset = this.offset;
        this.proseRejected = false;
    }

    rows(tokens) {
        return this.markdown.renderMarkdownTokens(tokens, this.width);
    }

    markdownRows(text) {
        return text.trim()
            ? this.rows(this.markdown.lexMarkdown(text))
            : [];
    }

    // Lines with no open inline constructs can be wrapped independently. A
    // later Setext underline or other continuation can still change the root
    // paragraph, so keep these rows reversible until a true blank separator.
    renderProse(text) {
        if (this.proseRejected) return undefined;
        const from = this.proseOffset;
        let end = from;
        let blank = false;
        for (let newline = text.indexOf("\n", end); newline !== -1; newline = text.indexOf("\n", end)) {
            const line = text.slice(end, newline);
            if (!line && end > this.offset) {
                blank = true;
                break;
            }
            if (!independentLine(line)) {
                this.resetProse();
                this.proseRejected = true;
                return undefined;
            }
            end = newline + 1;
        }
        const tail = text.slice(end);
        if (!blank && tail && !unfinishedProse(tail)) {
            this.resetProse();
            this.proseRejected = true;
            return undefined;
        }
        if (end > from) {
            this.prose.append(this.markdownRows(text.slice(from, end - 1)));
            this.proseOffset = end;
        }
        if (blank) {
            this.sealed.appendChunks(this.prose);
            // Lex the separator with the following block. Its blank row stays
            // mutable while more consecutive newlines arrive.
            this.offset = end - 1;
            this.resetProse();
            this.proseRejected = true;
            return undefined;
        }
        return [...this.prose.view(), Object.freeze(this.markdownRows(tail))];
    }

    renderBlocks(text) {
        const remainder = text.slice(this.offset);
        const tokens = this.markdown.lexMarkdown(remainder);
        if (Object.keys(tokens.links).length) {
            // Link definitions can resolve inline references in any earlier
            // block. Use the original whole-document parsing in that case.
            this.reset(this.width);
            this.canonical = true;
            return [Object.freeze(this.markdownRows(text))];
        }
        let cut = 0;
        let blockCount = 0;
        if (tokens.map((token) => token.raw).join("") === remainder) {
            let end = 0;
            tokens.forEach((token, index) => {
                end += token.raw.length;
                const following = tokens[index + 1];
                // A complete first line of a following root block proves a
                // list, table, fence or quote before the separator has ended.
                if (token.type === "space" && /\n[ \t]*\n/.test(token.raw)
                    && following && following.type !== "space"
                    && remainder.indexOf("\n", end) !== -1) {
                    cut = end;
                    blockCount = index + 1;
                }
            });
        }
        if (blockCount) {
            this.sealed.append(this.rows(tokens.slice(0, blockCount)));
            this.offset += cut;
            this.resetProse();
        }
        return [Object.freeze(this.rows(tokens.slice(blockCount)))];
    }

    render(text, width) {
        if (width !== this.width || this.text !== undefined && !text.startsWith(this.text)) {
            this.reset(width);
        }
        if (text === this.text) return this.rendered;
        let tail = [];
        if (!text.trim()) {
            tail = [];
        } else if (this.canonical) {
            tail = [Object.freeze(this.markdownRows(text))];
        } else {
            tail = this.renderProse(text) ?? this.renderBlocks(text);
        }
        this.text = text;
        this.rendered = [...this.sealed.view(), ...tail].filter((lines) => lines.length);
        return this.rendered;
    }
}
