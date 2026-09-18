# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown-it-py==4.0.0", "weasyprint==68.1"]
# ///
"""Render the checked public report; no GPU or model dependencies are needed."""

from pathlib import Path

from markdown_it import MarkdownIt
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 16mm 14mm;
  @bottom-right { content: counter(page); font-size: 8pt; color: #59616b; }
}
@page stage { size: A3 landscape; margin: 14mm; }
.wide-table { page: stage; break-before: page; break-after: page; }
.wide-table table { font-size: 7.3pt; line-height: 1.29; }
.wide-table th, .wide-table td { padding: 3pt; }
body { font-family: DejaVu Sans, sans-serif; font-size: 8.6pt;
       line-height: 1.36; color: #17212c; }
h1 { font-size: 20pt; line-height: 1.15; color: #102c48; }
h2 { font-size: 12pt; color: #102c48; margin-top: 17pt; }
h3 { font-size: 10pt; margin-top: 12pt; }
h1, h2, h3 { break-after: avoid; }
a { color: #195b8b; text-decoration: none; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; font-size: 7.4pt;
        margin: 9pt 0; }
th { background: #e9f0f6; text-align: left; }
th, td { padding: 4pt; border: 0.5pt solid #c7d1db; vertical-align: top; }
tr { break-inside: avoid; }
pre { background: #f1f4f7; padding: 7pt; white-space: pre-wrap; }
code { font-family: DejaVu Sans Mono, monospace; font-size: 7.4pt;
       overflow-wrap: anywhere; }
li { margin: 3pt 0; }
"""


def main():
    content = MarkdownIt("commonmark").enable("table").render((ROOT / "REPORT.md").read_text())
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Restoring M1/M8 and Eager/Compiled Agreement on RDNA4</title>"
        '<meta name="author" content="Terrydaktal">'
        f"<style>{CSS}</style></head><body>{content}</body></html>"
    )
    (ROOT / "report.html").write_text(document)
    HTML(string=document, base_url=str(ROOT)).write_pdf(ROOT / "report.pdf")
    print(ROOT / "report.pdf")


if __name__ == "__main__":
    main()
