"""Render a markdown file to a 1-page-friendly HTML page (compact styling).

Variant of md_to_pdf.py with tighter margins, smaller fonts, and dense
spacing for one-page handouts.
"""
from __future__ import annotations

import sys, pathlib, html, markdown

if len(sys.argv) != 3:
    print("usage: md_to_pdf_compact.py <input.md> <output.html>", file=sys.stderr)
    sys.exit(1)

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])

md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists", "smarty"])
body = md.convert(src.read_text(encoding="utf-8"))

css = """
@page { size: A4; margin: 10mm 12mm 10mm 12mm; }
html { font-family: "Charter", "Source Serif Pro", Georgia, serif; font-size: 8.5pt; line-height: 1.20; color: #1a1a1a; }
body { max-width: 100%; }
h1 { font-size: 14pt; margin: 0 0 4pt 0; border-bottom: 1.5px solid #1a1a1a; padding-bottom: 2pt; page-break-after: avoid; }
h2 { font-size: 10pt; margin: 6pt 0 3pt 0; border-bottom: 0.5px solid #888; padding-bottom: 1pt; page-break-after: avoid; }
h3 { font-size: 9pt; margin: 4pt 0 2pt 0; page-break-after: avoid; }
p { margin: 3pt 0; orphans: 2; widows: 2; }
hr { border: none; border-top: 0.5px solid #bbb; margin: 6pt 0; }
blockquote { margin: 4pt 0 4pt 8pt; padding-left: 6pt; border-left: 2px solid #888; color: #444; font-style: italic; }
code { font-family: "JetBrains Mono", monospace; font-size: 0.85em; background: #f4f4f4; padding: 0.5pt 2pt; border-radius: 1pt; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 7.5pt; width: 100%; page-break-inside: avoid; }
th, td { border: 0.5px solid #999; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #ececec; font-weight: 600; }
tr:nth-child(even) { background: #fafafa; }
ul, ol { margin: 3pt 0 3pt 0; padding-left: 14pt; }
li { margin: 1pt 0; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: #0050a0; text-decoration: none; }
"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(src.stem)}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""
dst.write_text(doc, encoding="utf-8")
print(f"wrote {dst} ({len(doc):,} bytes)")
