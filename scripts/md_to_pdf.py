#!/usr/bin/env python3
"""Render a markdown file to a print-ready HTML page."""
import sys, pathlib, html, markdown

if len(sys.argv) != 3:
    print("usage: md_to_pdf.py <input.md> <output.html>", file=sys.stderr)
    sys.exit(1)

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])

md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists", "smarty"])
body = md.convert(src.read_text(encoding="utf-8"))

css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: "Charter", "Source Serif Pro", Georgia, serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
body { max-width: 100%; }
h1 { font-size: 20pt; margin: 0 0 12pt 0; border-bottom: 2px solid #1a1a1a; padding-bottom: 4pt; page-break-after: avoid; }
h2 { font-size: 14pt; margin: 18pt 0 8pt 0; border-bottom: 1px solid #888; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 12pt 0 4pt 0; font-style: italic; page-break-after: avoid; }
p { margin: 6pt 0; orphans: 2; widows: 2; }
hr { border: none; border-top: 1px solid #bbb; margin: 16pt 0; }
blockquote { margin: 8pt 0 8pt 12pt; padding-left: 10pt; border-left: 3px solid #888; color: #444; font-style: italic; }
code { font-family: "JetBrains Mono", "Source Code Pro", monospace; font-size: 0.88em; background: #f4f4f4; padding: 0.5pt 3pt; border-radius: 2pt; }
pre { font-family: "JetBrains Mono", monospace; font-size: 0.85em; background: #f6f8fa; border: 1px solid #ddd; padding: 6pt 8pt; border-radius: 3pt; overflow-x: auto; page-break-inside: avoid; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #ececec; font-weight: 600; }
tr:nth-child(even) { background: #fafafa; }
ul, ol { margin: 6pt 0 6pt 0; padding-left: 18pt; }
li { margin: 2pt 0; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: #0050a0; text-decoration: none; }
a:hover { text-decoration: underline; }
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
