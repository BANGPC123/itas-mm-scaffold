"""Build the final report DOCX from its Markdown source, on the course template (T14).

Supports the subset used in docs/final_report/ITAS-MM_Final_Report.md: headings, paragraphs,
bullet and numbered lists, pipe tables, images with captions, and **bold** / `code` spans.
The template's styles and page setup are kept; its placeholder body is removed.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path("D:/DSP391m/1_mat1/Project Template.docx")
SOURCE = ROOT / "docs/final_report/ITAS-MM_Final_Report.md"
TARGET = ROOT / "docs/final_report/ITAS-MM_Final_Report.docx"
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def clear_body(document):
    body = document.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)


def style_or_none(document, name):
    try:
        return document.styles[name]
    except KeyError:
        return None


def add_runs(paragraph, text, size=None, italic=False):
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = "Consolas"
        else:
            run = paragraph.add_run(piece)
        if size:
            run.font.size = Pt(size)
        if italic:
            run.italic = True


def add_table(document, rows):
    header, body = rows[0], rows[1:]
    columns = len(header)
    table_style = style_or_none(document, "Table Grid")
    table = document.add_table(rows=1 + len(body), cols=columns)
    if table_style is not None:
        table.style = table_style
    for r, row in enumerate([header] + body):
        for c in range(columns):
            cell_text = row[c] if c < len(row) else ""
            cell = table.cell(r, c)
            cell.text = ""
            add_runs(cell.paragraphs[0], cell_text, size=9)
            if r == 0:
                for run in cell.paragraphs[0].runs:
                    run.bold = True
    document.add_paragraph()


def build():
    document = Document(str(TEMPLATE))
    clear_body(document)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    bullet_style = style_or_none(document, "List Bullet")
    index = 0
    paragraph_buffer = []

    def flush():
        if paragraph_buffer:
            paragraph = document.add_paragraph()
            add_runs(paragraph, " ".join(paragraph_buffer))
            paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush()
            index += 1
            continue
        if stripped.startswith("|"):
            flush()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                index += 1
            add_table(document, rows)
            continue
        if stripped.startswith("!["):
            flush()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            caption, image_path = match.group(1), (SOURCE.parent / match.group(2)).resolve()
            if not image_path.is_file():
                raise SystemExit(f"figure not found: {image_path}")
            document.add_picture(str(image_path), width=Inches(6.0))
            caption_paragraph = document.add_paragraph()
            add_runs(caption_paragraph, caption, size=9, italic=True)
            index += 1
            continue
        if stripped.startswith("# "):
            flush()
            document.add_heading(stripped[2:], level=0)
        elif stripped.startswith("## "):
            flush()
            document.add_heading(stripped[3:], level=1)
        elif stripped.startswith("### "):
            flush()
            document.add_heading(stripped[4:], level=2)
        elif stripped.startswith("- "):
            flush()
            paragraph = document.add_paragraph(style=bullet_style)
            add_runs(paragraph, stripped[2:])
        elif re.match(r"\d+\. ", stripped):
            # Keep the printed number: Word's list numbering would run on across separate lists.
            flush()
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.3)
            add_runs(paragraph, stripped)
        else:
            paragraph_buffer.append(stripped)
        index += 1
    flush()
    document.save(str(TARGET))
    print("wrote", TARGET)


if __name__ == "__main__":
    sys.exit(build())
