"""Genera el informe desde Markdown, sin cargar corpus, modelos ni métricas."""

from __future__ import annotations

import argparse
import html
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = A4
MARGIN_X = 63
MARGIN_TOP = 57
MARGIN_BOTTOM = 53
CONTENT_WIDTH = WIDTH - 2 * MARGIN_X


def inline(text: str) -> str:
    """Subconjunto explícito: enlaces, negrita, cursiva y código en línea."""
    value = html.escape(text, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<link href="{html.escape(m[2], quote=True)}">'
                  f"{m[1]}</link>",
        value,
    )
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body", fontName="Times-Roman", fontSize=11.5, leading=15.8,
        alignment=TA_JUSTIFY, firstLineIndent=18, spaceAfter=6,
        splitLongWords=True, allowWidows=0, allowOrphans=0,
    )
    block = ParagraphStyle("block", parent=body, alignment=TA_LEFT, firstLineIndent=0)
    table_header = ParagraphStyle(
        "table_header", parent=block, fontName="Times-Bold", fontSize=10.6,
        leading=12.8, spaceAfter=0,
    )
    meta = ParagraphStyle(
        "meta", parent=block, fontSize=11, leading=14, alignment=TA_CENTER,
        spaceAfter=4, keepWithNext=True,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=meta, fontName="Times-Bold", fontSize=20,
            leading=24, spaceBefore=5, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=meta, fontName="Times-Italic", spaceAfter=7,
        ),
        "meta": meta,
        "summary": ParagraphStyle(
            "summary", parent=block, alignment=TA_JUSTIFY, spaceBefore=7, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", parent=block, fontName="Times-Bold", fontSize=14,
            leading=18, spaceBefore=5, spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3", parent=block, fontName="Times-Bold", fontSize=11.5,
            leading=15, spaceBefore=5, spaceAfter=5,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "caption", parent=block, fontSize=10.6, leading=12.8,
            spaceBefore=4, spaceAfter=3, keepWithNext=True,
        ),
        "cell": ParagraphStyle(
            "cell", parent=block, fontSize=10.5, leading=12.6, spaceAfter=0,
        ),
        "cell_right": ParagraphStyle(
            "cell_right", parent=block, fontSize=10.5, leading=12.6,
            alignment=TA_RIGHT, spaceAfter=0,
        ),
        "table_header": table_header,
        "table_header_right": ParagraphStyle(
            "table_header_right", parent=table_header, alignment=TA_RIGHT,
        ),
        "reference": ParagraphStyle(
            "reference", parent=block, fontSize=10.5, leading=12, spaceAfter=4,
        ),
    }


def make_table(lines: list[str], style: dict) -> Table:
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")]
             for line in lines]
    alignments = [cell.endswith(":") for cell in cells[1]]
    rows = [cells[0], *cells[2:]]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError("Tabla Markdown con columnas inconsistentes")
    if count == 5:
        ratios = [0.22, 0.16, 0.20, 0.20, 0.22]
    elif count == 4:
        ratios = ([0.30, 0.10, 0.30, 0.30] if rows[0][1] == "n"
                  else [0.34, 0.21, 0.21, 0.24])
    elif rows[0][0] == "Elemento":
        ratios = [0.33, 0.16, 0.51]
    else:
        ratios = [0.42, 0.22, 0.36]
    if len(ratios) != count:
        raise ValueError("Número de columnas no soportado")
    header = [
        Paragraph(inline(cell), style["table_header_right" if aligned else "table_header"])
        for cell, aligned in zip(rows[0], alignments, strict=True)
    ]
    content = [header, *[
        [
            Paragraph(inline(cell), style["cell_right" if aligned else "cell"])
            for cell, aligned in zip(row, alignments, strict=True)
        ]
        for row in rows[1:]
    ]]
    table = Table(content, colWidths=[v * CONTENT_WIDTH for v in ratios],
                  repeatRows=1, hAlign="LEFT", spaceAfter=8)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
    ]))
    return table


def markdown_story(source: str) -> list:
    style = styles()
    lines = source.splitlines()
    story = []
    i = 0
    references = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(make_table(table_lines, style))
            continue
        heading = re.match(r"^(#{1,3}) (.+)$", line)
        if heading:
            key = {1: "title", 2: "h2", 3: "h3"}[len(heading[1])]
            if heading[2] == "Fuentes y trazabilidad":
                references = True
            story.append(Paragraph(inline(heading[2]), style[key]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            if lines[i].lstrip().startswith(("#", "|", "<!--")):
                break
            paragraph.append(lines[i].strip())
            i += 1
        content = inline(" ".join(paragraph))
        key = "reference" if references else "body"
        if line.startswith("**Tabla "):
            key = "caption"
            number, _, caption = content.partition("</b> ")
            content = f"{number}</b><br/><i>{caption}</i>"
        elif line.startswith("Comparación de Jaccard,"):
            key = "subtitle"
        elif line.startswith(("**Escuela Politécnica Nacional**", "**Integrantes.**", "Recuperación de Información ·")):
            key = "meta"
        elif line.startswith("**Resumen.**"):
            key = "summary"
        story.append(Paragraph(content, style[key]))
    return story


def decorate(canvas: Canvas, doc: BaseDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.drawRightString(WIDTH - MARGIN_X, HEIGHT - 31, str(doc.page))
    canvas.restoreState()


def generate(source: Path, target: Path) -> int:
    text = source.read_text(encoding="utf-8")
    text.encode("cp1252")
    authors = next(line.removeprefix("**Integrantes.** ") for line in text.splitlines()
                   if line.startswith("**Integrantes.** "))
    if text.count("<!-- pagebreak -->") != 4:
        raise ValueError("Se esperaban cinco páginas editoriales")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.pdf")
    document = BaseDocTemplate(
        str(temporary), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM, title="Recuperación de información financiera en FiQA",
        author=authors.rstrip("."),
        subject="Comparación de Jaccard, TF-IDF, BM25 y MiniLM/FAISS",
    )
    document.addPageTemplates(PageTemplate(
        id="report", frames=[Frame(
            MARGIN_X, MARGIN_BOTTOM, CONTENT_WIDTH, HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )], onPage=decorate,
    ))
    previous_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = str(int(datetime(2026, 11, 19, 5, tzinfo=UTC).timestamp()))
    try:
        document.build(markdown_story(text), canvasmaker=Canvas)
    finally:
        if previous_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous_epoch
    pages = len(PdfReader(temporary).pages)
    if pages != 5:
        raise ValueError(f"El PDF tiene {pages} páginas; revisar {temporary.name}")
    temporary.replace(target)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "informe_tecnico.pdf")
    args = parser.parse_args()
    pages = generate(HERE / "informe_tecnico.md", args.output)
    print(f"PDF generado: {pages} páginas")


if __name__ == "__main__":
    main()
