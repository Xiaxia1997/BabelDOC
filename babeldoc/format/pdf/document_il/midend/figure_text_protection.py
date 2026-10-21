"""Keep selectable text inside figures in the original PDF language."""

from babeldoc.format.pdf.document_il.il_version_1 import Box
from babeldoc.format.pdf.document_il.il_version_1 import Page
from babeldoc.format.pdf.document_il.il_version_1 import PdfCharacter

FIGURE_LABELS = frozenset({"figure", "image"})
TRANSLATABLE_LABELS = frozenset(
    {
        "figure_caption",
        "table_caption",
        "caption",
        "caption_hybrid",
        "table",
        "table_text",
        "table_cell",
        "table_cell_hybrid",
        "wired_table_cell",
        "wireless_table_cell",
        "table_footnote",
        "table_title",
    }
)


def figure_text_regions(page: Page) -> tuple[list[Box], list[Box]]:
    """Return figure bodies and text areas that must still be translated."""
    figures = [
        layout.box
        for layout in page.page_layout
        if layout.class_name in FIGURE_LABELS and layout.box is not None
    ]
    translatable = [
        layout.box
        for layout in page.page_layout
        if layout.class_name in TRANSLATABLE_LABELS and layout.box is not None
    ]
    return figures, translatable


def _contains(box: Box, x: float, y: float) -> bool:
    return box.x <= x <= box.x2 and box.y <= y <= box.y2


def is_figure_text(
    char: PdfCharacter, figures: list[Box], translatable: list[Box]
) -> bool:
    """Use character centers so a figure border does not capture nearby prose."""
    box = char.visual_bbox.box
    x, y = (box.x + box.x2) / 2, (box.y + box.y2) / 2
    return any(_contains(region, x, y) for region in figures) and not any(
        _contains(region, x, y) for region in translatable
    )
