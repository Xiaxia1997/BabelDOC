from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.midend.layout_parser import LayoutParser
from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder


def _box(x: float, y: float, x2: float, y2: float) -> il_version_1.Box:
    return il_version_1.Box(x=x, y=y, x2=x2, y2=y2)


def _line(text: str, *, x: float, y: float) -> il_version_1.PdfLine:
    chars = []
    for index, value in enumerate(text):
        char_box = _box(x + index * 5, y, x + (index + 1) * 5, y + 10)
        chars.append(
            il_version_1.PdfCharacter(
                box=char_box,
                visual_bbox=il_version_1.VisualBbox(box=char_box),
                char_unicode=value,
                xobj_id=0,
            )
        )
    return il_version_1.PdfLine(
        box=_box(x, y, x + len(text) * 5, y + 10),
        pdf_character=chars,
    )


def _char(value: str, box: il_version_1.Box) -> il_version_1.PdfCharacter:
    return il_version_1.PdfCharacter(
        box=box,
        visual_bbox=il_version_1.VisualBbox(box=box),
        char_unicode=value,
        xobj_id=0,
    )


def _paragraph(*lines: il_version_1.PdfLine) -> il_version_1.PdfParagraph:
    return il_version_1.PdfParagraph(
        pdf_paragraph_composition=[
            il_version_1.PdfParagraphComposition(pdf_line=line) for line in lines
        ],
        layout_id=1,
        layout_label="plain text",
        debug_id="source",
    )


def _curve(box: il_version_1.Box, *, filled: bool = True) -> il_version_1.PdfCurve:
    return il_version_1.PdfCurve(box=box, fill_background=filled)


def test_contained_title_layouts_become_one_text_region():
    page = il_version_1.Page(
        page_layout=[
            il_version_1.PageLayout(
                id=1, conf=0.4, class_name="title", box=_box(10, 20, 110, 40)
            ),
            il_version_1.PageLayout(
                id=2, conf=0.9, class_name="title", box=_box(30, 21, 105, 39)
            ),
            il_version_1.PageLayout(
                id=3,
                conf=0.9,
                class_name="plain text",
                box=_box(10, 50, 110, 80),
            ),
        ]
    )

    LayoutParser._merge_contained_text_layouts(page)

    assert len(page.page_layout) == 2
    title = next(layout for layout in page.page_layout if layout.class_name == "title")
    assert title.box == _box(10, 20, 110, 40)


def test_vertical_orphans_expand_neighboring_text_layout():
    layout = il_version_1.PageLayout(
        id=1,
        conf=0.9,
        class_name="title",
        box=_box(10, 20, 110, 30),
    )
    regular = _char("A", _box(20, 21, 25, 29))
    above = _char("ʼ", _box(30, 31, 33, 34))
    below = _char(",", _box(40, 16, 43, 19))
    page = il_version_1.Page(
        page_layout=[layout],
        pdf_character=[regular, above, below],
    )
    parser = LayoutParser.__new__(LayoutParser)

    parser._absorb_orphan_chars_into_text_layouts(page)

    assert layout.box == _box(10, 16, 110, 34)


def test_graphic_bullets_split_a_shared_layout_into_separate_paragraphs():
    paragraph = _paragraph(
        _line("First item", x=30, y=100),
        _line("Second item", x=30, y=80),
        _line("Third item", x=30, y=60),
    )
    page = il_version_1.Page(
        pdf_curve=[
            _curve(_box(20, 103, 23, 106)),
            _curve(_box(20, 83, 23, 86)),
            _curve(_box(20, 63, 23, 66)),
        ]
    )
    finder = ParagraphFinder.__new__(ParagraphFinder)

    paragraphs = finder._split_graphic_bullet_paragraphs(page, [paragraph])

    assert [item.unicode for item in paragraphs] == [
        "First item",
        "Second item",
        "Third item",
    ]


def test_inline_text_background_is_removed_but_bullet_and_banner_are_kept():
    paragraph = _paragraph(_line("Inline code", x=30, y=100))
    inline_background = _curve(_box(45, 99, 80, 111))
    bullet = _curve(_box(20, 103, 23, 106))
    banner = _curve(_box(10, 85, 120, 125))
    page = il_version_1.Page(
        pdf_paragraph=[paragraph],
        pdf_curve=[inline_background, bullet, banner],
    )
    finder = ParagraphFinder.__new__(ParagraphFinder)

    finder._remove_inline_text_backgrounds(page)

    assert page.pdf_curve == [bullet, banner]
