import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pymupdf import Document

import babeldoc.format.pdf.document_il.utils.extract_char
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.style_helper import GREEN
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class LayoutParser:
    stage_name = "Parse Page Layout"

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.model = translation_config.doc_layout_model

    def _save_debug_image(self, image: np.ndarray, layout, page_number: int):
        """Save debug image with drawn boxes if debug mode is enabled."""
        if not self.translation_config.debug:
            return

        debug_dir = Path(self.translation_config.get_working_file_path("ocr-box-image"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Draw boxes on the image
        debug_image = image.copy()
        for box in layout.boxes:
            x0, y0, x1, y1 = box.xyxy
            cv2.rectangle(
                debug_image,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                (0, 255, 0),
                2,
            )
            # Add text label
            cv2.putText(
                debug_image,
                layout.names[box.cls],
                (int(x0), int(y0) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        img_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)

        # Save the image
        output_path = debug_dir / f"{page_number}.jpg"
        cv2.imwrite(str(output_path), img_bgr)

    def _save_debug_box_to_page(self, page: il_version_1.Page):
        """Save debug boxes and text labels to the PDF page."""
        if not self.translation_config.debug:
            return

        color = GREEN

        for layout in page.page_layout:
            # Create a rectangle box
            scale_factor = 1
            if layout.class_name == "fallback_line":
                scale_factor = 0.1
            rect = il_version_1.PdfRectangle(
                box=il_version_1.Box(
                    x=layout.box.x,
                    y=layout.box.y,
                    x2=layout.box.x2,
                    y2=layout.box.y2,
                ),
                graphic_state=color,
                debug_info=True,
                line_width=0.4 * scale_factor,
            )
            page.pdf_rectangle.append(rect)

            # Create text label at top-left corner
            # Note: PDF coordinates are from bottom-left,
            # so we use y2 for top position
            style = il_version_1.PdfStyle(
                font_id="base",
                font_size=4 * scale_factor,
                graphic_state=color,
            )
            page.pdf_paragraph.append(
                il_version_1.PdfParagraph(
                    first_line_indent=False,
                    box=il_version_1.Box(
                        x=layout.box.x,
                        y=layout.box.y2,
                        x2=layout.box.x2,
                        y2=layout.box.y2 + 5,
                    ),
                    vertical=False,
                    pdf_style=style,
                    unicode=layout.class_name,
                    pdf_paragraph_composition=[
                        il_version_1.PdfParagraphComposition(
                            pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                                unicode=layout.class_name,
                                pdf_style=style,
                                debug_info=True,
                            ),
                        ),
                    ],
                    xobj_id=-1,
                ),
            )

    def process(self, docs: il_version_1.Document, mupdf_doc: Document):
        """Generate layouts for all pages that need to be translated."""
        # Get pages that need to be translated
        total = len(docs.page)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total * 2,
        ) as progress:
            # Process predictions for each page
            for page, layouts in self.model.handle_document(
                docs.page,
                mupdf_doc,
                self.translation_config,
                self._save_debug_image,
            ):
                page_layouts = []
                for layout in layouts.boxes:
                    # Convert coordinate system from picture to il
                    # system to the il coordinate system
                    x0, y0, x1, y1 = layout.xyxy
                    # pix = get_no_rotation_img(mupdf_doc[page.page_number])
                    # pix = mupdf_doc[page.page_number].get_pixmap()
                    # h, w = pix.height, pix.width
                    box = mupdf_doc[page.page_number].mediabox_size
                    b_h = math.ceil(box.y)
                    b_w = math.ceil(box.x)
                    # if b_h != h or b_w != w:
                    #     logger.warning(f"page {page.page_number} mediabox is not correct, b_h: {b_h}, h: {h}, b_w: {b_w}, w: {w}")
                    h, w = b_h, b_w
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    page_layout = il_version_1.PageLayout(
                        id=len(page_layouts) + 1,
                        box=il_version_1.Box(
                            x0.item(),
                            y0.item(),
                            x1.item(),
                            y1.item(),
                        ),
                        conf=layout.conf.item(),
                        class_name=layouts.names[layout.cls],
                    )
                    page_layouts.append(page_layout)

                page.page_layout = page_layouts
                self._rebuild_toc_layout_as_lines(page, mupdf_doc)
                # self.generate_fallback_line_layout_for_page(page)
                # self._save_debug_box_to_page(page)
                progress.advance(1)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for page in docs.page:
                    executor.submit(
                        self.generate_fallback_line_layout_for_page, page, progress
                    )
        return docs

    # 目录页（TOC）的排版结构性太强：每条目一行、点号引导线、右对齐页码。
    # DocLayout-YOLO 会把多条条目并成一个大文本块，且框常盖不满整行，
    # 导致同一行被劈进不同 layout（"2 Archite" | "cture"）。
    # 这里在目录页上用行级 layout 替换文本类 layout，绕开版面模型。
    # PageLayout 用 class_name（Layout 用 name，两者不通用，不能复用 is_text_layout）
    TEXT_CLASS_NAMES = frozenset({
        "plain text", "tiny text", "title", "abandon",
        "figure_caption", "table_caption", "table_text", "table_footnote",
        "paragraph_title", "abstract", "content", "figure_title",
        "table_title", "doc_title", "footnote", "header",
    })

    TOC_TITLE_PATTERN = re.compile(
        r"^\s*(contents|table\s+of\s+contents|目\s*录)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    # 至少这么多行带点号引导线才认为是目录
    TOC_MIN_DOT_LEADER_LINES = 3
    TOC_DOT_LEADER_PATTERN = re.compile(r"\.\s*\.\s*\.")

    def _is_toc_page(self, page: il_version_1.Page, mupdf_doc: Document) -> bool:
        """判断是否目录页：有 Contents/目录 标题，或多行带点号引导线。"""
        try:
            text = mupdf_doc[page.page_number].get_text("text")
        except Exception:
            return False
        if not text:
            return False

        head = "\n".join(text.splitlines()[:5])
        if self.TOC_TITLE_PATTERN.search(head):
            return True

        dot_lines = sum(
            1 for line in text.splitlines()
            if self.TOC_DOT_LEADER_PATTERN.search(line)
        )
        return dot_lines >= self.TOC_MIN_DOT_LEADER_LINES

    def _rebuild_toc_layout_as_lines(
        self, page: il_version_1.Page, mupdf_doc: Document
    ) -> None:
        """目录页：把文本类 layout 换成行级 layout，非文本类（图表等）保留。"""
        if not self._is_toc_page(page, mupdf_doc):
            return

        char_boxes = babeldoc.format.pdf.document_il.utils.extract_char.convert_page_to_char_boxes(
            page
        )
        if not char_boxes:
            return

        try:
            clusters = babeldoc.format.pdf.document_il.utils.extract_char.process_page_chars_to_lines(
                char_boxes
            )
        except Exception as e:
            logger.warning(
                "TOC line clustering failed on page %s: %s", page.page_number, e
            )
            return
        if not clusters:
            return

        text_layouts = [
            layout
            for layout in page.page_layout
            if layout.class_name in self.TEXT_CLASS_NAMES
        ]
        kept = [
            layout
            for layout in page.page_layout
            if layout.class_name not in self.TEXT_CLASS_NAMES
        ]

        line_layouts = []
        for cluster in clusters:
            boxes = [c[0] for c in cluster.chars]
            if not boxes:
                continue
            box = il_version_1.Box(
                min(b.x for b in boxes),
                min(b.y for b in boxes),
                max(b.x2 for b in boxes),
                max(b.y2 for b in boxes),
            )
            line_layouts.append(
                il_version_1.PageLayout(
                    id=0,  # 稍后统一编号
                    box=box,
                    conf=1.0,
                    class_name=self._dominant_class_name(box, text_layouts),
                )
            )

        if not line_layouts:
            return

        for idx, layout in enumerate(kept + line_layouts, start=1):
            layout.id = idx
        page.page_layout = kept + line_layouts

        logger.info(
            "TOC page %s: replaced %d text layouts with %d line layouts",
            page.page_number,
            len(text_layouts),
            len(line_layouts),
        )

    @staticmethod
    def _dominant_class_name(
        box: il_version_1.Box, text_layouts: list[il_version_1.PageLayout]
    ) -> str:
        """取与该行重叠面积最大的原 layout 的类别，没有则退回 plain text。"""
        best_name = "plain text"
        best_area = 0.0
        for layout in text_layouts:
            lb = layout.box
            w = min(box.x2, lb.x2) - max(box.x, lb.x)
            h = min(box.y2, lb.y2) - max(box.y, lb.y)
            if w <= 0 or h <= 0:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best_name = layout.class_name
        return best_name

    def generate_fallback_line_layout_for_page(self, page: il_version_1.Page, progress):
        try:
            exists_page_layouts = page.page_layout
            char_boxes = babeldoc.format.pdf.document_il.utils.extract_char.convert_page_to_char_boxes(
                page
            )
            if not char_boxes:
                return

            clusters = babeldoc.format.pdf.document_il.utils.extract_char.process_page_chars_to_lines(
                char_boxes
            )
            for cluster in clusters:
                boxes = [c[0] for c in cluster.chars]
                min_x = min(b.x for b in boxes)
                max_x = max(b.x2 for b in boxes)
                min_y = min(b.y for b in boxes)
                max_y = max(b.y2 for b in boxes)
                cluster.chars = il_version_1.Box(min_x, min_y, max_x, max_y)
                page_layout = il_version_1.PageLayout(
                    id=len(exists_page_layouts) + 1,
                    box=il_version_1.Box(
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                    ),
                    conf=1,
                    class_name="fallback_line",
                )
                exists_page_layouts.append(page_layout)
            self._save_debug_box_to_page(page)
        finally:
            progress.advance(1)
