"""Partial translated PDFs are merged back into their original document."""

from types import SimpleNamespace

from babeldoc.format.pdf import high_level


def test_translated_page_only_skips_original_bookmarks(monkeypatch):
    config = SimpleNamespace(
        use_alternating_pages_dual=False,
        only_include_translated_page=True,
    )

    def fail_if_opened(_path):
        raise AssertionError("Partial output must not copy full-document bookmarks")

    monkeypatch.setattr(high_level, "Document", fail_if_opened)

    high_level.migrate_toc(config, SimpleNamespace())
