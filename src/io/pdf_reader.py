"""
src.io.pdf_reader
====================
Reader for .pdf files. Extracts the embedded text layer only - a
scanned/image-only PDF with no text layer yields an empty string per
page, same as a corrupted or unreadable file would; it is never sent
through OCR here (see language_filter's MIN_WORDS_FOR_LANGUAGE_CHECK
guard, which keeps a near-empty document from being misjudged either
way).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from pypdf import PdfReader as _PypdfReader

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.utils.logger import get_logger

logger = get_logger("io.pdf_reader")

# pypdf logs a WARNING per malformed/unusual font it can't fully parse
# (e.g. "fontTools is required to fully parse...") - harmless for text
# extraction (it falls back to a partial mapping) but extremely noisy
# on real-world PDFs with embedded custom fonts, at real-world case
# volume. Text extraction quality is unaffected; only log verbosity is.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# How many blocked compiled modules one read may replace before giving up.
_MAX_SOURCE_FALLBACKS = 8


def _load_python_source_instead(exc: ImportError) -> bool:
    """Loads a module from its .py source when its compiled copy can't be
    loaded. pypdf reads CFF fonts through fontTools, which ships a few
    modules both compiled (bezierTools.cp312-win_amd64.pyd) and as source
    (bezierTools.py); Python picks the compiled one. On 2026-10-07 Windows
    Smart App Control blocked that file ("An Application Control policy has
    blocked this file"), so every PDF with such a font failed to read at
    all - "10th social em.pdf", 346 pages, came back with no text. Returns
    False when exc is anything else (a package that isn't installed, a
    module with no source beside it)."""
    blocked = getattr(exc, "path", None)
    if not blocked or isinstance(exc, ModuleNotFoundError):
        return False
    blocked = Path(blocked)
    short_name = blocked.name.split(".")[0]
    source = blocked.with_name(short_name + ".py")
    if blocked.suffix.lower() not in (".pyd", ".so") or not source.exists():
        return False
    # The error names only "bezierTools"; its package is the imported one
    # whose folder holds the file.
    parent_name = next((n for n, m in list(sys.modules.items())
                        if any(Path(p) == blocked.parent for p in getattr(m, "__path__", None) or ())), "")
    name = f"{parent_name}.{short_name}" if parent_name else short_name
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    if parent_name:
        setattr(sys.modules[parent_name], short_name, module)
    logger.warning("%s could not be loaded (%s) - using its Python source %s instead", blocked.name, exc, source.name)
    return True


class PdfReader(BaseReader):
    supported_extensions = ("pdf",)

    def _read(self, path: str) -> Document:
        fallbacks = 0
        while True:
            try:
                pages = self._extract_pages(path)
                break
            except ImportError as exc:
                fallbacks += 1
                if fallbacks > _MAX_SOURCE_FALLBACKS or not _load_python_source_instead(exc):
                    raise
        full_text = "\n\n".join(p.text for p in pages)
        return Document(
            source_path=path,
            source_format=SourceFormat.PDF,
            pages=pages,
            full_text=full_text,
            metadata={"page_count": len(pages)},
        )

    @staticmethod
    def _extract_pages(path: str) -> tuple[Page, ...]:
        reader = _PypdfReader(path)
        return tuple(
            Page(index=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        )
