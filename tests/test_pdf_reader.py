"""Unit tests for src.io.pdf_reader.PdfReader.

PdfReader is a heavily-used production input path (every real case scan
processes hundreds of real .pdf files - see output/scans/*.csv) that had
zero test coverage before this file (independent-audit finding #2).

Fixtures are built in-memory via pypdf.PdfWriter rather than checked-in
binary .pdf files, for two reasons: (1) a hand-written minimal PDF's xref
table is fragile and easy to get subtly wrong, where pypdf's own writer
guarantees a well-formed file its own reader can parse back; (2) it avoids
ever committing a real-world PDF (even an innocuous one) into a repo that
otherwise deliberately keeps real casework content out of version control
- see confirmed_labels.jsonl's same policy.

The "scanned/image-only PDF" and "corrupted PDF" behaviors below were also
verified directly against a real 98-page image-only PDF from an actual
case folder (confirmed success=True, full_text.strip() == "" - never
printed or copied, only its length was inspected) - the synthetic
fixtures here reproduce that same behavior for a reproducible, committed
regression test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.io.pdf_reader import PdfReader


def _write_pdf(path: Path, *, text: str | None) -> None:
    """Builds a minimal one-page PDF via pypdf's own writer. When `text`
    is given, the page gets a real content stream drawing that text with
    the standard (non-embedded) Helvetica font - a genuine text layer.
    When `text` is None, the page is left with no /Contents at all - the
    same shape a scanned/image-only PDF's page has (a valid page, zero
    extractable text)."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)

    if text is not None:
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
        stream_ref = writer._add_object(stream)

        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)

        resources = DictionaryObject()
        font_dict = DictionaryObject()
        font_dict[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = font_dict

        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = stream_ref

    with path.open("wb") as f:
        writer.write(f)


def test_pdf_with_real_text_layer_extracts_correctly(tmp_path):
    pdf_path = tmp_path / "with_text.pdf"
    _write_pdf(pdf_path, text="Suresh Kumar attended the meeting.")

    result = PdfReader().read(pdf_path)

    assert result.success
    assert result.document.full_text.strip() == "Suresh Kumar attended the meeting."
    assert result.document.metadata["page_count"] == 1


def test_scanned_image_only_pdf_returns_empty_text_not_a_crash(tmp_path):
    """A scanned/image-only PDF is a perfectly valid PDF with a page and
    no text layer - it must be read successfully with empty text, never
    raise. Distinguishing this from "genuinely read, found no names" is
    scan_directory.py's job (_pdf_has_extractable_text), not this
    reader's - the reader's only contract is: don't crash, report what's
    actually there (nothing)."""
    pdf_path = tmp_path / "scanned.pdf"
    _write_pdf(pdf_path, text=None)

    result = PdfReader().read(pdf_path)

    assert result.success
    assert result.document.full_text.strip() == ""
    assert result.document.metadata["page_count"] == 1


def test_corrupted_pdf_is_reported_as_a_failed_result_not_a_crash(tmp_path):
    """Matches src/io/base_reader.py's documented contract: read() never
    raises, converting any parse exception into a structured
    success=False DocumentResult - a single unreadable file must never
    abort a batch scan."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"this is not a valid pdf file at all" * 20)

    result = PdfReader().read(pdf_path)

    assert result.success is False
    assert result.document is None
    assert result.error


def test_missing_pdf_file_is_reported_as_a_failed_result():
    result = PdfReader().read("this_file_does_not_exist_at_all.pdf")

    assert result.success is False
    assert result.document is None
    assert "not found" in result.error.lower()


def test_pdf_reader_only_supports_pdf_extension():
    reader = PdfReader()
    assert reader.supports("file.pdf")
    assert reader.supports("FILE.PDF")
    assert not reader.supports("file.txt")


def _write_multipage_pdf(path: Path, pages: list[list[str]]) -> None:
    """Like _write_pdf, but several pages, each with several text lines
    (T* moves to the next line, which pypdf extracts as a newline)."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=300, height=300)
        ops = " T* ".join(f"({line}) Tj" for line in lines)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 14 TL 20 250 Td {ops} ET".encode("latin-1"))
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as f:
        writer.write(f)


def test_line_index_reports_page_and_line_within_page():
    from src.preprocessing.line_indexer import LineIndex

    text = "page one\n\nfirst\nsecond line\n\nx"
    index = LineIndex.build(text, page_start_offsets=[0, 10, 29])
    loc = index.locate(16, 22)  # "second" on page 2
    assert (loc.page_number, loc.line_number, loc.column_number) == (2, 2, 1)
    assert str(loc) == "page 2, line 2, col 1 (chars 16-22)"
    assert index.locate(29, 30).page_number == 3
    # Without pages: unchanged document-wide lines, no page in the text.
    plain = LineIndex.build(text).locate(16, 22)
    assert plain.page_number is None and str(plain) == "line 4, col 1 (chars 16-22)"


def test_pdf_mentions_are_located_by_page(tmp_path):
    """Pages used to be processed as one joined text and reported as a
    line of that join - a position no PDF viewer shows."""
    import copy

    from config import BASE_DIR, DEFAULT_CONFIG
    from src.pipeline.orchestrator import Pipeline

    pdf_path = tmp_path / "report.pdf"
    _write_multipage_pdf(pdf_path, [
        ["Quarterly report for the committee."],
        ["Summary of findings", "Dr. Suresh Kumar signed the report on Monday."],
        ["Appendix", "Notes", "Priya Raman confirmed the figures.", "Dr. Suresh Kumar agreed."],
    ])
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["feedback"]["enabled"] = False
    result = Pipeline(config=config, base_dir=BASE_DIR).run(pdf_path)

    found = {(p.display_text, m.candidate.location.page_number, m.candidate.location.line_number)
             for p in result.persons for m in p.mentions}
    assert found == {("Dr. Suresh Kumar", 2, 2), ("Dr. Suresh Kumar", 3, 4), ("Priya Raman", 3, 3)}


# --- A compiled helper module the operating system refuses to load ----------
# 2026-10-07: Windows Smart App Control blocked fontTools' compiled
# bezierTools, and every PDF with a CFF font failed to read. A file that is
# no real library gives the same kind of ImportError (name + path of the
# compiled file), so the fallback is tested without the policy.

def _blocked_module(tmp_path: Path, monkeypatch, package: str) -> ImportError:
    """A package whose `helper` module exists compiled (unloadable) and as
    source. Returns the ImportError importing it raises."""
    import importlib
    import importlib.machinery

    pkg = tmp_path / package
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helper.py").write_text("ORIGIN = 'python source'\n", encoding="utf-8")
    (pkg / f"helper{importlib.machinery.EXTENSION_SUFFIXES[0]}").write_bytes(b"not a real library")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        importlib.import_module(f"{package}.helper")
    except ImportError as exc:
        return exc
    raise AssertionError("the unloadable compiled module was imported")


def test_blocked_compiled_module_is_loaded_from_its_python_source(tmp_path, monkeypatch):
    import importlib

    from src.io.pdf_reader import _load_python_source_instead

    exc = _blocked_module(tmp_path, monkeypatch, "blockedpkg_a")
    assert _load_python_source_instead(exc) is True
    assert importlib.import_module("blockedpkg_a.helper").ORIGIN == "python source"


def test_a_package_that_is_not_installed_is_left_alone():
    from src.io.pdf_reader import _load_python_source_instead

    try:
        import a_package_that_does_not_exist_anywhere  # noqa: F401
    except ImportError as exc:
        assert _load_python_source_instead(exc) is False


def test_pdf_read_retries_after_a_blocked_compiled_module(tmp_path, monkeypatch):
    """The read that hit the blocked module is repeated once the Python
    source is in place - the file is read, not reported as failed."""
    exc = _blocked_module(tmp_path, monkeypatch, "blockedpkg_b")
    pdf_path = tmp_path / "with_text.pdf"
    _write_pdf(pdf_path, text="Suresh Kumar attended the meeting.")

    real_extract = PdfReader._extract_pages
    calls = []

    def blocked_once(path):
        calls.append(path)
        if len(calls) == 1:
            raise exc
        return real_extract(path)

    monkeypatch.setattr(PdfReader, "_extract_pages", staticmethod(blocked_once))
    result = PdfReader().read(pdf_path)

    assert result.success and len(calls) == 2
    assert result.document.full_text.strip() == "Suresh Kumar attended the meeting."
