"""
src.io.xml_reader
===================
Reader for .xml (Android shared_prefs/contacts exports, app data, office
XML). Every element's text becomes a line, and so does every attribute
value that contains a letter, as "attribute: value" - names often live in
attributes (<contact name="Andre Silva" .../>), not element text.

Malformed XML (common in carved/partial forensic exports) falls back to
the lenient HTML text extractor instead of failing, so its text is still
read. ElementTree does not resolve external entities, so parsing an
untrusted export can't reach outside the file.

A Cellebrite UFED report (report.xml, 2026-10-12) is read record by record
instead: one line per decoded record - a contact, a call, a chat - holding
the values of its fields, as a CSV row would. Read like any other XML, a
66 MB report became a line for every id, state and hash attribute of its
18,000 file nodes and 11,000 records, took 29 of a case scan's 44 minutes
and was left out of scans for that - and the phone's address book, which
exists nowhere else in the extraction, went with it: four of a case's
sixteen people stood only there. The report's listing of the extraction's
files is not read; those files are scanned themselves.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

from src.core.models import Document, Page, SourceFormat
from src.io.base_reader import BaseReader
from src.io.binary_check import binary_document, binary_format
from src.io.markup_text import CELL_SEPARATOR, html_to_text
from src.io.text_reader import TextReader
from src.utils.logger import get_logger

logger = get_logger("io.xml_reader")

_UFED_REPORT_NAMESPACE = b"pa.cellebrite.com/report"


class XmlReader(BaseReader):
    supported_extensions = ("xml",)

    def _read(self, path: str) -> Document:
        raw = Path(path).read_bytes()
        binary = binary_format(raw)
        if binary:
            return binary_document(path, SourceFormat.XML, binary)
        location_note = "line N of the extracted text/attribute lines"
        try:
            if _UFED_REPORT_NAMESPACE in raw[:4096]:
                full_text = "\n".join(ufed_record_lines(raw))
                location_note = "line N = decoded record N of the UFED report"
            else:
                full_text = "\n".join(xml_to_lines(ET.fromstring(raw)))
        except ET.ParseError as exc:
            logger.warning("%s is not well-formed XML (%s) - extracting its text leniently", path, exc)
            full_text = html_to_text(TextReader._decode(raw))
        return Document(
            source_path=path, source_format=SourceFormat.XML,
            pages=(Page(index=0, text=full_text),), full_text=full_text,
            metadata={"location_note": location_note},
        )


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]  # drop "{namespace}"


def xml_to_lines(root: ET.Element) -> list[str]:
    lines: list[str] = []

    def add(text: str | None, label: str = "") -> None:
        text = " ".join((text or "").split())
        if text and any(ch.isalpha() for ch in text):
            lines.append(f"{label}: {text}" if label else text)

    for element in root.iter():
        for attr, value in element.attrib.items():
            add(value, _local(attr))
        add(element.text)
        add(element.tail)
    return lines


def ufed_record_lines(raw: bytes) -> list[str]:
    """One line per decoded record (a <model> directly under a <modelType>)
    of a UFED report: the text values of its fields and of the records
    nested in it - a contact's numbers, a chat's messages - joined by
    CELL_SEPARATOR. Only values the report types as String: its flags,
    timestamps, numbers and option names ("Decoding", "True", "Unread") are
    no text. A value repeated within a record is written once."""
    lines: list[str] = []
    open_elements: list[ET.Element] = []
    for event, element in ET.iterparse(io.BytesIO(raw), events=("start", "end")):
        if event == "start":
            open_elements.append(element)
            continue
        open_elements.pop()
        tag = _local(element.tag)
        if tag == "file":
            element.clear()
        elif tag == "model" and open_elements and _local(open_elements[-1].tag) == "modelType":
            cells: list[str] = []
            for value in element.iter():
                if _local(value.tag) == "value" and value.get("type", "String") == "String":
                    text = " ".join((value.text or "").split())
                    if text and text not in cells:
                        cells.append(text)
            if cells:
                lines.append(CELL_SEPARATOR.join(cells))
            element.clear()
    return lines
