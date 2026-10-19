"""PDF-backed local retrieval corpus. No fallback to authoring text or cloud calls."""

from functools import lru_cache
import hashlib
from pathlib import Path
import re
from threading import RLock

_PDF_LOCK = RLock()

DIRECTORY = Path(__file__).with_name("documents")
FILES = {"technical": "Festo-381-Technical-Reference.pdf",
         "maintenance": "Festo-381-Maintenance-Scenario.pdf",
         "production": "Phone-Housing-Production-Playbook.pdf",
         "fault-service": "Festo-Pneumatic-Fault-Service-Guide.pdf"}


def document_path(document_id):
    if document_id not in FILES:
        raise ValueError("Unknown document")
    path = DIRECTORY / FILES[document_id]
    if not path.is_file():
        raise FileNotFoundError("Demo PDFs missing; run scripts/build-festo-knowledge.py")
    return path


@lru_cache(maxsize=12)
def _extract(document_id, data):
    # PDFium is not thread-safe; concurrent HTTP retrievals share this lock.
    with _PDF_LOCK:
        return _extract_serial(document_id, data)


def _extract_serial(document_id, data):
    import pypdfium2 as pdfium

    rows = []
    with pdfium.PdfDocument(data) as document:
        for number in range(len(document)):
            page = document[number]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range().replace("\r\n", "\n")
            finally:
                textpage.close()
                page.close()
            def field(name):
                match = re.search(rf"^{name}: (.+)$", text, re.MULTILINE)
                if not match:
                    raise ValueError(f"Invalid PDF metadata: {FILES[document_id]} page {number + 1}")
                return match.group(1).strip()
            content = text.split("CONTENT\n", 1)[1].split("END CONTENT", 1)[0]
            rows.append({"id": field("CHUNK_ID"), "title": field("TITLE"),
                         "origin": field("ORIGIN"), "source": field("SOURCE"),
                         "text": " ".join(content.split()), "document_id": document_id,
                         "filename": FILES[document_id], "page": number + 1,
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "citation": f"{FILES[document_id]}, p. {number + 1}"})
    return rows


def chunks():
    result = []
    for identifier in FILES:
        result.extend(_extract(identifier, document_path(identifier).read_bytes()))
    return result


def catalog():
    rows = chunks()
    return [{"id": identifier, "filename": filename,
             "pages": len([r for r in rows if r["document_id"] == identifier]),
             "sha256": next(r["sha256"] for r in rows if r["document_id"] == identifier)}
            for identifier, filename in FILES.items()]
