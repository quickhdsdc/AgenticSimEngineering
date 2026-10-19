"""Reproducibly generate selectable-text demonstration PDFs from reviewed sources.

Small standard PDF writer, Helvetica text and exact cross-reference offsets.
No raster-only pages: the retrieval service extracts text from these PDFs.
Only the four allowlisted generated documents are written, not vendor originals.
"""

import textwrap

from a4s.machines.festo_cp_lab.knowledge_sources import DOCUMENTS, FAULT_DOCUMENTS
from a4s.machines.festo_cp_lab.pdf_knowledge import DIRECTORY, FILES, catalog


def pdf_string(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages):
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for row in pages:
        profile = ("Rev. 2 | 2026-09-24 | Festo runtime simulation"
                   if row["id"] in {r["id"] for r in FAULT_DOCUMENTS}
                   else "Rev. 1 | 2026-09-13 | FestoWorkCenterStamping_381")
        page_id = len(objects) + 1
        kids.append(f"{page_id} 0 R")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode())
        source = row["source"] if row["id"] != "recovery-boundaries" else "A4S MES action contracts / architecture tutorial"
        # Metadata is kept on single lines (smaller font) for deterministic
        # page/chunk provenance extraction. Content is real PDF text, not JSON.
        lines = ["A4S / ENGINEERING KNOWLEDGE", "DEMONSTRATION COPY - NOT AN APPROVED SITE SOP", "",
                 f"CHUNK_ID: {row['id']}", f"TITLE: {row['title']}", f"ORIGIN: {row['origin']}",
                 f"SOURCE: {source}", "", "CONTENT", *textwrap.wrap(row["text"], 88), "END CONTENT", "",
                 profile,
                 "Exhibition product, fault and numerical assumptions are not manufacturer specifications.",
                 "Physical isolation / repair remains a qualified operator responsibility.",
                 f"Page {len(kids)}"]
        commands = ["BT /F1 9 Tf 42 748 Td 15 TL"]
        for line in lines:
            commands += ["/F1 7 Tf" if line.startswith("SOURCE:") else "/F1 9 Tf",
                         f"({pdf_string(line)}) Tj", "T*"]
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Count {len(kids)} /Kids [{' '.join(kids)}] >>".encode()
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


def main():
    groups = {"technical": DOCUMENTS[:3], "maintenance": [DOCUMENTS[4], DOCUMENTS[5]],
              "production": [DOCUMENTS[3], DOCUMENTS[6]], "fault-service": FAULT_DOCUMENTS}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for identifier, rows in groups.items():
        (DIRECTORY / FILES[identifier]).write_bytes(make_pdf(rows))
    for document in catalog():
        print(f"{document['filename']}: {document['pages']} extracted pages, SHA256 {document['sha256'][:12]}")


if __name__ == "__main__":
    main()
