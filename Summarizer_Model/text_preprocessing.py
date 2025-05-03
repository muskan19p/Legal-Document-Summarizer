import os
import re
import pdfplumber
import docx
from pathlib import Path

def load_document(path: str) -> str:
    """
    Detect file extension and route to the appropriate parser.
    Supports: .pdf, .docx, .doc, .txt
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext in {".docx"}:
        return parse_docx(path)
    elif ext == ".doc":
        # Convert .doc to .docx via LibreOffice, then parse
        converted_path = convert_doc_to_docx(path)
        return parse_docx(converted_path)
    elif ext == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported format: {ext}")


def convert_doc_to_docx(path: str) -> str:
    """
    Convert legacy .doc to .docx using libreoffice command-line.
    Requires LibreOffice installed and in PATH.
    Returns path to converted .docx file.
    """
    new_path = Path(path).with_suffix('.docx')
    cmd = f"libreoffice --headless --convert-to docx --outdir {new_path.parent} {path}"
    os.system(cmd)
    return str(new_path)


def parse_pdf(path: str) -> str:
    """
    Extract text from PDF using pdfplumber, preserving layout.
    """
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text_chunks.append(text)
    return "\n".join(text_chunks)


def parse_docx(path: str) -> str:
    """
    Extract text from .docx Word document using python-docx.
    """
    doc = docx.Document(path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)

def clean_text(raw: str) -> str:
    """
    Clean raw text by normalizing whitespace, fixing hyphenation,
    removing control characters, and preserving legal numbering.
    """
    text = raw

    # 1. Normalize Unicode
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # 2. Merge hyphenated line-breaks: 'obliga-\ntion' -> 'obligation'
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 3. Remove stray control characters
    text = re.sub(r"[\r\t\x0b\x0c]", " ", text)

    # 4. Collapse multiple newlines (>2 -> 2)
    text = re.sub(r" *\n{3,} *", "\n\n", text)

    # 5. Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 6. Trim spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # 7. Preserve legal numbering at line starts
    text = re.sub(r"\n\s*(\d+(?:\.\d+)*\.)", r"\n\1", text)

    # 8. Remove common header/footer patterns (e.g., page numbers)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove lines that are just page numbers
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # Remove lines that are common headers/footers
        if re.match(r'^\s*(Page|PAGE)\s*\d+\s*$', line):
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # 9. Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # 10. Remove hyperlinks
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    return text.strip()


def preprocess_document(path: str) -> str:
    """
    End-to-end preprocessing: load, extract, clean.
    """
    raw = load_document(path)
    clean = clean_text(raw)
    return clean
