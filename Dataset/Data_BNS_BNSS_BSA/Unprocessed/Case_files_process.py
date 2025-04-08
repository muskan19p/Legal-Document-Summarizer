import os
import re
import json
import fitz  # PyMuPDF

# Updated regex patterns with capturing groups for an optional section number.
patterns = {
    "BNS": re.compile(
        r'[\(\[]?(?:Offence\s+under\s+section\s+(\d+)\s+|as\s+per\s+provisions\s+in\s+)?'
        r'((?:BNS)|(?:B\.N\.S\.)|(?:Bhartiya Nyaya Sanhita)|(?:Bharatiya Nyaya Sanhita)|(?:BHARTIYA NYAYA SANHITA)|(?:BHARATIYA NYAYA SANHITA))'
        r'[\)\]]?',
        0  # case sensitive
    ),
    "BNSS": re.compile(
        r'[\(\[]?(?:Offence\s+under\s+section\s+(\d+)\s+|as\s+per\s+provisions\s+in\s+)?'
        r'((?:BNSS)|(?:B\.N\.S\.S\.)|(?:Bhartiya Nagrik Suraksha Sanhita)|(?:Bharatiya Nagrik Suraksha Sanhita)|(?:BHARTIYA NAGRIK SURAKSHA SANHITA)|(?:BHARATIYA NAGRIK SURAKSHA SANHITA))'
        r'[\)\]]?',
        0
    ),
    "BSA": re.compile(
        r'[\(\[]?(?:Offence\s+under\s+section\s+(\d+)\s+|as\s+per\s+provisions\s+in\s+)?'
        r'((?:BSA)|(?:B\.S\.A\.)|(?:Bhartiya Sakshya Adhiniyam)|(?:Bharatiya Sakshya Adhiniyam)|(?:BHARTIYA SAKSHYA ADHINIYAM)|(?:BHARATIYA SAKSHYA ADHINIYAM))'
        r'[\)\]]?',
        0
    )
}

def clean_text(text):
    """
    Remove URLs and any mention of 'indiankanoon' or 'Indian Kanoon' (case insensitive),
    remove isolated page numbers, and collapse extra whitespace.
    """
    text = re.sub(r'https?://\S*indiankanoon\S*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bindiankanoon\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bIndian\s+Kanoon\b', '', text, flags=re.IGNORECASE)
    # Remove lines that are only numbers (often page numbers)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    # Replace newlines with spaces and collapse multiple spaces.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return clean_text(text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def extract_metadata(text):
    """
    Extract minimal metadata from the text.
    Searches for a date in common formats (including dd Month, yyyy)
    and looks for a mention of a court.
    """
    metadata = {}
    date_regex = (
        r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|'
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b)'
    )
    date_match = re.search(date_regex, text)
    if date_match:
        metadata['judgment_date'] = date_match.group(0)
    court_match = re.search(r'\b(?:Supreme Court|High Court|District Court)[^,\n]*', text)
    if court_match:
        metadata['court'] = court_match.group(0)
    return metadata

def extract_case_id(text, default_id):
    """
    Extract a case id from the text using regex.
    It attempts to match patterns like:
      - "PETITION No. - 6601 of 2024"
      - "BA1 No.2007 of 2024"
      - "No. 6601 of 2024"
    The prefix is optional.
    Iterates over all matches and ignores any match where the preceding text (last 50 characters)
    ends with "Citation" (optionally followed by a colon and spaces).
    """
    pattern = re.compile(
        r'\b((?:[A-Z0-9]+(?:\s*[/-]\s*[A-Z0-9]+)?\s+))?No\.?\s*-?\s*(\d+)(?:\s+of\s+(\d{4}))?\b',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        start = match.start()
        preceding_text = text[max(0, start-50):start]
        if re.search(r'Citation\s*:?\s*$', preceding_text, re.IGNORECASE):
            continue
        prefix = match.group(1) or ""
        number = match.group(2)
        year = match.group(3) if match.group(3) else ""
        prefix = prefix.strip() if prefix else ""
        if prefix:
            case_id = f"{prefix} No. - {number}"
        else:
            case_id = f"No. - {number}"
        if year:
            case_id += f" of {year}"
        return case_id
    return default_id

def find_statute_mentions(text):
    """
    Split the text into sentences and search each sentence for mentions of the target statutes
    using regex-only matching.
    The context field is the entire cleaned sentence in which the section(s) are mentioned.
    Also uses an additional regex to extract a detailed section reference, capturing multiple
    section numbers (e.g., "221,132, 352 and 351(3)").
    """
    mentions = []
    section_pattern = re.compile(
        r'\b(?:Section(?:s)?|Sec\.?)\s*([0-9]+(?:\([^)]+\))?(?:\s*,\s*[0-9]+(?:\([^)]+\))?)*(?:\s*(?:and)\s*[0-9]+(?:\([^)]+\))?)?)'
    )
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        for statute, pattern in patterns.items():
            match = pattern.search(sentence)
            if match:
                sec_match = section_pattern.search(sentence)
                section = ""
                if sec_match:
                    section = "Sections " + sec_match.group(1)
                mentions.append({
                    "statute": statute,
                    "section": section,
                    "context": sentence.strip()
                })
    return mentions

def extract_case_title_from_pdf(pdf_path, filename):
    """
    Extract the case title from the beginning of the PDF document.
    It reads the first few pages, then extracts text until a date is encountered.
    If a match is found, returns that text; otherwise, falls back to the given filename.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Read the first 3 pages (adjust as needed)
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            text += page.get_text() + " "
        doc.close()
        text = ' '.join(text.split())
        # Regex pattern: capture text from the start until a date is found.
        title_pattern = re.compile(
            r'^(.*?)(?=\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b)',
            re.IGNORECASE
        )
        match = title_pattern.search(text)
        if match:
            return match.group(1).strip()
        else:
            return filename
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return filename

def process_pdfs(directory):
    """Process PDF files in the directory and extract case objects if a target statute is mentioned."""
    cases = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                if not text:
                    continue
                metadata = extract_metadata(text)
                statute_mentions = find_statute_mentions(text)
                if statute_mentions:
                    case_id = extract_case_id(text, os.path.splitext(file)[0])
                    # First, derive a title from the file name by removing the extension.
                    filename = os.path.splitext(file)[0]
                    # Remove any trailing suffix starting with "_on"
                    filename = re.sub(r'_on.*$', '', filename)
                    # Remove any trailing "_BNS"
                    filename = filename.replace("_BNS", "").strip()
                    # Optionally, compare with content-based extraction:
                    content_title = extract_case_title_from_pdf(pdf_path, filename)
                    # Choose the longer title (assuming it's more complete)
                    case_title = content_title if len(content_title) > len(filename) else filename
                    case_obj = {
                        "case_id": case_id,
                        "case_title": case_title,
                        "judgment_date": metadata.get("judgment_date", ""),
                        "statute_mentions": statute_mentions
                    }
                    cases.append(case_obj)
    return cases

if __name__ == "__main__":
    input_output = {
        r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BNS" : r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Case_Files\BNS_cases.json",
        r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BNSS" : r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Case_Files\BNSS_cases.json",
        r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BSA" : r"d:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Case_Files\BSA_cases.json"
    }

    for input_directory, output_json in input_output.items():
        cases = process_pdfs(input_directory)
        print(f"Processed {len(cases)} case files with target statute mentions.")
    
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"Saved extracted data to {output_json}")
