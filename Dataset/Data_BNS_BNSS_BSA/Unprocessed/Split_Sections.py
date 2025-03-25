import json
import re
import sys

def split_content_into_subsections(content):
    """
    Splits the content into parts based on numeric markers.
    Assumes that a valid subsection starts with a numeric marker like (1), (2), etc.
    """
    content = content.strip()
    # Use regex lookahead to split at every occurrence of a numeric marker.
    parts = re.split(r'(?=\(\d+\))', content)
    # Remove empty parts and strip whitespace.
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def process_sections(input_json):
    """
    Processes a list of section entries.

    For each section, if the content (or section_text) contains numeric subsection markers,
    it splits the text into parts. If the very first part does not start with a marker and
    there is at least one marker later on, that first part is treated as a preamble.
    
    Then for the remaining parts, it creates a JSON object for each subsection. If a non‑consecutive
    marker is encountered, its text (with its marker) is appended to the previous subsection.

    If no numeric marker is detected at all, the section is output as is.
    
    Each new object includes:
      - section_number
      - statute
      - For sections with subsections: subsection_number and subsection_text.
      - For sections without markers: section_text.
    """
    output_objects = []
    
    for entry in input_json:
        section_number = entry.get("section_number")
        statute = entry.get("statute")
        # Try to get text from "content" first; if not, then "section_text"
        content = entry.get("content") or entry.get("section_text", "")
        content = content.strip()
        
        if not content:
            continue
        
        parts = split_content_into_subsections(content)
        
        # If no part starts with a marker at all, output the section as is.
        if not any(re.match(r'^\(\d+\)', part) for part in parts):
            output_objects.append({
                "section_number": section_number,
                "statute": statute,
                "section_text": content
            })
            continue
        
        # If the first part doesn't start with a marker and there is more than one part,
        # treat it as a preamble and remove it from the parts.
        if len(parts) > 1 and not re.match(r'^\(\d+\)', parts[0]):
            preamble = parts.pop(0)
            # Optionally, you can store the preamble in an output object.
            output_objects.append({
                "section_number": section_number,
                "statute": statute,
                "preamble": preamble
            })
        
        expected_number = 1
        current_obj = None
        
        for part in parts:
            # Attempt to extract the numeric marker and the text that follows.
            marker_match = re.match(r'(\(\d+\))\s*(.*)', part, re.DOTALL)
            if marker_match:
                marker = marker_match.group(1)
                text = marker_match.group(2).strip()
                try:
                    current_number = int(marker.strip("()"))
                except Exception as e:
                    print(f"Warning: Could not parse marker {marker} in section {section_number}. Skipping this part.")
                    continue

                if current_number == expected_number:
                    # Create a new JSON object for a new subsection.
                    current_obj = {
                        "section_number": section_number,
                        "statute": statute,
                        "subsection_number": marker,
                        "subsection_text": text
                    }
                    output_objects.append(current_obj)
                    expected_number += 1
                else:
                    # Non-consecutive marker: append its marker and text to the previous subsection.
                    if current_obj:
                        append_text = f" {marker} {text}"
                        current_obj["subsection_text"] += append_text
                    else:
                        # If no previous subsection exists, create one.
                        current_obj = {
                            "section_number": section_number,
                            "statute": statute,
                            "subsection_number": marker,
                            "subsection_text": text
                        }
                        output_objects.append(current_obj)
                        expected_number = current_number + 1
            else:
                # No valid marker found; append the text to the previous subsection.
                if current_obj:
                    current_obj["subsection_text"] += " " + part
                else:
                    print(f"Warning: No valid marker found in a part in section {section_number}. Skipping.")
                    continue

    return output_objects

def main():
    
    
    input_output = {
        r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BNS_sections.json" : r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Statutes\BNS_sections.json",
        r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BNSS_sections.json" : r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Statutes\BNSS_sections.json",
        r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Unprocessed\BSA_sections.json" : r"D:\Legal-Document-Summarizer\Dataset\Data_BNS_BNSS_BSA\Statutes\BSA_sections.json"
    }

    for input_filename, output_filename in input_output.items():
        try:
            with open(input_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sys.exit(f"Error reading {input_filename}: {e}")
    
        if not isinstance(data, list):
            sys.exit("Input JSON must be a list of section entries.")
    
        output_data = process_sections(data)
    
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
        print(f"Extracted {len(output_data)} objects. Output written to {output_filename}")

if __name__ == "__main__":
    main()
