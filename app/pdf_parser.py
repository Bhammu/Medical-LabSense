"""
pdf_parser.py
Extracts lab values from uploaded PDF or image lab reports.
Uses pdfplumber for PDFs and pytesseract for images.
"""

import re
import pdfplumber
import pandas as pd
from PIL import Image
import io

# ── Patterns to search for in report text ────────────────────────────────────
# Each key maps to the feature name used by the model
# Values are (regex_pattern, unit_hint)

LAB_PATTERNS = {
    "WBC":    (r"WBC|White Blood Cell|Leukocyte",          r"[\d.]+(?:\s*×10[³3])?"),
    "RBC":    (r"RBC|Red Blood Cell|Erythrocyte",          r"[\d.]+"),
    "HGB":    (r"HGB|HB|Hemoglobin|Haemoglobin",          r"[\d.]+"),
    "HCT":    (r"HCT|Hematocrit|Haematocrit|PCV",         r"[\d.]+"),
    "MCV":    (r"MCV|Mean Corpuscular Volume",             r"[\d.]+"),
    "MCH":    (r"MCH|Mean Corpuscular Hemoglobin",         r"[\d.]+"),
    "MCHC":   (r"MCHC|Mean Corpuscular Hemoglobin Conc",  r"[\d.]+"),
    "RDW":    (r"RDW|Red Cell Distribution Width",        r"[\d.]+"),
    "PLT":    (r"PLT|Platelets?|Thrombocytes?",            r"[\d.]+"),
    "MPV":    (r"MPV|Mean Platelet Volume",                r"[\d.]+"),
    "PCT":    (r"PCT|Plateletcrit",                        r"[\d.]+"),
    "PDW":    (r"PDW|Platelet Distribution Width",        r"[\d.]+"),
    "NE#":    (r"NE#|NEU#|Neutrophil Count|NEUT#",        r"[\d.]+"),
    "LY#":    (r"LY#|LYM#|Lymphocyte Count|LYMPH#",       r"[\d.]+"),
    "MO#":    (r"MO#|MON#|Monocyte Count|MONO#",          r"[\d.]+"),
    "EO#":    (r"EO#|EOS#|Eosinophil Count",              r"[\d.]+"),
    "BA#":    (r"BA#|BAS#|Basophil Count",                r"[\d.]+"),
    "FERRITTE":(r"Ferritin|FERRITIN|Ferritte",            r"[\d.]+"),
    "FOLATE": (r"Folate|FOLATE|Folic Acid",               r"[\d.]+"),
    "B12":    (r"B12|Vitamin B12|Cobalamin|VITB12",       r"[\d.]+"),
    "TSD":    (r"TSD|Transferrin Saturation|TSAT|%Sat",   r"[\d.]+"),
    "SDTSD":  (r"SDTSD|TIBC|Total Iron Binding",         r"[\d.]+"),
    "SD":     (r"\bSD\b|Serum Iron|Iron Level",           r"[\d.]+"),
    "GENDER": (r"Sex|Gender|SEX",                         r"[MFmf01]"),
}

# Reference ranges for flagging
REFERENCE_RANGES = {
    "WBC":     (4.0,  11.0),
    "RBC":     (4.5,  5.9),
    "HGB":     (12.0, 17.0),
    "HCT":     (37.0, 52.0),
    "MCV":     (80.0, 100.0),
    "MCH":     (27.0, 33.0),
    "MCHC":    (32.0, 36.0),
    "RDW":     (11.5, 14.5),
    "PLT":     (150,  400),
    "MPV":     (7.5,  12.5),
    "FERRITTE":(12.0, 300.0),
    "FOLATE":  (3.0,  20.0),
    "B12":     (200,  900),
    "TSD":     (20.0, 50.0),
}


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file object."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        text = f"PDF_ERROR: {e}"
    return text


def extract_text_from_image(image_file) -> str:
    """Extract text from an image using OCR."""
    try:
        import pytesseract
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        return "OCR_UNAVAILABLE"
    except Exception as e:
        return f"IMAGE_ERROR: {e}"


def parse_gender(text: str) -> float:
    """Parse gender from text — returns 1 for Male, 0 for Female."""
    text_upper = text.upper()
    if re.search(r'\bMALE\b|\bM\b|\b1\b', text_upper):
        return 1.0
    elif re.search(r'\bFEMALE\b|\bF\b|\b0\b', text_upper):
        return 0.0
    return 0.0  # default


def extract_value_near_pattern(text: str, pattern: str) -> float | None:
    """
    Finds a numeric value near a lab test name in the report text.
    Looks for pattern then grabs the first number within ~60 chars.
    """
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for match in matches:
        # Look at text around the match
        start = match.end()
        snippet = text[start:start + 80]
        # Find first number in snippet
        num_match = re.search(r'(\d+\.?\d*)', snippet)
        if num_match:
            try:
                val = float(num_match.group(1))
                # Sanity check — avoid parsing dates or IDs
                if 0 < val < 100000:
                    return val
            except ValueError:
                continue
    return None


def parse_lab_report(file, file_type: str) -> dict:
    """
    Main parser — takes an uploaded file and returns a dict of lab values.
    file_type: 'pdf' or 'image'
    """
    # Extract raw text
    if file_type == "pdf":
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_image(file)

    extracted = {}
    not_found = []

    for feature, (pattern, _) in LAB_PATTERNS.items():
        if feature == "GENDER":
            extracted["GENDER"] = parse_gender(text)
            continue

        val = extract_value_near_pattern(text, pattern)
        if val is not None:
            extracted[feature] = val
        else:
            not_found.append(feature)

    return {
        "values": extracted,
        "not_found": not_found,
        "raw_text": text[:2000]  # first 2000 chars for debugging
    }


def flag_abnormal(values: dict) -> list:
    """Returns list of abnormal flags with direction."""
    flags = []
    for key, (low, high) in REFERENCE_RANGES.items():
        if key in values:
            v = values[key]
            if v < low:
                flags.append({"feature": key, "value": v, "status": "LOW",
                              "range": f"{low}–{high}"})
            elif v > high:
                flags.append({"feature": key, "value": v, "status": "HIGH",
                              "range": f"{low}–{high}"})
    return flags


def build_feature_row(values: dict, feature_cols: list) -> pd.DataFrame:
    """
    Converts extracted values into a DataFrame row matching model features.
    Missing values are filled with population medians.
    """
    MEDIANS = {
        "GENDER": 0, "WBC": 7.49, "NE#": 4.4, "LY#": 2.1,
        "MO#": 0.5, "EO#": 0.2, "BA#": 0.03, "RBC": 4.6,
        "HGB": 13.5, "HCT": 40.0, "MCV": 88.0, "MCH": 29.5,
        "MCHC": 33.5, "RDW": 13.0, "PLT": 240.0, "MPV": 10.0,
        "PCT": 0.24, "PDW": 12.0, "SD": 80.0, "SDTSD": 320.0,
        "TSD": 25.0, "FERRITTE": 60.0, "FOLATE": 8.0, "B12": 400.0,
    }

    row = {}
    for col in feature_cols:
        if col in values:
            row[col] = values[col]
        elif col in MEDIANS:
            row[col] = MEDIANS[col]
        else:
            row[col] = 0.0

    return pd.DataFrame([row])
