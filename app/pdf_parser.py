import re
import pdfplumber
import pandas as pd
from PIL import Image
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LAB_PATTERNS = {
    "WBC": ["WBC", "Leukocyte"],
    "RBC": ["RBC"],
    "HGB": ["Hemoglobin", "Haemoglobin", "HB"],
    "HCT": ["Hematocrit", "HCT"],
    "MCV": ["MCV"],
    "MCH": ["MCH"],
    "MCHC": ["MCHC"],
    "RDW": ["RDW"],
    "PLT": ["Platelet"],
    "MPV": ["MPV"],
    "PCT": ["PCT"],
    "PDW": ["PDW"],
    "NE#": ["Neutrophils"],
    "LY#": ["Lymphocytes"],
    "MO#": ["Monocytes"],
    "EO#": ["Eosinophils"],
    "BA#": ["Basophils"],
    "FERRITTE": ["Ferritin"],
    "FOLATE": ["Folate"],
    "B12": ["B12"],
    "TSD": ["Transferrin"],
    "SDTSD": ["TIBC"],
    "SD": ["Iron"],
    "GENDER": ["Gender", "Sex"]
}

REFERENCE_RANGES = {
    "WBC": (4.0, 11.0),
    "RBC": (4.5, 5.9),
    "HGB": (12.0, 17.0),
    "HCT": (37.0, 52.0),
    "MCV": (80.0, 100.0),
    "MCH": (27.0, 33.0),
    "MCHC": (32.0, 36.0),
    "PLT": (150, 400),
}

# ─────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def extract_ocr_data(image_file):
    import pytesseract

    img = Image.open(image_file)
    img = np.array(img)

    data = pytesseract.image_to_data(
        img,
        config='--psm 4',
        output_type=pytesseract.Output.DATAFRAME
    )

    data = data.dropna()
    return data


# ─────────────────────────────────────────
# OCR TABLE PARSING (CORE LOGIC)
# ─────────────────────────────────────────

def group_rows(df):
    rows = {}
    for _, row in df.iterrows():
        y = int(row['top'] // 10)  # group by vertical position
        if y not in rows:
            rows[y] = []
        rows[y].append(row['text'])

    return [" ".join(rows[k]) for k in sorted(rows.keys())]


def extract_from_rows(rows):
    extracted = {}

    for feature, keywords in LAB_PATTERNS.items():
        if feature == "GENDER":
            continue

        for row in rows:
            for kw in keywords:
                if kw.lower() in row.lower():
                    nums = re.findall(r'\d+\.?\d*', row)
                    if nums:
                        val = float(nums[0])
                        if 0 < val < 100000:
                            extracted[feature] = val
                            break
            if feature in extracted:
                break

    return extracted


# ─────────────────────────────────────────
# FALLBACK TEXT PARSER
# ─────────────────────────────────────────

def extract_from_text(text):
    extracted = {}
    lines = text.split("\n")

    for feature, keywords in LAB_PATTERNS.items():
        if feature == "GENDER":
            continue

        for line in lines:
            for kw in keywords:
                if kw.lower() in line.lower():
                    nums = re.findall(r'\d+\.?\d*', line)
                    if nums:
                        val = float(nums[0])
                        if 0 < val < 100000:
                            extracted[feature] = val
                            break
            if feature in extracted:
                break

    return extracted


# ─────────────────────────────────────────
# GENDER PARSER
# ─────────────────────────────────────────

def parse_gender(text):
    text = text.upper()
    if "MALE" in text:
        return 1.0
    elif "FEMALE" in text:
        return 0.0
    return 0.0


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────

def parse_lab_report(file, file_type):

    if file_type == "pdf":
        text = extract_text_from_pdf(file)
        extracted = extract_from_text(text)

    else:
        ocr_df = extract_ocr_data(file)
        rows = group_rows(ocr_df)
        extracted = extract_from_rows(rows)

        if len(extracted) < 3:
            text = " ".join(rows)
            fallback = extract_from_text(text)
            extracted.update(fallback)

        text = " ".join(rows)

    extracted["GENDER"] = parse_gender(text)

    not_found = [k for k in LAB_PATTERNS if k not in extracted]

    return {
        "values": extracted,
        "not_found": not_found,
        "raw_text": text[:2000]
    }


# ─────────────────────────────────────────
# FLAGS
# ─────────────────────────────────────────

def flag_abnormal(values):
    flags = []
    for key, (low, high) in REFERENCE_RANGES.items():
        if key in values:
            v = values[key]
            if v < low:
                flags.append({"feature": key, "value": v, "status": "LOW"})
            elif v > high:
                flags.append({"feature": key, "value": v, "status": "HIGH"})
    return flags


# ─────────────────────────────────────────
# MODEL INPUT
# ─────────────────────────────────────────

def build_feature_row(values, feature_cols):
    MEDIANS = {
        "GENDER": 0, "WBC": 7.49, "RBC": 4.6,
        "HGB": 13.5, "HCT": 40.0, "MCV": 88.0,
        "MCH": 29.5, "MCHC": 33.5, "PLT": 240.0
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