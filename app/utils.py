"""
utils.py
Shared helper functions for Medical Labsense app.
"""

import pandas as pd


FEATURE_FULL_NAMES = {
    "WBC":     "White Blood Cells",
    "RBC":     "Red Blood Cells",
    "HGB":     "Hemoglobin",
    "HCT":     "Hematocrit",
    "MCV":     "Mean Corpuscular Volume",
    "MCH":     "Mean Corpuscular Hemoglobin",
    "MCHC":    "MCH Concentration",
    "RDW":     "Red Cell Distribution Width",
    "PLT":     "Platelets",
    "MPV":     "Mean Platelet Volume",
    "PCT":     "Plateletcrit",
    "PDW":     "Platelet Distribution Width",
    "NE#":     "Neutrophil Count",
    "LY#":     "Lymphocyte Count",
    "MO#":     "Monocyte Count",
    "EO#":     "Eosinophil Count",
    "BA#":     "Basophil Count",
    "FERRITTE":"Ferritin",
    "FOLATE":  "Folate",
    "B12":     "Vitamin B12",
    "TSD":     "Transferrin Saturation",
    "SDTSD":   "Total Iron Binding Capacity",
    "SD":      "Serum Iron",
    "GENDER":  "Gender",
}

UNITS = {
    "WBC": "×10³/µL", "RBC": "×10⁶/µL", "HGB": "g/dL",
    "HCT": "%", "MCV": "fL", "MCH": "pg", "MCHC": "g/dL",
    "RDW": "%", "PLT": "×10³/µL", "MPV": "fL", "PCT": "%",
    "PDW": "%", "NE#": "×10³/µL", "LY#": "×10³/µL",
    "MO#": "×10³/µL", "EO#": "×10³/µL", "BA#": "×10³/µL",
    "FERRITTE": "ng/mL", "FOLATE": "ng/mL", "B12": "pg/mL",
    "TSD": "%", "SDTSD": "µg/dL", "SD": "µg/dL",
}


def format_feature_name(feature: str) -> str:
    return FEATURE_FULL_NAMES.get(feature, feature)


def format_unit(feature: str) -> str:
    return UNITS.get(feature, "")


def make_display_table(values: dict, flags: list) -> pd.DataFrame:
    """Builds a display-ready DataFrame of extracted lab values."""
    flag_map = {f["feature"]: f["status"] for f in flags}
    rows = []
    for feat, val in values.items():
        if feat == "GENDER":
            display_val = "Male" if val == 1 else "Female"
            unit = ""
        else:
            display_val = f"{val:.2f}"
            unit = format_unit(feat)

        status = flag_map.get(feat, "Normal")
        rows.append({
            "Test":   format_feature_name(feat),
            "Value":  display_val,
            "Unit":   unit,
            "Status": status,
        })
    return pd.DataFrame(rows)
