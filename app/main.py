"""
main.py — Medical Labsense Streamlit App
Run: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__))

from pdf_parser import parse_lab_report, flag_abnormal, build_feature_row
from predictor  import load_model, predict, get_shap_values, get_interpretation
from utils      import format_feature_name, make_display_table

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Labsense", page_icon="🩸",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
    border: 1px solid rgba(59,130,246,0.2);
}
.app-title { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
             color:white; margin:0; letter-spacing:-0.02em; }
.app-title span { color:#38bdf8; }
.app-subtitle { color:#94a3b8; font-size:1rem; margin-top:0.5rem; font-weight:300; }
.badge { display:inline-block; background:rgba(56,189,248,0.15); color:#38bdf8;
         border:1px solid rgba(56,189,248,0.3); padding:3px 12px; border-radius:20px;
         font-size:0.75rem; font-family:'DM Mono',monospace; margin-bottom:1rem; letter-spacing:0.08em; }
.result-positive { background:linear-gradient(135deg,#450a0a,#7f1d1d);
                   border:1px solid #ef4444; border-radius:12px; padding:1.5rem 2rem; margin-bottom:1rem; }
.result-negative { background:linear-gradient(135deg,#052e16,#14532d);
                   border:1px solid #22c55e; border-radius:12px; padding:1.5rem 2rem; margin-bottom:1rem; }
.result-label { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; margin:0; color:white; }
.result-confidence { font-family:'DM Mono',monospace; font-size:2.5rem; font-weight:500; margin:0; color:white; }
.flag-low  { background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.3);
             padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'DM Mono',monospace;
             margin:2px; display:inline-block; }
.flag-high { background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3);
             padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'DM Mono',monospace;
             margin:2px; display:inline-block; }
.flag-ok   { background:rgba(34,197,94,0.1); color:#4ade80; border:1px solid rgba(34,197,94,0.2);
             padding:3px 10px; border-radius:20px; font-size:0.8rem; font-family:'DM Mono',monospace;
             margin:2px; display:inline-block; }
.section-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#e2e8f0;
                 border-left:3px solid #38bdf8; padding-left:0.75rem; margin:1.5rem 0 1rem 0; }
.disclaimer { background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2);
              border-radius:8px; padding:0.75rem 1rem; font-size:0.82rem; color:#fbbf24; margin-top:1.5rem; }
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception:
        return None, None

model, feature_cols = get_model()


# ── Results renderer ──────────────────────────────────────────────────────────
def show_results(result, flags, shap_df, values):
    st.markdown("---")
    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

    col_pred, col_conf = st.columns([2, 1])
    with col_pred:
        css   = "result-positive" if result["prediction"] == 1 else "result-negative"
        icon  = "🔴" if result["prediction"] == 1 else "🟢"
        st.markdown(f"""
        <div class="{css}">
            <p style="font-size:0.8rem;color:#94a3b8;margin:0;font-family:'DM Mono',monospace;">PREDICTION</p>
            <p class="result-label">{icon} {result['label']}</p>
        </div>""", unsafe_allow_html=True)

    with col_conf:
        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                    padding:1.5rem;text-align:center;">
            <p style="font-size:0.75rem;color:#94a3b8;margin:0;font-family:'DM Mono',monospace;">CONFIDENCE</p>
            <p class="result-confidence">{result['confidence']}%</p>
        </div>""", unsafe_allow_html=True)

    # Probabilities
    st.markdown('<div class="section-title">Class probabilities</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Healthy", f"{result['prob_healthy']}%")
        st.progress(result["prob_healthy"] / 100)
    with c2:
        st.metric("Iron Anemia", f"{result['prob_anemia']}%")
        st.progress(result["prob_anemia"] / 100)

    # Flags
    st.markdown('<div class="section-title">Value flags</div>', unsafe_allow_html=True)
    if flags:
        pills = "".join(
            f'<span class="flag-{"low" if f["status"]=="LOW" else "high"}">'
            f'{f["feature"]} {f["status"]} ({f["value"]:.1f}) · ref {f["range"]}</span> '
            for f in flags
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown('<span class="flag-ok">✓ All values within reference ranges</span>',
                    unsafe_allow_html=True)

    # SHAP
    if shap_df is not None:
        st.markdown('<div class="section-title">Key drivers of this prediction</div>',
                    unsafe_allow_html=True)
        top = shap_df.head(10).copy()
        top["Feature"]   = top["feature"].apply(lambda f: f"{format_feature_name(f)} ({f})")
        top["Impact"]    = top["abs_shap"].round(4)
        top["Direction"] = top["shap_value"].apply(
            lambda v: "↑ Towards anemia" if v > 0 else "↓ Away from anemia")
        st.dataframe(top[["Feature", "Impact", "Direction"]],
                     use_container_width=True, hide_index=True)

    # Interpretation
    st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
    st.markdown(get_interpretation(result, flags))

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> Medical Labsense is a screening tool only.
        Results are not a clinical diagnosis. Always consult a licensed healthcare professional.
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="badge">◈ ANEMIA DETECTION · AI-POWERED</div>
    <p class="app-title">Medical <span>Labsense</span></p>
    <p class="app-subtitle">
        Upload a blood lab report or enter values manually to screen for iron deficiency anemia.
        Trained on 15,300 real patient records · XGBoost · 0.97 AUC-ROC
    </p>
</div>""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Place medical_labsense_model.pkl in the model/ folder.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Upload Report (PDF / Image)", "✏️ Enter Values Manually"])

# ─── Tab 1: Upload ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Upload your lab report</div>',
                unsafe_allow_html=True)
    st.caption("Supports PDF and image formats (JPG, PNG). Values extracted automatically.")

    uploaded = st.file_uploader("Drop file here", type=["pdf","jpg","jpeg","png"],
                                label_visibility="collapsed")

    if uploaded:
        ftype = "pdf" if uploaded.name.lower().endswith(".pdf") else "image"
        with st.spinner("🔍 Extracting lab values..."):
            parsed = parse_lab_report(uploaded, ftype)

        values, not_found = parsed["values"], parsed["not_found"]

        if not values:
            st.error("Could not extract values. Try Tab 2 to enter them manually.")
        else:
            st.success(f"✅ Extracted **{len(values)}** lab values.")
            if not_found:
                st.info(f"Not found (defaults used): {', '.join(not_found)}")

            flags = flag_abnormal(values)
            st.markdown('<div class="section-title">Extracted values</div>',
                        unsafe_allow_html=True)

            def color_status(val):
                if val == "LOW":  return "background-color:rgba(239,68,68,0.15);color:#f87171"
                if val == "HIGH": return "background-color:rgba(245,158,11,0.15);color:#fbbf24"
                return "background-color:rgba(34,197,94,0.08);color:#4ade80"

            st.dataframe(
                make_display_table(values, flags).style.map(color_status, subset=["Status"]),
                use_container_width=True, hide_index=True)

            row    = build_feature_row(values, feature_cols)
            result = predict(model, row)
            shap_df= get_shap_values(model, row, feature_cols)
            show_results(result, flags, shap_df, values)


# ─── Tab 2: Manual ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Enter lab values</div>', unsafe_allow_html=True)
    st.caption("Fill in from your blood report. Unknown fields use population defaults.")

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**CBC — Red Cell Indices**")
            HGB  = st.number_input("Hemoglobin (HGB) g/dL",     value=13.5, step=0.1)
            HCT  = st.number_input("Hematocrit (HCT) %",         value=40.0, step=0.5)
            RBC  = st.number_input("RBC ×10⁶/µL",                value=4.6,  step=0.1)
            MCV  = st.number_input("MCV fL",                      value=88.0, step=0.5)
            MCH  = st.number_input("MCH pg",                      value=29.5, step=0.5)
            MCHC = st.number_input("MCHC g/dL",                   value=33.5, step=0.5)
            RDW  = st.number_input("RDW %",                        value=13.0, step=0.1)

        with c2:
            st.markdown("**CBC — White Cell & Platelets**")
            WBC  = st.number_input("WBC ×10³/µL",                 value=7.5,  step=0.1)
            NE   = st.number_input("Neutrophils NE# ×10³/µL",     value=4.4,  step=0.1)
            LY   = st.number_input("Lymphocytes LY# ×10³/µL",     value=2.1,  step=0.1)
            MO   = st.number_input("Monocytes MO# ×10³/µL",       value=0.5,  step=0.05)
            EO   = st.number_input("Eosinophils EO# ×10³/µL",     value=0.2,  step=0.05)
            BA   = st.number_input("Basophils BA# ×10³/µL",       value=0.03, step=0.01)
            PLT  = st.number_input("Platelets PLT ×10³/µL",       value=240.0,step=5.0)
            MPV  = st.number_input("MPV fL",                       value=10.0, step=0.1)
            PCT  = st.number_input("Plateletcrit PCT %",           value=0.24, step=0.01)
            PDW  = st.number_input("PDW %",                        value=12.0, step=0.5)

        with c3:
            st.markdown("**Iron & Nutritional Markers**")
            FERRITTE = st.number_input("Ferritin ng/mL",           value=60.0, step=1.0)
            FOLATE   = st.number_input("Folate ng/mL",             value=8.0,  step=0.5)
            B12      = st.number_input("Vitamin B12 pg/mL",        value=400.0,step=10.0)
            TSD      = st.number_input("Transferrin Saturation %",  value=25.0, step=0.5)
            SDTSD    = st.number_input("TIBC µg/dL",                value=320.0,step=5.0)
            SD       = st.number_input("Serum Iron µg/dL",          value=80.0, step=1.0)
            st.markdown("**Demographics**")
            GENDER = 1.0 if st.selectbox("Gender", ["Female","Male"]) == "Male" else 0.0

        submitted = st.form_submit_button("🔬 Analyse", use_container_width=True)

    if submitted:
        values = {
            "GENDER":GENDER,"WBC":WBC,"NE#":NE,"LY#":LY,"MO#":MO,
            "EO#":EO,"BA#":BA,"RBC":RBC,"HGB":HGB,"HCT":HCT,
            "MCV":MCV,"MCH":MCH,"MCHC":MCHC,"RDW":RDW,"PLT":PLT,
            "MPV":MPV,"PCT":PCT,"PDW":PDW,"SD":SD,"SDTSD":SDTSD,
            "TSD":TSD,"FERRITTE":FERRITTE,"FOLATE":FOLATE,"B12":B12,
        }
        flags   = flag_abnormal(values)
        row     = build_feature_row(values, feature_cols)
        result  = predict(model, row)
        shap_df = get_shap_values(model, row, feature_cols)
        show_results(result, flags, shap_df, values)
