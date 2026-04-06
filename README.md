# Medical Labsense 🩸

AI-powered clinical lab test analyser that detects anemia types and nutritional deficiencies from blood work using an automated machine learning pipeline built on XGBoost, SMOTE and SHAP explainability.

## Features

- 📄 Upload a PDF or image of your lab report — values extracted automatically
- ✏️ Or enter values manually via a clean form interface
- 🔬 Predicts iron deficiency anemia with **0.97 AUC-ROC**
- 🧠 SHAP explainability — shows which values drove the prediction
- 🚩 Flags all out-of-range values with reference ranges
- ⚕️ Trained on **15,300 real patient records**

## Model

| Metric            | Score  |
|-------------------|--------|
| Balanced Accuracy | 0.9389 |
| Sensitivity       | 0.9389 |
| AUC-ROC           | 0.9720 |

**Algorithm:** XGBoost · **Imbalance handling:** Random Undersampling · **Search:** Latin Hypercube Sampling (AutoML-Med framework)

## Project Structure

```
medical-labsense/
├── app/
│   ├── main.py          # Streamlit app
│   ├── predictor.py     # Model loading & prediction
│   ├── pdf_parser.py    # Lab report text extraction
│   └── utils.py         # Helpers & reference ranges
├── model/
│   ├── medical_labsense_model.pkl
│   ├── medical_labsense_features.pkl
│   └── medical_labsense_results.json
├── pipeline/
│   └── medical_automl_pipeline.py
├── notebooks/
│   └── training.ipynb
├── data/
│   └── sample_report.pdf
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/medical-labsense.git
cd medical-labsense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model files to model/

# 4. Run the app
streamlit run app/main.py
```

## Dataset

Trained on the SKILICARSLAN Anemia Dataset — 15,300 patients with full CBC differential, iron studies, folate and B12 markers, with labeled anemia subtypes.

## Disclaimer

Medical Labsense is a **screening tool only** and is not a substitute for clinical diagnosis. Always consult a licensed healthcare professional before making medical decisions.
