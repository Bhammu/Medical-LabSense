import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, roc_auc_score,
    precision_score, f1_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import spearmanr
import xgboost as xgb
import shap
import warnings
import itertools
import json
from datetime import datetime

warnings.filterwarnings("ignore")

def generate_sample_lab_data(n_samples=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = n_samples

    data = {
        "WBC": rng.normal(7.5, 2.5, n).clip(2, 20),
        "RBC": rng.normal(4.8, 0.6, n).clip(2, 7),
        "Hemoglobin": rng.normal(14.0, 2.5, n).clip(6, 20),
        "Hematocrit": rng.normal(42.0, 6.0, n).clip(20, 60),
        "MCV": rng.normal(90.0, 10.0, n).clip(60, 120),
        "MCH": rng.normal(30.0, 4.0, n).clip(18, 42),
        "MCHC": rng.normal(33.0, 2.0, n).clip(28, 38),
        "Platelets": rng.normal(250, 70, n).clip(50, 600),
        "Glucose": rng.normal(95.0, 30.0, n).clip(50, 400),
        "BUN": rng.normal(15.0, 5.0, n).clip(5, 50),
        "Creatinine": rng.normal(1.0, 0.3, n).clip(0.4, 5),
        "Sodium": rng.normal(140, 4.0, n).clip(125, 155),
        "Potassium": rng.normal(4.0, 0.5, n).clip(2.5, 6.5),
        "Albumin": rng.normal(4.2, 0.5, n).clip(2, 6),
        "Cholesterol": rng.normal(195, 40, n).clip(100, 400),
        "LDL": rng.normal(120, 35, n).clip(40, 300),
        "HDL": rng.normal(55, 15, n).clip(20, 120),
        "Triglycerides": rng.normal(140, 60, n).clip(40, 600),
        "ALT": rng.normal(25, 15, n).clip(5, 200),
        "AST": rng.normal(28, 14, n).clip(5, 200),
        "Bilirubin": rng.normal(0.8, 0.4, n).clip(0.1, 5),
        "TSH": rng.normal(2.5, 1.5, n).clip(0.1, 15),
        "T4": rng.normal(8.0, 2.0, n).clip(3, 20),
        "Age": rng.randint(18, 80, n).astype(float),
        "Sex": rng.randint(0, 2, n).astype(float),
    }

    df = pd.DataFrame(data)

    labels = np.zeros(n, dtype=int)

    anemia_mask = (df["Hemoglobin"] < 11.5) & (df["RBC"] < 4.0) & (df["MCV"] < 80)
    labels[anemia_mask] = 1

    diabetes_mask = (df["Glucose"] > 140) & ~anemia_mask
    labels[diabetes_mask] = 2

    thyroid_mask = ((df["TSH"] < 0.5) | (df["TSH"] > 5.5)) & ~anemia_mask & ~diabetes_mask
    labels[thyroid_mask] = 3

    df["Target"] = labels

    for col in df.columns[:-1]:
        missing_idx = rng.choice(n, size=int(0.08 * n), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["LDL_HDL_ratio"] = df["LDL"] / (df["HDL"] + 1e-6)
    df["Total_HDL_ratio"] = df["Cholesterol"] / (df["HDL"] + 1e-6)
    df["MCH_MCV_ratio"] = df["MCH"] / (df["MCV"] + 1e-6)
    df["Hb_low_flag"] = (df["Hemoglobin"] < 12.0).astype(float)
    df["RBC_low_flag"] = (df["RBC"] < 4.0).astype(float)
    df["Glucose_high_flag"] = (df["Glucose"] > 126).astype(float)
    df["BUN_Creatinine_ratio"] = df["BUN"] / (df["Creatinine"] + 1e-6)
    df["TSH_abnormal_flag"] = ((df["TSH"] < 0.5) | (df["TSH"] > 5.0)).astype(float)
    df["TSH_T4_product"] = df["TSH"] * df["T4"]
    df["AST_ALT_ratio"] = df["AST"] / (df["ALT"] + 1e-6)
    df["WBC_high_flag"] = (df["WBC"] > 11.0).astype(float)

    return df

def latin_hypercube_sample(n_samples: int, param_ranges: dict, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    n_params = len(param_ranges)
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    intervals = np.zeros((n_samples, n_params))
    for i in range(n_params):
        perm = rng.permutation(n_samples)
        intervals[:, i] = (perm + rng.uniform(size=n_samples)) / n_samples

    configs = []
    for sample in intervals:
        config = {}
        for j, key in enumerate(keys):
            options = values[j]
            idx = int(sample[j] * len(options))
            idx = min(idx, len(options) - 1)
            config[key] = options[idx]
        configs.append(config)

    return configs

MODELS = {
    "xgboost": lambda seed: xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=seed, n_jobs=-1
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=seed, n_jobs=-1
    ),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        random_state=seed
    ),
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1
    ),
}

IMPUTERS = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "knn": KNNImputer(n_neighbors=5),
}

SCALERS = {
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler(),
    "none": None,
}

RESAMPLERS = {
    "smote": SMOTE(random_state=42),
    "adasyn": ADASYN(random_state=42),
    "undersample": RandomUnderSampler(random_state=42),
    "none": None,
}

def build_pipeline(config: dict, model_name: str, seed: int = 42):
    steps = []
    steps.append(("imputer", IMPUTERS[config["imputer"]]))

    if config["scaler"] != "none":
        steps.append(("scaler", SCALERS[config["scaler"]]))

    resampler = RESAMPLERS[config["resampler"]]
    if resampler is not None:
        steps.append(("resampler", resampler))
        pipeline_class = ImbPipeline
    else:
        pipeline_class = Pipeline

    if config["feature_selection"] == "kbest":
        steps.append(("selector", SelectKBest(f_classif, k=config.get("k_features", 20))))

    steps.append(("model", MODELS[model_name](seed)))

    return pipeline_class(steps)

def evaluate_pipeline(pipeline, X, y, cv_folds=5, seed=42):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "recall_macro": "recall_macro",
        "precision_macro": "precision_macro",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }

    try:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
        return {
            "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
            "sensitivity": scores["test_recall_macro"].mean(),
            "precision": scores["test_precision_macro"].mean(),
            "f1": scores["test_f1_macro"].mean(),
            "auc_roc": scores["test_roc_auc_ovr"].mean(),
            "status": "ok"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "balanced_accuracy": 0}

def run_automl_search(X, y, n_configs=20, models=None, seed=42, verbose=True):
    if models is None:
        models = ["xgboost", "random_forest", "logistic_regression"]

    param_ranges = {
        "imputer": ["mean", "median", "knn"],
        "scaler": ["standard", "robust", "minmax", "none"],
        "resampler": ["smote", "adasyn", "undersample", "none"],
        "feature_selection": ["kbest", "none"],
        "k_features": [10, 15, 20, 25, 30],
    }

    configs = latin_hypercube_sample(n_configs, param_ranges, random_state=seed)

    results = []
    total = len(configs) * len(models)
    count = 0

    print(f"\n{'='*60}")
    print(f"  AutoML-Med Search: {n_configs} configs × {len(models)} models = {total} runs")
    print(f"{'='*60}\n")

    for model_name in models:
        for i, config in enumerate(configs):
            count += 1
            if verbose:
                print(f"  [{count:3d}/{total}] model={model_name:<20} "
                      f"imputer={config['imputer']:<7} "
                      f"scaler={config['scaler']:<9} "
                      f"resampler={config['resampler']:<12}", end=" → ")

            pipeline = build_pipeline(config, model_name, seed)
            metrics = evaluate_pipeline(pipeline, X, y, seed=seed)

            if metrics["status"] == "ok":
                result = {
                    "model": model_name,
                    "config": config,
                    **metrics
                }
                results.append(result)
                if verbose:
                    print(f"bal_acc={metrics['balanced_accuracy']:.3f}  "
                          f"sensitivity={metrics['sensitivity']:.3f}  "
                          f"AUC={metrics['auc_roc']:.3f}")
            else:
                if verbose:
                    print(f"FAILED: {metrics.get('error', '')[:50]}")

    results.sort(key=lambda r: (r["balanced_accuracy"] + r["sensitivity"]) / 2, reverse=True)

    return results

def prcc_sensitivity_analysis(results):
    if len(results) < 5:
        return {}

    df = pd.DataFrame(results)
    target = df["balanced_accuracy"].rank()

    prcc_scores = {}
    config_df = pd.DataFrame(df["config"].tolist())

    for col in config_df.columns:
        encoded = pd.Categorical(config_df[col]).codes
        corr, pval = spearmanr(encoded, target)
        prcc_scores[col] = {"correlation": round(corr, 3), "p_value": round(pval, 3)}

    prcc_scores = dict(sorted(
        prcc_scores.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    ))

    return prcc_scores

def get_shap_importances(best_pipeline, X_train, X_test, feature_names):
    model = best_pipeline.named_steps["model"]

    preprocess_steps = [(k, v) for k, v in best_pipeline.steps if k != "model"]
    if preprocess_steps:
        from sklearn.pipeline import Pipeline as SKPipeline
        preproc = SKPipeline([(k, v) for k, v in preprocess_steps
                              if not hasattr(v, 'fit_resample')])
        X_train_t = preproc.fit_transform(X_train)
        X_test_t = preproc.transform(X_test)
    else:
        X_train_t = X_train.values
        X_test_t = X_test.values

    try:
        if isinstance(model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train_t)

        shap_values = explainer.shap_values(X_test_t)

        if isinstance(shap_values, list):
            shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_mean = np.abs(shap_values).mean(axis=0)

        n_features = min(len(shap_mean), len(feature_names))
        importance_df = pd.DataFrame({
            "feature": feature_names[:n_features],
            "importance": shap_mean[:n_features]
        }).sort_values("importance", ascending=False)

        return importance_df

    except Exception as e:
        print(f"  SHAP failed: {e}")
        return None

def run_full_pipeline(df: pd.DataFrame = None, target_col: str = "Target",
                      n_configs: int = 15, seed: int = 42):

    print("\n📋  Loading and engineering features...")
    if df is None:
        df = generate_sample_lab_data(n_samples=1500, random_state=seed)
        print(f"    Generated synthetic dataset: {df.shape[0]} patients, "
              f"{df.shape[1]-1} features")
        print(f"    Class distribution:\n{df[target_col].value_counts().to_string()}\n")

    df = engineer_features(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_automl_search(
        X_train, y_train,
        n_configs=n_configs,
        models=["xgboost", "random_forest", "logistic_regression"],
        seed=seed
    )

    print("\n📊  PRCC Sensitivity Analysis (which preprocessing matters most):")
    prcc = prcc_sensitivity_analysis(results)
    for param, scores in prcc.items():
        bar = "█" * int(abs(scores["correlation"]) * 20)
        print(f"    {param:<22} corr={scores['correlation']:+.3f}  {bar}")

    best = results[0]
    print(f"\n🏆  Best Pipeline:")
    print(f"    Model:        {best['model']}")
    print(f"    Config:       {best['config']}")
    print(f"    Bal. Acc:     {best['balanced_accuracy']:.4f}")
    print(f"    Sensitivity:  {best['sensitivity']:.4f}")
    print(f"    AUC-ROC:      {best['auc_roc']:.4f}")
    print(f"    F1 Macro:     {best['f1']:.4f}")

    print("\n🔧  Retraining best pipeline on full training set...")
    best_pipeline = build_pipeline(best["config"], best["model"], seed)
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)

    class_names = {0: "Healthy", 1: "Anemia", 2: "Diabetes", 3: "Thyroid"}
    target_names = [class_names.get(i, str(i)) for i in sorted(y.unique())]

    print("\n📈  Test Set Results:")
    print(f"    Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"    Sensitivity:       {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"    AUC-ROC (OvR):     ", end="")
    try:
        y_prob = best_pipeline.predict_proba(X_test)
        print(f"{roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'):.4f}")
    except:
        print("N/A")

    print(f"\n    Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n🧠  SHAP Feature Importances (top 10):")
    importance_df = get_shap_importances(best_pipeline, X_train, X_test, feature_names)
    if importance_df is not None:
        for _, row in importance_df.head(10).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best["model"],
        "best_config": best["config"],
        "cv_metrics": {k: v for k, v in best.items() if k not in ["model", "config", "status"]},
        "prcc_analysis": prcc,
        "top_features": importance_df.head(10).to_dict("records") if importance_df is not None else [],
        "all_results": [
            {"model": r["model"], "config": r["config"],
             "balanced_accuracy": r["balanced_accuracy"],
             "sensitivity": r["sensitivity"], "auc_roc": r["auc_roc"]}
            for r in results[:10]
        ]
    }

    with open("/mnt/user-data/outputs/automl_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅  Results saved to automl_results.json")
    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Best: {best['model']} | "
          f"Bal.Acc={best['balanced_accuracy']:.3f} | "
          f"Sensitivity={best['sensitivity']:.3f}")
    print(f"{'='*60}\n")

    return best_pipeline, results, prcc, importance_df

if __name__ == "__main__":
    best_pipeline, results, prcc, importances = run_full_pipeline(
        df=None,
        target_col="Target",
        n_configs=15,
        seed=42
    )