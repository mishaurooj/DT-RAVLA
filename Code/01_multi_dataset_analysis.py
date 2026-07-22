import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix
)

ROOT = r"D:\other\DT-RAVLA"
DATASET_DIR = os.path.join(ROOT, "Dataset")
RESULTS_DIR = os.path.join(ROOT, "Results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DATASETS = {
    "SWaT": {
        "folder": os.path.join(DATASET_DIR, "SWaT"),
        "files": ["merged.csv", "normal.csv", "attack.csv"]
    },
    "WADI": {
        "folder": os.path.join(DATASET_DIR, "WADI"),
        "files": ["WADI_14days_new.csv", "WADI_attackdataLABLE.csv"]
    }
}


def load_csv(path):
    print(f"\nLoading: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    print("Shape:", df.shape)
    return df


def find_label_column(df):
    candidates = [
        "Normal/Attack", "normal/attack", "Label", "label",
        "Attack", "attack", "Class", "class", "Y", "y"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        vals = df[c].dropna().astype(str).str.lower().unique()
        vals = set(vals[:30])
        if {"normal", "attack"} & vals:
            return c

    return None


def normalize_label(df, label_col):
    if label_col is None:
        return df, None

    raw = df[label_col].astype(str).str.strip().str.lower()

    mapping = {
        "normal": 0,
        "attack": 1,
        "a": 1,
        "n": 0,
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "nan": np.nan
    }

    df["binary_label"] = raw.map(mapping)

    if df["binary_label"].isna().sum() > 0:
        numeric = pd.to_numeric(df[label_col], errors="coerce")
        if numeric.notna().mean() > 0.80:
            df["binary_label"] = numeric.fillna(0).astype(int)

    return df, "binary_label"


def get_numeric_columns(df, label_col):
    ignore = {
        label_col, "binary_label",
        "Timestamp", "timestamp", "Date", "date", "Time", "time",
        "Row", "row", "Unnamed: 0"
    }

    cols = []

    for c in df.columns:
        if c in ignore:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.95:
                df[c] = x
                cols.append(c)

    return cols


def prepare_numeric_matrix(df, numeric_cols, sample_size=60000):
    temp = df[numeric_cols].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan)

    missing_ratio = temp.isna().mean()
    keep_cols = missing_ratio[missing_ratio < 0.40].index.tolist()
    temp = temp[keep_cols]

    temp = temp.apply(pd.to_numeric, errors="coerce")
    temp = temp.fillna(temp.median(numeric_only=True))
    temp = temp.fillna(0)

    nunique = temp.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    temp = temp[keep_cols]

    if len(temp) > sample_size:
        temp = temp.sample(sample_size, random_state=42)

    return temp, keep_cols


def dataset_profile(dataset_name, file_name, df, numeric_cols, label_col, original_label):
    info = {
        "dataset": dataset_name,
        "file": file_name,
        "samples": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": int(len(numeric_cols)),
        "original_label_column": original_label,
        "binary_label_column": label_col,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "first_20_features": numeric_cols[:20]
    }

    if label_col and label_col in df.columns:
        vc = df[label_col].value_counts(dropna=False).to_dict()
        info["label_distribution"] = {str(k): int(v) for k, v in vc.items()}

        normal = int(vc.get(0.0, vc.get(0, 0)))
        attack = int(vc.get(1.0, vc.get(1, 0)))
        total = normal + attack

        if total > 0:
            info["normal_samples"] = normal
            info["attack_samples"] = attack
            info["attack_ratio"] = float(attack / total)

    return info


def plot_label_distribution(dataset_name, file_name, df, label_col):
    if label_col is None or label_col not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    df[label_col].value_counts().sort_index().plot(kind="bar")
    plt.title(f"{dataset_name}: Label Distribution")
    plt.xlabel("0 = Normal, 1 = Attack")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{dataset_name}_{file_name}_label_distribution.png"), dpi=300)
    plt.close()


def plot_pca(dataset_name, file_name, df, numeric_cols, label_col, sample_size=60000):
    temp, used_cols = prepare_numeric_matrix(df, numeric_cols, sample_size)

    if temp.shape[0] == 0 or temp.shape[1] == 0:
        print(f"Skipping PCA for {dataset_name}-{file_name}: no usable numeric data.")
        return {
            "dataset": dataset_name,
            "file": file_name,
            "pc1": np.nan,
            "pc2": np.nan,
            "total": np.nan,
            "used_features": 0
        }

    X = StandardScaler().fit_transform(temp.values)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    if label_col and label_col in df.columns:
        y = df.loc[temp.index, label_col].fillna(0).astype(int).values
        plt.scatter(Z[:, 0], Z[:, 1], c=y, s=4, alpha=0.6)
        plt.colorbar(label="0 = Normal, 1 = Attack")
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.6)

    plt.title(f"{dataset_name}: PCA Industrial State Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{dataset_name}_{file_name}_pca.png"), dpi=300)
    plt.close()

    return {
        "dataset": dataset_name,
        "file": file_name,
        "pc1": float(pca.explained_variance_ratio_[0]),
        "pc2": float(pca.explained_variance_ratio_[1]),
        "total": float(pca.explained_variance_ratio_.sum()),
        "used_features": int(len(used_cols))
    }


def baseline_isolation_forest(dataset_name, file_name, df, numeric_cols, label_col):
    if label_col is None or label_col not in df.columns:
        return None

    temp_X, used_cols = prepare_numeric_matrix(df, numeric_cols, sample_size=120000)

    if temp_X.shape[0] == 0 or temp_X.shape[1] == 0:
        print(f"Skipping Isolation Forest for {dataset_name}-{file_name}: no usable numeric data.")
        return None

    y = df.loc[temp_X.index, label_col].fillna(0).astype(int).values

    if len(np.unique(y)) < 2:
        print(f"Skipping Isolation Forest for {dataset_name}-{file_name}: only one class found.")
        return None

    X = StandardScaler().fit_transform(temp_X.values)

    contamination = max(0.001, min(0.30, y.mean()))

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    pred_raw = model.fit_predict(X)
    pred = np.where(pred_raw == -1, 1, 0)
    scores = -model.decision_function(X)

    metrics = {
        "dataset": dataset_name,
        "file": file_name,
        "model": "IsolationForest",
        "used_features": int(len(used_cols)),
        "samples_used": int(len(y)),
        "attack_ratio": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores))
    }

    cm = confusion_matrix(y, pred)
    pd.DataFrame(
        cm,
        index=["true_normal", "true_attack"],
        columns=["pred_normal", "pred_attack"]
    ).to_csv(os.path.join(TABLE_DIR, f"{dataset_name}_{file_name}_isolation_forest_cm.csv"))

    return metrics


def save_descriptive_statistics(dataset_name, file_name, df, numeric_cols):
    temp, used_cols = prepare_numeric_matrix(df, numeric_cols, sample_size=len(df))
    if temp.shape[1] == 0:
        return

    desc = temp.describe().T
    desc["missing_ratio_original"] = df[used_cols].isna().mean().values
    desc.to_csv(os.path.join(TABLE_DIR, f"{dataset_name}_{file_name}_descriptive_statistics.csv"))


def save_feature_missing_report(dataset_name, file_name, df, numeric_cols):
    report = pd.DataFrame({
        "feature": numeric_cols,
        "missing_count": [int(df[c].isna().sum()) for c in numeric_cols],
        "missing_ratio": [float(df[c].isna().mean()) for c in numeric_cols],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in numeric_cols]
    })
    report.to_csv(os.path.join(TABLE_DIR, f"{dataset_name}_{file_name}_feature_quality.csv"), index=False)


def analyze_all():
    profiles = []
    pca_rows = []
    baseline_rows = []

    for dataset_name, cfg in DATASETS.items():
        folder = cfg["folder"]

        for file in cfg["files"]:
            path = os.path.join(folder, file)

            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            safe_file = file.replace(".csv", "").replace(" ", "_")

            df = load_csv(path)

            original_label = find_label_column(df)
            df, label_col = normalize_label(df, original_label)
            numeric_cols = get_numeric_columns(df, label_col)

            print("Original label column:", original_label)
            print("Binary label column:", label_col)
            print("Numeric features:", len(numeric_cols))

            profile = dataset_profile(
                dataset_name, safe_file, df, numeric_cols, label_col, original_label
            )
            profiles.append(profile)

            save_descriptive_statistics(dataset_name, safe_file, df, numeric_cols)
            save_feature_missing_report(dataset_name, safe_file, df, numeric_cols)
            plot_label_distribution(dataset_name, safe_file, df, label_col)

            pca_info = plot_pca(dataset_name, safe_file, df, numeric_cols, label_col)
            pca_rows.append(pca_info)

            baseline = baseline_isolation_forest(dataset_name, safe_file, df, numeric_cols, label_col)
            if baseline is not None:
                baseline_rows.append(baseline)

            del df

    profile_df = pd.DataFrame(profiles)
    pca_df = pd.DataFrame(pca_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    profile_df.to_csv(os.path.join(TABLE_DIR, "Table_1_dataset_characteristics_raw.csv"), index=False)
    pca_df.to_csv(os.path.join(TABLE_DIR, "pca_explained_variance.csv"), index=False)
    baseline_df.to_csv(os.path.join(TABLE_DIR, "baseline_isolation_forest_results.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump(profiles, f, indent=4)

    print("\nAnalysis complete.")
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    analyze_all()