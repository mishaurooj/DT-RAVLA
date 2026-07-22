import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import skew, kurtosis

DATA_DIR = r"D:\other\DT-RAVLA\SWaTDataset"
OUTPUT_DIR = r"D:\other\DT-RAVLA\analysis_outputs"

MERGED_FILE = os.path.join(DATA_DIR, "merged.csv")
NORMAL_FILE = os.path.join(DATA_DIR, "normal.csv")
ATTACK_FILE = os.path.join(DATA_DIR, "attack.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tables"), exist_ok=True)


def load_csv_safely(path):
    print(f"\nLoading: {path}")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    print(f"Shape: {df.shape}")
    return df


def clean_column_names(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_label_column(df):
    possible = ["Normal/Attack", "normal/attack", "Label", "label", "Attack", "attack", "Class", "class"]
    for col in possible:
        if col in df.columns:
            return col

    for col in df.columns:
        vals = df[col].dropna().astype(str).str.lower().unique()
        if any(v in vals for v in ["normal", "attack", "a", "n"]):
            return col

    return None


def normalize_labels(df, label_col):
    if label_col is None:
        return df, None

    y_raw = df[label_col].astype(str).str.strip().str.lower()

    mapping = {
        "normal": 0,
        "attack": 1,
        "a": 1,
        "n": 0,
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1
    }

    df["binary_label"] = y_raw.map(mapping)

    unknown = df["binary_label"].isna().sum()
    if unknown > 0:
        print(f"Warning: {unknown} labels could not be mapped.")

    return df, "binary_label"


def get_numeric_features(df, label_cols):
    ignore = set(label_cols)
    ignore.update(["Timestamp", "timestamp", "Date", "Time", "date", "time"])

    numeric_cols = []
    for col in df.columns:
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            converted = pd.to_numeric(df[col], errors="coerce")
            valid_ratio = converted.notna().mean()
            if valid_ratio > 0.95:
                df[col] = converted
                numeric_cols.append(col)

    return numeric_cols


def save_basic_info(df, name, label_col=None):
    info = {
        "name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "label_column": label_col
    }

    if label_col and label_col in df.columns:
        info["label_distribution"] = df[label_col].value_counts(dropna=False).to_dict()

    with open(os.path.join(OUTPUT_DIR, f"{name}_basic_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(json.dumps(info, indent=4)[:3000])


def plot_missing_values(df, name):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print(f"No missing values found in {name}.")
        return

    plt.figure(figsize=(14, 6))
    missing.head(50).plot(kind="bar")
    plt.title(f"Top Missing Values: {name}")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{name}_missing_values.png"), dpi=300)
    plt.close()


def plot_label_distribution(df, name, label_col):
    if label_col is None:
        return

    plt.figure(figsize=(6, 4))
    df[label_col].value_counts().sort_index().plot(kind="bar")
    plt.title(f"Label Distribution: {name}")
    plt.xlabel("0 = Normal, 1 = Attack")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{name}_label_distribution.png"), dpi=300)
    plt.close()


def save_descriptive_statistics(df, name, numeric_cols):
    desc = df[numeric_cols].describe().T
    desc["missing"] = df[numeric_cols].isna().sum()
    desc["skewness"] = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    desc["kurtosis"] = df[numeric_cols].apply(lambda x: kurtosis(x.dropna()))
    desc.to_csv(os.path.join(OUTPUT_DIR, "tables", f"{name}_descriptive_statistics.csv"))


def plot_correlation(df, name, numeric_cols):
    if len(numeric_cols) > 80:
        selected = numeric_cols[:80]
    else:
        selected = numeric_cols

    corr = df[selected].corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Heatmap: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{name}_correlation_heatmap.png"), dpi=300)
    plt.close()

    corr.to_csv(os.path.join(OUTPUT_DIR, "tables", f"{name}_correlation_matrix.csv"))


def plot_time_series_samples(df, name, numeric_cols, label_col=None, max_features=12):
    selected = numeric_cols[:max_features]

    for col in selected:
        plt.figure(figsize=(16, 4))
        plt.plot(df[col].values, linewidth=0.7)

        if label_col and label_col in df.columns:
            attack_idx = np.where(df[label_col].values == 1)[0]
            if len(attack_idx) > 0:
                plt.scatter(
                    attack_idx,
                    df[col].values[attack_idx],
                    s=3,
                    label="Attack points"
                )
                plt.legend()

        plt.title(f"{name}: Time Series of {col}")
        plt.xlabel("Time index")
        plt.ylabel(col)
        plt.tight_layout()
        safe_col = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{name}_timeseries_{safe_col}.png"), dpi=300)
        plt.close()


def pca_visualization(df, name, numeric_cols, label_col=None, sample_size=50000):
    temp = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) > sample_size:
        temp = temp.sample(sample_size, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(temp)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    if label_col and label_col in df.columns:
        y = df.loc[temp.index, label_col]
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=4, alpha=0.6)
        plt.colorbar(label="0 = Normal, 1 = Attack")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=4, alpha=0.6)

    plt.title(f"PCA Visualization: {name}")
    plt.xlabel(f"PC1")
    plt.ylabel(f"PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", f"{name}_pca.png"), dpi=300)
    plt.close()

    explained = {
        "PC1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "PC2_explained_variance": float(pca.explained_variance_ratio_[1]),
        "Total_2D_explained_variance": float(pca.explained_variance_ratio_.sum())
    }

    with open(os.path.join(OUTPUT_DIR, f"{name}_pca_explained_variance.json"), "w") as f:
        json.dump(explained, f, indent=4)


def baseline_unsupervised_detection(df, name, numeric_cols, label_col):
    if label_col is None:
        print("No label column found. Skipping baseline detection.")
        return

    temp = df[numeric_cols + [label_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) > 100000:
        temp = temp.sample(100000, random_state=42)

    X = temp[numeric_cols].values
    y = temp[label_col].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        n_estimators=200,
        contamination=max(0.001, min(0.3, y.mean())),
        random_state=42,
        n_jobs=-1
    )

    pred_raw = clf.fit_predict(X_scaled)
    pred = np.where(pred_raw == -1, 1, 0)

    report = classification_report(y, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred)

    with open(os.path.join(OUTPUT_DIR, "tables", f"{name}_isolation_forest_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, "tables", f"{name}_isolation_forest_confusion_matrix.csv"))

    print("\nIsolation Forest baseline:")
    print(classification_report(y, pred, zero_division=0))
    print(cm)


def generate_swat_graph_template(numeric_cols):
    graph = {
        "description": "Initial SWaT graph template. Modify edges using actual SWaT topology if needed.",
        "nodes": numeric_cols,
        "edges": []
    }

    stage_groups = {}
    for col in numeric_cols:
        prefix = col.split("_")[0] if "_" in col else col[:4]
        stage_groups.setdefault(prefix, []).append(col)

    for _, cols in stage_groups.items():
        for i in range(len(cols) - 1):
            graph["edges"].append([cols[i], cols[i + 1]])

    with open(os.path.join(OUTPUT_DIR, "swat_initial_graph_template.json"), "w") as f:
        json.dump(graph, f, indent=4)

    print(f"Graph template saved with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")


def analyze_file(path, name):
    df = load_csv_safely(path)
    df = clean_column_names(df)

    label_original = detect_label_column(df)
    df, label_col = normalize_labels(df, label_original)

    label_cols = []
    if label_original:
        label_cols.append(label_original)
    if label_col:
        label_cols.append(label_col)

    numeric_cols = get_numeric_features(df, label_cols)

    print(f"\nDetected original label column: {label_original}")
    print(f"Detected binary label column: {label_col}")
    print(f"Numeric feature count: {len(numeric_cols)}")

    save_basic_info(df, name, label_col)
    plot_missing_values(df, name)
    plot_label_distribution(df, name, label_col)
    save_descriptive_statistics(df, name, numeric_cols)
    plot_correlation(df, name, numeric_cols)
    plot_time_series_samples(df, name, numeric_cols, label_col)
    pca_visualization(df, name, numeric_cols, label_col)
    baseline_unsupervised_detection(df, name, numeric_cols, label_col)
    generate_swat_graph_template(numeric_cols)

    df.head(1000).to_csv(os.path.join(OUTPUT_DIR, "tables", f"{name}_head_1000.csv"), index=False)

    return {
        "name": name,
        "shape": df.shape,
        "label_original": label_original,
        "label_col": label_col,
        "numeric_cols": numeric_cols
    }


def main():
    results = {}

    if os.path.exists(MERGED_FILE):
        results["merged"] = analyze_file(MERGED_FILE, "merged")
    else:
        print(f"Missing file: {MERGED_FILE}")

    if os.path.exists(NORMAL_FILE):
        results["normal"] = analyze_file(NORMAL_FILE, "normal")
    else:
        print(f"Missing file: {NORMAL_FILE}")

    if os.path.exists(ATTACK_FILE):
        results["attack"] = analyze_file(ATTACK_FILE, "attack")
    else:
        print(f"Missing file: {ATTACK_FILE}")

    summary = {}
    for key, value in results.items():
        summary[key] = {
            "shape": value["shape"],
            "label_original": value["label_original"],
            "label_col": value["label_col"],
            "numeric_feature_count": len(value["numeric_cols"]),
            "first_20_numeric_features": value["numeric_cols"][:20]
        }

    with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nAnalysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()