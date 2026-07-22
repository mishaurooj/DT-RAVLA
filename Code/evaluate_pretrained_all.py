from __future__ import annotations

"""
DT-RAVLA strict pretrained evaluation pipeline.

Purpose
-------
This script evaluates existing pretrained DT-RAVLA checkpoints and saved
scikit-learn baselines without retraining them. It recreates the original
preprocessing and temporal splits, calibrates thresholds on validation data
only, evaluates in-domain performance, cross-process transfer, robustness,
efficiency, and optional true ablation checkpoints, then exports CSV, LaTeX,
and publication-ready PDF figures.

Expected files
--------------
D:/other/DT-RAVLA/Results/models/
    SWaT_DT_RAVLA_seed42.pt
    SWaT_DT_RAVLA_seed52.pt
    SWaT_DT_RAVLA_seed62.pt
    SWaT_DT_RAVLA_seed72.pt
    SWaT_DT_RAVLA_seed82.pt
    WADI_DT_RAVLA_seed42.pt
    WADI_DT_RAVLA_seed52.pt
    WADI_DT_RAVLA_seed62.pt
    WADI_DT_RAVLA_seed72.pt
    WADI_DT_RAVLA_seed82.pt
    SWaT_LogisticRegression.joblib
    SWaT_RandomForest.joblib

Optional true ablation checkpoints
----------------------------------
    SWaT_No_Transformer_seed42.pt
    SWaT_Mean_Pooling_seed42.pt
    SWaT_No_Class_Weight_seed42.pt
    SWaT_No_Robust_Scaling_seed42.pt
    SWaT_No_Tokenisation_seed42.pt

Important
---------
A true ablation must load a checkpoint trained with that ablation. This script
does not disable trained components only at test time and call the result an
ablation. Missing ablation checkpoints are logged and excluded.

Run
---
python evaluate_pretrained_all.py
"""

import copy
import json
import math
import random
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(r"D:\other\DT-RAVLA")
SWAT_PATH = ROOT / "Dataset" / "SWaT" / "merged.csv"
WADI_NORMAL_PATH = ROOT / "Dataset" / "WADI" / "WADI_14days_new.csv"
WADI_ATTACK_PATH = ROOT / "Dataset" / "WADI" / "WADI_attackdataLABLE.csv"

SOURCE_RESULTS = ROOT / "Results"
MODELS_DIR = SOURCE_RESULTS / "models"
OUTPUT_DIR = SOURCE_RESULTS / "Reevaluation"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"

WADI_ATTACK_HEADER = 1
WADI_LABEL_COLUMN = "Attack LABLE (1:No Attack, -1:Attack)"

SEEDS = [42, 52, 62, 72, 82]
WINDOW = 64
STRIDE = 8
MAX_WINDOWS = 50_000
BATCH_SIZE = 256
TOKEN_DIM = 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD_OBJECTIVE = "mcc"
MAX_VALIDATION_FPR = 0.01
NORMAL_QUANTILE = 0.995

ROBUSTNESS_CONDITIONS = [
    "Clean",
    "Gaussian_5pct",
    "Gaussian_10pct",
    "Missing_10pct",
    "Missing_20pct",
    "Frozen_10pct",
    "Bias_10pct",
    "Temporal_Jitter",
]

ABLATION_CONFIGS = {
    "Full": "DT_RAVLA",
    "No Transformer": "No_Transformer",
    "Mean Pooling": "Mean_Pooling",
    "No Class Weight": "No_Class_Weight",
    "No Robust Scaling": "No_Robust_Scaling",
    "No Tokenisation": "No_Tokenisation",
}


# =============================================================================
# Reproducibility and I/O
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dirs() -> None:
    for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, DIAGNOSTIC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def canon(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_csv(path: Path, header: int = 0) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in [None, "utf-8", "latin1"]:
        try:
            kwargs = {"header": header, "low_memory": False}
            if encoding is not None:
                kwargs["encoding"] = encoding
            return pd.read_csv(path, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def normalize_label(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    values = set(numeric.dropna().unique())

    if values and values.issubset({-1, 1}):
        return numeric.map({1: 0, -1: 1})
    if values and values.issubset({0, 1}):
        return numeric

    return raw.map(
        {
            "normal": 0,
            "benign": 0,
            "no attack": 0,
            "attack": 1,
            "anomaly": 1,
            "false": 0,
            "true": 1,
            "n": 0,
            "a": 1,
        }
    )


def infer_features(
    dataframe: pd.DataFrame,
    ignored_columns: Iterable[str],
    minimum_valid_ratio: float = 0.50,
) -> Tuple[List[str], List[Dict[str, object]]]:
    ignored = {canon(column) for column in ignored_columns}
    retained: List[str] = []
    dropped: List[Dict[str, object]] = []

    for column in dataframe.columns:
        if canon(column) in ignored:
            continue

        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        valid_ratio = float(numeric.notna().mean())
        unique_count = int(numeric.nunique(dropna=True))

        if valid_ratio >= minimum_valid_ratio and unique_count > 1:
            dataframe[column] = numeric
            retained.append(column)
        else:
            dropped.append(
                {
                    "column": column,
                    "valid_ratio": valid_ratio,
                    "unique_count": unique_count,
                }
            )

    return retained, dropped


# =============================================================================
# Dataset preparation
# =============================================================================

def load_swat() -> Tuple[pd.DataFrame, List[str]]:
    dataframe = read_csv(SWAT_PATH)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    label_candidates = [
        column
        for column in dataframe.columns
        if canon(column) in {"normal_attack", "label", "attack_label", "binary_label"}
    ]
    if not label_candidates:
        raise RuntimeError("SWaT label column was not found.")

    source_label = label_candidates[0]
    dataframe["binary_label"] = normalize_label(dataframe[source_label])
    dataframe = dataframe[dataframe["binary_label"].notna()].reset_index(drop=True)
    dataframe["binary_label"] = dataframe["binary_label"].astype(int)

    features, dropped = infer_features(
        dataframe,
        {
            source_label,
            "binary_label",
            "timestamp",
            "date",
            "time",
            "row",
            "index",
        },
    )

    if not features:
        raise RuntimeError("No usable SWaT process variables were found.")

    print(
        f"SWaT: rows={len(dataframe):,}, features={len(features)}, "
        f"normal={(dataframe.binary_label == 0).sum():,}, "
        f"attack={(dataframe.binary_label == 1).sum():,}"
    )
    return dataframe, features


def load_wadi() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    normal = read_csv(WADI_NORMAL_PATH, header=0)
    attack = read_csv(WADI_ATTACK_PATH, header=WADI_ATTACK_HEADER)

    normal.columns = [str(column).strip() for column in normal.columns]
    attack.columns = [str(column).strip() for column in attack.columns]

    if WADI_LABEL_COLUMN not in attack.columns:
        raise RuntimeError(f"Missing WADI label column: {WADI_LABEL_COLUMN!r}")

    attack["binary_label"] = normalize_label(attack[WADI_LABEL_COLUMN])
    attack = attack[attack["binary_label"].notna()].reset_index(drop=True)
    attack["binary_label"] = attack["binary_label"].astype(int)

    normal_features, _ = infer_features(
        normal,
        {"timestamp", "date", "time", "row", "index"},
    )
    attack_features, _ = infer_features(
        attack,
        {
            WADI_LABEL_COLUMN,
            "binary_label",
            "timestamp",
            "date",
            "time",
            "row",
            "index",
        },
    )

    normal_map = {canon(column): column for column in normal_features}
    attack_map = {canon(column): column for column in attack_features}
    common = sorted(set(normal_map).intersection(attack_map))

    if len(common) < 20:
        raise RuntimeError(f"Only {len(common)} WADI variables align across files.")

    normal_aligned = pd.DataFrame(
        {
            key: pd.to_numeric(normal[normal_map[key]], errors="coerce")
            for key in common
        }
    )
    attack_aligned = pd.DataFrame(
        {
            key: pd.to_numeric(attack[attack_map[key]], errors="coerce")
            for key in common
        }
    )
    attack_aligned["binary_label"] = attack["binary_label"].to_numpy()

    usable: List[str] = []
    for key in common:
        combined = pd.concat(
            [normal_aligned[key], attack_aligned[key]],
            ignore_index=True,
        )
        if combined.notna().mean() >= 0.60 and combined.nunique(dropna=True) > 1:
            usable.append(key)

    if len(usable) < 20:
        raise RuntimeError(f"Only {len(usable)} usable aligned WADI variables.")

    print(
        f"WADI: normal rows={len(normal_aligned):,}, "
        f"attack-file rows={len(attack_aligned):,}, aligned features={len(usable)}"
    )
    return normal_aligned, attack_aligned, usable


def chronological_split(
    dataframe: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count = len(dataframe)
    train_end = int(count * train_fraction)
    validation_end = int(count * (train_fraction + validation_fraction))
    return (
        dataframe.iloc[:train_end].copy(),
        dataframe.iloc[train_end:validation_end].copy(),
        dataframe.iloc[validation_end:].copy(),
    )


def class_temporal_split(
    dataframe: pd.DataFrame,
    label_value: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = (
        dataframe[dataframe["binary_label"] == label_value]
        .copy()
        .reset_index(drop=True)
    )
    return chronological_split(subset, train_fraction, validation_fraction)


def combine_swat_splits(
    dataframe: pd.DataFrame,
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame],
    pd.DataFrame,
]:
    normal_train, normal_val, normal_test = class_temporal_split(dataframe, 0)
    attack_train, attack_val, attack_test = class_temporal_split(dataframe, 1)

    fit_dataframe = pd.concat([normal_train, attack_train], ignore_index=True)
    return (
        (normal_train, attack_train),
        (normal_val, attack_val),
        (normal_test, attack_test),
        fit_dataframe,
    )


class Preprocessor:
    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.features: List[str] = []
        self.medians = pd.Series(dtype=float)

    def fit(self, dataframe: pd.DataFrame, features: Sequence[str]) -> "Preprocessor":
        values = dataframe[list(features)].replace([np.inf, -np.inf], np.nan)
        medians = values.median(numeric_only=True).reindex(features).fillna(0.0)
        values = values.fillna(medians).fillna(0.0)

        variable_mask = values.nunique(dropna=True) > 1
        self.features = variable_mask[variable_mask].index.tolist()
        self.medians = medians.reindex(self.features).fillna(0.0)

        if not self.features:
            raise RuntimeError("No variable features remain after preprocessing.")

        matrix = np.nan_to_num(
            values[self.features].to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.scaler.fit(matrix)
        return self

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        values = (
            dataframe[self.features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
            .fillna(0.0)
        )
        matrix = np.nan_to_num(
            values.to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.clip(
            self.scaler.transform(matrix).astype(np.float32),
            -10.0,
            10.0,
        )


def semantic_tokens(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(
        np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
        -10.0,
        10.0,
    )
    quantiles = np.quantile(
        matrix,
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        axis=1,
    ).T

    adjacent = (
        np.mean(np.abs(np.diff(matrix, axis=1)), axis=1)
        if matrix.shape[1] > 1
        else np.zeros(len(matrix), dtype=np.float32)
    )

    output = np.column_stack(
        [
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.min(axis=1),
            matrix.max(axis=1),
            np.median(matrix, axis=1),
            np.mean(np.abs(matrix), axis=1),
            np.sqrt(np.mean(matrix ** 2, axis=1)),
            np.mean(matrix > 0, axis=1),
            np.mean(matrix < 0, axis=1),
            np.mean(matrix == 0, axis=1),
            quantiles,
            adjacent,
        ]
    )
    return np.nan_to_num(output).astype(np.float32)


def labelled_windows(
    token_matrix: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    window_labels: List[int] = []

    for end in range(WINDOW, len(token_matrix) + 1, STRIDE):
        windows.append(token_matrix[end - WINDOW : end])
        window_labels.append(int(labels[end - 1]))
        if len(windows) >= MAX_WINDOWS:
            break

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(window_labels, dtype=np.int64),
    )


def normal_windows(token_matrix: np.ndarray) -> np.ndarray:
    windows: List[np.ndarray] = []
    for end in range(WINDOW, len(token_matrix) + 1, STRIDE):
        windows.append(token_matrix[end - WINDOW : end])
        if len(windows) >= MAX_WINDOWS:
            break
    return np.asarray(windows, dtype=np.float32)


def windows_from_class_parts(
    preprocessor: Preprocessor,
    normal_dataframe: pd.DataFrame,
    attack_dataframe: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    normal_tokens = semantic_tokens(preprocessor.transform(normal_dataframe))
    attack_tokens = semantic_tokens(preprocessor.transform(attack_dataframe))

    normal_x, normal_y = labelled_windows(
        normal_tokens,
        np.zeros(len(normal_dataframe), dtype=np.int64),
    )
    attack_x, attack_y = labelled_windows(
        attack_tokens,
        np.ones(len(attack_dataframe), dtype=np.int64),
    )

    x = np.concatenate([normal_x, attack_x], axis=0)
    y = np.concatenate([normal_y, attack_y], axis=0)
    return x, y


# =============================================================================
# Exact model architectures used by the uploaded training code
# =============================================================================

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 128
        self.inp = nn.Sequential(
            nn.Linear(TOKEN_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=3)
        self.pool = nn.Sequential(nn.Linear(hidden, 1), nn.Softmax(dim=1))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.temporal(self.inp(x))
        weights = self.pool(hidden)
        pooled = (weights * hidden).sum(dim=1)
        return pooled, hidden, weights.squeeze(-1)


class SupModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled, hidden, attention = self.encoder(x)
        logits = torch.clamp(self.head(pooled).squeeze(-1), -20.0, 20.0)
        return {
            "logits": logits,
            "embedding": pooled,
            "attention": attention,
        }


class OCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, TOKEN_DIM),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled, hidden, attention = self.encoder(x)
        return {
            "reconstruction": self.decoder(hidden),
            "embedding": pooled,
            "attention": attention,
        }


class MeanPoolingSupModel(nn.Module):
    """
    Exact architecture for a separately trained mean-pooling ablation.
    Use only with a Mean_Pooling checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.encoder.pool = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder.temporal(self.encoder.inp(x))
        pooled = hidden.mean(dim=1)
        logits = torch.clamp(self.head(pooled).squeeze(-1), -20.0, 20.0)
        return {"logits": logits, "embedding": pooled}


class NoTransformerSupModel(nn.Module):
    """
    Exact architecture for a separately trained no-transformer ablation.
    Use only with a No_Transformer checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.encoder.temporal = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled, hidden, attention = self.encoder(x)
        logits = torch.clamp(self.head(pooled).squeeze(-1), -20.0, 20.0)
        return {
            "logits": logits,
            "embedding": pooled,
            "attention": attention,
        }


def model_for_configuration(configuration: str) -> nn.Module:
    if configuration in {
        "DT_RAVLA",
        "No_Class_Weight",
        "No_Robust_Scaling",
        "No_Tokenisation",
    }:
        return SupModel()
    if configuration == "No_Transformer":
        return NoTransformerSupModel()
    if configuration == "Mean_Pooling":
        return MeanPoolingSupModel()
    raise ValueError(f"Unsupported configuration: {configuration}")


def load_state_dict_strict(model: nn.Module, checkpoint: Path) -> nn.Module:
    state = torch.load(checkpoint, map_location=DEVICE)

    if isinstance(state, nn.Module):
        loaded = state
    else:
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise TypeError(f"Unsupported checkpoint format: {checkpoint}")

        model.load_state_dict(state_dict, strict=True)
        loaded = model

    loaded.to(DEVICE)
    loaded.eval()
    return loaded


# =============================================================================
# Inference
# =============================================================================

def make_loader(x: np.ndarray) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


@torch.inference_mode()
def supervised_probabilities(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    probabilities: List[np.ndarray] = []

    for (batch,) in make_loader(x):
        logits = model(batch.to(DEVICE))["logits"]
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities).astype(np.float64)


@torch.inference_mode()
def reconstruction_scores(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []

    for (batch,) in make_loader(x):
        batch = batch.to(DEVICE)
        reconstruction = model(batch)["reconstruction"]
        error = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
        scores.append(error.cpu().numpy())

    return np.concatenate(scores).astype(np.float64)


@torch.inference_mode()
def inference_latency_ms(
    model: nn.Module,
    sample: np.ndarray,
    repetitions: int = 100,
) -> float:
    tensor = torch.from_numpy(sample[:1]).to(DEVICE)
    model.eval()

    for _ in range(10):
        _ = model(tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repetitions):
        _ = model(tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / repetitions


# =============================================================================
# Thresholds and metrics
# =============================================================================

def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        raise ValueError("Validation scores contain no finite values.")

    quantiles = np.linspace(0.001, 0.999, 999)
    values = np.unique(np.quantile(finite, quantiles))
    return np.concatenate(
        [
            [np.nextafter(finite.min(), -np.inf)],
            values,
            [np.nextafter(finite.max(), np.inf)],
        ]
    )


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    objective: str = THRESHOLD_OBJECTIVE,
    max_fpr: Optional[float] = MAX_VALIDATION_FPR,
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        return float(np.quantile(scores, NORMAL_QUANTILE))

    best_threshold = float(np.median(scores))
    best_key = (-np.inf, -np.inf, np.inf)

    for threshold in candidate_thresholds(scores):
        prediction = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            labels,
            prediction,
            labels=[0, 1],
        ).ravel()

        fpr = fp / max(tn + fp, 1)
        recall = tp / max(tp + fn, 1)

        if max_fpr is not None and fpr > max_fpr:
            continue

        if objective == "mcc":
            primary = matthews_corrcoef(labels, prediction)
        elif objective == "f2":
            primary = fbeta_score(labels, prediction, beta=2, zero_division=0)
        elif objective == "balanced_accuracy":
            primary = balanced_accuracy_score(labels, prediction)
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")

        key = (primary, recall, -fpr)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)

    if not np.isfinite(best_key[0]):
        return select_threshold(labels, scores, objective=objective, max_fpr=None)

    return best_threshold


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (
            (probabilities >= lower)
            & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        )
        if mask.any():
            value += mask.mean() * abs(
                labels[mask].mean() - probabilities[mask].mean()
            )

    return float(value)


def metric_pack(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    if probabilities is None:
        lower = np.nanmin(scores)
        upper = np.nanmax(scores)
        probabilities = (scores - lower) / max(upper - lower, 1e-12)

    probabilities = np.clip(
        np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )

    return {
        "Accuracy": accuracy_score(labels, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(labels, prediction),
        "Precision": precision_score(labels, prediction, zero_division=0),
        "Recall_DR": recall_score(labels, prediction, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(labels, prediction, zero_division=0),
        "F2": fbeta_score(labels, prediction, beta=2, zero_division=0),
        "AUROC": (
            roc_auc_score(labels, scores)
            if len(np.unique(labels)) == 2
            else np.nan
        ),
        "AUPRC": (
            average_precision_score(labels, scores)
            if len(np.unique(labels)) == 2
            else np.nan
        ),
        "MCC": matthews_corrcoef(labels, prediction),
        "Cohen_Kappa": cohen_kappa_score(labels, prediction),
        "FPR_FAR": fpr,
        "FNR_MissRate": fnr,
        "NPV": npv,
        "Brier": brier_score_loss(labels, probabilities),
        "ECE": expected_calibration_error(labels, probabilities),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Threshold": float(threshold),
        "Collapsed_All_Attack": bool(prediction.all()),
        "Collapsed_All_Normal": bool((prediction == 0).all()),
    }


def aggregate_mean_std(
    raw: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = {"Seed"}
    numeric_columns = [
        column
        for column in raw.select_dtypes(include=np.number).columns
        if column not in excluded
    ]

    grouped = raw.groupby(list(group_columns), dropna=False)
    mean = grouped[numeric_columns].mean().add_suffix("_Mean")
    std = grouped[numeric_columns].std(ddof=1).fillna(0.0).add_suffix("_Std")
    return pd.concat([mean, std], axis=1).reset_index()


# =============================================================================
# Corruptions
# =============================================================================

def corrupt_windows(
    x: np.ndarray,
    condition: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corrupted = x.copy()

    if condition == "Clean":
        return corrupted

    if condition.startswith("Gaussian_"):
        severity = float(re.search(r"(\d+)", condition).group(1)) / 100.0
        scale = np.std(corrupted, axis=(0, 1), keepdims=True)
        scale = np.where(scale > 1e-8, scale, 1.0)
        noise = rng.normal(0.0, severity, size=corrupted.shape).astype(np.float32)
        return corrupted + noise * scale

    if condition.startswith("Missing_"):
        severity = float(re.search(r"(\d+)", condition).group(1)) / 100.0
        mask = rng.random(corrupted.shape) < severity
        corrupted[mask] = 0.0
        return corrupted

    if condition.startswith("Frozen_"):
        severity = float(re.search(r"(\d+)", condition).group(1)) / 100.0
        count = max(1, int(round(corrupted.shape[2] * severity)))
        dimensions = rng.choice(corrupted.shape[2], count, replace=False)
        corrupted[:, 1:, dimensions] = corrupted[:, :1, dimensions]
        return corrupted

    if condition.startswith("Bias_"):
        severity = float(re.search(r"(\d+)", condition).group(1)) / 100.0
        count = max(1, int(round(corrupted.shape[2] * severity)))
        dimensions = rng.choice(corrupted.shape[2], count, replace=False)
        corrupted[:, :, dimensions] += 0.5
        return corrupted

    if condition == "Temporal_Jitter":
        for index in range(len(corrupted)):
            if rng.random() < 0.50:
                permutation = np.arange(corrupted.shape[1])
                swap_at = rng.integers(0, corrupted.shape[1] - 1)
                permutation[swap_at], permutation[swap_at + 1] = (
                    permutation[swap_at + 1],
                    permutation[swap_at],
                )
                corrupted[index] = corrupted[index, permutation]
        return corrupted

    raise ValueError(f"Unknown corruption condition: {condition}")


# =============================================================================
# LaTeX export
# =============================================================================

def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def save_latex_table(
    summary: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    caption: str,
    label: str,
    path: Path,
) -> None:
    alignment = "l" * len(group_columns) + "c" * len(metrics)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(
            [column.replace("_", " ") for column in group_columns]
            + [metric.replace("_", " ") for metric in metrics]
        )
        + r" \\",
        r"\hline",
    ]

    for _, row in summary.iterrows():
        values = [str(row[column]) for column in group_columns]
        values.extend(
            format_mean_std(row[f"{metric}_Mean"], row[f"{metric}_Std"])
            for metric in metrics
        )
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Figures
# =============================================================================

def plot_bar_summary(
    summary: pd.DataFrame,
    label_column: str,
    metric: str,
    path: Path,
) -> None:
    labels = summary[label_column].astype(str).tolist()
    means = summary[f"{metric}_Mean"].to_numpy()
    errors = summary[f"{metric}_Std"].to_numpy()

    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=errors, capsize=4)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_ylim(0.0, 1.02)
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    labels: np.ndarray,
    scores: np.ndarray,
    prefix: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)

    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    axis.plot(fpr, tpr)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_roc.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    axis.plot(recall, precision)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_pr.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_robustness(
    summary: pd.DataFrame,
    dataset: str,
    path: Path,
) -> None:
    subset = summary[summary["Dataset"] == dataset].copy()
    order = [condition for condition in ROBUSTNESS_CONDITIONS
             if condition in set(subset["Condition"])]
    subset = subset.set_index("Condition").reindex(order)

    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(order))

    for metric in ["Recall_DR", "Precision", "FPR_FAR", "MCC"]:
        axis.plot(
            x,
            subset[f"{metric}_Mean"],
            marker="o",
            label=metric.replace("_", " "),
        )

    axis.set_xticks(x, order, rotation=25, ha="right")
    axis.set_ylim(-0.1, 1.02)
    axis.set_ylabel("Score")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Statistical tests
# =============================================================================

def paired_significance(
    raw: pd.DataFrame,
    metric: str,
    reference_configuration: str = "Full",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    reference = raw[raw["Configuration"] == reference_configuration][
        ["Seed", metric]
    ].rename(columns={metric: "Reference"})

    for configuration in sorted(set(raw["Configuration"]) - {reference_configuration}):
        comparison = raw[raw["Configuration"] == configuration][
            ["Seed", metric]
        ].rename(columns={metric: "Comparison"})

        merged = reference.merge(comparison, on="Seed", how="inner")
        if len(merged) < 2:
            continue

        difference = merged["Reference"] - merged["Comparison"]
        t_statistic, t_p = stats.ttest_rel(
            merged["Reference"],
            merged["Comparison"],
            nan_policy="omit",
        )

        try:
            w_statistic, w_p = stats.wilcoxon(
                merged["Reference"],
                merged["Comparison"],
                zero_method="wilcox",
            )
        except ValueError:
            w_statistic, w_p = np.nan, np.nan

        standard_deviation = difference.std(ddof=1)
        cohen_d = (
            difference.mean() / standard_deviation
            if standard_deviation > 0
            else np.nan
        )

        rows.append(
            {
                "Reference": reference_configuration,
                "Comparison": configuration,
                "Metric": metric,
                "Pairs": len(merged),
                "Mean_Difference": difference.mean(),
                "Paired_t": t_statistic,
                "Paired_t_p": t_p,
                "Wilcoxon_W": w_statistic,
                "Wilcoxon_p": w_p,
                "Cohen_dz": cohen_d,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main evaluation
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    prepare_output_dirs()
    print(f"Device: {DEVICE}")
    print(f"Loading pretrained models from: {MODELS_DIR}")
    print(f"Saving reevaluation outputs to: {OUTPUT_DIR}")

    swat, swat_features = load_swat()
    wadi_normal, wadi_attack, wadi_features = load_wadi()

    # -------------------------------------------------------------------------
    # Frozen SWaT split and preprocessing
    # -------------------------------------------------------------------------
    swat_train_parts, swat_val_parts, swat_test_parts, swat_fit = (
        combine_swat_splits(swat)
    )
    swat_preprocessor = Preprocessor().fit(swat_fit, swat_features)

    swat_train = windows_from_class_parts(
        swat_preprocessor,
        swat_train_parts[0],
        swat_train_parts[1],
    )
    swat_val = windows_from_class_parts(
        swat_preprocessor,
        swat_val_parts[0],
        swat_val_parts[1],
    )
    swat_test = windows_from_class_parts(
        swat_preprocessor,
        swat_test_parts[0],
        swat_test_parts[1],
    )

    # -------------------------------------------------------------------------
    # Frozen WADI split and preprocessing
    # -------------------------------------------------------------------------
    wadi_normal_train, wadi_normal_val, _ = chronological_split(
        wadi_normal,
        0.70,
        0.15,
    )
    _, wadi_attack_val, wadi_attack_test = chronological_split(
        wadi_attack,
        0.20,
        0.20,
    )

    wadi_preprocessor = Preprocessor().fit(
        wadi_normal_train,
        wadi_features,
    )

    wadi_train_normal = normal_windows(
        semantic_tokens(wadi_preprocessor.transform(wadi_normal_train))
    )
    wadi_val_normal = normal_windows(
        semantic_tokens(wadi_preprocessor.transform(wadi_normal_val))
    )

    wadi_val = labelled_windows(
        semantic_tokens(wadi_preprocessor.transform(wadi_attack_val)),
        wadi_attack_val["binary_label"].to_numpy(np.int64),
    )
    wadi_test = labelled_windows(
        semantic_tokens(wadi_preprocessor.transform(wadi_attack_test)),
        wadi_attack_test["binary_label"].to_numpy(np.int64),
    )

    protocol = {
        "Device": str(DEVICE),
        "Seeds": SEEDS,
        "Window": WINDOW,
        "Stride": STRIDE,
        "Token_Dimension": TOKEN_DIM,
        "SWaT_Features": len(swat_preprocessor.features),
        "WADI_Features": len(wadi_preprocessor.features),
        "SWaT_Windows": {
            "Train": len(swat_train[1]),
            "Validation": len(swat_val[1]),
            "Test": len(swat_test[1]),
        },
        "WADI_Windows": {
            "Normal_Train": len(wadi_train_normal),
            "Normal_Validation": len(wadi_val_normal),
            "Labelled_Validation": len(wadi_val[1]),
            "Test": len(wadi_test[1]),
        },
        "Threshold_Objective": THRESHOLD_OBJECTIVE,
        "Maximum_Validation_FPR": MAX_VALIDATION_FPR,
        "Normal_Quantile": NORMAL_QUANTILE,
    }
    (DIAGNOSTIC_DIR / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    benchmark_rows: List[Dict[str, object]] = []
    transfer_rows: List[Dict[str, object]] = []
    robustness_rows: List[Dict[str, object]] = []
    efficiency_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []

    swat_seed_outputs: Dict[int, Dict[str, object]] = {}

    # -------------------------------------------------------------------------
    # Existing scikit-learn baselines
    # -------------------------------------------------------------------------
    baseline_features_val = np.concatenate(
        [
            swat_val[0].mean(axis=1),
            swat_val[0].std(axis=1),
            swat_val[0].min(axis=1),
            swat_val[0].max(axis=1),
        ],
        axis=1,
    )
    baseline_features_test = np.concatenate(
        [
            swat_test[0].mean(axis=1),
            swat_test[0].std(axis=1),
            swat_test[0].min(axis=1),
            swat_test[0].max(axis=1),
        ],
        axis=1,
    )

    for model_name in ["LogisticRegression", "RandomForest"]:
        checkpoint = MODELS_DIR / f"SWaT_{model_name}.joblib"
        if not checkpoint.exists():
            print(f"SKIP missing baseline: {checkpoint.name}")
            continue

        model = joblib.load(checkpoint)
        validation_probability = model.predict_proba(baseline_features_val)[:, 1]
        threshold = select_threshold(
            swat_val[1],
            validation_probability,
        )
        test_probability = model.predict_proba(baseline_features_test)[:, 1]
        metrics = metric_pack(
            swat_test[1],
            test_probability,
            threshold,
            test_probability,
        )
        metrics.update(
            {
                "Dataset": "SWaT",
                "Model": model_name,
                "Seed": -1,
                "Protocol": "Deterministic pretrained baseline",
            }
        )
        benchmark_rows.append(metrics)

    # -------------------------------------------------------------------------
    # Full pretrained models
    # -------------------------------------------------------------------------
    for seed in SEEDS:
        seed_all(seed)

        # Supervised SWaT
        swat_checkpoint = MODELS_DIR / f"SWaT_DT_RAVLA_seed{seed}.pt"
        if swat_checkpoint.exists():
            model = load_state_dict_strict(SupModel(), swat_checkpoint)

            validation_probability = supervised_probabilities(model, swat_val[0])
            supervised_threshold = select_threshold(
                swat_val[1],
                validation_probability,
            )
            test_probability = supervised_probabilities(model, swat_test[0])

            metrics = metric_pack(
                swat_test[1],
                test_probability,
                supervised_threshold,
                test_probability,
            )
            metrics.update(
                {
                    "Dataset": "SWaT",
                    "Model": "DT-RAVLA-Supervised",
                    "Seed": seed,
                    "Protocol": "In-domain",
                }
            )
            benchmark_rows.append(metrics)

            latency = inference_latency_ms(model, swat_test[0])
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            efficiency_rows.append(
                {
                    "Dataset": "SWaT",
                    "Model": "DT-RAVLA-Supervised",
                    "Seed": seed,
                    "Parameters": parameter_count,
                    "Model_Size_MB": swat_checkpoint.stat().st_size / (1024 ** 2),
                    "Latency_ms_per_window": latency,
                    "Windows_per_second": 1000.0 / latency,
                }
            )

            swat_seed_outputs[seed] = {
                "model": model,
                "validation_probability": validation_probability,
                "test_probability": test_probability,
                "threshold": supervised_threshold,
            }

            # ROC and PR for the first seed only
            if seed == SEEDS[0]:
                plot_roc_pr(
                    swat_test[1],
                    test_probability,
                    FIGURE_DIR / "swat_dt_ravla",
                )

            # Cross-process transfer
            target_probability_val = supervised_probabilities(model, wadi_val[0])
            target_probability_test = supervised_probabilities(model, wadi_test[0])

            # Protocol 1: strict zero-shot, source threshold unchanged
            zero_shot = metric_pack(
                wadi_test[1],
                target_probability_test,
                supervised_threshold,
                target_probability_test,
            )
            zero_shot.update(
                {
                    "Direction": "SWaT_to_WADI",
                    "Protocol": "Zero-shot source threshold",
                    "Target_Label_Fraction": 0.0,
                    "Seed": seed,
                }
            )
            transfer_rows.append(zero_shot)

            # Protocol 2: target-normal calibration, no attack labels
            target_normal_scores = supervised_probabilities(model, wadi_val_normal)
            normal_calibrated_threshold = float(
                np.quantile(target_normal_scores, NORMAL_QUANTILE)
            )
            normal_calibrated = metric_pack(
                wadi_test[1],
                target_probability_test,
                normal_calibrated_threshold,
                target_probability_test,
            )
            normal_calibrated.update(
                {
                    "Direction": "SWaT_to_WADI",
                    "Protocol": "Target-normal calibrated",
                    "Target_Label_Fraction": 0.0,
                    "Seed": seed,
                }
            )
            transfer_rows.append(normal_calibrated)

            # Protocol 3: 1% few-shot threshold calibration, encoder unchanged
            rng = np.random.default_rng(seed)
            validation_indices = np.arange(len(wadi_val[1]))
            positive_indices = validation_indices[wadi_val[1] == 1]
            negative_indices = validation_indices[wadi_val[1] == 0]

            positive_count = max(1, int(math.ceil(0.01 * len(positive_indices))))
            negative_count = max(1, int(math.ceil(0.01 * len(negative_indices))))

            selected_indices = np.concatenate(
                [
                    rng.choice(
                        positive_indices,
                        size=min(positive_count, len(positive_indices)),
                        replace=False,
                    ),
                    rng.choice(
                        negative_indices,
                        size=min(negative_count, len(negative_indices)),
                        replace=False,
                    ),
                ]
            )

            few_shot_threshold = select_threshold(
                wadi_val[1][selected_indices],
                target_probability_val[selected_indices],
                max_fpr=None,
            )
            few_shot = metric_pack(
                wadi_test[1],
                target_probability_test,
                few_shot_threshold,
                target_probability_test,
            )
            few_shot.update(
                {
                    "Direction": "SWaT_to_WADI",
                    "Protocol": "One-percent few-shot calibration",
                    "Target_Label_Fraction": 0.01,
                    "Seed": seed,
                }
            )
            transfer_rows.append(few_shot)

            # Robustness for the supervised model
            for condition in ROBUSTNESS_CONDITIONS:
                corrupted = corrupt_windows(swat_test[0], condition, seed)
                corrupted_probability = supervised_probabilities(model, corrupted)
                robust_metrics = metric_pack(
                    swat_test[1],
                    corrupted_probability,
                    supervised_threshold,
                    corrupted_probability,
                )
                robust_metrics.update(
                    {
                        "Dataset": "SWaT",
                        "Model": "DT-RAVLA-Supervised",
                        "Condition": condition,
                        "Seed": seed,
                    }
                )
                robustness_rows.append(robust_metrics)
        else:
            print(f"SKIP missing supervised checkpoint: {swat_checkpoint.name}")

        # WADI one-class
        wadi_checkpoint = MODELS_DIR / f"WADI_DT_RAVLA_seed{seed}.pt"
        if wadi_checkpoint.exists():
            one_class_model = load_state_dict_strict(OCModel(), wadi_checkpoint)

            validation_scores = reconstruction_scores(
                one_class_model,
                wadi_val[0],
            )
            one_class_threshold = select_threshold(
                wadi_val[1],
                validation_scores,
            )
            test_scores = reconstruction_scores(
                one_class_model,
                wadi_test[0],
            )

            one_class_metrics = metric_pack(
                wadi_test[1],
                test_scores,
                one_class_threshold,
            )
            one_class_metrics.update(
                {
                    "Dataset": "WADI",
                    "Model": "DT-RAVLA-OneClass",
                    "Seed": seed,
                    "Protocol": "In-domain normal-only",
                }
            )
            benchmark_rows.append(one_class_metrics)

            latency = inference_latency_ms(one_class_model, wadi_test[0])
            parameter_count = sum(
                parameter.numel() for parameter in one_class_model.parameters()
            )
            efficiency_rows.append(
                {
                    "Dataset": "WADI",
                    "Model": "DT-RAVLA-OneClass",
                    "Seed": seed,
                    "Parameters": parameter_count,
                    "Model_Size_MB": wadi_checkpoint.stat().st_size / (1024 ** 2),
                    "Latency_ms_per_window": latency,
                    "Windows_per_second": 1000.0 / latency,
                }
            )

            for condition in ROBUSTNESS_CONDITIONS:
                corrupted = corrupt_windows(wadi_test[0], condition, seed)
                corrupted_scores = reconstruction_scores(
                    one_class_model,
                    corrupted,
                )
                robust_metrics = metric_pack(
                    wadi_test[1],
                    corrupted_scores,
                    one_class_threshold,
                )
                robust_metrics.update(
                    {
                        "Dataset": "WADI",
                        "Model": "DT-RAVLA-OneClass",
                        "Condition": condition,
                        "Seed": seed,
                    }
                )
                robustness_rows.append(robust_metrics)
        else:
            print(f"SKIP missing one-class checkpoint: {wadi_checkpoint.name}")

    # -------------------------------------------------------------------------
    # Optional true ablation checkpoints
    # -------------------------------------------------------------------------
    for display_name, file_configuration in ABLATION_CONFIGS.items():
        for seed in SEEDS:
            checkpoint = MODELS_DIR / f"SWaT_{file_configuration}_seed{seed}.pt"

            if display_name == "Full":
                checkpoint = MODELS_DIR / f"SWaT_DT_RAVLA_seed{seed}.pt"

            if not checkpoint.exists():
                print(f"SKIP missing ablation checkpoint: {checkpoint.name}")
                continue

            ablation_model = load_state_dict_strict(
                model_for_configuration(file_configuration),
                checkpoint,
            )
            validation_probability = supervised_probabilities(
                ablation_model,
                swat_val[0],
            )
            threshold = select_threshold(
                swat_val[1],
                validation_probability,
            )
            test_probability = supervised_probabilities(
                ablation_model,
                swat_test[0],
            )

            metrics = metric_pack(
                swat_test[1],
                test_probability,
                threshold,
                test_probability,
            )
            metrics.update(
                {
                    "Dataset": "SWaT",
                    "Configuration": display_name,
                    "Seed": seed,
                    "Checkpoint": checkpoint.name,
                }
            )
            ablation_rows.append(metrics)

    # -------------------------------------------------------------------------
    # Save benchmark results
    # -------------------------------------------------------------------------
    benchmark_raw = pd.DataFrame(benchmark_rows)
    benchmark_raw.to_csv(
        TABLE_DIR / "Table_2_Benchmark_Raw_All_Seeds.csv",
        index=False,
    )
    benchmark_summary = aggregate_mean_std(
        benchmark_raw,
        ["Dataset", "Model", "Protocol"],
    )
    benchmark_summary.to_csv(
        TABLE_DIR / "Table_2_Benchmark_Mean_Std.csv",
        index=False,
    )
    save_latex_table(
        benchmark_summary,
        ["Dataset", "Model", "Protocol"],
        [
            "Balanced_Accuracy",
            "Precision",
            "Recall_DR",
            "F1",
            "AUPRC",
            "MCC",
        ],
        (
            "Pretrained-model performance under the frozen evaluation protocol. "
            "Values are mean $\\pm$ standard deviation over available seeds."
        ),
        "tab:pretrained_benchmark",
        TABLE_DIR / "Table_2_Benchmark.tex",
    )

    # -------------------------------------------------------------------------
    # Save transfer results
    # -------------------------------------------------------------------------
    transfer_raw = pd.DataFrame(transfer_rows)
    transfer_raw.to_csv(
        TABLE_DIR / "Table_3_Cross_Process_Raw_All_Seeds.csv",
        index=False,
    )
    transfer_summary = aggregate_mean_std(
        transfer_raw,
        ["Direction", "Protocol", "Target_Label_Fraction"],
    )
    transfer_summary.to_csv(
        TABLE_DIR / "Table_3_Cross_Process_Mean_Std.csv",
        index=False,
    )
    save_latex_table(
        transfer_summary,
        ["Direction", "Protocol", "Target_Label_Fraction"],
        [
            "Balanced_Accuracy",
            "Precision",
            "Recall_DR",
            "Specificity",
            "F1",
            "MCC",
        ],
        (
            "Cross-process evaluation using the source threshold, target-normal "
            "calibration, and one-percent labelled target calibration."
        ),
        "tab:cross_process",
        TABLE_DIR / "Table_3_Cross_Process.tex",
    )

    # -------------------------------------------------------------------------
    # Save robustness results
    # -------------------------------------------------------------------------
    robustness_raw = pd.DataFrame(robustness_rows)
    robustness_raw.to_csv(
        TABLE_DIR / "Table_4_Robustness_Raw_All_Seeds.csv",
        index=False,
    )
    robustness_summary = aggregate_mean_std(
        robustness_raw,
        ["Dataset", "Model", "Condition"],
    )
    robustness_summary.to_csv(
        TABLE_DIR / "Table_4_Robustness_Mean_Std.csv",
        index=False,
    )
    save_latex_table(
        robustness_summary,
        ["Dataset", "Model", "Condition"],
        ["Precision", "Recall_DR", "FPR_FAR", "F1", "MCC"],
        (
            "Robustness under controlled corruption. Every condition uses the "
            "threshold selected on clean validation data."
        ),
        "tab:robustness",
        TABLE_DIR / "Table_4_Robustness.tex",
    )

    for dataset in robustness_summary["Dataset"].unique():
        plot_robustness(
            robustness_summary,
            dataset,
            FIGURE_DIR / f"Robustness_{dataset}.pdf",
        )

    # -------------------------------------------------------------------------
    # Save efficiency results
    # -------------------------------------------------------------------------
    efficiency_raw = pd.DataFrame(efficiency_rows)
    efficiency_raw.to_csv(
        TABLE_DIR / "Table_5_Efficiency_Raw_All_Seeds.csv",
        index=False,
    )
    efficiency_summary = aggregate_mean_std(
        efficiency_raw,
        ["Dataset", "Model"],
    )
    efficiency_summary.to_csv(
        TABLE_DIR / "Table_5_Efficiency_Mean_Std.csv",
        index=False,
    )
    save_latex_table(
        efficiency_summary,
        ["Dataset", "Model"],
        [
            "Parameters",
            "Model_Size_MB",
            "Latency_ms_per_window",
            "Windows_per_second",
        ],
        "Model size and inference efficiency measured on the evaluation device.",
        "tab:efficiency",
        TABLE_DIR / "Table_5_Efficiency.tex",
    )

    # -------------------------------------------------------------------------
    # Save true ablation results, when checkpoints exist
    # -------------------------------------------------------------------------
    missing_ablation_report = []
    for display_name, file_configuration in ABLATION_CONFIGS.items():
        available = sum(
            (
                MODELS_DIR
                / (
                    f"SWaT_DT_RAVLA_seed{seed}.pt"
                    if display_name == "Full"
                    else f"SWaT_{file_configuration}_seed{seed}.pt"
                )
            ).exists()
            for seed in SEEDS
        )
        missing_ablation_report.append(
            {
                "Configuration": display_name,
                "Expected_Seeds": len(SEEDS),
                "Available_Checkpoints": available,
                "Status": (
                    "Complete"
                    if available == len(SEEDS)
                    else "Incomplete, excluded from strict comparison"
                ),
            }
        )

    pd.DataFrame(missing_ablation_report).to_csv(
        DIAGNOSTIC_DIR / "ablation_checkpoint_availability.csv",
        index=False,
    )

    if ablation_rows:
        ablation_raw = pd.DataFrame(ablation_rows)
        ablation_raw.to_csv(
            TABLE_DIR / "Table_6_Ablation_Raw_All_Seeds.csv",
            index=False,
        )
        ablation_summary = aggregate_mean_std(
            ablation_raw,
            ["Dataset", "Configuration"],
        )
        ablation_summary.to_csv(
            TABLE_DIR / "Table_6_Ablation_Mean_Std.csv",
            index=False,
        )
        save_latex_table(
            ablation_summary,
            ["Dataset", "Configuration"],
            [
                "Balanced_Accuracy",
                "Precision",
                "Recall_DR",
                "F1",
                "AUPRC",
                "MCC",
            ],
            (
                "True component ablation using separately trained checkpoints. "
                "Incomplete configurations are excluded."
            ),
            "tab:true_ablation",
            TABLE_DIR / "Table_6_Ablation.tex",
        )

        plot_bar_summary(
            ablation_summary,
            "Configuration",
            "MCC",
            FIGURE_DIR / "Ablation_MCC.pdf",
        )
        plot_bar_summary(
            ablation_summary,
            "Configuration",
            "Recall_DR",
            FIGURE_DIR / "Ablation_Recall.pdf",
        )

        significance_frames = []
        for metric in ["MCC", "F1", "Recall_DR", "AUPRC"]:
            frame = paired_significance(ablation_raw, metric)
            if not frame.empty:
                significance_frames.append(frame)

        if significance_frames:
            significance = pd.concat(significance_frames, ignore_index=True)
            significance.to_csv(
                TABLE_DIR / "Table_7_Statistical_Significance.csv",
                index=False,
            )

    # -------------------------------------------------------------------------
    # Summary figures
    # -------------------------------------------------------------------------
    neural_summary = benchmark_summary[
        benchmark_summary["Model"].str.contains("DT-RAVLA", regex=False)
    ].copy()

    if not neural_summary.empty:
        neural_summary["Display"] = (
            neural_summary["Dataset"] + "\n" + neural_summary["Model"]
        )
        plot_bar_summary(
            neural_summary,
            "Display",
            "MCC",
            FIGURE_DIR / "Benchmark_MCC.pdf",
        )
        plot_bar_summary(
            neural_summary,
            "Display",
            "Recall_DR",
            FIGURE_DIR / "Benchmark_Recall.pdf",
        )

    # -------------------------------------------------------------------------
    # Final manifest
    # -------------------------------------------------------------------------
    manifest = {
        "Completed": True,
        "Device": str(DEVICE),
        "Models_Directory": str(MODELS_DIR),
        "Output_Directory": str(OUTPUT_DIR),
        "No_Retraining": True,
        "Thresholds_Selected_Using_Test_Labels": False,
        "Robustness_Uses_Clean_Validation_Threshold": True,
        "True_Ablations_Require_Separate_Checkpoints": True,
        "Files": sorted(
            str(path.relative_to(OUTPUT_DIR))
            for path in OUTPUT_DIR.rglob("*")
            if path.is_file()
        ),
    }
    (OUTPUT_DIR / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nStrict pretrained reevaluation completed.")
    print(f"Tables:  {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Audit:   {DIAGNOSTIC_DIR}")


if __name__ == "__main__":
    main()
