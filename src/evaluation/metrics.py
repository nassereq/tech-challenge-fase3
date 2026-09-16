"""Metricas de avaliacao para classificacao binaria com foco em robustez a
desbalanceamento (ROC-AUC, PR-AUC, F1, Precisao, Recall, Matriz de Confusao)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(name: str, estimator, X_test, y_test) -> dict:
    """Calcula o conjunto completo de metricas de avaliacao para um modelo
    ja treinado, no conjunto de teste (holdout estratificado)."""
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]

    return {
        "modelo": name,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("modelo")
    return df.sort_values("roc_auc", ascending=False).round(4)


def plot_confusion_matrices(estimators: dict, X_test, y_test, output_path: str) -> None:
    fig, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 4.5))
    if len(estimators) == 1:
        axes = [axes]
    for ax, (name, estimator) in zip(axes, estimators.items()):
        y_pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Nao alfabetizado", "Alfabetizado"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(estimators: dict, X_test, y_test, output_path_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, estimator in estimators.items():
        RocCurveDisplay.from_estimator(estimator, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_title("Curvas ROC -- comparacao de modelos (holdout, 20% dos municipios)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{output_path_prefix}_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, estimator in estimators.items():
        PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test, name=name, ax=ax)
    ax.set_title("Curvas Precisao-Recall -- comparacao de modelos (holdout, 20% dos municipios)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{output_path_prefix}_pr.png", dpi=150)
    plt.close(fig)
