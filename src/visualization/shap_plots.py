"""Interpretabilidade do modelo campeao via Feature Importance e SHAP Values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def plot_feature_importance(estimator, feature_names, output_path: str, top_n: int = 20) -> pd.Series:
    """Feature importance nativa do modelo de arvore (MDI)."""
    classifier = estimator.named_steps["classifier"]
    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importances))))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_title("Feature Importance (MDI) -- modelo campeao")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return importances


def compute_shap_values(estimator, X_train_transformed, X_test_transformed):
    """Cria o explainer SHAP apropriado (TreeExplainer para modelos de
    arvore) usando o treino como background e explica o teste.

    Para classificadores binarios baseados em arvore, o TreeExplainer
    retorna, em algumas versoes, uma saida com 3 dimensoes
    (amostras x atributos x classes); nesse caso selecionamos a fatia da
    classe positiva ("alfabetizado"=1), que e a que interessa ao negocio.
    """
    classifier = estimator.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(np.asarray(X_test_transformed, dtype=float))
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return explainer, shap_values


def plot_shap_summary(shap_values, feature_names, output_path: str) -> None:
    shap_values.feature_names = list(feature_names)
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP Summary Plot -- impacto das variaveis na predicao")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values, feature_names, output_path: str) -> None:
    shap_values.feature_names = list(feature_names)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP -- importancia media absoluta por variavel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_waterfall(shap_values, index: int, output_path: str) -> None:
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"SHAP Waterfall -- explicacao individual (amostra #{index})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_values, feature_name: str, output_path: str) -> None:
    plt.figure()
    shap.plots.scatter(shap_values[:, feature_name], show=False)
    plt.title(f"SHAP Dependence Plot -- {feature_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
