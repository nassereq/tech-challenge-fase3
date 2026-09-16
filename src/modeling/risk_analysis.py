"""Aplicacoes estrategicas do modelo: ranking de risco municipal e
agrupamento (clustering) de municipios com padroes socioeducacionais
semelhantes -- respondem diretamente as perguntas de negocio do desafio.

Como a base (real, `gold_preview`) e um corte transversal com uma linha
por municipio no ano-alvo (2024), essas funcoes operam diretamente sobre
o DataFrame de features, sem necessidade de agregar por ano."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_municipio_risk_ranking(df: pd.DataFrame, estimator, feature_cols: list[str]) -> pd.DataFrame:
    """Usa o modelo campeao para estimar a probabilidade de o municipio
    atingir a meta de alfabetizacao, classificando-os em faixas de risco
    (quanto menor a probabilidade, maior o risco)."""
    ranking = df[["id_municipio", "nome_municipio", "sigla_uf", "regiao", "meta_pct"]].copy()
    ranking["proba_atingir_meta"] = estimator.predict_proba(df[feature_cols])[:, 1]
    ranking = ranking.sort_values("proba_atingir_meta")

    def classifica_risco(p: float) -> str:
        if p < 0.40:
            return "Alto risco"
        if p < 0.60:
            return "Risco moderado"
        return "Baixo risco"

    ranking["faixa_risco"] = ranking["proba_atingir_meta"].apply(classifica_risco)
    return ranking


def plot_top_risk_municipios(ranking: pd.DataFrame, output_path: str, top_n: int = 15) -> None:
    top = ranking.head(top_n).copy()
    top["label"] = top["nome_municipio"] + " (" + top["sigla_uf"] + ")"

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * top_n)))
    colors = top["faixa_risco"].map({
        "Alto risco": "#d62728", "Risco moderado": "#ff7f0e", "Baixo risco": "#2ca02c",
    })
    ax.barh(top["label"], top["proba_atingir_meta"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidade estimada de atingir a meta")
    ax.set_title(f"Top {top_n} municipios com maior risco educacional (2024)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cluster_municipios(
    df: pd.DataFrame, cluster_features: list[str], n_clusters: int = 4
) -> tuple[pd.DataFrame, KMeans]:
    """Agrupa municipios por padrao socioeducacional usando K-Means sobre
    indicadores padronizados."""
    subset = (
        df[["id_municipio", "nome_municipio", "sigla_uf", "regiao"] + cluster_features]
        .dropna()
        .reset_index(drop=True)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset[cluster_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    subset["cluster"] = kmeans.fit_predict(X_scaled)
    return subset, kmeans


def plot_clusters(subset: pd.DataFrame, x: str, y: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=subset, x=x, y=y, hue="cluster", style="regiao", palette="Set2", s=90, ax=ax
    )
    ax.set_title("Agrupamento de municipios por padrao socioeducacional")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
