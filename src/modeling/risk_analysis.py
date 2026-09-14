"""Aplicacoes estrategicas do modelo: ranking de risco municipal e
agrupamento (clustering) de municipios com padroes socioeducacionais
semelhantes -- respondem diretamente as perguntas de negocio do desafio."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_municipio_risk_ranking(
    df: pd.DataFrame, estimator, feature_cols: list[str], ano_referencia: int
) -> pd.DataFrame:
    """Usa o modelo campeao para estimar, para o ano de referencia, a
    probabilidade media de alfabetizacao por municipio e classifica os
    municipios em faixas de risco (quanto menor a probabilidade media,
    maior o risco de nao atingir a meta)."""
    subset = df[df["ano"] == ano_referencia].copy()
    subset["proba_alfabetizacao"] = estimator.predict_proba(subset[feature_cols])[:, 1]

    ranking = (
        subset.groupby(["id_municipio", "nome_municipio", "sigla_uf", "regiao"])
        .agg(
            proba_media_alfabetizacao=("proba_alfabetizacao", "mean"),
            n_alunos=("proba_alfabetizacao", "size"),
            meta_pct_municipio=("meta_pct_municipio_lag1", "first"),
        )
        .reset_index()
        .sort_values("proba_media_alfabetizacao")
    )

    def classifica_risco(p: float) -> str:
        if p < 0.40:
            return "Alto risco"
        if p < 0.60:
            return "Risco moderado"
        return "Baixo risco"

    ranking["faixa_risco"] = ranking["proba_media_alfabetizacao"].apply(classifica_risco)
    return ranking


def plot_top_risk_municipios(ranking: pd.DataFrame, output_path: str, top_n: int = 15) -> None:
    top = ranking.head(top_n).copy()
    top["label"] = top["nome_municipio"] + " (" + top["sigla_uf"] + ")"

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * top_n)))
    colors = top["faixa_risco"].map({
        "Alto risco": "#d62728", "Risco moderado": "#ff7f0e", "Baixo risco": "#2ca02c",
    })
    ax.barh(top["label"], top["proba_media_alfabetizacao"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidade media de alfabetizacao (estimada)")
    ax.set_title(f"Top {top_n} municipios com maior risco educacional")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cluster_municipios(
    df: pd.DataFrame, cluster_features: list[str], ano_referencia: int, n_clusters: int = 4
) -> tuple[pd.DataFrame, KMeans]:
    """Agrupa municipios (na ultima observacao disponivel) por padrao
    socioeducacional usando K-Means sobre indicadores padronizados."""
    subset = (
        df[df["ano"] == ano_referencia]
        .groupby(["id_municipio", "nome_municipio", "sigla_uf", "regiao"])[cluster_features]
        .mean()
        .dropna()
        .reset_index()
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
