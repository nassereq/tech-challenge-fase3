"""Graficos diagnosticos de Analise Exploratoria de Dados (EDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")


def plot_target_distribution(df: pd.DataFrame, target: str, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    df[target].value_counts().sort_index().plot(
        kind="bar", ax=axes[0], color=["#d62728", "#2ca02c"]
    )
    axes[0].set_xticklabels(["Nao alfabetizado", "Alfabetizado"], rotation=0)
    axes[0].set_title("Distribuicao geral do alvo")
    axes[0].set_ylabel("N. de alunos")

    taxa_ano = df.groupby("ano")[target].mean()
    taxa_ano.plot(kind="bar", ax=axes[1], color="#1f77b4")
    axes[1].set_title("Taxa de alfabetizacao por ano")
    axes[1].set_ylabel("% alfabetizados")
    axes[1].set_xticklabels(taxa_ano.index, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], output_path: str) -> None:
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4c72b0")
        ax.set_title(col, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_boxplots_outliers(df: pd.DataFrame, numeric_cols: list[str], output_path: str) -> None:
    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#dd8452")
        ax.set_title(col, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], target: str, output_path: str) -> None:
    corr_cols = numeric_cols + [target]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de correlacao -- atributos numericos vs. alvo")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_categorical_vs_target(df: pd.DataFrame, cat_col: str, target: str, output_path: str) -> None:
    taxa = df.groupby(cat_col)[target].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(taxa)), 4.5))
    taxa.plot(kind="bar", ax=ax, color="#55a868")
    ax.axhline(df[target].mean(), color="black", linestyle="--", label="Media geral")
    ax.set_ylabel("Taxa de alfabetizacao")
    ax.set_title(f"Taxa de alfabetizacao por {cat_col}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
