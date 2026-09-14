"""Carregamento e engenharia de atributos da base Gold (Fase 2) para o
problema de classificacao binaria "aluno alfabetizado x nao alfabetizado".

Regra de negocio identificada na base: `alfabetizado` e definido de forma
deterministica como `proficiencia_saeb >= ponto_corte` (743 pontos). Por isso
`proficiencia_saeb` e `ponto_corte` sao removidos do conjunto de atributos:
usa-los seria vazamento direto do alvo (a variavel e literalmente o rotulo
disfarcado), nao um preditor de negocio.

Alem disso, indicadores municipais agregados (ex.: `pct_alfabetizados` do
proprio ano) sao calculados a partir dos mesmos alunos que compoem o alvo
daquele ano/municipio (vazamento agregado). Para evitar esse leakage
temporal, todo indicador historico municipal/estadual entra no modelo
defasado em um ano (`shift(1)` por municipio), representando apenas
informacao que already existiria no passado quando o aluno for avaliado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REGIAO_POR_UF = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte",
    "PA": "Norte", "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste",
    "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
}

# Colunas que configuram vazamento direto ou agregado do alvo e nunca devem
# entrar como feature do modelo.
LEAKAGE_COLUMNS = [
    "proficiencia_saeb",
    "ponto_corte",
    "pct_alfabetizados",
    "n_avaliados",
    "gap_meta_pct",
    "atingiu_meta",
    "delta_pp_ano_anterior",
]

TARGET = "alfabetizado"
ID_COLUMNS = ["id_aluno", "id_municipio", "id_uf", "ano"]


def load_raw_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Le as tabelas de dominio e fato da camada Gold/amostra da Fase 2."""
    data_dir = Path(data_dir)
    sample_dir = data_dir / "sample"
    gold_dir = data_dir / "gold"

    tables = {
        "alunos": pd.read_csv(sample_dir / "alunos.csv"),
        "municipio": pd.read_csv(sample_dir / "municipio.csv"),
        "uf": pd.read_csv(sample_dir / "uf.csv"),
        "meta_municipio": pd.read_csv(sample_dir / "meta_municipio.csv"),
        "meta_uf": pd.read_csv(sample_dir / "meta_uf.csv"),
        "meta_brasil": pd.read_csv(sample_dir / "meta_brasil.csv"),
        "indicador_municipio": pd.read_csv(sample_dir / "indicador_municipio.csv"),
        "evolucao_municipio": pd.read_csv(gold_dir / "evolucao_temporal_municipio.csv"),
    }
    return tables


def _build_municipio_history(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Constroi o historico municipal ano a ano e aplica defasagem (t-1)
    nos indicadores derivados de desempenho, evitando leakage agregado."""
    hist = tables["indicador_municipio"][
        ["ano", "id_municipio", "pct_alfabetizados", "n_avaliados"]
    ].merge(
        tables["meta_municipio"][["ano", "id_municipio", "meta_pct"]],
        on=["ano", "id_municipio"],
        how="left",
    )
    hist = hist.rename(columns={"meta_pct": "meta_pct_municipio"})
    hist["gap_meta_pct_municipio"] = hist["pct_alfabetizados"] - hist["meta_pct_municipio"]
    hist["atingiu_meta_municipio"] = (hist["gap_meta_pct_municipio"] >= 0).astype(int)

    hist = hist.sort_values(["id_municipio", "ano"])
    lag_cols = [
        "pct_alfabetizados",
        "n_avaliados",
        "meta_pct_municipio",
        "gap_meta_pct_municipio",
        "atingiu_meta_municipio",
    ]
    for col in lag_cols:
        hist[f"{col}_lag1"] = hist.groupby("id_municipio")[col].shift(1)
    hist["delta_pct_alfabetizados_lag1"] = hist.groupby("id_municipio")["pct_alfabetizados"].diff().shift(1)

    keep_cols = ["ano", "id_municipio"] + [f"{c}_lag1" for c in lag_cols] + [
        "delta_pct_alfabetizados_lag1"
    ]
    return hist[keep_cols]


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta as dimensoes territoriais/socioeconomicas e as metas vigentes
    (nao-vazadoras) ao nivel-aluno, retornando o dataset pronto para
    treino/teste (ainda sem split)."""
    df = tables["alunos"].copy()

    df = df.merge(
        tables["municipio"][["id_municipio", "nome_municipio"]],
        on="id_municipio",
        how="left",
    )
    df = df.merge(
        tables["uf"][["id_uf", "nome_uf"]],
        on="id_uf",
        how="left",
    )
    df["regiao"] = df["sigla_uf"].map(REGIAO_POR_UF)

    # Metas vigentes no proprio ano: sao alvos de politica publica definidos
    # a priori (nao derivam do desempenho realizado), portanto nao configuram
    # leakage -- sao legitimas como feature de contexto.
    df = df.merge(
        tables["meta_uf"][["ano", "id_uf", "meta_pct"]].rename(columns={"meta_pct": "meta_pct_uf"}),
        on=["ano", "id_uf"],
        how="left",
    )
    df = df.merge(
        tables["meta_brasil"][["ano", "meta_pct"]].rename(columns={"meta_pct": "meta_pct_brasil"}),
        on="ano",
        how="left",
    )

    # Historico municipal defasado (t-1): informacao que um gestor teria
    # disponivel *antes* do resultado do ano corrente.
    municipio_hist = _build_municipio_history(tables)
    df = df.merge(municipio_hist, on=["ano", "id_municipio"], how="left")

    # Atributo temporal simples: quantos anos de serie o municipio ja possui
    # ate o ano corrente (proxy de maturidade de monitoramento).
    df["ano_indice"] = df["ano"] - df["ano"].min()

    return df


def temporal_train_test_split(
    df: pd.DataFrame, test_years: tuple[int, ...] = (2023,)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide treino/teste por ano (holdout temporal), simulando o caso de
    uso real: treinar com anos passados e prever o ano mais recente.
    Isso evita qualquer contaminacao entre observacoes do mesmo periodo e
    testa a capacidade de generalizacao para o futuro."""
    is_test = df["ano"].isin(test_years)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (numericas, categoricas) excluindo alvo, IDs e colunas de
    leakage."""
    exclude = set(ID_COLUMNS) | {TARGET} | set(LEAKAGE_COLUMNS) | {"nome_municipio", "nome_uf"}
    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns if c not in exclude
    ]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category", "str"]).columns if c not in exclude
    ]
    return numeric_cols, categorical_cols


if __name__ == "__main__":
    tables = load_raw_tables(Path(__file__).resolve().parents[2] / "data" / "raw")
    feature_df = build_feature_table(tables)
    train_df, test_df = temporal_train_test_split(feature_df)
    print("Shape total:", feature_df.shape)
    print("Treino (2021-2022):", train_df.shape, "| Teste (2023):", test_df.shape)
    num_cols, cat_cols = get_feature_columns(feature_df)
    print("Numericas:", num_cols)
    print("Categoricas:", cat_cols)
    print("Missing no treino (lag do primeiro ano observado):")
    print(train_df[num_cols].isna().mean().sort_values(ascending=False).head(10))
