"""Carregamento e engenharia de atributos da base Gold real (Fase 2) para o
problema de classificacao binaria "municipio atingiu a meta de alfabetizacao
x nao atingiu".

## Historico desta escolha de dados

A primeira versao deste pipeline usava `data/raw/sample` e `data/raw/gold`,
uma amostra SINTETICA de demonstracao (81 municipios fictícios, 2021-2023)
criada pelo gerador `pipelines/batch/generate_sample_data.py` da Fase 2 para
permitir rodar a pipeline sem credenciais de nuvem. Ao investigar um salto
implausivel na taxa de alfabetizacao daquela amostra (quase dobrou entre
2021 e 2023), descobrimos que o crescimento era um artefato mecanico do
gerador (uma meta que cresce ~10 p.p./ano por construcao, alimentando um
sorteio de proficiencia com corte fixo) -- nao um fenomeno real.

Durante essa investigacao encontramos, dentro do proprio repositorio da
Fase 2, uma segunda fonte: `reports/gold_preview/`, um espelho real da
camada Gold gerado a partir de dados oficiais da Base dos Dados / CNCA
(Compromisso Nacional Crianca Alfabetizada), cobrindo **5.516 municipios
brasileiros reais** (codigos IBGE genuinos) em 2023-2024. Esta versao do
pipeline usa exclusivamente essa fonte real.

## Por que o alvo passou a ser o MUNICIPIO, nao o aluno

A tabela real de alunos individuais só existe no BigQuery (nunca foi
exportada localmente); apenas o indicador agregado por municipio esta
disponivel como CSV. Modelar no nivel de municipio nao e uma limitacao
imposta por conveniencia: e exatamente uma das perguntas de negocio do
desafio ("quais municipios apresentam maior risco?", "como prever
municipios que podem nao atingir metas futuras?"), e permite usar dados
100% reais em vez de uma proxy sintetica em nivel de aluno.

## Vazamento identificado nesta base

`atingiu_meta` (2024) e definido deterministicamente por
`pct_alfabetizados_2024 >= meta_pct_2024` (confirmado com 100% de
correspondencia). Por isso os valores de 2024 de `pct_alfabetizados`,
`gap_meta_pct`, `delta_pp_ano_anterior` e `n_avaliados` sao excluidos do
conjunto de features -- o modelo so pode usar informacao que existia
*antes* do resultado de 2024: o desempenho do proprio municipio e da UF
em 2023 (defasado em um ano), e as metas vigentes para 2024 (que sao
definidas a priori, nao derivadas do resultado).
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

ANO_ALVO = 2024
ANO_LAG = 2023

# Colunas do ano-alvo (2024) derivadas do proprio resultado: vazamento
# direto ou agregado, nunca entram como feature.
LEAKAGE_COLUMNS = [
    "pct_alfabetizados",
    "gap_meta_pct",
    "delta_pp_ano_anterior",
    "n_avaliados",
    "camada",
    "ponto_corte",
]

TARGET = "atingiu_meta"
ID_COLUMNS = ["id_municipio", "id_uf", "ano"]
NAME_COLUMNS = ["nome_municipio", "fonte_meta"]


def load_raw_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Le as tabelas da camada Gold real (`reports/gold_preview` da Fase 2,
    copiadas para `data/raw/gold_preview`)."""
    gold_dir = Path(data_dir) / "gold_preview"
    tables = {
        "evolucao_municipio": pd.read_csv(gold_dir / "evolucao_temporal_municipio.csv"),
        "evolucao_uf": pd.read_csv(gold_dir / "evolucao_temporal_uf.csv"),
        "comparativo_uf": pd.read_csv(gold_dir / "comparativo_meta_resultado_uf.csv"),
        "comparativo_brasil": pd.read_csv(gold_dir / "comparativo_meta_resultado_brasil.csv"),
    }
    return tables


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Constroi a tabela municipio x atributos para o ano-alvo (2024),
    juntando o historico defasado (2023) do proprio municipio e da UF."""
    mun = tables["evolucao_municipio"]

    cohort = mun[mun["ano"] == ANO_ALVO].copy()
    cohort[TARGET] = cohort[TARGET].astype(bool).astype(int)

    lag_municipio = (
        mun[mun["ano"] == ANO_LAG][["id_municipio", "pct_alfabetizados"]]
        .rename(columns={"pct_alfabetizados": "pct_alfabetizados_lag1"})
    )
    cohort = cohort.merge(lag_municipio, on="id_municipio", how="left")

    uf = tables["evolucao_uf"]
    lag_uf = (
        uf[uf["ano"] == ANO_LAG][["sigla_uf", "pct_alfabetizados"]]
        .rename(columns={"pct_alfabetizados": "pct_alfabetizados_uf_lag1"})
    )
    cohort = cohort.merge(lag_uf, on="sigla_uf", how="left")

    comp_uf = tables["comparativo_uf"]
    meta_uf_alvo = (
        comp_uf[comp_uf["ano"] == ANO_ALVO][["sigla_uf", "meta_pct_uf"]]
    )
    cohort = cohort.merge(meta_uf_alvo, on="sigla_uf", how="left")

    cohort["regiao"] = cohort["sigla_uf"].map(REGIAO_POR_UF)

    return cohort.reset_index(drop=True)


def stratified_train_test_split(
    df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide treino/teste com amostragem estratificada pelo alvo.

    Os dados sao um corte transversal (todos os municipios avaliados no
    mesmo ano-alvo, 2024) -- nao ha estrutura de painel/repeticao a
    proteger aqui (diferente da versao anterior com dados sinteticos
    multi-ano), entao a divisao aleatoria estratificada e a escolha
    correta e padrao da literatura para esse cenario."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna (numericas, categoricas) excluindo alvo, IDs, nomes e
    colunas de leakage."""
    exclude = set(ID_COLUMNS) | {TARGET} | set(LEAKAGE_COLUMNS) | set(NAME_COLUMNS)
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
    print("Shape do cohort 2024:", feature_df.shape)
    print("Taxa de atingiu_meta:", feature_df[TARGET].mean())
    train_df, test_df = stratified_train_test_split(feature_df)
    print("Treino:", train_df.shape, "| Teste:", test_df.shape)
    num_cols, cat_cols = get_feature_columns(feature_df)
    print("Numericas:", num_cols)
    print("Categoricas:", cat_cols)
    print("Missing (treino):")
    print(train_df[num_cols].isna().mean().sort_values(ascending=False))
