"""Pipeline de pre-processamento Scikit-learn (ColumnTransformer) para o
problema de classificacao binaria de alfabetizacao.

O pre-processamento e ajustado (fit) exclusivamente no conjunto de treino e
aplicado (transform) ao teste, sempre dentro de um `Pipeline` unico junto ao
estimador -- isso garante que nenhuma estatistica (media, moda, escala)
calculada com dados de teste vaze para o treino.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Cria o ColumnTransformer com:
    - Numericas: imputacao pela mediana (robusta a outliers e a alta
      proporcao de NaN estrutural dos lags do primeiro ano) + padronizacao.
    - Categoricas: imputacao pela moda + One-Hot Encoding (baixa
      cardinalidade: UF e regiao).
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> np.ndarray:
    """Recupera os nomes das features pos-transformacao (uteis para
    Feature Importance / SHAP)."""
    return preprocessor.get_feature_names_out()
