"""Treinamento e otimizacao de hiperparametros dos modelos supervisionados.

Compara 3 abordagens conforme exigido no desafio:
  1. Baseline linear regularizado -- Regressao Logistica (L2/L1 via saga).
  2. Random Forest -- ensemble de arvores (bagging).
  3. Gradient Boosting -- XGBoost.

Cada modelo e envolvido em um `Pipeline` unico (pre-processamento +
estimador) e otimizado com `RandomizedSearchCV`/`GridSearchCV` usando
`StratifiedKFold`, garantindo que a imputacao/escalonamento sejam
recalculados a cada fold (sem leakage entre folds de validacao cruzada).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42


@dataclass
class ModelSearchResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_results: dict
    best_cv_roc_auc: float


def _cv() -> StratifiedGroupKFold:
    """StratifiedGroupKFold agrupado por municipio: como o mesmo municipio
    aparece em varios anos (estrutura de painel), um KFold ingenuo por linha
    deixaria observacoes do mesmo municipio em treino e validacao
    simultaneamente, vazando caracteristicas municipais entre os folds.
    Agrupar por `id_municipio` garante que cada municipio fique inteiramente
    em um unico fold."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_search_space(preprocessor: ColumnTransformer) -> dict[str, dict]:
    """Define, para cada modelo, o Pipeline completo e a grade de busca de
    hiperparametros a ser explorada."""

    logreg_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE, solver="lbfgs", penalty="l2",
        )),
    ])
    logreg_grid = {
        "classifier__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "classifier__class_weight": [None, "balanced"],
    }

    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_grid = {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    }

    xgb_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])
    xgb_grid = {
        "classifier__n_estimators": [150, 300, 500],
        "classifier__max_depth": [2, 3, 4, 6],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.85, 1.0],
        "classifier__colsample_bytree": [0.7, 0.85, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
    }

    return {
        "logistic_regression": {"pipeline": logreg_pipe, "grid": logreg_grid, "search": "grid"},
        "random_forest": {"pipeline": rf_pipe, "grid": rf_grid, "search": "random"},
        "xgboost": {"pipeline": xgb_pipe, "grid": xgb_grid, "search": "random"},
    }


def run_model_search(
    X_train, y_train, preprocessor: ColumnTransformer, groups=None, n_iter: int = 40
) -> dict[str, ModelSearchResult]:
    """Executa a busca de hiperparametros para os 3 modelos e retorna os
    melhores estimadores (ja re-ajustados no treino completo).

    `groups` deve ser o `id_municipio` de cada linha de `X_train`, usado
    pelo StratifiedGroupKFold para nao vazar municipios entre treino e
    validacao durante a validacao cruzada."""
    search_space = build_search_space(preprocessor)
    cv = _cv()
    results: dict[str, ModelSearchResult] = {}

    for name, spec in search_space.items():
        if spec["search"] == "grid":
            searcher = GridSearchCV(
                estimator=spec["pipeline"],
                param_grid=spec["grid"],
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                estimator=spec["pipeline"],
                param_distributions=spec["grid"],
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            )
        searcher.fit(X_train, y_train, groups=groups)
        results[name] = ModelSearchResult(
            name=name,
            best_estimator=searcher.best_estimator_,
            best_params=searcher.best_params_,
            cv_results=searcher.cv_results_,
            best_cv_roc_auc=searcher.best_score_,
        )
        print(f"[{name}] melhor ROC-AUC (CV 5-fold): {searcher.best_score_:.4f}")
        print(f"[{name}] melhores hiperparametros: {searcher.best_params_}")

    return results
