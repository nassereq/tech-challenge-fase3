# Tech Challenge — Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto integrador da Fase 3 da Pós Tech (FIAP), que utiliza a camada **Gold** de dados construída na
Fase 2 (pipeline de engenharia de dados sobre o Indicador Criança Alfabetizada) para desenvolver um
**modelo supervisionado de Machine Learning** capaz de prever se um aluno será considerado **alfabetizado**
ou **não alfabetizado**, e para gerar inteligência analítica aplicável a políticas públicas educacionais.

> 📓 O desenvolvimento completo, documentado passo a passo (EDA, prevenção de leakage, pipeline,
> modelagem, interpretabilidade e aplicação estratégica), está em
> [`notebooks/01_pipeline_predicao_alfabetizacao.ipynb`](notebooks/01_pipeline_predicao_alfabetizacao.ipynb).

---

## Sumário

1. [Contexto do problema e objetivo analítico](#1-contexto-do-problema-e-objetivo-analítico)
2. [Descrição da base utilizada](#2-descrição-da-base-utilizada)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Como reproduzir o pipeline](#4-como-reproduzir-o-pipeline)
5. [Etapas de pré-processamento e prevenção de data leakage](#5-etapas-de-pré-processamento-e-prevenção-de-data-leakage)
6. [Metodologia de modelagem](#6-metodologia-de-modelagem)
7. [Métricas de avaliação e comparação de modelos](#7-métricas-de-avaliação-e-comparação-de-modelos)
8. [Interpretação dos resultados (SHAP) e insights de negócio](#8-interpretação-dos-resultados-shap-e-insights-de-negócio)
9. [Aplicação prática para políticas públicas](#9-aplicação-prática-para-políticas-públicas)
10. [Limitações do projeto](#10-limitações-do-projeto)
11. [Possíveis evoluções futuras](#11-possíveis-evoluções-futuras)

---

## 1. Contexto do problema e objetivo analítico

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e social do
Brasil. Compreender apenas o retrato atual não é suficiente para apoiar decisões estratégicas: gestores
públicos precisam **antecipar riscos**, **identificar regiões vulneráveis** e **compreender quais fatores
mais impactam os indicadores educacionais** antes que o ciclo letivo termine.

Na Fase 2 deste Tech Challenge foi construído o pipeline de engenharia de dados (camadas Bronze → Silver →
Gold) que integrou o Indicador Criança Alfabetizada a metas nacionais/estaduais/municipais e a dimensões
territoriais. Nesta Fase 3, esses dados Gold viram matéria-prima analítica.

**Objetivo analítico:** desenvolver um modelo supervisionado de classificação binária capaz de prever se um
aluno será considerado **alfabetizado (1)** ou **não alfabetizado (0)**, a partir de variáveis
educacionais, territoriais e de metas — e usar esse modelo para responder às perguntas estratégicas do
desafio:

- Quais fatores mais impactam a alfabetização?
- Quais municípios apresentam maior risco de não atingir metas educacionais?
- Quais regiões possuem padrões semelhantes (agrupamentos)?

O foco não é apenas atingir métricas altas, mas produzir **inteligência aplicável** ao contexto
educacional brasileiro — inclusive quando as métricas revelam os limites do que a base de dados disponível
permite concluir (ver [Seção 10](#10-limitações-do-projeto)).

## 2. Descrição da base utilizada

Todos os dados usados neste projeto vêm da camada **Gold**/amostra sintética offline gerada na Fase 2
(`data/raw/`), sem qualquer fonte externa adicional nesta entrega:

| Arquivo | Granularidade | Descrição |
|---|---|---|
| `sample/alunos.csv` | aluno x ano | Proficiência SAEB e rótulo `alfabetizado` (2021-2023, 1.940 alunos) |
| `sample/municipio.csv`, `sample/uf.csv` | município / UF | Dimensões territoriais (81 municípios, 27 UFs) |
| `sample/meta_municipio.csv`, `sample/meta_uf.csv`, `sample/meta_brasil.csv` | município/UF/Brasil x ano | Metas oficiais de % de alfabetização por nível federativo |
| `sample/indicador_municipio.csv` | município x ano | Indicador histórico realizado (% alfabetizados, nº avaliados) |
| `gold/evolucao_temporal_municipio.csv` | município x ano | Série temporal consolidada com variação ano a ano |

> **Nota de transparência:** conforme o `manifest.json` da própria Fase 2, esta é uma **amostra sintética
> offline** ("dados sintéticos de demonstração"), criada para viabilizar o desenvolvimento do pipeline sem
> depender de credenciais de produção (Base dos Dados / BigQuery). Os padrões (crescimento da alfabetização
> ao longo do tempo, desigualdade regional) foram desenhados para serem plausíveis, mas os números
> absolutos **não devem ser lidos como estatística oficial**. A metodologia — pipeline Scikit-learn,
> prevenção de leakage, tuning, interpretabilidade — é diretamente reaproveitável ao substituir a fonte por
> dados reais do INEP/Censo Escolar/IBGE, como discutido na [Seção 11](#11-possíveis-evoluções-futuras).

## 3. Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/                     # Dados Gold/amostra copiados da Fase 2 (somente leitura)
│   │   ├── gold/
│   │   └── sample/
│   └── processed/                # Reservado para datasets intermediários (não versionado)
├── notebooks/
│   └── 01_pipeline_predicao_alfabetizacao.ipynb   # Notebook integrador, executado ponta a ponta
├── src/
│   ├── preprocessing/
│   │   ├── data_loading.py       # Carga, engenharia de atributos, lag features, split temporal
│   │   └── pipeline.py           # ColumnTransformer (imputação + encoding + scaling)
│   ├── modeling/
│   │   ├── train.py              # 3 modelos + GridSearch/RandomizedSearch + StratifiedGroupKFold
│   │   └── risk_analysis.py      # Ranking de risco municipal + clustering regional (K-Means)
│   ├── evaluation/
│   │   └── metrics.py            # ROC-AUC, PR-AUC, F1, precisão, recall, matriz de confusão
│   └── visualization/
│       ├── eda_plots.py          # Gráficos diagnósticos de EDA
│       └── shap_plots.py         # Feature Importance + SHAP (summary/bar/waterfall/dependence)
├── reports/
│   ├── images/                   # Todos os gráficos gerados pelo pipeline
│   └── roteiro_video_executivo.md
├── requirements.txt
├── README.md
└── .gitignore
```

## 4. Como reproduzir o pipeline

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Executa o pipeline completo (EDA -> modelagem -> avaliação -> SHAP -> insights)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_pipeline_predicao_alfabetizacao.ipynb

# Ou explore módulo a módulo:
python -m src.preprocessing.data_loading
```

Todos os gráficos são regravados em `reports/images/` a cada execução — o pipeline é **determinístico**
(`random_state=42` fixado em todos os componentes estocásticos: split, KFold, RandomizedSearchCV,
RandomForest, XGBoost, K-Means), garantindo replicabilidade.

## 5. Etapas de pré-processamento e prevenção de data leakage

### 5.1 Engenharia de atributos

A tabela de alunos é enriquecida com dimensões territoriais (`municipio`, `uf`, `regiao`, derivada da UF) e
com as metas vigentes de alfabetização (município, UF e Brasil) para o ano corrente do aluno.

### 5.2 Dois vazamentos identificados e tratados

Esta foi a descoberta mais importante da análise exploratória:

1. **Leakage direto:** o rótulo `alfabetizado` é definido deterministicamente por
   `proficiencia_saeb >= 743` (ponto de corte) — confirmado com 100% de correspondência nos dados. Usar
   `proficiencia_saeb`/`ponto_corte` como *feature* seria usar o próprio rótulo disfarçado, sem qualquer
   valor preditivo real (o gestor não possui essa nota *antes* da avaliação — é exatamente o que
   precisamos prever a partir de outras variáveis). Ambas as colunas foram **excluídas do conjunto de
   features** (`LEAKAGE_COLUMNS` em `src/preprocessing/data_loading.py`).

2. **Leakage agregado:** o indicador `pct_alfabetizados` de um município em um ano é calculado a partir dos
   próprios alunos daquele município/ano — incluindo o aluno sendo predito. Usar esse valor do **ano
   corrente** como feature vazaria (de forma agregada) informação do próprio alvo. **Solução:** todo
   indicador histórico municipal (`pct_alfabetizados`, `meta_pct_municipio`, `gap_meta_pct_municipio`,
   `atingiu_meta_municipio`, `n_avaliados`) entra no modelo **defasado em um ano** (`shift(1)` por
   município), representando apenas informação que já existiria no passado no momento da predição.

Essa defasagem gera `NaN` estruturais para o primeiro ano da série (2021, sem ano anterior disponível) —
tratados explicitamente na imputação (não são erros de coleta, são ausência estrutural esperada).

### 5.3 Divisão treino/teste — holdout temporal (não aleatório)

Como os dados têm estrutura de painel (o mesmo município aparece em múltiplos anos) e a pergunta de
negócio central é prever municípios em risco **no futuro**, a divisão treino/teste é feita **por ano**:
treino = 2021-2022, teste = 2023. Isso evita (a) vazamento de municípios entre treino/teste no mesmo ano e
(b) simula o uso real do modelo — treinar com o passado, prever o próximo ciclo letivo.

### 5.4 Pipeline Scikit-learn integrado

Pré-processamento e modelo vivem em um único `sklearn.pipeline.Pipeline`, nunca ajustados fora dele:

- **Numéricas:** `SimpleImputer(strategy="median")` + `StandardScaler`.
- **Categóricas** (`sigla_uf`, `regiao`): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder`.

Isso garante que toda estatística de imputação/escala seja aprendida **apenas no treino** (ou apenas no
fold de treino, durante a validação cruzada) e apenas aplicada ao teste — nunca reajustada nele.

## 6. Metodologia de modelagem

Foram comparadas 3 abordagens supervisionadas, cada uma dentro do seu próprio `Pipeline`:

| Modelo | Papel | Biblioteca |
|---|---|---|
| Regressão Logística regularizada (L2) | Baseline linear, interpretável | `scikit-learn` |
| Random Forest | Ensemble de árvores (bagging) | `scikit-learn` |
| XGBoost | Gradient Boosting | `xgboost` |

**Validação cruzada:** `StratifiedGroupKFold` (5 folds), agrupado por `id_municipio`. Como o mesmo
município aparece em 2021 *e* 2022 no treino, um `KFold`/`StratifiedKFold` ingênuo por linha deixaria
observações do mesmo município em treino e validação simultaneamente — um vazamento sutil de
características municipais entre folds. Agrupar por município garante que cada município fique inteiramente
em um único fold durante o tuning de hiperparâmetros.

**Otimização de hiperparâmetros:** `GridSearchCV` para a Regressão Logística (espaço pequeno: `C`,
`class_weight`) e `RandomizedSearchCV` (40 iterações) para Random Forest e XGBoost (espaços maiores:
profundidade, nº de estimadores, taxa de aprendizado, subsample, regularização), otimizando **ROC-AUC**.

## 7. Métricas de avaliação e comparação de modelos

Métricas robustas a desbalanceamento, calculadas no holdout temporal (ano 2023):

| Modelo | ROC-AUC | PR-AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | **0,583** | **0,769** | 0,803 | 0,779 | 0,829 |
| XGBoost | 0,582 | 0,765 | 0,438 | 0,763 | 0,307 |
| Regressão Logística | 0,559 | 0,759 | 0,834 | 0,733 | 0,968 |

⭐ **Modelo campeão: Random Forest** (maior ROC-AUC no holdout).

**ROC-AUC em validação cruzada (StratifiedGroupKFold, treino 2021-2022):** Regressão Logística 0,564 |
Random Forest 0,553 | XGBoost 0,534 — na mesma faixa do holdout, **sem sinal de overfitting** (a diferença
entre CV e teste é pequena para os três modelos).

### Leitura honesta dos resultados

O ROC-AUC no holdout (0,56-0,58) é modesto — mas essa é uma leitura **importante e honesta**, não uma
falha de metodologia: a proximidade entre o desempenho em validação cruzada e no holdout temporal mostra
que o teto observado reflete uma **limitação genuína da granularidade dos dados disponíveis** (apenas 2
anos de histórico por município, sem covariáveis socioeconômicas reais como renda, IDH ou infraestrutura
escolar), e não um problema de vazamento não tratado ou overfitting. O PR-AUC (0,76-0,77), bem acima da
prevalência da classe positiva no teste (73,3%), mostra que os modelos capturam sinal útil para
**ranquear relativamente** o risco entre municípios em um mesmo período — a aplicação estratégica da
[Seção 9](#9-aplicação-prática-para-políticas-públicas).

## 8. Interpretação dos resultados (SHAP) e insights de negócio

Usando `Feature Importance` (MDI) e `SHAP TreeExplainer` sobre o modelo campeão (Random Forest):

Os 5 atributos de maior importância (MDI), em ordem exata:

| Ranking | Atributo | Importância |
|---|---|---|
| 1º | `pct_alfabetizados_lag1` — taxa de alfabetização do município no ano anterior | 13,6% |
| 2º | `delta_pct_alfabetizados_lag1` — variação da taxa em relação ao ano anterior | 11,7% |
| 3º | `n_avaliados_lag1` — proxy do porte/volume de alunos avaliados no município | 10,5% |
| 4º | `meta_pct_uf` — meta estadual vigente no ano | 10,1% |
| 5º | `gap_meta_pct_municipio_lag1` — distância do município em relação à sua meta no ano anterior | 8,0% |

- **`pct_alfabetizados_lag1`** é, disparadamente, o atributo mais influente — municípios com trajetória
  histórica forte tendem a continuar alfabetizando bem (efeito de persistência institucional).
- **`delta_pct_alfabetizados_lag1`** reforça esse efeito: municípios em trajetória de melhora recente pesam
  positivamente na predição.
- **`n_avaliados_lag1`** sugere que o porte da rede municipal correlaciona com a capacidade de execução da
  política de alfabetização.
- **`meta_pct_uf`** e **`gap_meta_pct_municipio_lag1`** capturam, respectivamente, o efeito da meta estadual
  vigente e o quão perto (ou longe) o município já estava de cumpri-la.
- As dummies de **`regiao`**/`sigla_uf` aparecem com importância individual menor (ex.: `regiao_Sudeste`
  em torno de 4%) — relevantes o suficiente para confirmar heterogeneidade regional nos dados, mas
  individualmente menos determinantes que o histórico do próprio município. Como discutido na
  [Seção 3.4 do notebook](notebooks/01_pipeline_predicao_alfabetizacao.ipynb) (hipótese H4, refutada pelos
  dados), o sentido dessa heterogeneidade **nesta amostra sintética não deve ser generalizado** como
  retrato real da desigualdade educacional brasileira.

**Resposta à pergunta de negócio "quais fatores mais impactam a alfabetização?"**: nesta base, o
desempenho histórico recente do próprio município e a cobertura das metas nacionais/estaduais são os
sinais mais fortes disponíveis — reforçando que **continuidade de políticas municipais bem-sucedidas** e
**metas nacionais ambiciosas e bem comunicadas** caminham juntas.

## 9. Aplicação prática para políticas públicas

O pipeline gera dois artefatos diretamente acionáveis por secretarias de educação (ver
`src/modeling/risk_analysis.py` e Seção 10 do notebook):

1. **Ranking de risco municipal:** para o ano mais recente, o modelo estima a probabilidade média de
   alfabetização por município e classifica-os em **Alto risco / Risco moderado / Baixo risco**
   (`reports/images/risk_top_municipios.png`). Isso permite priorizar visitas técnicas, reforço de
   material didático e acompanhamento pedagógico nos municípios mais vulneráveis **antes** do resultado
   final do ciclo letivo.
2. **Agrupamento regional (K-Means, k=4):** municípios são agrupados por padrão socioeducacional
   (desempenho histórico, meta vigente, gap em relação à meta), revelando perfis de maturidade
   educacional que podem orientar estratégias diferenciadas por grupo (ex.: grupos com gap persistente
   precisam de intervenção estrutural; grupos já acima da meta podem servir de referência/mentoria para
   os demais).

Ambos os artefatos respondem diretamente às perguntas "quais municípios apresentam maior risco
educacional?" e "quais regiões possuem padrões semelhantes?" propostas no desafio.

## 10. Limitações do projeto

- **Base sintética de demonstração:** os dados usados são uma amostra sintética offline da Fase 2, não a
  base oficial do INEP/Censo Escolar. Os padrões qualitativos (crescimento temporal, desigualdade
  regional) foram desenhados para serem plausíveis, mas os números absolutos não devem ser interpretados
  como estatística oficial.
- **Poucos anos de histórico (2021-2023):** apenas 2 transições ano-a-ano por município para calcular os
  atributos de lag, limitando a capacidade do modelo de aprender tendências de médio prazo.
- **Ausência de covariáveis socioeconômicas reais:** a base não contém renda, IDH, infraestrutura escolar,
  formação docente ou indicadores do Censo Escolar/PNAD — variáveis que, na literatura educacional, são
  fortemente associadas à alfabetização e que aqui não puderam ser incorporadas.
- **Forte drift temporal:** a taxa de alfabetização quase dobrou entre 2021 (36,7%) e 2023 (73,3%),
  provavelmente refletindo uma mudança de política nacional. Extrapolar esse tipo de salto usando apenas
  o histórico municipal é, comprovadamente (Seção 7), uma tarefa genuinamente difícil — o modelo tem
  desempenho consistente mas modesto (ROC-AUC ~0,58) tanto em validação cruzada quanto no holdout futuro.
- **Cardinalidade municipal reduzida (81 municípios):** limita a capacidade do `OneHotEncoder` de UF/
  região de capturar nuances territoriais mais finas sem incorrer em overfitting.

## 11. Possíveis evoluções futuras

- **Enriquecer a base com fontes reais** citadas no edital: IBGE (dados populacionais/territoriais),
  Censo Escolar (infraestrutura, formação docente, dependência administrativa), FUNDEB (investimento por
  aluno), PNAD (renda familiar), Atlas do Desenvolvimento Humano (IDHM) e Cadastro Único (vulnerabilidade
  social) — substituindo diretamente `data/raw/` mantendo o mesmo pipeline de features/leakage/modelagem.
- **Aumentar a granularidade temporal**, incorporando mais anos de série histórica assim que disponíveis,
  para permitir modelos de série temporal por município (ex.: features de tendência/sazonalidade mais
  robustas, ou modelos hierárquicos bayesianos).
- **Re-treino periódico (MLOps):** dado o drift identificado, um modelo em produção precisaria de
  monitoramento de *data drift* e re-treino a cada novo ciclo de resultados, não um treino único estático.
- **Modelos hierárquicos/mistos** (ex.: efeitos aleatórios por município/UF) para capturar melhor a
  estrutura de painel sem depender apenas de features de lag manuais.
- **Calibração de probabilidade** (Platt scaling / isotonic regression) antes de expor o ranking de risco
  a gestores, garantindo que a probabilidade estimada seja diretamente interpretável como "chance real".
- **Painel interativo** (ex. Streamlit/Power BI) consumindo o ranking de risco e os clusters para uso
  direto por secretarias de educação, fora do notebook.

---

## Vídeo executivo

O roteiro cronometrado (até 5 minutos) para a apresentação executiva simulando uma reunião com
secretários de educação está em
[`reports/roteiro_video_executivo.md`](reports/roteiro_video_executivo.md).
