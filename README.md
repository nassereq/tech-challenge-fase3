# Tech Challenge — Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto integrador da Fase 3 da Pós Tech (FIAP), que utiliza a camada **Gold real** de dados construída na
Fase 2 (Base dos Dados / CNCA) para desenvolver um **modelo supervisionado de Machine Learning** capaz de
prever se um **município** brasileiro atingirá sua meta de alfabetização, e para gerar inteligência
analítica aplicável a políticas públicas educacionais.

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
mais impactam os indicadores educacionais** antes que o ciclo de avaliação termine.

Na Fase 2 deste Tech Challenge foi construído o pipeline de engenharia de dados (camadas Bronze → Silver →
Gold) que integrou o Indicador Criança Alfabetizada a metas nacionais/estaduais/municipais e a dimensões
territoriais, usando dados oficiais da **Base dos Dados / CNCA** (Compromisso Nacional Criança
Alfabetizada). Nesta Fase 3, esses dados Gold viram matéria-prima analítica.

**Objetivo analítico:** desenvolver um modelo supervisionado de classificação binária capaz de prever se um
**município** atingirá **(1)** ou não atingirá **(0)** sua meta de alfabetização, a partir de variáveis
territoriais e de metas — e usar esse modelo para responder às perguntas estratégicas do desafio:

- Quais fatores mais impactam a alfabetização?
- Quais municípios apresentam maior risco de não atingir metas educacionais?
- Quais regiões possuem padrões semelhantes (agrupamentos)?
- Como prever municípios que podem não atingir metas futuras?

## 2. Descrição da base utilizada

### 2.1 Como chegamos a esta base (transparência sobre uma correção de rota)

A primeira versão deste pipeline foi construída sobre `data/raw/sample` e `data/raw/gold` — uma **amostra
sintética de demonstração** (81 municípios fictícios, 2021-2023), gerada pelo script
`pipelines/batch/generate_sample_data.py` da Fase 2 para permitir rodar a pipeline localmente, sem
credenciais de nuvem. Ao revisar os resultados, notamos uma taxa de alfabetização que quase dobrava entre
2021 e 2023 — implausível demais para ser real. Investigamos o gerador e confirmamos: o salto era um
**artefato mecânico da amostra sintética** (uma meta que cresce ~10 p.p./ano por construção, alimentando um
sorteio de proficiência com corte fixo), sem relação com nenhum fenômeno educacional real.

Essa investigação nos levou a procurar, dentro do próprio repositório da Fase 2, uma fonte melhor — e a
encontramos: `reports/gold_preview/`, um espelho real da camada Gold, gerado a partir de dados oficiais da
Base dos Dados / CNCA, cobrindo **5.516 municípios brasileiros reais** (códigos IBGE genuínos) em
2023-2024. **É essa a base usada nesta versão do projeto.**

### 2.2 Por que o alvo é o município, e não o aluno

O enunciado do desafio menciona prever se um "aluno" será alfabetizado. Investigamos a disponibilidade de
dados e confirmamos que a tabela real de alunos individuais existe apenas no BigQuery da Fase 2 (nunca foi
exportada localmente como arquivo) — apenas o indicador agregado por município está disponível como CSV.
Modelar no nível de **município** não é uma limitação de conveniência: é exatamente uma das perguntas de
negócio explícitas do desafio ("quais municípios apresentam maior risco?", "como prever municípios que
podem não atingir metas futuras?"), respondida aqui com dados **100% reais**, em vez de uma proxy sintética
em nível de aluno.

### 2.3 Tabelas utilizadas

| Arquivo | Granularidade | Descrição |
|---|---|---|
| `gold_preview/evolucao_temporal_municipio.csv` | município x ano (2023-2024) | Indicador realizado, meta, gap e se atingiu a meta — **fonte principal** |
| `gold_preview/evolucao_temporal_uf.csv` | UF x ano | Indicador agregado estadual, usado para o histórico defasado da UF |
| `gold_preview/comparativo_meta_resultado_uf.csv` | UF x ano | Meta estadual vigente por ano |
| `gold_preview/comparativo_meta_resultado_brasil.csv` | Brasil x ano | Contexto nacional (não usado como feature, apenas EDA) |

**Fonte oficial:** Base dos Dados / CNCA (Compromisso Nacional Criança Alfabetizada). Ponto de corte SAEB
de referência: 743 pontos (não utilizado diretamente nesta modelagem em nível de município).

## 3. Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/
│   │   ├── gold_preview/         # Dados Gold REAIS (Base dos Dados / CNCA) -- fonte principal
│   │   ├── gold/                 # Amostra sintetica (versao anterior, mantida por proveniencia)
│   │   └── sample/               # Amostra sintetica (versao anterior, mantida por proveniencia)
│   └── processed/                # Reservado para datasets intermediarios (nao versionado)
├── notebooks/
│   └── 01_pipeline_predicao_alfabetizacao.ipynb   # Notebook integrador, executado ponta a ponta
├── src/
│   ├── preprocessing/
│   │   ├── data_loading.py       # Carga dos dados reais, engenharia de atributos, lag features, split
│   │   └── pipeline.py           # ColumnTransformer (imputacao + encoding + scaling)
│   ├── modeling/
│   │   ├── train.py              # 3 modelos + GridSearch/RandomizedSearch + StratifiedKFold
│   │   └── risk_analysis.py      # Ranking de risco municipal + clustering regional (K-Means)
│   ├── evaluation/
│   │   └── metrics.py            # ROC-AUC, PR-AUC, F1, precisao, recall, matriz de confusao
│   └── visualization/
│       ├── eda_plots.py          # Graficos diagnosticos de EDA
│       └── shap_plots.py         # Feature Importance + SHAP (summary/bar/waterfall/dependence)
├── reports/
│   ├── images/                   # Todos os graficos gerados pelo pipeline
│   ├── apresentacao_executiva/   # Slides de apoio ao video executivo (HTML autocontido)
│   └── roteiro_video_executivo.md
├── requirements.txt
├── README.md
└── .gitignore
```

> A amostra sintética original (`data/raw/sample`, `data/raw/gold`) foi **mantida no repositório** por
> proveniência e para documentar o processo de investigação descrito na Seção 2.1, mas **não é mais usada**
> pelo pipeline atual.

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

Para o ano-alvo (2024, quando meta e resultado estão disponíveis para todos os 5.516 municípios), a
tabela é enriquecida com:

- `pct_alfabetizados_lag1`: desempenho do **próprio município** em 2023 (defasado);
- `pct_alfabetizados_uf_lag1`: desempenho agregado da **UF** em 2023 (defasado);
- `meta_pct`: meta vigente do município para 2024 (definida a priori pelo CNCA);
- `meta_pct_uf`: meta vigente da UF para 2024;
- `sigla_uf`, `regiao` (derivada da UF), `nivel_meta` (se a meta é específica do município ou um
  *fallback* estadual).

### 5.2 Vazamento identificado e tratado

`atingiu_meta` (2024) é definido deterministicamente por `pct_alfabetizados_2024 >= meta_pct_2024`
(confirmado com **100% de correspondência** nos dados). Por isso `pct_alfabetizados` do ano-alvo,
`gap_meta_pct`, `delta_pp_ano_anterior` e `n_avaliados` foram **excluídos do conjunto de features** — o
modelo só pode usar informação que já existiria *antes* do resultado de 2024.

Essa defasagem gera `NaN` estruturais para os 597 municípios (10,6%) que só aparecem no ano-alvo, sem uma
linha de 2023 correspondente — tratados explicitamente na imputação (não são erros de coleta, são ausência
estrutural esperada, provavelmente municípios que entraram na cobertura da avaliação em 2024).

### 5.3 Divisão treino/teste

Diferente de uma base em painel multi-ano, esta é um **corte transversal**: todos os municípios são
avaliados no mesmo ano-alvo (2024), sem repetição entre linhas. A divisão treino/teste é uma amostragem
**aleatória estratificada pelo alvo** (80% / 20%) — a escolha correta e padrão da literatura para esse
cenário, sem necessidade de agrupamento por município (diferente da versão sintética anterior).

### 5.4 Pipeline Scikit-learn integrado

- **Numéricas:** `SimpleImputer(strategy="median")` + `StandardScaler`.
- **Categóricas** (`sigla_uf`, `regiao`, `nivel_meta`): `SimpleImputer(strategy="most_frequent")` +
  `OneHotEncoder`.

Pré-processamento e modelo vivem em um único `sklearn.pipeline.Pipeline`, nunca ajustados fora dele.

## 6. Metodologia de modelagem

| Modelo | Papel | Biblioteca |
|---|---|---|
| Regressão Logística regularizada (L2) | Baseline linear, interpretável | `scikit-learn` |
| Random Forest | Ensemble de árvores (bagging) | `scikit-learn` |
| XGBoost | Gradient Boosting | `xgboost` |

**Validação cruzada:** `StratifiedKFold` (5 folds) — suficiente aqui porque cada linha é um município
distinto, sem estrutura de painel a proteger.

**Otimização de hiperparâmetros:** `GridSearchCV` para a Regressão Logística e `RandomizedSearchCV`
(40 iterações) para Random Forest e XGBoost, otimizando **ROC-AUC**.

## 7. Métricas de avaliação e comparação de modelos

Métricas calculadas no holdout estratificado (20% dos municípios, n=1.104):

| Modelo | ROC-AUC | PR-AUC | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | **0,787** | **0,811** | 0,729 | 0,719 | 0,740 |
| Random Forest | 0,786 | 0,805 | 0,703 | 0,722 | 0,685 |
| Regressão Logística | 0,777 | 0,796 | 0,715 | 0,697 | 0,733 |

⭐ **Modelo campeão: XGBoost** (maior ROC-AUC no holdout).

**ROC-AUC em validação cruzada (StratifiedKFold, treino):** Regressão Logística 0,776 | Random Forest
0,780 | XGBoost 0,787 — muito próximo do holdout, **sem sinal de overfitting**.

### Leitura dos resultados

Com dados reais, o desempenho é substancialmente melhor e mais estável do que na primeira versão (baseada
em amostra sintética, que alcançava ROC-AUC ~0,58): os atributos reais desta base (histórico
municipal/estadual, meta vigente, UF) carregam sinal genuíno e substancial sobre o resultado. A
proximidade entre CV e holdout mostra que o resultado é robusto, não uma coincidência do split.

## 8. Interpretação dos resultados (SHAP) e insights de negócio

Os 5 atributos de maior importância (gain, XGBoost campeão):

| Ranking | Atributo | Importância |
|---|---|---|
| 1º | `sigla_uf_RS` (município é do Rio Grande do Sul) | 16,3% |
| 2º | `sigla_uf_MG` (município é de Minas Gerais) | 14,4% |
| 3º | `sigla_uf_BA` (município é da Bahia) | 13,2% |
| 4º | `regiao_Sul` | 9,2% |
| 5º | `pct_alfabetizados_uf_lag1` (desempenho da UF em 2023) | 5,7% |

**O paradoxo do Sul (achado central desta análise):** o modelo campeão apoia suas decisões fortemente em
dummies de UF específicas — um reflexo direto de um padrão real e verificado nos dados: o **Rio Grande do
Sul (RS)** tem desempenho histórico absoluto **mediano** (`pct_alfabetizados_lag1` médio ≈ 53,8%, não é a
pior UF do Brasil nesse quesito), mas sua **meta média é desproporcionalmente alta** (≈ 71,4%, um gap de
-17,6 p.p.) — por isso apenas **10,5%** dos municípios gaúchos atingem a meta em 2024, a **menor taxa entre
todas as UFs**.

Isso é bem diferente do perfil da **Bahia (BA)**: desempenho absoluto genuinamente baixo
(`pct_alfabetizados_lag1` ≈ 36,6%) combinado com uma meta proporcionalmente mais modesta (≈ 43,7%), ainda
assim insuficiente para boa parte dos municípios baianos (apenas 18,9% atingem a meta). O **Ceará (CE)**
ilustra o caso de sucesso: desempenho absoluto altíssimo (90,3%) superando uma meta também ambiciosa
(78,9%) — consistente com a fama nacional do programa estadual de alfabetização do Ceará (PAIC), que
inspirou o próprio Compromisso Nacional Criança Alfabetizada.

**Resposta à pergunta de negócio "quais fatores mais impactam a alfabetização?"**: nesta base real,
**existem dois perfis distintos de risco educacional** que um número único (taxa de alfabetização) esconde:
(1) municípios com desempenho absoluto genuinamente baixo, que precisam de investimento estrutural em
capacidade de alfabetização (perfil Bahia); e (2) municípios com desempenho absoluto razoável, mas metas
desproporcionalmente ambiciosas em relação à sua trajetória, que precisam de aceleração ou recalibração de
meta (perfil Rio Grande do Sul). Confundir os dois perfis levaria a políticas públicas erradas.

## 9. Aplicação prática para políticas públicas

1. **Ranking de risco municipal:** o modelo estima a probabilidade de cada município atingir sua meta em
   2024, classificando-os em **Alto risco / Risco moderado / Baixo risco**
   (`reports/images/risk_top_municipios.png`; 1.585 municípios em alto risco, 1.722 em risco moderado,
   2.209 em baixo risco). **Limitação honesta:** os 15 municípios de maior risco no ranking são todos do
   RS — reflexo do peso que o modelo dá à UF. O ranking prioriza bem *entre* estados, mas precisa de mais
   variáveis para diferenciar risco *dentro* de um mesmo estado (Seção 10).
2. **Agrupamento regional (K-Means, k=4):** os municípios se dividem em 4 perfis nítidos — **Referência**
   (desempenho e meta altos, já superando), **Consistente** (desempenho e meta compatíveis), **Moderado**,
   e **Atenção prioritária** (desempenho baixo) — permitindo estratégias diferenciadas por grupo em vez de
   uma meta única nacional.

Ambos os artefatos respondem diretamente às perguntas "quais municípios apresentam maior risco
educacional?", "quais regiões possuem padrões semelhantes?" e "como prever municípios que podem não
atingir metas futuras?" propostas no desafio.

## 10. Limitações do projeto

- **Granularidade municipal, não individual:** a tabela real de alunos só existe no BigQuery da Fase 2
  (nunca exportada localmente); o modelo prevê o resultado agregado do município, não de um aluno
  específico.
- **Corte transversal de um único ano-alvo (2024):** apenas 2023 está disponível como ano de referência
  (lag), limitando a capacidade de capturar tendências de médio prazo por município.
- **Ausência de covariáveis socioeconômicas reais:** a base Gold desta fase não inclui renda, IDH,
  infraestrutura escolar ou formação docente — variáveis que poderiam explicar melhor a variação *dentro*
  de uma mesma UF (ver "paradoxo do Sul" acima).
- **Dependência forte da identidade da UF:** o modelo campeão usa dummies de UF como atributos de maior
  peso, o que funciona bem para comparar estados, mas produz um ranking de risco pouco diferenciado dentro
  de um mesmo estado (todos os 15 municípios de maior risco são do RS).
- **597 municípios sem histórico de 2023:** para esses casos, o modelo depende mais fortemente da meta e
  do contexto estadual do que do histórico próprio, aumentando a incerteza da predição individual.

## 11. Possíveis evoluções futuras

- **Enriquecer a base com fontes reais** citadas no edital: IBGE (dados populacionais/territoriais),
  Censo Escolar (infraestrutura, formação docente, dependência administrativa), FUNDEB (investimento por
  aluno), PNAD (renda familiar), Atlas do Desenvolvimento Humano (IDHM) — para diferenciar risco *dentro*
  de uma mesma UF, resolvendo a limitação central identificada nesta versão.
- **Obter acesso à tabela real de alunos no BigQuery** para retornar à granularidade individual pedida no
  enunciado original, mantendo a mesma metodologia de prevenção de leakage.
- **Aumentar a granularidade temporal**, incorporando mais anos de série histórica do CNCA assim que
  disponíveis, para features de tendência mais robustas por município.
- **Investigar a política de definição de metas** (por que RS recebe metas tão acima de seu histórico
  recente?) junto às áreas de negócio/CNCA — um achado desta análise que merece validação qualitativa.
- **Modelos hierárquicos** (efeitos aleatórios por UF) para separar explicitamente o efeito de política de
  metas por estado do efeito de características do próprio município.
- **Painel interativo** (ex. Streamlit/Power BI) consumindo o ranking de risco e os clusters para uso
  direto por secretarias de educação, fora do notebook.

---

## Vídeo executivo

O roteiro cronometrado (até 5 minutos) para a apresentação executiva simulando uma reunião com
secretários de educação está em
[`reports/roteiro_video_executivo.md`](reports/roteiro_video_executivo.md).

Os slides de apoio (11 telas, seguindo os 6 blocos do roteiro) estão em
[`reports/apresentacao_executiva/apresentacao.html`](reports/apresentacao_executiva/apresentacao.html) —
um arquivo único e autocontido: basta abrir com dois cliques em qualquer navegador (não depende de
internet, conta ou instalação). Use as setas do teclado (ou os botões na tela) para navegar, `F` para
tela cheia. O código-fonte editável de cada slide (`.dc.html`) fica na mesma pasta.
