# Roteiro — Vídeo Executivo (até 5 minutos)

**Formato:** simulação de reunião executiva com secretários de educação e gestores públicos, apresentando
o problema, os principais insights e o valor estratégico da solução de Machine Learning desenvolvida.

**Participantes sugeridos:** 1 a 2 apresentadores (papel de "cientista de dados" apresentando aos
"gestores públicos"). Se gravado em grupo, dividir os blocos abaixo entre os integrantes.

**Materiais de apoio a ter em tela:** `reports/images/eda_target_distribution.png`,
`reports/images/eda_regiao_vs_target.png`, tabela comparativa de modelos (Seção 7 do README),
`reports/images/shap_summary.png`, `reports/images/risk_top_municipios.png`,
`reports/images/cluster_municipios.png`.

---

## Bloco 1 — Abertura e problema educacional (0:00 – 0:45)

**Tom:** direto, contextualizando a urgência do tema para quem decide política pública, não para um
público técnico.

> "Bom dia a todos. Obrigado pelo tempo de vocês. Hoje eu quero apresentar como transformamos os dados que
> já coletamos sobre o Indicador Criança Alfabetizada em uma ferramenta prática de decisão.
>
> O problema que estamos resolvendo é simples de enunciar e difícil de resolver: **como saber, ainda
> durante o ano letivo, quais municípios correm risco de não alfabetizar suas crianças a tempo — antes que
> o resultado final chegue?**
>
> Hoje esse diagnóstico só existe *depois* do resultado. Nosso objetivo foi antecipar esse sinal."

**Tela:** `eda_target_distribution.png` — mostrar a evolução da taxa de alfabetização 2021-2023.

## Bloco 2 — O que construímos (0:45 – 1:45)

> "Partindo da base Gold que estruturamos na fase anterior — indicadores de alfabetização, metas
> municipais, estaduais e nacionais, e dados territoriais — construímos um modelo de Machine Learning que
> aprende com o histórico de cada município para estimar a probabilidade de um aluno ser alfabetizado.
>
> Testamos três abordagens diferentes: um modelo estatístico mais simples e interpretável, e dois modelos
> de conjunto de árvores mais sofisticados. O melhor deles — uma Random Forest — foi escolhido depois de um
> processo rigoroso de comparação e otimização.
>
> Um ponto que quero destacar porque é o que dá confiança nesse tipo de análise: **encontramos e corrigimos
> dois vazamentos de informação** durante a construção do modelo — casos em que, sem querer, o modelo
> poderia estar 'colando' a partir do próprio resultado que deveria prever. Corrigimos isso construindo o
> modelo para usar apenas informação que um gestor teria disponível **antes** do resultado sair, o que torna
> a ferramenta genuinamente preditiva, e não uma ilusão estatística."

**Tela:** tabela comparativa de modelos (README, Seção 7).

## Bloco 3 — Principais insights (1:45 – 3:00)

> "Três achados relevantes para orientar decisão:
>
> **Primeiro:** o fator que mais explica a alfabetização de um município é o **histórico recente do
> próprio município** — quem vinha melhorando, tende a continuar melhorando. Isso é uma boa notícia: mostra
> que investimento continuado em política municipal de alfabetização tem efeito persistente, e retrocessos
> também se propagam.
>
> **Segundo:** as **metas estaduais vigentes** aparecem entre os fatores mais relevantes — o que sugere
> que metas bem comunicadas e ambiciosas realmente correlacionam com o resultado. Vale um adendo de
> honestidade técnica aqui: nesta amostra de demonstração, a própria meta cresce de forma artificial ano a
> ano por construção do gerador de dados — então essa correlação específica não deve ser lida como prova
> de causalidade real. É um bom exemplo de por que sempre auditamos como os dados foram gerados antes de
> tirar conclusões.
>
> **Terceiro:** existe **heterogeneidade regional significativa** nos dados analisados — o que reforça um
> ponto metodológico importante: como ainda trabalhamos com uma amostra sintética de demonstração, não
> podemos usar essa base para tirar conclusão sobre qual região do Brasil está pior hoje. O valor real aqui
> é outro: já validamos que o modelo é sensível a diferenças regionais e consegue quantificar esse tipo de
> gap — assim que plugarmos dados reais do INEP, a mesma metodologia vai gerar o diagnóstico regional
> verdadeiro, região a região."

**Tela:** `shap_summary.png` (fatores de maior impacto) e `eda_regiao_vs_target.png` (desigualdade
regional).

## Bloco 4 — Da predição à ação: risco municipal e agrupamentos (3:00 – 4:00)

> "Mas o mais importante para vocês, gestores, não é o modelo — é o que ele permite fazer. Duas entregas
> práticas:
>
> A primeira é um **ranking de risco por município**, atualizado a cada novo ciclo de dados, classificando
> cada município em alto risco, risco moderado ou baixo risco de não atingir a meta de alfabetização. Isso
> permite priorizar visitas técnicas, reforço pedagógico e distribuição de recursos **antes** do resultado
> final — não depois.
>
> A segunda é um **agrupamento de municípios com padrões semelhantes** — o que nos permite recomendar
> estratégias diferenciadas por grupo, em vez de uma política única para 5.570 realidades diferentes.
> Municípios com desempenho já consolidado podem, inclusive, servir de referência e mentoria para os
> grupos que ainda enfrentam gap persistente em relação à meta."

**Tela:** `risk_top_municipios.png` e `cluster_municipios.png`.

## Bloco 5 — Limites, honestidade técnica e próximos passos (4:00 – 4:40)

> "Quero ser transparente sobre os limites desse primeiro modelo, porque isso é parte de fazer ciência de
> dados responsável em política pública: o modelo hoje tem um poder preditivo **moderado, não perfeito** —
> e isso é esperado, porque ainda não incorporamos variáveis-chave como renda familiar, infraestrutura
> escolar e formação docente, disponíveis em bases como Censo Escolar, PNAD e FUNDEB. Essa é exatamente a
> próxima etapa que propomos: enriquecer a base com essas fontes, sem precisar refazer o pipeline que já
> construímos — ele foi desenhado para isso.
>
> Também recomendamos que, uma vez em produção com dados reais, o modelo seja **retreinado a cada novo
> ciclo de dados** — indicadores educacionais mudam com o tempo, e um modelo estático perde precisão. Isso
> é ainda mais importante do que parece nesta amostra de demonstração: nós mesmos auditamos o gerador dos
> dados sintéticos e descobrimos que o forte crescimento da taxa entre 2021 e 2023 é um artefato de como a
> amostra foi construída, não um padrão real do país. Contar essa descoberta para vocês agora é parte do
> nosso compromisso de transparência: preferimos admitir os limites da base a apresentar um número
> chamativo sem explicá-lo."

## Bloco 6 — Encerramento (4:40 – 5:00)

> "Em resumo: entregamos um pipeline reprodutível, auditável e já com prevenção explícita de vieses
> técnicos, que transforma o indicador que vocês já acompanham em um **radar de risco antecipado** e em
> **agrupamentos acionáveis** de municípios. O próximo passo é validar esse radar em campo, com dados reais
> de um ou dois estados-piloto, e evoluir a base de variáveis junto com as secretarias regionais.
>
> Fico à disposição para aprofundar qualquer um desses pontos. Obrigado."

---

## Checklist de gravação

- [ ] Ensaiar cronometrando cada bloco (o roteiro soma ~5:00 no ritmo de fala pausado/executivo).
- [ ] Ter as imagens de `reports/images/` abertas/prontas para compartilhar tela na ordem indicada.
- [ ] Evitar jargão técnico não explicado (ROC-AUC, SHAP, etc.) — traduzir sempre para a implicação de
  negócio, como feito nas falas acima.
- [ ] Fechar com um pedido de próximo passo concreto (piloto com dados reais), não apenas uma conclusão
  genérica.
