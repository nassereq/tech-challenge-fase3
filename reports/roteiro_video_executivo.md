# Roteiro — Vídeo Executivo (até 5 minutos)

**Formato:** simulação de reunião executiva com secretários de educação e gestores públicos, apresentando
o problema, os principais insights e o valor estratégico da solução de Machine Learning desenvolvida.

**Participantes sugeridos:** 1 a 2 apresentadores (papel de "cientista de dados" apresentando aos
"gestores públicos"). Se gravado em grupo, dividir os blocos abaixo entre os integrantes.

**Materiais de apoio a ter em tela:** os 11 slides em
`reports/apresentacao_executiva/apresentacao.html`, seguindo a mesma ordem dos blocos abaixo.

---

## Bloco 1 — Abertura e problema educacional (0:00 – 0:45)

**Tom:** direto, contextualizando a urgência do tema para quem decide política pública, não para um
público técnico.

> "Bom dia a todos. Obrigado pelo tempo de vocês. Hoje eu quero apresentar como transformamos os dados
> oficiais do Compromisso Nacional Criança Alfabetizada em uma ferramenta prática de decisão.
>
> O problema que estamos resolvendo é simples de enunciar e difícil de resolver: **como saber, ainda
> durante o ciclo de avaliação, quais municípios correm risco de não atingir sua meta de alfabetização —
> antes que o resultado final chegue?**
>
> Hoje esse diagnóstico só existe *depois* do resultado. Nosso objetivo foi antecipar esse sinal, usando
> dados reais de 5.516 municípios brasileiros."

**Tela:** Slide 1 (título) → Slide 2 (o problema).

## Bloco 2 — O que construímos (0:45 – 1:45)

> "Partindo da base Gold oficial da Base dos Dados e do CNCA — indicadores de alfabetização, metas
> municipais e estaduais, e dados territoriais — construímos um modelo de Machine Learning que aprende com
> o histórico de cada município para estimar a probabilidade de ele atingir sua meta.
>
> Testamos três abordagens diferentes: um modelo estatístico mais simples e interpretável, e dois modelos
> de conjunto de árvores mais sofisticados. O melhor deles — um XGBoost — foi escolhido depois de um
> processo rigoroso de comparação e otimização.
>
> Um ponto que quero destacar porque é o que dá confiança nesse tipo de análise: **encontramos e corrigimos
> um vazamento de informação** durante a construção do modelo — o resultado do próprio município no ano
> avaliado não podia ser usado para prevê-lo. Corrigimos isso construindo o modelo para usar apenas
> informação que um gestor teria disponível **antes** do resultado sair.
>
> E quero ser transparente sobre um outro ponto: a primeira versão deste projeto usava uma amostra
> sintética de demonstração. Ao notar um número que não fazia sentido, investigamos, confirmamos que era um
> artefato dos dados de teste, e voltamos à base real oficial da Fase 2 — os números que vocês vão ver
> agora são dados reais do Brasil, não uma simulação."

**Tela:** Slide 3 (pipeline) → Slide 4 (vazamento corrigido).

## Bloco 3 — Comparação de modelos e principal insight (1:45 – 3:00)

> "O modelo campeão atinge uma capacidade preditiva consistente — a mesma performance na validação e no
> teste, o que nos dá confiança de que não é sorte.
>
> E aqui está o achado mais importante da análise, que eu chamo de **o paradoxo do Sul**: se vocês olharem
> só a taxa de municípios que atingiram a meta por região, o Sul aparece com a **pior** taxa do Brasil —
> soaria como se o Sul tivesse o pior ensino. Mas não é isso que está acontecendo.
>
> Quando olhamos o desempenho absoluto do Rio Grande do Sul, ele está na **média** do país. O problema é
> que a meta definida para o RS é **muito mais ambiciosa**, proporcionalmente, do que a de outros estados —
> por isso só 1 em cada 10 municípios gaúchos consegue atingi-la.
>
> Compare com a Bahia: desempenho absoluto genuinamente baixo, mas meta proporcionalmente mais modesta —
> ainda assim, menos de 1 em cada 5 municípios baianos atinge a meta.
>
> São dois problemas completamente diferentes que exigem soluções diferentes: o Rio Grande do Sul talvez
> precise de mais tempo ou uma meta recalibrada; a Bahia precisa de investimento estrutural em capacidade
> de alfabetização. Tratar os dois com a mesma política seria um erro."

**Tela:** Slide 5 (comparação de modelos) → Slide 6/7 (paradoxo do Sul).

## Bloco 4 — Da predição à ação: risco municipal e agrupamentos (3:00 – 4:00)

> "Duas entregas práticas para vocês, gestores:
>
> A primeira é um **ranking de risco por município**: 1.585 municípios em alto risco, 1.722 em risco
> moderado, 2.209 em baixo risco — atualizado a cada novo ciclo de dados. Isso permite priorizar visitas
> técnicas e apoio pedagógico **antes** do resultado final.
>
> Um ponto de honestidade técnica sobre esse ranking: hoje ele é fortemente influenciado pela identidade do
> estado — os municípios de maior risco no nosso modelo são, em sua maioria, do Rio Grande do Sul, pelo
> motivo que acabei de explicar. Isso significa que o modelo já é útil para **comparar estados entre si**,
> mas ainda precisa de mais variáveis — renda, infraestrutura escolar — para diferenciar risco **dentro**
> de um mesmo estado com mais precisão.
>
> A segunda entrega é um **agrupamento de municípios em 4 perfis**: municípios de referência, que já
> superam metas ambiciosas; municípios consistentes; municípios com desempenho moderado; e municípios que
> precisam de atenção prioritária. Isso permite recomendar estratégias diferenciadas por perfil, em vez de
> uma meta única nacional."

**Tela:** Slide 8 (ranking de risco) → Slide 9 (clusters).

## Bloco 5 — Limites, honestidade técnica e próximos passos (4:00 – 4:40)

> "Quero ser transparente sobre os limites desse modelo. Primeiro: ele prevê o resultado do **município**,
> não de um aluno específico — a base de alunos individuais só existe na nuvem da equipe da Fase 2, e não
> tivemos acesso a ela nesta etapa. Segundo: ainda não incorporamos variáveis-chave como renda familiar,
> infraestrutura escolar e formação docente, que ajudariam a explicar por que municípios de um mesmo estado
> variam tanto entre si.
>
> Essa é exatamente a próxima etapa que propomos: enriquecer a base com essas fontes — Censo Escolar, PNAD,
> FUNDEB — sem precisar refazer o pipeline que já construímos, porque ele foi desenhado para isso.
>
> Também recomendamos investigar, junto à equipe responsável pelas metas, por que alguns estados recebem
> metas tão acima do seu histórico recente — um achado que veio da análise de dados e que merece uma
> conversa com quem define essas metas."

**Tela:** Slide 10 (limitações).

## Bloco 6 — Encerramento (4:40 – 5:00)

> "Em resumo: entregamos um pipeline reprodutível, auditável, construído sobre dados reais de 5.516
> municípios brasileiros, que já identificou um padrão real e acionável — o paradoxo do Sul — que passaria
> despercebido em um olhar superficial dos dados.
>
> O próximo passo é aprofundar essa análise com dados socioeconômicos reais e validar o ranking de risco
> com as secretarias estaduais. Fico à disposição para aprofundar qualquer um desses pontos. Obrigado."

**Tela:** Slide 11 (encerramento).

---

## Checklist de gravação

- [ ] Ensaiar cronometrando cada bloco (o roteiro soma ~5:00 no ritmo de fala pausado/executivo).
- [ ] Ter os slides de `reports/apresentacao_executiva/apresentacao.html` prontos, na ordem indicada.
- [ ] Evitar jargão técnico não explicado (ROC-AUC, SHAP, etc.) — traduzir sempre para a implicação de
  negócio, como feito nas falas acima.
- [ ] Fechar com um pedido de próximo passo concreto (validação com secretarias estaduais), não apenas uma
  conclusão genérica.
