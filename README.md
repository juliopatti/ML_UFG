# Estudo de Classificação de Textos com Classes Desbalanceadas

## 1) Ambiente de Desenvolvimento

```bash
# Criar e ativar o ambiente virtual
python3.11 -m venv venv
source venv/bin/activate

# Atualizar o pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Registrar kernel no Jupyter
python -m ipykernel install --user --name=venv --display-name "ml_kernel"

# Baixar o modelo do SpaCy
python -m spacy download en_core_web_sm
```

> **Observação**: o arquivo `requirements.txt` está organizado na mesma ordem dos comandos de instalação.

## 2) Organização dos Estudos

Os experimentos e análises estão documentados em **notes 0 a 8**, cobrindo: exploração do conjunto de dados, preparação e limpeza, desenho de métricas para desbalanceamento, construção de pipelines de vetorização (TF–IDF), seleção/redução de atributos, reamostragem (com ênfase em *Random Under Sampling*), validação cruzada estratificada, otimização de limiares de decisão e comparação de modelos clássicos (por exemplo, Naive Bayes, Regressão Logística e LightGBM).

## 3) Contexto do Trabalho

O avanço das aplicações de *machine learning* (ML) tem sido marcado por uma forte dependência de conjuntos de dados clássicos, muitas vezes utilizados como ponto de partida para investigações práticas e empíricas. Em ambientes controlados, diferentes técnicas alcançam desempenhos satisfatórios. Contudo, a transposição desses métodos para cenários não acadêmicos apresenta desafios substanciais, entre os quais o **desbalanceamento de classes** é um dos mais críticos. Em situações reais, raramente é possível ampliar de forma significativa o número de exemplos da classe minoritária; quando isso é viável, uma avaliação justa exige manter as proporções originais. Por isso, **amostragens estratificadas** em validação e teste tornam-se imprescindíveis.

A literatura técnica descreve diversas estratégias para mitigar o desbalanceamento, como reamostragem, ajustes de pesos de classe e abordagens híbridas. Entre as opções de reamostragem, técnicas baseadas em vizinhança e variações de *oversampling* e *undersampling* são frequentemente discutidas. Entretanto, é comum observar aplicações inadequadas — por exemplo, ao tratar atributos categóricos ou ao posicionar a reamostragem fora do fluxo correto de *cross-validation* — o que compromete a validade experimental e destaca a necessidade de rigor metodológico. Em dados de alta dimensionalidade, especialmente quando a similaridade local deixa de ser informativa, abordagens simples e eficientes como o **Random Under Sampling (RUS)** costumam oferecer uma relação custo–benefício atraente.

Esse dilema é ainda mais evidente em tarefas de **Processamento de Linguagem Natural (PLN)**. Representações clássicas como **Bag of Words (BoW)** e **Term Frequency–Inverse Document Frequency (TF–IDF)** geram matrizes esparsas de grande dimensionalidade, frequentemente com vocabulários que ultrapassam centenas de milhares de termos. Embora modelos de linguagem de larga escala (LLMs) capturem dependências semânticas com mais riqueza, sua adoção demanda infraestrutura e custo computacional elevados, o que limita o uso em muitos contextos aplicados. Além disso, nem todos os problemas apresentam natureza textual, o que reforça o valor das representações clássicas em diferentes domínios.

Apesar de suas limitações, **BoW** e **TF–IDF** permanecem relevantes na prática por contarem com implementações maduras e altamente otimizadas que manipulam estruturas esparsas de forma eficiente. Quando combinadas com **classificadores consolidados** — como Naive Bayes, Regressão Logística e algoritmos de árvores com *boosting* — essas representações formam pipelines capazes de escalar para grandes volumes de dados, mantendo custos controlados e interpretabilidade razoável.

Neste estudo, investigamos um problema de **classificação binária extremamente desbalanceado** sobre o conjunto de resumos de livros (**Book Summaries Dataset**), com a tarefa de distinguir **ficção** de **não ficção**. O desafio é acentuado pela forte desproporção (cerca de 97,5% da classe majoritária) e pela elevada dimensionalidade (vocabulário superior a 100 mil termos). As investigações abrangem tópicos centrais: escolha e desenho de **métricas apropriadas** para desbalanceamento (com ênfase em *Average Precision/AUPRC*, *Balanced Accuracy* e *Precision* macro), **seleção/redução de atributos**, **reamostragem adequada** ao longo da validação, **ajuste de limiares de decisão**, **otimização de hiperparâmetros** baseada em busca bayesiana e **validação cruzada** com estratificação para estimativas realistas de generalização.

### Contribuições

O trabalho aporta três frentes principais:

1. **Análise crítica** de métodos de reamostragem em ambientes textuais de alta dimensionalidade, discutindo custos e impactos práticos;
2. **Avaliação sistemática** de modelos clássicos de classificação em cenários de severo desbalanceamento, com métricas alinhadas ao problema;
3. **Proposição de um fluxo reprodutível e rigoroso** para estudos de classificação textual, , conciliando práticas consolidadas e empíricas, do pré-processamento ao ajuste de limiares.

## 4) Reprodutibilidade (visão geral)

- **Vetorização**: TF–IDF com pré-processamento textual simples;  
- **Seleção/Redução**: técnicas de filtragem por relevância (quando aplicáveis);  
- **Reamostragem**: RUS aplicado **dentro** dos *folds* de validação;  
- **Modelagem**: Naive Bayes, Regressão Logística e LightGBM;  
- **Otimização**: busca bayesiana de hiperparâmetros;  
- **Threshold**: ajuste de limiar com probabilidades *out-of-fold*;  
- **Métricas**: foco em AUPRC, *Balanced Accuracy* e *Precision* macro;  
- **Validação**: *k*-fold estratificado, com consolidação OOF para estimativa estável.

## 5) Licenças

- **Código e textos deste repositório**: licenciados sob **Creative Commons – Atribuição (CC BY 4.0)**.  
- **Conjunto de dados – Book Summaries Dataset**: disponibilizado sob **licença Creative Commons** pelo(s) detentor(es) do dataset. Verifique os termos específicos do dataset antes de uso comercial ou redistribuição.

---

Caso encontre algum problema de reprodutibilidade, verifique a versão do Python, do `pip` e das bibliotecas listadas no `requirements.txt`. Ajustes de semente aleatória também podem ser necessários para replicar números exatos de métricas em ambientes distintos. 
**Obs:** Utilizou-se para todas as sementes o número 2025 (ano de confecção deste trabalho)
