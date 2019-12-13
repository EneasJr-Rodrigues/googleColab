# googleColab
Notebook from google colab - Projeto Machine Learning para predição de dados bancários

**_dataset explainated
2
explicação do dataset:_**
3

4
funções criadas:
5

6
- **transformVectorize**- Função criada para transformar texto em features, usei essa função para criar a variável de entrada para predição do modelo machine learning supervisionado
7

8
- **modelLogistcRegression**- Função criada para gerar predições do modelo de machine learning onde eu passo as features da função anterior e a variável categórica
9

10
- **recoveryLogistcRegression**-Função criada para recuperar arquivo físico criado com base no modelo de machine learning, Isso torna a leitura dos dados e das predições mais rápidas
11

12
- **confusionMatrix**-Função criada para formatar a matriz de confusão com base nas variáveis de teste e predição do modelo de machine learning, com ela conseguimos analisar a precisão do modelo, o quanto as predições estão acertando positivamente.
13

14
 **bibliotecas installadas**
15

16
! pip install plotly
17

18
**_libraries import_**
19

20
import pandas as pd
21
import numpy as np
22

23
# Importações das libraries de machine learning Regressão Logistica
24
from sklearn.linear_model import LogisticRegression
25
from sklearn.model_selection import train_test_split
26
from sklearn.feature_extraction.text import TfidfVectorizer
27
from sklearn.metrics import classification_report, confusion_matrix
28

29
# Biblioteca usada para salvar o modelo criado da Regressão Logistica em arquivo fisico (velocidade de leitura é bem maior)
30
from sklearn.externals import joblib
31

32
# Bibliotecas para plot de gráficos KPI
33
import matplotlib.pyplot as plt
34
import plotly.graph_objs as go
35
import plotly.offline as py
36
import plotly
37

38
# Bibliotecas de formatações de algumas tabelas (Foco na matriz de confusão)
39
from tkinter import font
40
import seaborn
41

42
# Bibliotecas de importações basicas de log
43
import logging
44
import sys
45

46
# Bibliotecas de log warning - otimos o resultado no notebook
47
import warnings
48
warnings.filterwarnings('ignore')
49

50
## Funcões do method HEAD de Pandas para configurar o display
51
pd.set_option('display.max_colwidth', -1)
52
pd.set_option('display.max_rows', 10)
53

54
**Questionário de Respostas do DataSet bank-full.csv**
55

56
**1. Qual profissão tem mais tendência a fazer um empréstimo? De qual tipo?**
57
Resposta:
58

59
- **Primeira Parte**-Analisando o dataset foi-se necessário criar um modelo de predição de dados baseados em aprendizado de maquina, o modelo escolhido foi Regressão Logística, pois o mesmo nos permite criar predições com variáveis categóricas, no caso eu usei a coluna 'LOAN' (se o cliente possui ou não empréstimo), transformei o texto do campo 'JOB' em features (variável estatística das palavras), como precisamos predizer quais são as profissões tendenciosas, podemos criar um modelo que vai predizer quais profissões tem tendência a se fazer um empréstimo.
60
Usei a função nativa de sklearn classification_report para analisarmos a precisão, recall, f1-score do modelo, com ele podemos avaliar o quanto o modelo está aderente ao cenário exposto (dataset bank), com base na média total em um primeiro treinamento do modelo podemos concluir que o modelo está em média acima de 50% com predições corretas
61

62
- **Segunda Parte**- usei a matriz de confusão para avaliar o recall que são os falsos positivos e os falsos negativos, com isso temos uma visão clara da aderência das predições.
63

64
- **Explicação rápida sobre as métricas da matriz de confusão:**
65

66
- **Verdadeiro positivo (true positive — TP)** - Ocorre quando no conjunto real, a classe que estamos buscando foi prevista corretamente.
67

68
- **Falso Positivo - (false positive — FP)**- Ocorre quando no conjunto real, a classe que estamos buscando prever foi prevista incorretamente
69

70
- **Falso verdadeiro (true negative — TN)**-Ocorre quando no conjunto real, a classe que não estamos buscando prever foi prevista
71

72
- **Falso negativo (false negative — FN)**-Ocorre quando no conjunto real, a classe que não estamos buscando prever foi prevista incorretamente.
73

74
Analisando o **Falso positivo** da categoria não (é a variável categórica do dataset), em correlação a predição os valores estão altos o que indica que é necessário calibrar o modelo, no caso podemos pegar uma amostra maior com um maior numero de com valores mais variável para que o modelo possa se aproximar do menor ponto de erro, como faz outros modelos de machine learning (Regressão Logística) onde a função do gradiente executa diversos ciclos para aperfeiçoamento onde o índice de erro fica no ponto mais próximo.
75

76
Usamos a variável de predição para popular o dataset com valores preditos, a amostra de predições foi de **13K** aproximadamente
77
Com os valores preditos conseguimos chegar no resultado de **61%** para profissões que tem uma tendência a fazer empréstimo, em destaque está a profissão **blue-collar** que tem **30%** de propensão à fazer empréstimos.
78

79
**2. Fazendo uma relação entre número de contatos e sucesso da campanha quais
80
são os pontos relevantes a serem observados?**
81
**Resposta**
82
Nos podemos visualizar no fator de correlação usando o mapa de calor, optei pelo method spearman ,porque as variáveis se relacionam monotonicamente entre si, mas não necessariamente de maneira linear, e também por se tratar de previsões variáveis.
83

84
Podemos observar de acordo com as cores, a coluna "**previous**" (numero de contatos da campanha anterior) ela correlaciona com a "**pdays**" onde a contagem de dias do ultimo contato tem forte influencia no sucesso da campanha, também podemos ver que a duração da ultima chamada tem forte influencia com o **pdays** (contagem de dias do ultimo contato), eles também tem influencia no resultado de sucesso.
85

86
**3. Baseando-se nos resultados de adesão desta campanha qual o número médio e
87
o máximo de ligações que você indica para otimizar a adesão?**
88
**Resposta:**
89
Usei o **describe** pois se trata do melhor method para calcular algumas métricas estatísticas.
90
O **pountcome** (status da campanha) é o melhor cenário para analisar e otimizar a adesão, com a analise numérica do algoritmo o status '**success**' e o '**other**' tem a mediana próxima uma da outra, com o menor índice de contatos que ficam entre 11 mínimo e 16 máximo de contatos, e as medianas estão próximas uma da outra.
91

92
**4. O resultado da campanha anterior tem relevância na campanha atual?**
93
**Resposta**
94
De acordo com o mapa de calor, existe uma forte correlação entre "**previous**" e "**pdays**", a quantidade de contatos está relacionada com a quantidade de dias do ultimo contato, podendo modificar o cenário atual.
95

96
**5. Qual o fator determinante para que o banco exija um seguro de crédito?**
97
**Resposta:**
98
O Fator determinante estão correlacionados diretamente entre "**age**" e o "**balance**", fiz mais 2 analises levando em consideração se o cliente possui ou não imóvel e se ele tem empréstimo no banco, o fator da idade e do balance perde força onde os dias de contato e o balance ganham mais força. Finalizando podemos concluir que se o cliente tiver uma idade acima de 40 anos e se já possuir empréstimo, pode ser o fator determinante para que o banco exija seguro de crédito
99

100
**6) Quais são as características mais proeminentes de um cliente que possua empréstimo imobiliário?**
101
Resposta:
102
O balance e a idade tem forte influencia na decisão, de acordo com o mapa de calor. Com base na analise anterior, utilizei as mesmas métricas para determinar as características influenciadas no resultado.
103
defini 2 faixas de idade para compor esse raciocínio, clientes com idades superiores a 40 anos e inferiores a 40 anos.
104
No fator de clientes abaixo de 40 anos podemos observar que o balance tem a maior relevância, já clientes acima de 40 o balance ganha maior força
105

106
**links de pesquisas**
107
https://medium.com/brdata/correla%C3%A7%C3%A3o-direto-ao-ponto-9ec1d48735fb
108
https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
109
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
110
https://medium.com/data-hackers/entendendo-o-que-%C3%A9-matriz-de-confus%C3%A3o-com-python-114e683ec509
