# googleColab
Notebook from google colab - Projeto Machine Learning para predição de dados bancários

#### - Ferramentas utilizadas:
##### - Microsoft Excel, Jupyter-notebook, Google Colab

#### - dataset explainated

#### Explicação do dataset:
#### funções criadas:

##### - transformVectorize - Função criada para transformar texto em features, usei essa função para criar a variável de entrada para predição do modelo machine learning supervisionado

##### - modelLogistcRegression - Função criada para gerar predições do modelo de machine learning onde eu passo as features da função anterior e a variável categórica

#### - recoveryLogistcRegression - Função criada para recuperar arquivo físico criado com base no modelo de machine learning, Isso torna a leitura dos dados e das predições mais rápidas

#### - confusionMatrix - Função criada para formatar a matriz de confusão com base nas variáveis de teste e predição do modelo de machine learning, com ela conseguimos analisar a precisão do modelo, o quanto as predições estão acertando positivamente.


 ### bibliotecas installadas

! pip install plotly

### libraries import

import pandas as pd
import numpy as np


## Importações das libraries de machine learning Regressão Logistica

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

## Biblioteca usada para salvar o modelo criado da Regressão Logistica em arquivo fisico (velocidade de leitura é bem maior)

from sklearn.externals import joblib

## Bibliotecas para plot de gráficos KPI

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import plotly

## Bibliotecas de formatações de algumas tabelas (Foco na matriz de confusão)

from tkinter import font
import seaborn

## Bibliotecas de importações basicas de log

import logging
import sys

## Bibliotecas de log warning - otimos o resultado no notebook

import warnings
warnings.filterwarnings('ignore')

## Funcões do method HEAD de Pandas para configurar o display

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 10)

## Questionário de Respostas do DataSet bank-full.csv

### 1. Qual profissão tem mais tendência a fazer um empréstimo? De qual tipo?
#### Resposta:

**Primeira Parte** - Analisando o dataset foi-se necessário criar um modelo de predição de dados baseados em aprendizado de maquina, o modelo escolhido foi Regressão Logística, pois o mesmo nos permite criar predições com variáveis categóricas, no caso eu usei a coluna 'LOAN' (se o cliente possui ou não empréstimo), transformei o texto do campo 'JOB' em features (variável estatística das palavras), como precisamos predizer quais são as profissões tendenciosas, podemos criar um modelo que vai predizer quais profissões tem tendência a se fazer um empréstimo.

Usei a função nativa de sklearn classification_report para analisarmos a precisão, recall, f1-score do modelo, com ele podemos avaliar o quanto o modelo está aderente ao cenário exposto (dataset bank), com base na média total em um primeiro treinamento do modelo podemos concluir que o modelo está em média acima de 50% com predições corretas



- **Segunda Parte**- usei a matriz de confusão para avaliar o recall que são os falsos positivos e os falsos negativos, com isso temos uma visão clara da aderência das predições.


#### Explicação rápida sobre as métricas da matriz de confusão:

- **Verdadeiro positivo (true positive — TP)** - Ocorre quando no conjunto real, a classe que estamos buscando foi prevista corretamente.

- **Falso Positivo - (false positive — FP)**- Ocorre quando no conjunto real, a classe que estamos buscando prever foi prevista incorretamente

- **Falso verdadeiro (true negative — TN)**-Ocorre quando no conjunto real, a classe que não estamos buscando prever foi prevista

- **Falso negativo (false negative — FN)**-Ocorre quando no conjunto real, a classe que não estamos buscando prever foi prevista incorretamente.

Analisando o **Falso positivo** da categoria não (é a variável categórica do dataset), em correlação a predição os valores estão altos o que indica que é necessário calibrar o modelo, no caso podemos pegar uma amostra maior com um maior numero de com valores mais variável para que o modelo possa se aproximar do menor ponto de erro, como faz outros modelos de machine learning (Regressão Linear) onde a função do gradiente executa diversos ciclos para aperfeiçoamento onde o índice de erro fica no ponto mais baixo de erro.

Usamos a variável de predição para popular o dataset com valores preditos, a amostra de predições foi de **13K** aproximadamente

Com os valores preditos conseguimos chegar no resultado de **61%** para profissões que tem uma tendência a fazer empréstimo, em destaque está a profissão **blue-collar** que tem **30%** de propensão à fazer empréstimos.

## 2. Fazendo uma relação entre número de contatos e sucesso da campanha quais são os pontos relevantes a serem observados?
**Resposta**
Nos podemos visualizar no fator de correlação usando o mapa de calor, optei pelo method spearman ,porque as variáveis se relacionam monotonicamente entre si, mas não necessariamente de maneira linear, e também por se tratar de previsões variáveis.

Podemos observar de acordo com as cores, a coluna "**previous**" (numero de contatos da campanha anterior) ela correlaciona com a "**pdays**" onde a contagem de dias do ultimo contato tem forte influencia no sucesso da campanha, também podemos ver que a duração da ultima chamada tem forte influencia com o **pdays** (contagem de dias do ultimo contato), eles também tem influencia no resultado de sucesso.

## 3. Baseando-se nos resultados de adesão desta campanha qual o número médio e o máximo de ligações que você indica para otimizar a adesão?
**Resposta:**
Usei o **describe** pois se trata do melhor method para calcular algumas métricas estatísticas.
- **pountcome** (status da campanha) é o melhor cenário para analisar e otimizar a adesão, com a analise numérica do algoritmo o status '**success**' e o '**other**' tem a mediana próxima uma da outra, com o menor índice de contatos que ficam entre 11 mínimo e 16 máximo de contatos, e as medianas estão próximas uma da outra.

## 4. O resultado da campanha anterior tem relevância na campanha atual?
**Resposta**
De acordo com o mapa de calor, existe uma forte correlação entre "**previous**" e "**pdays**", a quantidade de contatos está relacionada com a quantidade de dias do ultimo contato, podendo modificar o cenário atual.

## 5. Qual o fator determinante para que o banco exija um seguro de crédito?
**Resposta:**
O Fator determinante estão correlacionados diretamente entre "**age**" e o "**balance**", fiz mais 2 analises levando em consideração se o cliente possui ou não imóvel e se ele tem empréstimo no banco, o fator da idade e do balance perde força onde os dias de contato e o balance ganham mais força. Finalizando podemos concluir que se o cliente tiver uma idade acima de 40 anos e se já possuir empréstimo, pode ser o fator determinante para que o banco exija seguro de crédito

## 6) Quais são as características mais proeminentes de um cliente que possua empréstimo imobiliário?
**Resposta:**
Com base na analise anterior, utilizei as mesmas métricas para determinar as características influenciadas e traçar o perfil do cleinte que já possui um empréstimo imobiliário.
Criei alguns cenários distintos levando em considearação se o cliente é ou não casado e qual o grau de escolaridade ele possui, defini uma faixa de idade para analise, clientes maiores de 40 anos e clientes menores ou iguais a 40 anos, com isso aumentamos nossa precisão e podemos analisar melhor o balanciamento do entre as correlações.
O mapa de calor varia bastante entre os perfis mas o perfil que tem maior aderência, possuiem as seguintes caracteristicas:
Grande parte possui menos de 40 anos, não são casados, o grau de escolaridade é desconhecido ou não possui, o "balance" possui uma forte correlação entre o "balance" e "previous" pois esses são seus maiores influenciadores, podemos concluir que esse é o perfil proeminentes de um cliente que possui emprestimo imobiliario.

### links de pesquisas
https://medium.com/brdata/correla%C3%A7%C3%A3o-direto-ao-ponto-9ec1d48735fb
https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://medium.com/data-hackers/entendendo-o-que-%C3%A9-matriz-de-confus%C3%A3o-com-python-114e683ec509
