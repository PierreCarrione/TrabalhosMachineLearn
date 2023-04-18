import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#Lendo do dataset disponível em https://archive.ics.uci.edu/ml/datasets/iris
dados = pd.read_csv("iris.data", names=['sepal length','sepal width','petal length','petal width','target'])
dados.head(5)


# Verificando o número de clusters
X = np.array(dados.drop('target', axis=1))
numCluster = 6
cost = []

for i in range(1, numCluster):
    kmeans = KMeans(n_clusters=i, random_state=0, max_iter=300)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)

plt.plot(range(1, 6), cost)
plt.xlabel("Número de clusters")
plt.ylabel("Custo")
plt.show()
#A partir do gráfico, pela ténica do cotovelo, pode-se ver que 3 será um bom número para clusters


# Calculando o kmeans para 3 clusters.
kmeans = KMeans(n_clusters = 3, random_state = 0, max_iter = 500)
prediction = kmeans.fit_predict(X)
print(prediction)

dados['K-classes'] = prediction
print(dados)


# Plotando os resultados
fig, axes = plt.subplots(1,2,figsize=(15,12),constrained_layout=True)
fig.suptitle("Valores referências")
sns.scatterplot(dados['sepal length'], dados['sepal width'],hue=dados['target'],ax = axes[0])
sns.scatterplot(dados['petal length'], dados['petal width'],hue=dados['target'],ax = axes[1])
figg, axess = plt.subplots(1,2,figsize=(15,12),constrained_layout=True)
figg.suptitle("Valores obtidos pelo Kmeans")
sns.scatterplot(dados['sepal length'], dados['sepal width'],hue=dados['K-classes'],ax = axess[0])
sns.scatterplot(dados['petal length'], dados['petal width'],hue=dados['K-classes'],ax = axess[1])