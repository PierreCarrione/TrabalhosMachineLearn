import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


#Link do dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# Lendo o dataset e ajustando índices sem valores
df = pd.read_csv("breast-cancer-wisconsin.data",names=['id bumber','Clump Thicknes','Cell Size','Cell Shape','Marginal Adhesion',
                   'Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])
df = df.drop('id bumber',axis = 1)
df = df.replace(["?"], 1)
print(df)

# Separando os dataset
x_data = df.loc[:,['Clump Thicknes','Cell Size','Cell Shape','Marginal Adhesion',
                   'Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']].values
print(x_data)

y_data = df.loc[: , ['Class']].values

# Treinando sem pca
x_date = StandardScaler().fit_transform(x_data)
x_train,x_teste,y_train,y_teste = train_test_split(x_date, y_data, test_size = 0.3, random_state = 42)
clf = MLPClassifier(hidden_layer_sizes = (10,5), random_state = 1, max_iter = 2000)

clf.fit(x_train,y_train)
y_predict = clf.predict(x_teste)
acuracia_inicial = accuracy_score(y_teste, y_predict)
print("Acuracia: %.2f" % (acuracia_inicial*100) + "%")


# Treinando com pca
pca = PCA(n_components = 2)
reducted = pca.fit(x_data).transform(x_data)
x_train,x_teste,y_train,y_teste = train_test_split(reducted, y_data, test_size = 0.3, random_state = 42)

clf.fit(x_train, y_train)
y_predict = clf.predict(x_teste)
acuracia_final = accuracy_score(y_teste, y_predict)

print("Acuracia: %.2f" % (acuracia_final*100) + "%")


# Resultados
print("Acuracia inicial: %.2f" % (acuracia_inicial*100) + "%")
print("Acuracia pós PCA: %.2f" % (acuracia_final*100) + "%")