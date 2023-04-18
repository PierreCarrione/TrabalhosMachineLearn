import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# Lendo o csv e ajustando os dados
# Dataset: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
df = pd.read_csv("fetal_health.csv")

x_data = np.array(df[['baseline value','abnormal_short_term_variability',
                      'percentage_of_time_with_abnormal_long_term_variability',
                      'histogram_mean','histogram_median']])
y_data = np.array(df['fetal_health'])
print(df.describe())


# Padronizando os dados pelo zscore e separando o dataset
scaler = StandardScaler()
scaler.fit(x_data)
x_scaled = scaler.transform(x_data)

x_treino, x_teste, y_treino, y_teste = train_test_split(x_scaled,y_data, test_size=0.3, random_state=42)
x_teste, x_desenvolvimento,y_teste,y_desenvolvimento = train_test_split(x_teste, y_teste, test_size=0.5, random_state=42)


# Rede neural
clf_1 = MLPClassifier(hidden_layer_sizes=(200,100),activation='identity',learning_rate='constant').fit(x_treino,y_treino)
clf_2 = MLPClassifier(hidden_layer_sizes=(160,80),activation='tanh',learning_rate='invscaling').fit(x_teste,y_teste)
clf_3 = MLPClassifier(hidden_layer_sizes=(300,150),activation='relu', learning_rate='adaptive').fit(x_desenvolvimento,y_desenvolvimento)

y_treino_predict = clf_1.predict(x_treino)
y_teste_predict = clf_2.predict(x_teste)
y_desenvolvimento_predict = clf_3.predict(x_desenvolvimento)


# Regressão Logística
reglog= LogisticRegression()
reglog.fit(x_teste,y_teste)
y_teste_reglog_predict = reglog.predict(x_teste)


# Resultado dataset teste Rede Neural
print(classification_report(y_teste, y_teste_predict))
plt.plot(clf_2.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# Resultado dataset teste Regressão Logística
print(classification_report(y_teste, y_teste_reglog_predict))
plt.plot()
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# Matriz de confusão rede neural
confusion_matrix(y_teste,y_teste_predict)
matrix = plot_confusion_matrix(clf_2,x_teste,y_teste)
print(matrix)


# Matriz de confusão Regressão Logística
confusion_matrix(y_teste,y_teste_reglog_predict)
matrix = plot_confusion_matrix(reglog,x_teste,y_teste)
print(matrix)