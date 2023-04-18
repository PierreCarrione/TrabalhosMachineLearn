import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


# Função que irá fazer o cálculo de uma regressão linear simples
def regressaoLinear(vetMatriz, vetorY):
    vectorX = np.array(vetMatriz)
    vectorY = np.array(vetorY)
    x_treino, x_teste, y_treino, y_teste = train_test_split(vectorX, vectorY, test_size=0.2, random_state=42)
    b_treino = round((len(x_treino) * np.sum([x_treino * y_treino]) - np.sum([x_treino]) * np.sum([y_treino])) / (
                len(x_treino) * np.dot(x_treino, x_treino) - np.power(np.sum([x_treino]), 2)), 2)
    a_treino = round((np.sum([y_treino]) - b_treino * np.sum([x_treino])) / len(y_treino), 2)
    b_teste = round((len(x_teste) * np.sum([x_teste * y_teste]) - np.sum([x_teste]) * np.sum([y_teste])) / (
                len(x_teste) * np.dot(x_teste, x_teste) - np.power(np.sum([x_teste]), 2)), 2)
    a_teste = round((np.sum([y_teste]) - b_teste * np.sum([x_teste])) / len(y_teste), 2)

    print("Coeficientes de treino:\nA = ", a_treino, "\nB = ", b_treino)
    print("Coeficientes de teste:\nA = ", a_teste, "\nB = ", b_teste)

    #------ Variáveis Treino ------#
    residuos_treino = []
    predicoes_treino = []
    sumQuadradoResiduos_treino = 0
    erroAbsMedio_treino = 0
    mediaValores_treino = np.sum([y_treino]) / len(y_treino)
    varianciaMediaValores_treino = 0
    erroQuadraticoMedio_treino = 0
    r2_treino = None

    #------ Variáveis Teste ------#
    residuos_teste = []
    predicoes_teste = []
    sumQuadradoResiduos_teste = 0
    erroAbsMedio_teste = 0
    mediaValores_teste = np.sum([y_teste]) / len(y_teste)
    varianciaMediaValores_teste = 0
    erroQuadraticoMedio_teste = 0
    r2_teste = None


    #-------------------------------------- Treino --------------------------------------#
    for i in range(len(x_treino)):
        predicoes_treino.append(round((b_treino * x_treino[i]) + a_treino, 2))
        residuos_treino.append(round(predicoes_treino[i] - y_treino[i], 2))
        sumQuadradoResiduos_treino = sumQuadradoResiduos_treino + round(pow(residuos_treino[i], 2), 2)
        erroAbsMedio_treino = erroAbsMedio_treino + round(abs(residuos_treino[i]), 2)
        varianciaMediaValores_treino = varianciaMediaValores_treino + round(
            pow(y_treino[i] - mediaValores_treino, 2), 2)


    #-------------------------------------- Teste --------------------------------------#
    for i in range(len(x_teste)):
        predicoes_teste.append(round((b_teste * x_teste[i]) + a_teste, 2))
        residuos_teste.append(round(predicoes_teste[i] - y_teste[i], 2))
        sumQuadradoResiduos_teste = sumQuadradoResiduos_teste + round(pow(residuos_teste[i], 2), 2)
        erroAbsMedio_teste = erroAbsMedio_teste + round(abs(residuos_teste[i]), 2)
        varianciaMediaValores_teste = varianciaMediaValores_teste + round(pow(y_teste[i] - mediaValores_teste, 2),
                                                                          2)

    #----------------------------- Treino -----------------------------#
    sumQuadradoResiduos_treino = round(sumQuadradoResiduos_treino, 2)
    erroQuadraticoMedio_treino = sumQuadradoResiduos_treino / len(y_treino)  # MSE
    erroQuadraticoMedio_treino = round(erroQuadraticoMedio_treino, 2)
    erroAbsMedio_treino = erroAbsMedio_treino / len(y_treino)  # MAE
    erroAbsMedio_treino = round(erroAbsMedio_treino, 2)
    r2_treino = 1 - (sumQuadradoResiduos_treino / varianciaMediaValores_treino)
    r2_treino = round(r2_treino, 2)

    #----------------------------- Teste -----------------------------#
    sumQuadradoResiduos_teste = round(sumQuadradoResiduos_teste, 2)
    erroQuadraticoMedio_teste = sumQuadradoResiduos_teste / len(y_teste)  # MSE
    erroQuadraticoMedio_teste = round(erroQuadraticoMedio_teste, 2)
    erroAbsMedio_teste = erroAbsMedio_teste / len(y_teste)  # MAE
    erroAbsMedio_teste = round(erroAbsMedio_teste, 2)
    r2_teste = 1 - (sumQuadradoResiduos_teste / varianciaMediaValores_teste)
    r2_teste = round(r2_teste, 2)

    print("---------------------------- Valores Treino ----------------------------")
    print("Soma do Erro Quadrático(SSE) : ", sumQuadradoResiduos_treino, "\nErro Absoluto Médio(MAE) : ",
          erroAbsMedio_treino,
          "\nErro Quadratico Médio(MSE) : ", erroQuadraticoMedio_treino)
    print("R2 : ", r2_treino)
    print("---------------------------- Valores Teste ----------------------------")
    print("Soma do Erro Quadrático(SSE) : ", sumQuadradoResiduos_teste, "\nErro Absoluto Médio(MAE) : ",
          erroAbsMedio_teste,
          "\nErro Quadratico Médio(MSE) : ", erroQuadraticoMedio_teste)
    print("R2 : ", r2_teste)

    plt.subplot(1, 2, 1)
    plt.scatter(x_treino, y_treino, color='blue')#plotando os pontos no grafico
    plt.title("Modelo Treino")
    plt.plot(x_treino, b_treino * x_treino + a_treino)#plotando a equação no grafico
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.subplot(1, 2, 2)
    plt.scatter(x_teste, y_teste, color='blue')
    plt.title("Modelo Teste")
    plt.plot(x_teste, b_teste * x_teste + a_teste)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.show()
#---------------------------- Fim função regressaoLinear ----------------------------#


# Função que irá cálcular o Gradiente Descendente
def gradienteDescedente(matrisX, vetorY, learningRate):
    matrizX = np.array(matrisX)
    vectorY = np.array(vetorY)
    maior = 0
    menor = 99999
    thetas = []
    residuos = []
    predicoes = None
    sumQuadradoResiduos = 0#SSE
    erroAbsMedio = 0#MAE
    mediaValores = np.sum([vectorY]) / len(vectorY)
    varianciaMediaValores = 0
    erroQuadraticoMedio = None#MSE
    r2 = None
    custoInicial = 0
    custoNovo = 0

    for i in range(len(matrizX)):#Verificando o maior e menor valor para servir de seed para o random
        for j in range(len(matrizX[i])):
            if (matrizX[i][j] > maior):
                maior = matrizX[i][j]
            if (matrizX[i][j] < menor):
                menor = matrizX[i][j]

    maior = int(maior)
    menor = int(menor)
    thetas.append(random.randrange(menor,maior))#Valor de theta0

    for i in range(len(matrizX[i])):#Gerando os demais thetas
        thetas.append(random.randrange(menor, maior))

    print("Thetas iniciais gerados aleatoriamente \u03F40 , \u03F41, ... ,\u03F4n: ", thetas)
    thetas = np.array(thetas)
    derivadaCusto = np.array(np.full(len(thetas), 0))
    predicoes = np.array(np.full(len(vectorY), 0))
    residuos = np.array(np.full(len(vectorY), 0))

    # Gerando as predições pelo thetas gerados
    for i in range(len(matrizX)):
        for j in range(len(matrizX[i])):
            predicoes[i] = predicoes[i] + thetas[j + 1] * matrizX[i][j]
        predicoes[i] = predicoes[i] + thetas[0]
        residuos[i] = predicoes[i] - vectorY[i]
        custoInicial = custoInicial + pow(predicoes[i] - vectorY[i],2)
        sumQuadradoResiduos = sumQuadradoResiduos + pow(residuos[i], 2)
        erroAbsMedio = erroAbsMedio + abs(residuos[i])
        varianciaMediaValores = varianciaMediaValores + pow(vectorY[i] - mediaValores, 2)

    custoInicial = custoInicial / len(vectorY)

    for i in range(len(residuos)):
        for j in range(len(derivadaCusto)):
            if j == 0:
                derivadaCusto[j] = derivadaCusto[j] + residuos[i]
            else:
                derivadaCusto[j] = derivadaCusto[j] + (residuos[i] * matrizX[i][j - 1])

    for i in range(len(derivadaCusto)):
        derivadaCusto[i] = (derivadaCusto[i] * 2) / len(vectorY)#Calculando o somatório das derivadas
        thetas[i] = thetas[i] - (learningRate * derivadaCusto[i])#Calculando novo theta

    contador = 0
    plt.scatter(contador, custoInicial)
    plt.xlabel("Iterações")
    plt.ylabel("Custo")
    print("Custo inicial : ", custoInicial)

    while True:
        predicoes.fill(0)
        residuos.fill(0)
        sumQuadradoResiduos = 0
        erroAbsMedio = 0
        varianciaMediaValores = 0

        for i in range(len(matrizX)):
            for j in range(len(matrizX[i])):
                predicoes[i] = predicoes[i] + thetas[j + 1] * matrizX[i][j]
            predicoes[i] = predicoes[i] + thetas[0]
            residuos[i] = predicoes[i] - vectorY[i]
            custoNovo = custoNovo + pow(predicoes[i] - vectorY[i], 2)
            sumQuadradoResiduos = sumQuadradoResiduos + pow(residuos[i], 2)
            erroAbsMedio = erroAbsMedio + abs(residuos[i])
            varianciaMediaValores = varianciaMediaValores + pow(vectorY[i] - mediaValores, 2)

        custoNovo = custoNovo / len(vectorY)
        print("Custo novo : ", custoNovo)
        contador += 1
        plt.scatter(contador, custoNovo)
        plt.pause(1)
        plt.draw()
        for i in range(len(residuos)):
            for j in range(len(derivadaCusto)):
                if j == 0:
                    derivadaCusto[j] = derivadaCusto[j] + residuos[i]
                else:
                    derivadaCusto[j] = derivadaCusto[j] + (residuos[i] * matrizX[i][j - 1])

        for i in range(len(derivadaCusto)):
            derivadaCusto[i] = (derivadaCusto[i] * 2) / len(vectorY)#Calculando o somatório das derivadas
            thetas[i] = thetas[i] - (learningRate * derivadaCusto[i])#calculando novo theta

        if custoInicial - custoNovo <= 0.01:
            break
        else:
            custoInicial = custoNovo
            custoNovo = 0

    plt.show()
    erroQuadraticoMedio = sumQuadradoResiduos / len(vectorY)#MSE
    erroAbsMedio = erroAbsMedio / len(vectorY)#MAE
    r2 = 1 - (sumQuadradoResiduos / varianciaMediaValores)
    print("Soma do Erro Quadrático(SSE) : ", sumQuadradoResiduos, "\nErro Absoluto Médio(MAE) : ", erroAbsMedio,
          "\nErro Quadratico Médio(MSE) : ", erroQuadraticoMedio)
    print("R2 : ", r2)
#---------------------------- Fim função gradiente ----------------------------#


# Lendo o dataset disponível em https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/RegressaoLinear/FuelConsumptionCo2.csv
df = pd.read_csv("FuelConsumptionCo2.csv")
vetorX = np.array(df[['ENGINESIZE','FUELCONSUMPTION_COMB']])

# Resolução por regressão linear
regressaoLinear(df['ENGINESIZE'].tolist(),df['CO2EMISSIONS'].tolist())

# Resolução pelo gradiente Descendente
gradienteDescedente(vetorX,df['CO2EMISSIONS'].tolist(),0.001)