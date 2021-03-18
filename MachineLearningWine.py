import pandas as pd

# Lê o arquivo wine_dataset.csv e atribui a arquivo.
arquivo = pd.read_csv('/home/yzx01/Documents/wine_dataset.csv')

# Visualiza o cabeçalho de wine dataset
arquivo.head()

# Substitui red por 0 na coluna style.
arquivo['style'] = arquivo['style'].replace('red', 0)
# Substitui white por 1 na coluna style.
arquivo['style'] = arquivo['style'].replace('white', 1)

# Coluna style atribuida a variável y.
y = arquivo['style']
# Coluna style exceto tudo que tem valor um na coluna style, atribuida a variável x.
x = arquivo.drop('style', axis=1)

from sklearn.model_selection import train_test_split

# Variáveis para treinar e testar, test_size=0.3 serve pegar apenas 30% de todos os dados para teste,
# enquanto os outros 70%, são utilizados para treinar.
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import ExtraTreesClassifier

# Algoritmo de aprendizado de máquina, para criar o modelo
modelo = ExtraTreesClassifier()
# Treina o modelo
modelo.fit(x_treino, y_treino)

# Pontua em quantos porcentos o algoritmo está acertando, baseado no que foi aprendido. Compara com o resultado real
# para ver se acertou ou errou.
resultado = modelo.score(x_teste, y_teste)
print("Acurácia: ", resultado)

#Imprime 3 valores de x_teste e y_teste. Não são os valores das linhas 400, 401, 402, 403, porque o algoritmo pega
# somente 30% de todos os valores para teste, por isso são valores aleatórios.
y_teste[400:403]
x_teste[400:403]

previsoes = modelo.predict(x_teste[400:403])
previsoes
