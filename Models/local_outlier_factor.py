import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# carregar base de dados

dados = pd.read_csv("caminho.csv")

# variáveis numéricas do modelo

features = [
    "var1", 
    "var2",
    "var3"
]

x = dados[features]


# criando o modelo
modelo = LocalOutlierFactor(
    n_neighbors = 20,
    contamination = "auto"
)

# treinar modelo
modelo.fit_predict(x)

# anomalia score

dados["anomalia_score"] = -modelo.negative_outlier_factor_(x)

# classificar como anomalo ou normal

dados["anomalia"] = modelo.fit_predict(x)

# ordenar resultados do anomalia score

dados = dados.sort_values(
    by = "anomalia_score",
    ascending = False
)

dados.to_csv("caminho.csv", index = True)
