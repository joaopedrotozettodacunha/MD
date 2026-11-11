import pandas as pd
from sklearn.ensemble import IsolationForest

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
modelo = IsolationForest(
    n_estimators = 200,
    contamination = "auto",
    random_state = 42
)

# treinar modelo
modelo.fit(x)

# anomalia score

dados["anomalia_score"] = -modelo.score_samples(x)

# classificar como anomalo ou normal

dados["anomalia"] = modelo.predict(x)

# ordenar resultados do anomalia score

dados = dados.sort_values(
    by = "anomalia_score",
    ascending = False
)

dados.to_csv("caminho.csv", index = True)
