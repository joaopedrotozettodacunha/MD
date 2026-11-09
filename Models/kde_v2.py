import pandas as pd 
import numpy as np

from sklearn.neighbors import KernelDensity

dados = pd.read_csv("caminho.csv")

#selecionar variáveis
features = [
    "var1", 
    "var2",
    "var3"
]

x = dados[features]

# REMOCAO DE OUTLIERS

percentual_remocao = 0.05

kde_filtro = KernelDensity(
    bandwidth = 0.5,
    kernel = "gaussian"
)

kde_filtro.fit(x)

# calcular densidades

densidade_log = kde_filtro.score_samples(x)

limiar = np.quantile(densidade_log, percentual_remocao)

# manter tuplas com densidade acima do limiar

dados_filtrados = densidade_log >= limiar

x_filtrado = x[dados_filtrados]

# MODELO KDE 

kde = KernelDensity(
    bandwidth = 0.5,
    kernel = "gaussian"
)

kde.fit(x_filtrado)

densidade_log_final = kde.score_samples(x)

dados["anomalia_score"] = -densidade_log_final

# identificar observações removidas
dados["removido_treinamento"] = ~dados_filtrados

#ordenar das mais anomala pra a menos anomala
dados = dados.sort_values(
    by = "anomalia_score",
    ascending = False
)

dados.to_csv("caminho.csv", index = False)

print("Total de observações:", len(dados))
print("Observações removidas:", (~dados_filtrados).sum())
print("Observações mantidas:", dados_filtrados.sum())

print("\nPartidas mais anômalas:")
print([["anomalia_score", "removido_treinamento"]].head(20))