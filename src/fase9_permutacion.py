"""
Fase 9 (paso 5) - Prueba de permutacion agrupada por dimension, 1 000 repeticiones.

  - Modelo: el modelo final ajustado (fase9_ajuste_hiperparametros.py),
    entrenado una sola vez con 2012-2021. Aqui NO se reentrena: solo predice.
  - Datos: toda la prueba 2022-2025.
  - En cada repeticion se sortea UN orden de filas y se aplica a TODAS las
    columnas de la dimension a la vez (la cobertura entra con todas sus
    variables indicadoras). Las otras dimensiones no se tocan.
  - Caida = AUC-ROC original - AUC-ROC con la dimension permutada.
  - Valor p = (1 + repeticiones con caida <= 0) / (1 000 + 1). Significativa si p < 0.05.
    El minimo posible es 1 / 1 001 = 0.000999.
  - Semilla 42: cada dimension usa un generador nuevo con semilla 42.

Guarda el avance al terminar cada dimension (permutacion_avance.csv). Si el
script se interrumpe, al volver a correrlo salta las dimensiones ya hechas.

Genera (en resultados/fase9/):
  permutacion_avance.csv (las 1 000 repeticiones de cada dimension)
  permutacion_1000.csv, fig_permutacion_1000.png
"""
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import SEED
from fase8_interpretacion import HIPOTESIS, INDICADORES_POR_DIMENSION, columnas_de_indicador
from fase8_modelo import NOMBRES
from fase9_comun import CARPETA_FASE9, datos_separados

N_REPETICIONES = 1_000
ALFA = 0.05
RUTA_AVANCE = CARPETA_FASE9 / "permutacion_avance.csv"


def permutar_dimension(modelo, X, y, columnas):
    rng = np.random.default_rng(SEED)
    filas = []
    inicio = time.time()
    for r in range(1, N_REPETICIONES + 1):
        orden = rng.permutation(len(X))
        X_perm = X.copy()
        X_perm[columnas] = X[columnas].to_numpy()[orden]  # misma permutacion para todas las columnas
        prob = modelo.predict_proba(X_perm)[:, 1]
        filas.append({"repeticion": r, "auc_roc": roc_auc_score(y, prob), "auc_pr": average_precision_score(y, prob)})
        if r % 100 == 0:
            print(f"    {r}/{N_REPETICIONES} ({time.time() - inicio:.0f} s)", flush=True)
    return pd.DataFrame(filas)


def main():
    paquete = joblib.load(CARPETA_FASE9 / "modelo_final_ajustado.joblib")
    modelo, cols_features = paquete["modelo"], paquete["cols_features"]
    _, _, prueba, _, cols_clase = datos_separados()
    X, y = prueba[cols_features], prueba["positivo"]

    prob_base = modelo.predict_proba(X)[:, 1]
    roc_base, pr_base = roc_auc_score(y, prob_base), average_precision_score(y, prob_base)
    print(f"Modelo final: {NOMBRES[paquete['algoritmo']]}. Prueba 2022-2025: {len(X)} filas, {int(y.sum())} con incendio. "
          f"AUC-ROC original {roc_base:.4f}, AUC-PR original {pr_base:.4f}")

    avance = pd.read_csv(RUTA_AVANCE) if RUTA_AVANCE.exists() else pd.DataFrame()
    for dimension, indicadores in INDICADORES_POR_DIMENSION.items():
        if len(avance) and (avance["dimension"] == dimension).sum() == N_REPETICIONES:
            print(f"{HIPOTESIS[dimension]} {dimension}: ya estaba hecha, se salta")
            continue
        columnas = [c for ind in indicadores for c in columnas_de_indicador(ind, cols_clase)]
        print(f"{HIPOTESIS[dimension]} {dimension} ({len(columnas)} columnas: {columnas})", flush=True)
        resultado = permutar_dimension(modelo, X, y, columnas)
        resultado.insert(0, "dimension", dimension)
        avance = pd.concat([avance[avance["dimension"] != dimension] if len(avance) else avance, resultado], ignore_index=True)
        avance.to_csv(RUTA_AVANCE, index=False)

    filas = []
    for dimension in INDICADORES_POR_DIMENSION:
        sub = avance[avance["dimension"] == dimension]
        caida_roc, caida_pr = roc_base - sub["auc_roc"].to_numpy(), pr_base - sub["auc_pr"].to_numpy()
        p_roc = (1 + np.sum(caida_roc <= 0)) / (N_REPETICIONES + 1)
        p_pr = (1 + np.sum(caida_pr <= 0)) / (N_REPETICIONES + 1)
        filas.append({"hipotesis": HIPOTESIS[dimension], "dimension": dimension, "modelo": NOMBRES[paquete["algoritmo"]],
                      "auc_roc_original": roc_base, "caida_auc_roc_media": caida_roc.mean(), "caida_auc_roc_desv": caida_roc.std(ddof=1),
                      "p_valor_auc_roc": p_roc, "significativa": p_roc < ALFA,
                      "auc_pr_original": pr_base, "caida_auc_pr_media": caida_pr.mean(), "p_valor_auc_pr": p_pr,
                      "n_repeticiones": len(sub)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(CARPETA_FASE9 / "permutacion_1000.csv", index=False)
    print("\n=== Permutacion agrupada, 1 000 repeticiones ===")
    print(tabla.round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    dims = list(INDICADORES_POR_DIMENSION)
    ax.boxplot([roc_base - avance.loc[avance["dimension"] == d, "auc_roc"] for d in dims],
               tick_labels=[f"{HIPOTESIS[d]}\n{d}" for d in dims])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Caída del AUC-ROC al permutar")
    ax.set_title(f"Permutación agrupada por dimensión ({NOMBRES[paquete['algoritmo']]}, 1 000 repeticiones)")
    fig.tight_layout()
    fig.savefig(CARPETA_FASE9 / "fig_permutacion_1000.png", dpi=150)


if __name__ == "__main__":
    main()
