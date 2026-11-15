"""
Fase 9 (paso 3) - Ajuste de hiperparametros de RF y XGBoost.

Usa SOLO 2012-2021. Los anios 2022-2025 no se leen en ningun momento de
este script. Validacion: GroupKFold de 5 pliegues por bloques de 5 x 5 km
dentro de 2012-2021. Metrica: AUC-PR medio de los 5 pliegues.

Busqueda pequena (8 combinaciones por algoritmo). Cada rejilla incluye la
configuracion de la Fase 8, asi el ajuste solo puede mantenerla o mejorarla.
El resto de hiperparametros (300 arboles, pesos de clase, semilla 42) sale
de construir_modelos() de fase7_modelo_c4.py.

El modelo final es la combinacion (de cualquiera de los dos algoritmos) con
mayor AUC-PR medio. Se reentrena con todo 2012-2021 y se guarda junto con sus
predicciones fuera de pliegue (las usa la calibracion isotonica).

Genera (en resultados/fase9/):
  ajuste_hiperparametros.csv, modelo_final_ajustado.joblib,
  predicciones_fuera_de_pliegue_2012_2021.parquet
"""
import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fase7_modelo_c4 import construir_modelos
from fase8_modelo import NOMBRES, peso_positivo
from fase9_comun import CARPETA_FASE9, datos_separados, pliegues_espaciales

REJILLAS = {
    "random_forest": {"max_depth": [None, 12], "min_samples_leaf": [1, 10], "max_features": ["sqrt", 0.5]},
    "xgboost": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1], "min_child_weight": [1, 10]},
}
CONFIG_FASE8 = {
    "random_forest": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "xgboost": {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1},
}


def combinaciones(rejilla):
    claves = list(rejilla)
    return [dict(zip(claves, valores)) for valores in itertools.product(*rejilla.values())]


def construir(algoritmo, params, y_entrenamiento):
    return construir_modelos(peso_positivo(y_entrenamiento))[algoritmo].set_params(**params)


def evaluar_combinacion(algoritmo, params, entrenamiento, cols_features, pliegues):
    X, y = entrenamiento[cols_features], entrenamiento["positivo"]
    fuera_de_pliegue = np.zeros(len(entrenamiento))
    auc_pr, auc_roc = [], []
    for idx_ent, idx_val in pliegues:
        modelo = construir(algoritmo, params, y.iloc[idx_ent])
        modelo.fit(X.iloc[idx_ent], y.iloc[idx_ent])
        prob = modelo.predict_proba(X.iloc[idx_val])[:, 1]
        fuera_de_pliegue[idx_val] = prob
        auc_pr.append(average_precision_score(y.iloc[idx_val], prob))
        auc_roc.append(roc_auc_score(y.iloc[idx_val], prob))
    return auc_pr, auc_roc, fuera_de_pliegue


def main():
    _, entrenamiento, _, cols_features, _ = datos_separados()  # la prueba 2022-2025 se descarta aqui
    assert entrenamiento["anio"].max() <= 2021
    pliegues = pliegues_espaciales(entrenamiento)
    print(f"Entrenamiento 2012-2021: {len(entrenamiento)} filas, {entrenamiento['positivo'].sum()} positivos, "
          f"{entrenamiento['bloque_espacial'].nunique()} bloques en 5 pliegues")

    filas, oof = [], {}
    for algoritmo, rejilla in REJILLAS.items():
        for i, params in enumerate(combinaciones(rejilla), start=1):
            inicio = time.time()
            auc_pr, auc_roc, fuera = evaluar_combinacion(algoritmo, params, entrenamiento, cols_features, pliegues)
            clave = (algoritmo, str(params))
            oof[clave] = fuera
            filas.append({"algoritmo": algoritmo, "params": str(params), **{f"param_{k}": v for k, v in params.items()},
                          "es_config_fase8": params == CONFIG_FASE8[algoritmo],
                          "auc_pr_media": np.mean(auc_pr), "auc_pr_desv": np.std(auc_pr, ddof=1),
                          "auc_roc_media": np.mean(auc_roc), "auc_roc_desv": np.std(auc_roc, ddof=1),
                          **{f"auc_pr_pliegue{k}": v for k, v in enumerate(auc_pr, start=1)}})
            print(f"[{algoritmo} {i}/8] {params}: AUC-PR {np.mean(auc_pr):.4f} (+/- {np.std(auc_pr, ddof=1):.4f}), "
                  f"AUC-ROC {np.mean(auc_roc):.4f} ({time.time() - inicio:.0f} s)", flush=True)

    resultados = pd.DataFrame(filas).sort_values("auc_pr_media", ascending=False)
    resultados.to_csv(CARPETA_FASE9 / "ajuste_hiperparametros.csv", index=False)

    print("\n=== Mejor combinacion de cada algoritmo ===")
    for algoritmo in REJILLAS:
        mejor = resultados[resultados["algoritmo"] == algoritmo].iloc[0]
        fase8 = resultados[(resultados["algoritmo"] == algoritmo) & resultados["es_config_fase8"]].iloc[0]
        print(f"{NOMBRES[algoritmo]}: {mejor['params']} AUC-PR {mejor['auc_pr_media']:.4f} "
              f"(configuracion de la Fase 8: {fase8['auc_pr_media']:.4f})")

    final = resultados.iloc[0]
    params_final = eval(final["params"])  # el texto viene de str(dict) generado arriba
    print(f"\nModelo final: {NOMBRES[final['algoritmo']]} {params_final} (AUC-PR espacial medio {final['auc_pr_media']:.4f})")

    modelo = construir(final["algoritmo"], params_final, entrenamiento["positivo"])
    modelo.fit(entrenamiento[cols_features], entrenamiento["positivo"])
    joblib.dump({"modelo": modelo, "algoritmo": final["algoritmo"], "params": params_final, "cols_features": cols_features,
                 "auc_pr_espacial_2012_2021": final["auc_pr_media"]}, CARPETA_FASE9 / "modelo_final_ajustado.joblib")

    pred = entrenamiento[["cell_id", "anio", "mes", "positivo"]].copy()
    pred["prob_fuera_de_pliegue"] = oof[(final["algoritmo"], final["params"])]
    pred.to_parquet(CARPETA_FASE9 / "predicciones_fuera_de_pliegue_2012_2021.parquet", index=False)
    print("Guardado: modelo_final_ajustado.joblib y predicciones_fuera_de_pliegue_2012_2021.parquet")


if __name__ == "__main__":
    main()
