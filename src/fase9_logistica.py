"""
Fase 9 (paso 1) - Regresion logistica de referencia.

Mismos 8 indicadores que RF y XGBoost: las 7 variables numericas se
estandarizan (media 0, desviacion 1) y la cobertura MapBiomas entra como
variables indicadoras 0/1, sin estandarizar. Pesos de clase balanceados.

  - Prueba temporal: entrena con 2012-2021, evalua en 2022-2025.
  - Validacion espacial: los mismos 5 pliegues por bloques de 5 x 5 km que
    fase8_modelo.py (sobre toda la tabla 2012-2025, para poder comparar).

Los valores de RF y XGBoost se leen de los resultados de la Fase 8
(no se reentrenan).

Lee:
  data/processed/fase8_tabla_modelado.parquet (via fase8_modelo.cargar_datos)
  resultados/tabla_tratamiento_vs_control.csv, resultados/metricas_pliegues_espaciales.csv
Genera (en resultados/fase9/):
  logistica_modelo.joblib, predicciones_prueba_logistica.parquet,
  logistica_coeficientes.csv, logistica_pliegues.csv,
  comparacion_modelos.csv, comparacion_pliegues.csv
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ROOT_DIR, SEED
from fase7_modelo_c4 import DIMENSIONES_NUM
from fase9_comun import CARPETA_FASE9, datos_separados, pliegues_espaciales

COLS_NUMERICAS = [c for cols in DIMENSIONES_NUM.values() for c in cols]


def construir_logistica():
    estandarizar = ColumnTransformer([("numericas", StandardScaler(), COLS_NUMERICAS)], remainder="passthrough")
    modelo = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
    return Pipeline([("estandarizar", estandarizar), ("logistica", modelo)])


def metricas(y, prob):
    return {"auc_roc": roc_auc_score(y, prob), "auc_pr": average_precision_score(y, prob)}


def main():
    tabla, entrenamiento, prueba, cols_features, cols_clase = datos_separados()

    # --- Prueba temporal ---
    modelo = construir_logistica()
    modelo.fit(entrenamiento[cols_features], entrenamiento["positivo"])
    prob_prueba = modelo.predict_proba(prueba[cols_features])[:, 1]
    temporal = metricas(prueba["positivo"], prob_prueba)
    print(f"Prueba 2022-2025: AUC-ROC={temporal['auc_roc']:.4f}  AUC-PR={temporal['auc_pr']:.4f}")

    joblib.dump({"modelo": modelo, "cols_features": cols_features}, CARPETA_FASE9 / "logistica_modelo.joblib")
    pred = prueba[["cell_id", "anio", "mes", "positivo"]].copy()
    pred["prob_logistica"] = prob_prueba
    pred.to_parquet(CARPETA_FASE9 / "predicciones_prueba_logistica.parquet", index=False)

    # Coeficientes: como las numericas estan estandarizadas, su tamano se puede comparar entre si
    nombres = modelo.named_steps["estandarizar"].get_feature_names_out()
    coef = pd.DataFrame({"variable": [n.split("__", 1)[1] for n in nombres],
                         "coeficiente": modelo.named_steps["logistica"].coef_[0]})
    coef["razon_de_odds"] = np.exp(coef["coeficiente"])
    coef.to_csv(CARPETA_FASE9 / "logistica_coeficientes.csv", index=False)
    print(coef.round(4).to_string(index=False))

    # --- Validacion espacial (mismos pliegues que la Fase 8) ---
    filas = []
    for pliegue, (idx_ent, idx_val) in enumerate(pliegues_espaciales(tabla), start=1):
        m = construir_logistica()
        m.fit(tabla.loc[idx_ent, cols_features], tabla.loc[idx_ent, "positivo"])
        prob = m.predict_proba(tabla.loc[idx_val, cols_features])[:, 1]
        filas.append({"pliegue": pliegue, "tratamiento": "logistica", "n_prueba": len(idx_val),
                      "positivos_prueba": int(tabla.loc[idx_val, "positivo"].sum()), **metricas(tabla.loc[idx_val, "positivo"], prob)})
        print(f"  pliegue {pliegue}: AUC-ROC={filas[-1]['auc_roc']:.4f}  AUC-PR={filas[-1]['auc_pr']:.4f}")
    pliegues_log = pd.DataFrame(filas)
    pliegues_log.to_csv(CARPETA_FASE9 / "logistica_pliegues.csv", index=False)

    # --- Tabla comparativa con RF, XGBoost y el azar (Fase 8) ---
    fase8 = pd.read_csv(ROOT_DIR / "resultados" / "tabla_tratamiento_vs_control.csv")
    fase8 = fase8[["tratamiento", "metrica", "prueba_temporal", "espacial_media", "espacial_desv"]]
    nuevas = pd.DataFrame([
        {"tratamiento": "Regresión logística", "metrica": nombre, "prueba_temporal": temporal[col],
         "espacial_media": pliegues_log[col].mean(), "espacial_desv": pliegues_log[col].std()}
        for nombre, col in [("AUC-ROC", "auc_roc"), ("AUC-PR", "auc_pr")]
    ])
    comparacion = pd.concat([nuevas, fase8], ignore_index=True).sort_values(["metrica", "prueba_temporal"], ascending=[False, False])
    comparacion.to_csv(CARPETA_FASE9 / "comparacion_modelos.csv", index=False)
    print("\n=== Comparacion de modelos ===")
    print(comparacion.round(4).to_string(index=False))

    # --- Misma validacion que el ajuste de hiperparametros (bloques de 5 km solo dentro de 2012-2021) ---
    # Permite comparar la logistica con los AUC-PR de ajuste_hiperparametros.csv en igualdad de condiciones.
    auc_pr_ent = []
    for idx_ent, idx_val in pliegues_espaciales(entrenamiento):
        m = construir_logistica()
        m.fit(entrenamiento.loc[idx_ent, cols_features], entrenamiento.loc[idx_ent, "positivo"])
        prob = m.predict_proba(entrenamiento.loc[idx_val, cols_features])[:, 1]
        auc_pr_ent.append(average_precision_score(entrenamiento.loc[idx_val, "positivo"], prob))
    pd.DataFrame({"pliegue": range(1, 6), "auc_pr": auc_pr_ent}).to_csv(CARPETA_FASE9 / "logistica_pliegues_2012_2021.csv", index=False)
    print(f"\nValidacion espacial solo 2012-2021 (como el ajuste): AUC-PR {np.mean(auc_pr_ent):.4f} (+/- {np.std(auc_pr_ent, ddof=1):.4f})")

    pliegues_fase8 = pd.read_csv(ROOT_DIR / "resultados" / "metricas_pliegues_espaciales.csv")
    por_pliegue = pd.concat([pliegues_fase8, pliegues_log], ignore_index=True)
    por_pliegue = por_pliegue.pivot_table(index="pliegue", columns="tratamiento", values=["auc_roc", "auc_pr"])
    por_pliegue.columns = [f"{metrica}_{trat}" for metrica, trat in por_pliegue.columns]
    por_pliegue.to_csv(CARPETA_FASE9 / "comparacion_pliegues.csv")
    print("\n=== Por pliegue ===")
    print(por_pliegue.round(4).to_string())


if __name__ == "__main__":
    main()
