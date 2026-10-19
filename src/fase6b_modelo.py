"""
Fase 6b (pasos 6-8) - Reentrena RF y XGBoost sobre la tabla corregida
(eventos, definicion extendida, solo VIIRS 2012-2024, celdas con >=20% de
bosque, predictores de accesibilidad revisados), con validacion espacial
y temporal, comparando contra la linea base del azar, y SHAP agrupado en
4 dimensiones. Luego repite todo con la definicion ESTRICTA como analisis
de sensibilidad.

Linea base del azar:
  - AUC-ROC de un clasificador sin ninguna capacidad de discriminacion es
    siempre 0.5, sin importar el desbalance.
  - AUC-PR de un clasificador sin ninguna capacidad de discriminacion es
    igual a la PREVALENCIA (proporcion de positivos) del conjunto que se
    esta evaluando -- por eso se reporta junto a cada AUC-PR obtenido,
    para saber cuanto mejora realmente el modelo sobre "no aprender nada".

4 dimensiones para SHAP (distinto de la Fase 6 original, que usaba 3):
  - climaticos: precip_mm, dias_secos_mes, temp_max_c, vpd_kpa, spi_1m, spi_3m
  - topograficos: elevacion, pendiente, orientacion
  - vegetacion: ndmi, pct_bosque_1km, clase_mapbiomas (dummies)
  - antropogenicos: pct_perdida_bosque_ese_anio, dist_agropecuario_m,
    dist_borde_bosque_m (borde bosque/no-bosque, proxy de presion humana),
    dist_rio_m (unico predictor de OSM que queda; se agrupa aqui porque en
    la Amazonia los rios son la principal via de acceso y asentamiento
    humano, no un rasgo puramente fisico del terreno)
"""
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import geopandas as gpd

from config import DATA_PROCESSED_DIR, FIGURES_DIR, SEED

ANIO_CORTE_TEST = 2022  # entrena 2012-2021, prueba 2022-2024
TAMANO_BLOQUE_M = 5000
N_FOLDS_ESPACIAL = 5
N_MUESTRA_SHAP = 15000

COLS_CLIMA = ["precip_mm", "dias_secos_mes", "temp_max_c", "vpd_kpa", "spi_1m", "spi_3m"]
COLS_TOPO = ["elevacion", "pendiente", "orientacion"]
COLS_VEGETACION_NUM = ["ndmi", "pct_bosque_1km"]
COLS_ANTROPOGENICO = ["pct_perdida_bosque_ese_anio", "dist_agropecuario_m", "dist_borde_bosque_m", "dist_rio_m"]


def cargar_datos(ruta_tabla):
    tabla = pd.read_parquet(ruta_tabla)

    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    celdas["centroide"] = celdas.geometry.centroid
    celdas["bx"] = (celdas["centroide"].x // TAMANO_BLOQUE_M).astype(int)
    celdas["by"] = (celdas["centroide"].y // TAMANO_BLOQUE_M).astype(int)
    celdas["bloque_espacial"] = celdas["bx"].astype(str) + "_" + celdas["by"].astype(str)
    tabla = tabla.merge(celdas[["cell_id", "bloque_espacial"]], on="cell_id", how="left")

    dummies_clase = pd.get_dummies(tabla["clase_mapbiomas"].astype(int), prefix="clase")
    tabla = pd.concat([tabla, dummies_clase], axis=1)

    cols_features = COLS_CLIMA + COLS_TOPO + COLS_VEGETACION_NUM + COLS_ANTROPOGENICO + list(dummies_clase.columns)
    return tabla, cols_features, list(dummies_clase.columns)


def metricas(y_true, y_prob, umbral=0.5):
    prevalencia = y_true.mean()
    y_pred = (y_prob >= umbral).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
        "auc_roc_azar": 0.5,
        "auc_pr": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan"),
        "auc_pr_azar_prevalencia": prevalencia,
        "sensibilidad": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "positivos_reales": int(y_true.sum()),
        "positivos_predichos": int(y_pred.sum()),
        "n_filas": len(y_true),
    }


def construir_modelos(peso_positivo):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=peso_positivo,
        n_jobs=-1, random_state=SEED, eval_metric="aucpr",
    )
    return {"random_forest": rf, "xgboost": xgb}


def validacion_temporal(tabla, cols_features, etiqueta=""):
    print(f"\n=== [{etiqueta}] Validación temporal (entrena 2012-2021, prueba 2022-2024) ===")
    train = tabla[tabla["anio"] < ANIO_CORTE_TEST]
    test = tabla[tabla["anio"] >= ANIO_CORTE_TEST]
    print(f"Train: {len(train)} filas, {train['positivo'].sum()} positivos")
    print(f"Test:  {len(test)} filas, {test['positivo'].sum()} positivos")

    if test["positivo"].sum() == 0:
        print("AVISO: el conjunto de prueba tiene 0 positivos. AUC-ROC/AUC-PR no son calculables de forma confiable. Se omite.")
        return {}

    X_train, y_train = train[cols_features], train["positivo"]
    X_test, y_test = test[cols_features], test["positivo"]
    peso = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    resultados = {}
    for nombre, modelo in construir_modelos(peso).items():
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        m = metricas(y_test, y_prob)
        resultados[nombre] = m
        print(f"[{nombre}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    return resultados


def validacion_espacial(tabla, cols_features, etiqueta=""):
    n_bloques = tabla["bloque_espacial"].nunique()
    n_folds = min(N_FOLDS_ESPACIAL, n_bloques)
    print(f"\n=== [{etiqueta}] Validación espacial (GroupKFold, {n_folds} folds, {n_bloques} bloques) ===")

    if tabla["positivo"].sum() < n_folds:
        print(f"AVISO: solo hay {tabla['positivo'].sum()} positivos totales, menos que el numero de folds ({n_folds}). "
              "Es muy probable que algunos folds queden sin positivos en el conjunto de prueba. Se procede de todas formas y se reporta el resultado agregado (out-of-fold) con esta salvedad explicita.")

    X = tabla[cols_features]
    y = tabla["positivo"]
    grupos = tabla["bloque_espacial"]
    peso = (y == 0).sum() / max((y == 1).sum(), 1)

    gkf = GroupKFold(n_splits=n_folds)
    resultados = {}
    for nombre in ["random_forest", "xgboost"]:
        y_prob_oof = np.zeros(len(tabla))
        for fold, (idx_train, idx_test) in enumerate(gkf.split(X, y, groups=grupos), start=1):
            modelo = construir_modelos(peso)[nombre]
            modelo.fit(X.iloc[idx_train], y.iloc[idx_train])
            y_prob_oof[idx_test] = modelo.predict_proba(X.iloc[idx_test])[:, 1]
            print(f"  [{nombre}] fold {fold}/{n_folds} listo")
        m = metricas(y, y_prob_oof)
        resultados[nombre] = m
        print(f"[{nombre}] (out-of-fold) " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    return resultados


def grupo_de(col, cols_clase):
    if col in COLS_CLIMA:
        return "climaticos"
    if col in COLS_TOPO:
        return "topograficos"
    if col in COLS_VEGETACION_NUM or col in cols_clase:
        return "vegetacion"
    return "antropogenicos"


def analisis_shap(tabla, cols_features, cols_clase, etiqueta=""):
    print(f"\n=== [{etiqueta}] SHAP (4 dimensiones) ===")
    if tabla["positivo"].sum() < 5:
        print("AVISO: muy pocos positivos para un SHAP confiable. Se calcula de todas formas, a modo exploratorio.")

    n_muestra = min(N_MUESTRA_SHAP, len(tabla))
    muestra = tabla.sample(n=n_muestra, random_state=SEED)
    peso = (tabla["positivo"] == 0).sum() / max((tabla["positivo"] == 1).sum(), 1)

    modelo = construir_modelos(peso)["xgboost"]
    modelo.fit(tabla[cols_features], tabla["positivo"])

    explainer = shap.TreeExplainer(modelo)
    valores_shap = explainer.shap_values(muestra[cols_features])

    importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=cols_features).sort_values(ascending=False)
    print("\nTop 15 variables por importancia SHAP media:")
    print(importancia.head(15))

    por_grupo = importancia.groupby([grupo_de(c, cols_clase) for c in importancia.index]).sum()
    por_grupo_pct = (por_grupo / por_grupo.sum() * 100).sort_values(ascending=False)
    print("\nImportancia SHAP por dimensión (%):")
    print(por_grupo_pct)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    por_grupo_pct.plot(kind="bar", ax=ax, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax.set_ylabel("% de la importancia SHAP total")
    ax.set_title(f"Importancia SHAP por dimensión — {etiqueta}")
    fig.tight_layout()
    sufijo = etiqueta.lower().replace(" ", "_")
    fig.savefig(FIGURES_DIR / f"fase6b_shap_por_dimension_{sufijo}.png", dpi=150)

    return importancia, por_grupo_pct


def ejecutar(ruta_tabla, etiqueta, nombre_salida_csv):
    print(f"\n{'#'*70}\n# {etiqueta}\n{'#'*70}")
    tabla, cols_features, cols_clase = cargar_datos(ruta_tabla)
    print(f"Filas: {len(tabla)}, features: {len(cols_features)}, positivos: {tabla['positivo'].sum()} ({tabla['positivo'].mean():.4%})")

    res_temporal = validacion_temporal(tabla, cols_features, etiqueta)
    res_espacial = validacion_espacial(tabla, cols_features, etiqueta)

    if tabla["positivo"].sum() >= 5:
        importancia, por_grupo = analisis_shap(tabla, cols_features, cols_clase, etiqueta)
    else:
        print(f"\nSHAP omitido para '{etiqueta}': menos de 5 positivos totales, no es representativo.")
        importancia, por_grupo = None, None

    filas = {}
    for modelo in res_espacial:
        filas[("espacial", modelo)] = res_espacial[modelo]
    for modelo in res_temporal:
        filas[("temporal", modelo)] = res_temporal[modelo]
    if filas:
        resumen = pd.DataFrame(filas)
        resumen.to_csv(DATA_PROCESSED_DIR / nombre_salida_csv)
        print(f"\nGuardado: {nombre_salida_csv}")
    return res_temporal, res_espacial, importancia, por_grupo


def main():
    print("PARTE 1: definición EXTENDIDA (principal)")
    ejecutar(
        DATA_PROCESSED_DIR / "fase6b_tabla_modelado.parquet",
        "Extendida",
        "fase6b_resumen_metricas_extendida.csv",
    )

    print("\n\nPARTE 2: definición ESTRICTA (análisis de sensibilidad, paso 8)")
    ejecutar(
        DATA_PROCESSED_DIR / "fase6b_tabla_modelado_estricta.parquet",
        "Estricta",
        "fase6b_resumen_metricas_estricta.csv",
    )


if __name__ == "__main__":
    main()
