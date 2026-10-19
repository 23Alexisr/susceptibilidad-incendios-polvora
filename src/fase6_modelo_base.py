"""
Fase 6 - Modelos base (Random Forest y XGBoost), sin optimizacion fina.

Por que la exactitud (accuracy) enganya con este desbalance: con una
proporcion de positivos de ~0.31% (1 positivo cada ~320 filas), un modelo
que SIEMPRE predice "no hay incendio" acierta el 99.7% de las veces sin
haber aprendido nada util. Por eso se reportan AUC-ROC, AUC-PR
(precision-recall, mas informativa que ROC bajo desbalance fuerte),
sensibilidad (recall de la clase positiva) y F1: estas si distinguen un
modelo que realmente detecta incendios de uno que solo predice la clase
mayoritaria.

Dos validaciones, pedidas explicitamente por la consigna:
  1. Bloques espaciales (GroupKFold sobre celdas agrupadas en bloques de
     5x5 km): mide si el modelo generaliza a LUGARES nuevos que no vio en
     entrenamiento. Es la validacion que usa el criterio de viabilidad de
     la Fase 7 (AUC-ROC >= 0.70).
  2. Particion temporal (entrena 2001-2018, prueba 2019-2024): mide si el
     modelo generaliza a AÑOS futuros que no vio en entrenamiento, que es
     el uso real que tendria un sistema de alerta temprana.

SHAP: se calcula sobre una muestra (no las 651,744 filas completas, por
tiempo de computo) y se agrupa en 3 dimensiones pedidas por la consigna:
clima, uso de suelo, fisicos/accesibilidad. NDMI se cuenta dentro de
"clima" porque refleja humedad de la vegetacion impulsada por la lluvia
reciente, no una caracteristica fija del uso de suelo.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from config import DATA_PROCESSED_DIR, FIGURES_DIR, SEED

ANIO_CORTE_TEST = 2019  # entrena 2001-2018, prueba 2019-2024
TAMANO_BLOQUE_M = 5000  # bloques espaciales de 5x5 km para la validacion espacial
N_FOLDS_ESPACIAL = 5
N_MUESTRA_SHAP = 20000

COLS_CLIMA = ["precip_mm", "dias_secos_mes", "temp_max_c", "vpd_kpa", "ndmi", "spi_1m", "spi_3m"]
COLS_USO_SUELO_NUM = ["pct_bosque_1km", "pct_perdida_bosque_ese_anio", "dist_borde_bosque_m", "dist_agropecuario_m"]
COLS_FISICOS = ["elevacion", "pendiente", "orientacion", "dist_via_m", "dist_centro_poblado_m", "dist_rio_m"]


def cargar_datos():
    tabla = pd.read_parquet(DATA_PROCESSED_DIR / "tabla_modelado.parquet")

    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    celdas["centroide"] = celdas.geometry.centroid
    celdas["bx"] = (celdas["centroide"].x // TAMANO_BLOQUE_M).astype(int)
    celdas["by"] = (celdas["centroide"].y // TAMANO_BLOQUE_M).astype(int)
    celdas["bloque_espacial"] = celdas["bx"].astype(str) + "_" + celdas["by"].astype(str)

    tabla = tabla.merge(celdas[["cell_id", "bloque_espacial"]], on="cell_id", how="left")

    dummies_clase = pd.get_dummies(tabla["clase_mapbiomas"].astype(int), prefix="clase")
    tabla = pd.concat([tabla, dummies_clase], axis=1)

    cols_features = COLS_CLIMA + COLS_USO_SUELO_NUM + COLS_FISICOS + list(dummies_clase.columns)
    return tabla, cols_features


def metricas(y_true, y_prob, umbral=0.5):
    y_pred = (y_prob >= umbral).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "sensibilidad": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "positivos_reales": int(y_true.sum()),
        "positivos_predichos": int(y_pred.sum()),
    }


def construir_modelos(peso_positivo):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=peso_positivo,
        n_jobs=-1,
        random_state=SEED,
        eval_metric="aucpr",
    )
    return {"random_forest": rf, "xgboost": xgb}


def validacion_temporal(tabla, cols_features):
    print("\n=== Validacion temporal (entrena 2001-2018, prueba 2019-2024) ===")
    train = tabla[tabla["anio"] < ANIO_CORTE_TEST]
    test = tabla[tabla["anio"] >= ANIO_CORTE_TEST]
    print(f"Train: {len(train)} filas, {train['positivo'].sum()} positivos")
    print(f"Test:  {len(test)} filas, {test['positivo'].sum()} positivos")

    X_train, y_train = train[cols_features], train["positivo"]
    X_test, y_test = test[cols_features], test["positivo"]
    peso = (y_train == 0).sum() / (y_train == 1).sum()

    resultados = {}
    for nombre, modelo in construir_modelos(peso).items():
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        m = metricas(y_test, y_prob)
        resultados[nombre] = m
        print(f"\n[{nombre}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    return resultados


def validacion_espacial(tabla, cols_features):
    print(f"\n=== Validacion espacial (GroupKFold, {N_FOLDS_ESPACIAL} folds por bloques de {TAMANO_BLOQUE_M}m) ===")
    print(f"Numero de bloques espaciales distintos: {tabla['bloque_espacial'].nunique()}")

    X = tabla[cols_features]
    y = tabla["positivo"]
    grupos = tabla["bloque_espacial"]
    peso = (y == 0).sum() / (y == 1).sum()

    gkf = GroupKFold(n_splits=N_FOLDS_ESPACIAL)
    resultados = {}
    for nombre in ["random_forest", "xgboost"]:
        y_prob_oof = np.zeros(len(tabla))
        for fold, (idx_train, idx_test) in enumerate(gkf.split(X, y, groups=grupos), start=1):
            modelo = construir_modelos(peso)[nombre]
            modelo.fit(X.iloc[idx_train], y.iloc[idx_train])
            y_prob_oof[idx_test] = modelo.predict_proba(X.iloc[idx_test])[:, 1]
            print(f"  [{nombre}] fold {fold}/{N_FOLDS_ESPACIAL} listo")
        m = metricas(y, y_prob_oof)
        resultados[nombre] = m
        print(f"[{nombre}] (out-of-fold, todas las celdas) " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    return resultados


def analisis_shap(tabla, cols_features):
    print(f"\n=== SHAP (muestra de {N_MUESTRA_SHAP} filas) ===")
    muestra = tabla.sample(n=min(N_MUESTRA_SHAP, len(tabla)), random_state=SEED)
    X = muestra[cols_features]
    y = muestra["positivo"]
    peso = (tabla["positivo"] == 0).sum() / (tabla["positivo"] == 1).sum()

    # Se entrena XGBoost sobre TODA la tabla (no solo la muestra) para que
    # el modelo explicado sea representativo; SHAP se calcula solo sobre
    # la muestra por tiempo de computo.
    modelo = construir_modelos(peso)["xgboost"]
    modelo.fit(tabla[cols_features], tabla["positivo"])

    explainer = shap.TreeExplainer(modelo)
    valores_shap = explainer.shap_values(X)

    importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=cols_features).sort_values(ascending=False)
    print("\nTop 15 variables por importancia SHAP media (|valor SHAP|):")
    print(importancia.head(15))

    def grupo_de(col):
        if col in COLS_CLIMA:
            return "clima"
        if col in COLS_USO_SUELO_NUM or col.startswith("clase_"):
            return "uso_suelo"
        return "fisicos_accesibilidad"

    por_grupo = importancia.groupby([grupo_de(c) for c in importancia.index]).sum()
    por_grupo_pct = (por_grupo / por_grupo.sum() * 100).sort_values(ascending=False)
    print("\nImportancia SHAP total por dimension:")
    print(por_grupo_pct)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    importancia.head(15).sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importancia SHAP media (|valor SHAP|)")
    ax.set_title("Top 15 predictores - XGBoost (SHAP)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fase6_shap_top15.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    por_grupo_pct.plot(kind="bar", ax=ax2, color=["#4c72b0", "#55a868", "#c44e52"])
    ax2.set_ylabel("% de la importancia SHAP total")
    ax2.set_title("Importancia SHAP por dimension")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fase6_shap_por_dimension.png", dpi=150)

    importancia.to_csv(DATA_PROCESSED_DIR / "fase6_shap_importancia.csv")
    print(f"\nFiguras guardadas: fase6_shap_top15.png, fase6_shap_por_dimension.png")

    return importancia, por_grupo_pct


def main():
    tabla, cols_features = cargar_datos()
    print(f"Filas: {len(tabla)}, features: {len(cols_features)}")
    print(f"Proporcion de positivos global: {tabla['positivo'].mean():.4%}")
    print(
        "\nNota sobre la exactitud (accuracy): con este desbalance, predecir SIEMPRE 'no incendio' "
        f"da un accuracy de {(1 - tabla['positivo'].mean()):.4%} sin ningun poder predictivo real. "
        "Por eso no se reporta como metrica principal."
    )

    res_temporal = validacion_temporal(tabla, cols_features)
    res_espacial = validacion_espacial(tabla, cols_features)
    importancia, por_grupo = analisis_shap(tabla, cols_features)

    resumen = pd.DataFrame(
        {
            ("temporal", modelo): res_temporal[modelo]
            for modelo in res_temporal
        }
        | {
            ("espacial", modelo): res_espacial[modelo]
            for modelo in res_espacial
        }
    )
    resumen.to_csv(DATA_PROCESSED_DIR / "fase6_resumen_metricas.csv")
    print("\n=== Resumen final ===")
    print(resumen)


if __name__ == "__main__":
    main()
