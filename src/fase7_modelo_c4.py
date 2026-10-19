"""
Fase 7_c4 (paso 6) - Version paralela de fase7_modelo.py: reentrena RF y
XGBoost sobre fase7_tabla_modelado_c4.parquet (MapBiomas C4, periodo
2012-2025). Mismas validaciones, SEED, SHAP y permutacion agrupada que la
version C3 (fase7_modelo.py), sin cambios metodologicos -- solo cambia la
tabla de entrada y que el test temporal ahora es 2022-2025 (no 2022-2024).
No modifica ningun archivo del pipeline C3 (todas las salidas llevan
sufijo _c4).
"""
import platform
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from config import DATA_PROCESSED_DIR, FIGURES_DIR, SEED

ANIO_CORTE_TEST = 2022  # entrena 2012-2021, prueba 2022-2025 (antes 2022-2024)
TAMANO_BLOQUE_M = 5000
N_FOLDS_ESPACIAL = 5
N_MUESTRA_SHAP = 15000
N_PERMUTACIONES = 100

DIMENSIONES_NUM = {
    "climatica": ["precip_mm", "temp_max_c"],
    "topografica": ["elevacion", "pendiente"],
    "ambiental": ["ndvi"],
    "antropogenica": ["dist_carretera_m", "dist_zona_residencial_m"],
}

# AUC-ROC del modelo de 8 indicadores con C3 (2012-2024, fase7_resumen_metricas.csv),
# para comparar el efecto de migrar a C4 + extender el periodo.
C3_AUC_ESPACIAL = {"random_forest": 0.892366, "xgboost": 0.909111}
C3_AUC_TEMPORAL = {"random_forest": 0.872837, "xgboost": 0.895072}


def cargar_datos():
    tabla = pd.read_parquet(DATA_PROCESSED_DIR / "fase7_tabla_modelado_c4.parquet")

    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    celdas["centroide"] = celdas.geometry.centroid
    celdas["bx"] = (celdas["centroide"].x // TAMANO_BLOQUE_M).astype(int)
    celdas["by"] = (celdas["centroide"].y // TAMANO_BLOQUE_M).astype(int)
    celdas["bloque_espacial"] = celdas["bx"].astype(str) + "_" + celdas["by"].astype(str)
    tabla = tabla.merge(celdas[["cell_id", "bloque_espacial"]], on="cell_id", how="left")

    dummies_clase = pd.get_dummies(tabla["clase_mapbiomas"].astype(int), prefix="clase")
    tabla = pd.concat([tabla, dummies_clase], axis=1)
    cols_clase = list(dummies_clase.columns)

    dimensiones = {**DIMENSIONES_NUM, "ambiental": DIMENSIONES_NUM["ambiental"] + cols_clase}
    cols_features = (
        dimensiones["climatica"] + dimensiones["topografica"] + dimensiones["ambiental"] + dimensiones["antropogenica"]
    )
    return tabla, cols_features, cols_clase, dimensiones


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


def validacion_temporal(tabla, cols_features):
    print(f"\n=== Validación temporal C4 (entrena 2012-2021, prueba 2022-2025) ===")
    train = tabla[tabla["anio"] < ANIO_CORTE_TEST]
    test = tabla[tabla["anio"] >= ANIO_CORTE_TEST]
    print(f"Train: {len(train)} filas, {train['positivo'].sum()} positivos")
    print(f"Test:  {len(test)} filas, {test['positivo'].sum()} positivos")

    X_train, y_train = train[cols_features], train["positivo"]
    X_test, y_test = test[cols_features], test["positivo"]
    peso = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    resultados = {}
    modelos_entrenados = {}
    for nombre, modelo in construir_modelos(peso).items():
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        m = metricas(y_test, y_prob)
        resultados[nombre] = m
        modelos_entrenados[nombre] = modelo
        print(f"[{nombre}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    return resultados, modelos_entrenados, X_test, y_test


def validacion_espacial(tabla, cols_features):
    n_bloques = tabla["bloque_espacial"].nunique()
    print(f"\n=== Validación espacial C4 (GroupKFold, {N_FOLDS_ESPACIAL} folds, {n_bloques} bloques de {TAMANO_BLOQUE_M}m) ===")

    X = tabla[cols_features]
    y = tabla["positivo"]
    grupos = tabla["bloque_espacial"]
    peso = (y == 0).sum() / max((y == 1).sum(), 1)

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
        print(f"[{nombre}] (out-of-fold) " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    return resultados


def grupo_de(col, cols_clase):
    if col in DIMENSIONES_NUM["climatica"]:
        return "climatica"
    if col in DIMENSIONES_NUM["topografica"]:
        return "topografica"
    if col == "ndvi" or col in cols_clase:
        return "ambiental"
    return "antropogenica"


def analisis_shap(tabla, cols_features, cols_clase):
    print(f"\n=== SHAP C4 (muestra de {N_MUESTRA_SHAP} filas, XGBoost entrenado en toda la tabla) ===")
    n_muestra = min(N_MUESTRA_SHAP, len(tabla))
    muestra = tabla.sample(n=n_muestra, random_state=SEED)
    peso = (tabla["positivo"] == 0).sum() / max((tabla["positivo"] == 1).sum(), 1)

    modelo = construir_modelos(peso)["xgboost"]
    modelo.fit(tabla[cols_features], tabla["positivo"])

    explainer = shap.TreeExplainer(modelo)
    valores_shap = explainer.shap_values(muestra[cols_features])

    importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=cols_features).sort_values(ascending=False)
    print("\nImportancia SHAP media (|valor SHAP|) por variable:")
    print(importancia)

    por_grupo = importancia.groupby([grupo_de(c, cols_clase) for c in importancia.index]).sum()
    por_grupo_pct = (por_grupo / por_grupo.sum() * 100).sort_values(ascending=False)
    print("\nImportancia SHAP total por dimensión (2 indicadores cada una):")
    print(por_grupo_pct)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    importancia.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importancia SHAP media (|valor SHAP|)")
    ax.set_title("Importancia SHAP por variable — C4, 2012-2025 (XGBoost)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fase7_shap_por_variable_c4.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    por_grupo_pct.plot(kind="bar", ax=ax2, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax2.set_ylabel("% de la importancia SHAP total")
    ax2.set_title("Importancia SHAP por dimensión — C4, 2012-2025")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fase7_shap_por_dimension_c4.png", dpi=150)

    importancia.to_csv(DATA_PROCESSED_DIR / "fase7_shap_importancia_variable_c4.csv")
    por_grupo_pct.to_csv(DATA_PROCESSED_DIR / "fase7_shap_importancia_dimension_c4.csv")
    print("\nFiguras guardadas: fase7_shap_por_variable_c4.png, fase7_shap_por_dimension_c4.png")

    return importancia, por_grupo_pct


def importancia_permutacion_agrupada(modelo, X_test, y_test, dimensiones, n_rep=N_PERMUTACIONES, seed=SEED):
    rng = np.random.default_rng(seed)
    y_prob_base = modelo.predict_proba(X_test)[:, 1]
    auc_base = roc_auc_score(y_test, y_prob_base)

    resultados = []
    for nombre_dim, cols_grupo in dimensiones.items():
        cols_presentes = [c for c in cols_grupo if c in X_test.columns]
        drops = np.empty(n_rep)
        aucs_perm = np.empty(n_rep)
        for r in range(n_rep):
            idx_perm = rng.permutation(len(X_test))
            X_perm = X_test.copy()
            for col in cols_presentes:
                X_perm[col] = X_test[col].to_numpy()[idx_perm]
            y_prob_perm = modelo.predict_proba(X_perm)[:, 1]
            auc_perm = roc_auc_score(y_test, y_prob_perm)
            aucs_perm[r] = auc_perm
            drops[r] = auc_base - auc_perm

        p_valor = (1 + np.sum(aucs_perm >= auc_base)) / (n_rep + 1)
        resultados.append(
            {
                "dimension": nombre_dim,
                "n_indicadores": len(cols_presentes),
                "auc_base": auc_base,
                "auc_permutado_medio": aucs_perm.mean(),
                "auc_permutado_std": aucs_perm.std(ddof=1),
                "caida_auc_media": drops.mean(),
                "caida_auc_std": drops.std(ddof=1),
                "p_valor": p_valor,
                "n_permutaciones": n_rep,
            }
        )
        print(
            f"  [{nombre_dim}] caida AUC media = {drops.mean():.4f} (+/- {drops.std(ddof=1):.4f}), "
            f"p = {p_valor:.4f}  ({'significativo' if p_valor < 0.05 else 'no significativo'} a alfa=0.05)"
        )
    return pd.DataFrame(resultados)


def reportar_versiones():
    import geopandas
    import ee

    info = {
        "python": sys.version.split()[0],
        "sistema_operativo": platform.platform(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "pandas": pd.__version__,
        "geopandas": geopandas.__version__,
        "earthengine-api": ee.__version__,
    }
    print("\n=== Versiones exactas del entorno ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    pd.Series(info).to_csv(DATA_PROCESSED_DIR / "fase7_versiones_entorno_c4.csv", header=["version"])
    return info


def main():
    tabla, cols_features, cols_clase, dimensiones = cargar_datos()
    print(f"Filas: {len(tabla)}, features: {len(cols_features)} (8 indicadores + {len(cols_clase)} dummies de clase_mapbiomas)")
    print(f"Positivos: {tabla['positivo'].sum()} ({tabla['positivo'].mean():.4%})")
    print(f"Dimensiones y columnas:")
    for dim, cols in dimensiones.items():
        print(f"  {dim}: {cols}")

    res_temporal, modelos_temporales, X_test, y_test = validacion_temporal(tabla, cols_features)
    res_espacial = validacion_espacial(tabla, cols_features)
    importancia, por_grupo = analisis_shap(tabla, cols_features, cols_clase)

    print(f"\n=== Importancia por permutación agrupada por dimensión C4 (test temporal 2022-2025, {N_PERMUTACIONES} repeticiones) ===")
    resultados_permutacion = []
    for nombre_modelo, modelo in modelos_temporales.items():
        print(f"\n--- {nombre_modelo} ---")
        df_perm = importancia_permutacion_agrupada(modelo, X_test, y_test, dimensiones, n_rep=N_PERMUTACIONES, seed=SEED)
        df_perm.insert(0, "modelo", nombre_modelo)
        resultados_permutacion.append(df_perm)
    resultados_permutacion = pd.concat(resultados_permutacion, ignore_index=True)
    resultados_permutacion.to_csv(DATA_PROCESSED_DIR / "fase7_importancia_permutacion_dimension_c4.csv", index=False)
    print(f"\nGuardado: fase7_importancia_permutacion_dimension_c4.csv")

    info_versiones = reportar_versiones()

    resumen = pd.DataFrame(
        {("espacial", modelo): res_espacial[modelo] for modelo in res_espacial}
        | {("temporal", modelo): res_temporal[modelo] for modelo in res_temporal}
    )
    resumen.to_csv(DATA_PROCESSED_DIR / "fase7_resumen_metricas_c4.csv")

    print("\n" + "=" * 70)
    print("=== COMPARACIÓN C4 (2012-2025) vs. C3 (2012-2024), modelo de 8 indicadores (AUC-ROC) ===")
    print("=" * 70)
    comparacion = pd.DataFrame(
        {
            "c4_temporal_2012_2025": {m: res_temporal[m]["auc_roc"] for m in res_temporal},
            "c3_temporal_2012_2024": C3_AUC_TEMPORAL,
            "c4_espacial": {m: res_espacial[m]["auc_roc"] for m in res_espacial},
            "c3_espacial": C3_AUC_ESPACIAL,
        }
    )
    print(comparacion)
    comparacion.to_csv(DATA_PROCESSED_DIR / "fase7_comparacion_c3_c4.csv")

    print("\n=== Resumen final C4 (espacial / temporal) ===")
    print(resumen)


if __name__ == "__main__":
    main()
