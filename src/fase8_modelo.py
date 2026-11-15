"""
Fase 8 (paso 2) - Experimento computacional: Random Forest y XGBoost
(tratamientos) frente al azar (control), con las mismas condiciones para
ambos: misma tabla, misma particion temporal, mismos bloques espaciales y
semilla 42. Los hiperparametros son los de fase7_modelo_c4.py (se importa
construir_modelos para garantizar que sean identicos); no se optimizan.

Diferencias con fase7_modelo_c4.py:
  - Usa fase8_tabla_modelado.parquet (NDVI imputado con promedio).
  - La validacion espacial reporta AUC-ROC y AUC-PR de CADA pliegue y su
    media y desviacion estandar (antes solo una metrica sobre todas las
    predicciones fuera de pliegue juntas, que tambien se conserva).
  - El peso de clase de cada pliegue se calcula solo con su entrenamiento.
  - El modelo final se elige por mayor AUC-PR medio en la validacion espacial.

Lee:
  data/processed/fase8_tabla_modelado.parquet
  data/processed/grilla_1km.gpkg (capa celdas_utm, para los bloques de 5 x 5 km)
Genera:
  data/processed/fase8_modelos_temporales.joblib (modelos entrenados 2012-2021)
  data/processed/fase8_predicciones_prueba.parquet
  resultados/metricas_pliegues_espaciales.csv
  resultados/tabla_tratamiento_vs_control.csv
  resultados/modelo_final.txt
  resultados/fig_curvas_roc_pr_prueba.png, resultados/fig_tratamiento_vs_control.png
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from config import DATA_PROCESSED_DIR, ROOT_DIR, SEED
from fase7_modelo_c4 import DIMENSIONES_NUM, construir_modelos

RESULTADOS_DIR = ROOT_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

ANIO_CORTE_TEST = 2022
TAMANO_BLOQUE_M = 5000
N_FOLDS_ESPACIAL = 5
NOMBRES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def cargar_datos():
    tabla = pd.read_parquet(DATA_PROCESSED_DIR / "fase8_tabla_modelado.parquet")

    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    centroides = celdas.geometry.centroid
    bx = (centroides.x // TAMANO_BLOQUE_M).astype(int).astype(str)
    by = (centroides.y // TAMANO_BLOQUE_M).astype(int).astype(str)
    celdas["bloque_espacial"] = bx + "_" + by
    tabla = tabla.merge(celdas[["cell_id", "bloque_espacial"]], on="cell_id", how="left")

    # clase_mapbiomas es categorica: se codifica con variables indicadoras (0/1)
    dummies_clase = pd.get_dummies(tabla["clase_mapbiomas"].astype(int), prefix="clase").astype(int)
    tabla = pd.concat([tabla, dummies_clase], axis=1)
    cols_clase = list(dummies_clase.columns)

    cols_features = (
        DIMENSIONES_NUM["climatica"] + DIMENSIONES_NUM["topografica"]
        + DIMENSIONES_NUM["ambiental"] + cols_clase + DIMENSIONES_NUM["antropogenica"]
    )
    return tabla, cols_features, cols_clase


def peso_positivo(y) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def metricas_azar(y, rng) -> dict:
    # El azar teorico: AUC-ROC = 0.5 y AUC-PR = prevalencia. Se agrega un
    # clasificador aleatorio real (puntajes uniformes) como comprobacion.
    puntaje = rng.random(len(y))
    return {
        "auc_roc": 0.5,
        "auc_pr": float(y.mean()),
        "auc_roc_empirico": roc_auc_score(y, puntaje),
        "auc_pr_empirico": average_precision_score(y, puntaje),
    }


def validacion_temporal(tabla, cols_features, rng):
    train = tabla[tabla["anio"] < ANIO_CORTE_TEST]
    test = tabla[tabla["anio"] >= ANIO_CORTE_TEST]
    print(f"\n=== Validacion temporal: entrena 2012-2021 ({len(train)} filas, {train['positivo'].sum()} pos), "
          f"prueba 2022-2025 ({len(test)} filas, {test['positivo'].sum()} pos) ===")

    X_train, y_train = train[cols_features], train["positivo"]
    X_test, y_test = test[cols_features], test["positivo"]

    resultados, modelos, probs = {}, {}, {}
    for nombre, modelo in construir_modelos(peso_positivo(y_train)).items():
        inicio = time.time()
        modelo.fit(X_train, y_train)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        resultados[nombre] = {"auc_roc": roc_auc_score(y_test, y_prob), "auc_pr": average_precision_score(y_test, y_prob)}
        modelos[nombre], probs[nombre] = modelo, y_prob
        print(f"[{nombre}] AUC-ROC={resultados[nombre]['auc_roc']:.4f}  AUC-PR={resultados[nombre]['auc_pr']:.4f}  ({time.time() - inicio:.0f} s)")

    resultados["azar"] = metricas_azar(y_test, rng)
    print(f"[azar] AUC-ROC=0.5  AUC-PR(prevalencia)={resultados['azar']['auc_pr']:.4f}  "
          f"(empirico: {resultados['azar']['auc_roc_empirico']:.4f} / {resultados['azar']['auc_pr_empirico']:.4f})")
    return resultados, modelos, probs, test


def validacion_espacial(tabla, cols_features, rng):
    X, y, grupos = tabla[cols_features], tabla["positivo"], tabla["bloque_espacial"]
    print(f"\n=== Validacion espacial: GroupKFold {N_FOLDS_ESPACIAL} pliegues, {grupos.nunique()} bloques de 5 x 5 km ===")

    filas = []
    oof = {nombre: np.zeros(len(tabla)) for nombre in NOMBRES}
    for pliegue, (idx_train, idx_test) in enumerate(GroupKFold(n_splits=N_FOLDS_ESPACIAL).split(X, y, groups=grupos), start=1):
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
        azar = metricas_azar(y_test, rng)
        filas.append({"pliegue": pliegue, "tratamiento": "azar", "n_prueba": len(idx_test), "positivos_prueba": int(y_test.sum()),
                      "auc_roc": azar["auc_roc"], "auc_pr": azar["auc_pr"]})
        for nombre, modelo in construir_modelos(peso_positivo(y_train)).items():
            modelo.fit(X.iloc[idx_train], y_train)
            y_prob = modelo.predict_proba(X.iloc[idx_test])[:, 1]
            oof[nombre][idx_test] = y_prob
            filas.append({"pliegue": pliegue, "tratamiento": nombre, "n_prueba": len(idx_test), "positivos_prueba": int(y_test.sum()),
                          "auc_roc": roc_auc_score(y_test, y_prob), "auc_pr": average_precision_score(y_test, y_prob)})
            print(f"  pliegue {pliegue} [{nombre}] AUC-ROC={filas[-1]['auc_roc']:.4f} AUC-PR={filas[-1]['auc_pr']:.4f} "
                  f"(prevalencia {azar['auc_pr']:.4f}, {int(y_test.sum())} positivos)")

    por_pliegue = pd.DataFrame(filas)
    agregadas_oof = {nombre: {"auc_roc": roc_auc_score(y, oof[nombre]), "auc_pr": average_precision_score(y, oof[nombre])} for nombre in NOMBRES}
    return por_pliegue, agregadas_oof


def graficar_curvas(y_test, probs, ruta):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for nombre, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f"{NOMBRES[nombre]} (AUC={roc_auc_score(y_test, y_prob):.3f})")
        ax2.plot(rec, prec, label=f"{NOMBRES[nombre]} (AP={average_precision_score(y_test, y_prob):.4f})")
    ax1.plot([0, 1], [0, 1], "k--", label="Azar (0.5)")
    ax2.axhline(y_test.mean(), color="k", ls="--", label=f"Azar (prevalencia={y_test.mean():.4f})")
    ax1.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos (sensibilidad)", title="Curva ROC - prueba 2022-2025")
    ax2.set(xlabel="Exhaustividad (recall)", ylabel="Precision", title="Curva precision-exhaustividad - prueba 2022-2025")
    ax2.set_yscale("log")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


def graficar_tratamiento_vs_control(tabla_comp, ruta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colores = {"Random Forest": "#4c72b0", "XGBoost": "#dd8452", "Azar (control)": "#999999"}
    for ax, metrica in zip(axes, ["AUC-ROC", "AUC-PR"]):
        sub = tabla_comp[tabla_comp["metrica"] == metrica]
        x = np.arange(2)
        for i, trat in enumerate(colores):
            fila = sub[sub["tratamiento"] == trat].iloc[0]
            ax.bar(x + (i - 1) * 0.27, [fila["prueba_temporal"], fila["espacial_media"]], 0.27, color=colores[trat], label=trat)
            # la prueba temporal es un solo valor; solo la validacion espacial tiene dispersion entre pliegues
            ax.errorbar(x[1] + (i - 1) * 0.27, fila["espacial_media"], yerr=fila["espacial_desv"], color="black", capsize=3)
        ax.set_xticks(x, ["Prueba temporal\n2022-2025", "Validacion espacial\n(media de 5 pliegues)"])
        ax.set_title(metrica)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Tratamientos vs. control (azar)")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(ruta, dpi=150)


def main():
    rng = np.random.default_rng(SEED)
    tabla, cols_features, cols_clase = cargar_datos()
    print(f"Filas: {len(tabla)}, positivos: {tabla['positivo'].sum()} ({tabla['positivo'].mean():.4%})")
    print(f"Variables de entrada ({len(cols_features)}): {cols_features}")

    res_temporal, modelos, probs, test = validacion_temporal(tabla, cols_features, rng)
    por_pliegue, oof = validacion_espacial(tabla, cols_features, rng)
    por_pliegue.to_csv(RESULTADOS_DIR / "metricas_pliegues_espaciales.csv", index=False)

    resumen_esp = por_pliegue.groupby("tratamiento")[["auc_roc", "auc_pr"]].agg(["mean", "std"])
    print("\nValidacion espacial (media y desviacion estandar de 5 pliegues):")
    print(resumen_esp.round(4))

    etiquetas = {"random_forest": "Random Forest", "xgboost": "XGBoost", "azar": "Azar (control)"}
    filas = []
    for clave, nombre in etiquetas.items():
        for metrica, col in [("AUC-ROC", "auc_roc"), ("AUC-PR", "auc_pr")]:
            filas.append({
                "tratamiento": nombre, "metrica": metrica,
                "prueba_temporal": res_temporal[clave][col],
                "espacial_media": resumen_esp.loc[clave, (col, "mean")],
                "espacial_desv": resumen_esp.loc[clave, (col, "std")],
                "espacial_fuera_de_pliegue": oof[clave][col] if clave in oof else (0.5 if col == "auc_roc" else tabla["positivo"].mean()),
            })
    tabla_comp = pd.DataFrame(filas)
    tabla_comp.to_csv(RESULTADOS_DIR / "tabla_tratamiento_vs_control.csv", index=False)
    print("\n=== Tabla tratamiento vs. control ===")
    print(tabla_comp.round(4).to_string(index=False))

    media_pr = {n: resumen_esp.loc[n, ("auc_pr", "mean")] for n in NOMBRES}
    final = max(media_pr, key=media_pr.get)
    texto = (
        f"modelo_final={final}\n"
        f"criterio=mayor AUC-PR medio en la validacion espacial (5 pliegues)\n"
        + "".join(f"auc_pr_espacial_media_{n}={v:.6f}\n" for n, v in media_pr.items())
        + f"auc_roc_prueba_final={res_temporal[final]['auc_roc']:.6f}\n"
        + f"auc_pr_prueba_final={res_temporal[final]['auc_pr']:.6f}\n"
    )
    (RESULTADOS_DIR / "modelo_final.txt").write_text(texto, encoding="utf-8")
    print(f"\nModelo final: {final} (AUC-PR espacial medio {media_pr[final]:.4f})")

    joblib.dump({"modelos": modelos, "cols_features": cols_features, "cols_clase": cols_clase, "modelo_final": final},
                DATA_PROCESSED_DIR / "fase8_modelos_temporales.joblib")
    pred = test[["cell_id", "anio", "mes", "positivo"]].copy()
    for nombre, y_prob in probs.items():
        pred[f"prob_{nombre}"] = y_prob
    pred.to_parquet(DATA_PROCESSED_DIR / "fase8_predicciones_prueba.parquet", index=False)

    graficar_curvas(test["positivo"], probs, RESULTADOS_DIR / "fig_curvas_roc_pr_prueba.png")
    graficar_tratamiento_vs_control(tabla_comp, RESULTADOS_DIR / "fig_tratamiento_vs_control.png")
    print("Figuras guardadas en resultados/")


if __name__ == "__main__":
    main()
