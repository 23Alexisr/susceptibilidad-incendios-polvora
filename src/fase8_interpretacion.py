"""
Fase 8 (paso 3) - Interpretacion del modelo final y pruebas de hipotesis.

  - SHAP (TreeExplainer) del modelo final, entrenado con 2012-2021, sobre una
    muestra aleatoria de 15 000 registros de la prueba 2022-2025. Los valores
    de las variables indicadoras de clase_mapbiomas se suman para obtener el
    aporte del indicador completo (los valores SHAP son aditivos).
  - Permutacion agrupada por dimension en toda la prueba 2022-2025, 100
    iteraciones: se desordenan juntas las columnas de una dimension y se mide
    cuanto cae el AUC. Valor p = (1 + veces que el AUC permutado >= AUC
    original) / (100 + 1). HE1 climatica, HE2 topografica, HE3 ambiental,
    HE4 antropogenica.
  - HG: AUC-ROC del modelo final en la prueba frente a 0.5, con intervalo de
    confianza bootstrap (1 000 remuestreos) y prueba U de Mann-Whitney
    unilateral (equivalente a contrastar AUC > 0.5).

Lee:
  data/processed/fase8_modelos_temporales.joblib
  data/processed/fase8_tabla_modelado.parquet (via fase8_modelo.cargar_datos)
Genera:
  resultados/shap_importancia_indicador.csv, resultados/shap_importancia_dimension.csv
  resultados/permutacion_dimension.csv, resultados/hipotesis.csv
  resultados/fig_shap_indicador.png, resultados/fig_shap_dimension.png,
  resultados/fig_shap_beeswarm.png, resultados/fig_permutacion_dimension.png
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score

from config import DATA_PROCESSED_DIR, ROOT_DIR, SEED
from fase8_modelo import ANIO_CORTE_TEST, NOMBRES, cargar_datos

RESULTADOS_DIR = ROOT_DIR / "resultados"
N_MUESTRA_SHAP = 15_000
N_PERMUTACIONES = 100
N_BOOTSTRAP = 1_000
ALFA = 0.05

INDICADORES_POR_DIMENSION = {
    "climatica": ["precip_mm", "temp_max_c"],
    "topografica": ["elevacion", "pendiente"],
    "ambiental": ["ndvi", "clase_mapbiomas"],
    "antropogenica": ["dist_carretera_m", "dist_zona_residencial_m"],
}
HIPOTESIS = {"climatica": "HE1", "topografica": "HE2", "ambiental": "HE3", "antropogenica": "HE4"}
NOMBRE_INDICADOR = {
    "precip_mm": "Precipitación", "temp_max_c": "Temperatura máxima", "elevacion": "Elevación",
    "pendiente": "Pendiente", "ndvi": "NDVI", "clase_mapbiomas": "Cobertura (MapBiomas)",
    "dist_carretera_m": "Distancia a carretera", "dist_zona_residencial_m": "Distancia a centro poblado",
}


def columnas_de_indicador(indicador, cols_clase):
    return cols_clase if indicador == "clase_mapbiomas" else [indicador]


def shap_clase_positiva(explainer, X):
    valores = explainer.shap_values(X, check_additivity=False)
    if isinstance(valores, list):  # versiones antiguas de shap: una matriz por clase
        return valores[1]
    valores = np.asarray(valores)
    return valores[:, :, 1] if valores.ndim == 3 else valores


def calcular_shap(modelo, X, n_jobs=8, tamano_bloque=500):
    # Random Forest con arboles profundos es lento en TreeExplainer: se reparte por bloques
    explainer = shap.TreeExplainer(modelo)
    bloques = [X.iloc[i:i + tamano_bloque] for i in range(0, len(X), tamano_bloque)]
    partes = Parallel(n_jobs=n_jobs)(delayed(shap_clase_positiva)(explainer, b) for b in bloques)
    return np.vstack(partes)


def agregar_por_indicador(valores_shap, cols_features, cols_clase):
    df = pd.DataFrame(valores_shap, columns=cols_features)
    return pd.DataFrame({ind: df[columnas_de_indicador(ind, cols_clase)].sum(axis=1)
                         for cols in INDICADORES_POR_DIMENSION.values() for ind in cols})


def analisis_shap(modelo, nombre_modelo, test, cols_features, cols_clase):
    muestra = test.sample(n=min(N_MUESTRA_SHAP, len(test)), random_state=SEED)
    print(f"\n=== SHAP ({nombre_modelo}, {len(muestra)} registros de prueba, {int(muestra['positivo'].sum())} con incendio) ===")
    inicio = time.time()
    valores = calcular_shap(modelo, muestra[cols_features])
    print(f"SHAP calculado en {time.time() - inicio:.0f} s")

    por_indicador = agregar_por_indicador(valores, cols_features, cols_clase)
    importancia = por_indicador.abs().mean().sort_values(ascending=False)
    dimension_de = {ind: dim for dim, cols in INDICADORES_POR_DIMENSION.items() for ind in cols}
    por_dimension = importancia.groupby(importancia.index.map(dimension_de)).sum().sort_values(ascending=False)
    pct_dimension = por_dimension / por_dimension.sum() * 100

    tabla_ind = pd.DataFrame({"indicador": importancia.index, "nombre": importancia.index.map(NOMBRE_INDICADOR),
                              "dimension": importancia.index.map(dimension_de), "shap_abs_medio": importancia.values,
                              "pct": (importancia / importancia.sum() * 100).values})
    tabla_dim = pd.DataFrame({"dimension": por_dimension.index, "shap_abs_medio": por_dimension.values, "pct": pct_dimension.values})
    tabla_ind.to_csv(RESULTADOS_DIR / "shap_importancia_indicador.csv", index=False)
    tabla_dim.to_csv(RESULTADOS_DIR / "shap_importancia_dimension.csv", index=False)
    print(tabla_ind.round(4).to_string(index=False))
    print(tabla_dim.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(tabla_ind["nombre"][::-1], tabla_ind["shap_abs_medio"][::-1], color="#4c72b0")
    ax.set_xlabel("Importancia SHAP media (|valor SHAP|)")
    ax.set_title(f"Importancia SHAP por indicador - {NOMBRES[nombre_modelo]}, prueba 2022-2025")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "fig_shap_indicador.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tabla_dim["dimension"], tabla_dim["pct"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    ax.set_ylabel("% de la importancia SHAP total")
    ax.set_title("Importancia SHAP por dimensión")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "fig_shap_dimension.png", dpi=150)

    valores_x = muestra[list(por_indicador.columns.drop("clase_mapbiomas"))].copy()
    valores_x["clase_mapbiomas"] = muestra["clase_mapbiomas"].to_numpy()
    valores_x = valores_x[por_indicador.columns]
    plt.figure()
    shap.summary_plot(por_indicador.to_numpy(), valores_x, feature_names=[NOMBRE_INDICADOR[c] for c in por_indicador.columns],
                      show=False, max_display=8)
    plt.title(f"Valores SHAP por registro - {NOMBRES[nombre_modelo]}")
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "fig_shap_beeswarm.png", dpi=150)
    plt.close("all")
    return tabla_ind, tabla_dim


def permutacion_agrupada(modelo, X_test, y_test, cols_clase, rng):
    prob_base = modelo.predict_proba(X_test)[:, 1]
    roc_base, pr_base = roc_auc_score(y_test, prob_base), average_precision_score(y_test, prob_base)
    filas = []
    for dimension, indicadores in INDICADORES_POR_DIMENSION.items():
        cols = [c for ind in indicadores for c in columnas_de_indicador(ind, cols_clase)]
        roc_perm, pr_perm = np.empty(N_PERMUTACIONES), np.empty(N_PERMUTACIONES)
        inicio = time.time()
        for r in range(N_PERMUTACIONES):
            orden = rng.permutation(len(X_test))
            X_perm = X_test.copy()
            # misma permutacion para todas las columnas de la dimension: conserva su relacion interna
            X_perm[cols] = X_test[cols].to_numpy()[orden]
            prob = modelo.predict_proba(X_perm)[:, 1]
            roc_perm[r], pr_perm[r] = roc_auc_score(y_test, prob), average_precision_score(y_test, prob)
        p_roc = (1 + np.sum(roc_perm >= roc_base)) / (N_PERMUTACIONES + 1)
        p_pr = (1 + np.sum(pr_perm >= pr_base)) / (N_PERMUTACIONES + 1)
        filas.append({
            "hipotesis": HIPOTESIS[dimension], "dimension": dimension, "columnas": len(cols),
            "auc_roc_original": roc_base, "auc_roc_permutado_medio": roc_perm.mean(),
            "reduccion_auc_roc_media": roc_base - roc_perm.mean(), "reduccion_auc_roc_desv": roc_perm.std(ddof=1),
            "p_valor_auc_roc": p_roc,
            "auc_pr_original": pr_base, "reduccion_auc_pr_media": pr_base - pr_perm.mean(),
            "reduccion_auc_pr_desv": pr_perm.std(ddof=1), "p_valor_auc_pr": p_pr,
            "significativa": p_roc < ALFA, "n_permutaciones": N_PERMUTACIONES,
            "_reducciones_roc": roc_base - roc_perm,
        })
        print(f"  {HIPOTESIS[dimension]} {dimension}: reduccion AUC-ROC {roc_base - roc_perm.mean():.4f} "
              f"(+/- {roc_perm.std(ddof=1):.4f}), p={p_roc:.4f} | reduccion AUC-PR {pr_base - pr_perm.mean():.5f}, p={p_pr:.4f} "
              f"({time.time() - inicio:.0f} s)")
    return filas


def prueba_hg(y_test, prob, rng):
    auc = roc_auc_score(y_test, prob)
    y, p = y_test.to_numpy(), np.asarray(prob)
    remuestreos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() > 0:
            remuestreos.append(roc_auc_score(y[idx], p[idx]))
    remuestreos = np.array(remuestreos)
    ic_inf, ic_sup = np.percentile(remuestreos, [2.5, 97.5])
    u = mannwhitneyu(p[y == 1], p[y == 0], alternative="greater")
    return {"auc_roc_prueba": auc, "ic95_inferior": ic_inf, "ic95_superior": ic_sup,
            "p_valor_mann_whitney": u.pvalue, "supera_azar": bool(ic_inf > 0.5 and u.pvalue < ALFA)}


def main():
    paquete = joblib.load(DATA_PROCESSED_DIR / "fase8_modelos_temporales.joblib")
    modelos, cols_features, cols_clase, final = (paquete[k] for k in ["modelos", "cols_features", "cols_clase", "modelo_final"])
    tabla, cols_check, _ = cargar_datos()
    assert cols_check == cols_features
    test = tabla[tabla["anio"] >= ANIO_CORTE_TEST].reset_index(drop=True)
    X_test, y_test = test[cols_features], test["positivo"]
    print(f"Modelo final: {final}. Prueba 2022-2025: {len(test)} registros, {int(y_test.sum())} con incendio")

    rng = np.random.default_rng(SEED)
    hg = prueba_hg(y_test, modelos[final].predict_proba(X_test)[:, 1], rng)
    print(f"\nHG: AUC-ROC={hg['auc_roc_prueba']:.4f}, IC95 [{hg['ic95_inferior']:.4f}, {hg['ic95_superior']:.4f}], "
          f"p (Mann-Whitney)={hg['p_valor_mann_whitney']:.2e} -> {'se acepta' if hg['supera_azar'] else 'no se acepta'} que supera 0.5")

    analisis_shap(modelos[final], final, test, cols_features, cols_clase)

    filas = []
    orden_modelos = [final] + [m for m in modelos if m != final]
    for nombre in orden_modelos:
        print(f"\n=== Permutacion agrupada por dimension ({nombre}, {N_PERMUTACIONES} iteraciones) ===")
        for fila in permutacion_agrupada(modelos[nombre], X_test, y_test, cols_clase, np.random.default_rng(SEED)):
            fila["modelo"] = nombre
            fila["es_modelo_final"] = nombre == final
            filas.append(fila)
    reducciones = {(f["modelo"], f["dimension"]): f.pop("_reducciones_roc") for f in filas}
    permutacion = pd.DataFrame(filas)
    permutacion.to_csv(RESULTADOS_DIR / "permutacion_dimension.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    dims = list(INDICADORES_POR_DIMENSION)
    ax.boxplot([reducciones[(final, d)] for d in dims], tick_labels=[f"{HIPOTESIS[d]}\n{d}" for d in dims])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Reducción del AUC-ROC al permutar")
    ax.set_title(f"Permutación agrupada por dimensión ({NOMBRES[final]}, 100 iteraciones)")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "fig_permutacion_dimension.png", dpi=150)

    final_perm = permutacion[permutacion["es_modelo_final"]]
    filas_hip = [{"hipotesis": "HG", "contraste": "AUC-ROC del modelo final en la prueba > 0.5",
                  "estadistico": hg["auc_roc_prueba"], "p_valor": hg["p_valor_mann_whitney"],
                  "detalle": f"IC95 bootstrap [{hg['ic95_inferior']:.4f}, {hg['ic95_superior']:.4f}]",
                  "decision": "se acepta" if hg["supera_azar"] else "no se acepta"}]
    for _, f in final_perm.iterrows():
        filas_hip.append({"hipotesis": f["hipotesis"], "contraste": f"La dimension {f['dimension']} reduce el AUC-ROC al permutarse",
                          "estadistico": f["reduccion_auc_roc_media"], "p_valor": f["p_valor_auc_roc"],
                          "detalle": f"desv {f['reduccion_auc_roc_desv']:.4f}; reduccion AUC-PR {f['reduccion_auc_pr_media']:.5f} (p={f['p_valor_auc_pr']:.4f})",
                          "decision": "se acepta" if f["significativa"] and f["reduccion_auc_roc_media"] > 0 else "no se acepta"})
    hipotesis = pd.DataFrame(filas_hip)
    hipotesis.to_csv(RESULTADOS_DIR / "hipotesis.csv", index=False)
    print("\n=== Decision sobre las hipotesis (modelo final) ===")
    print(hipotesis.to_string(index=False))


if __name__ == "__main__":
    main()
