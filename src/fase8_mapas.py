"""
Fase 8 (paso 4) - Mapas mensuales de susceptibilidad (enero a diciembre) para
las 2 263 celdas de la grilla, y datos del visor web.

Predictores de cada mapa:
  - Clima (precip_mm, temp_max_c) y NDVI: climatologia de la celda para ese
    mes, es decir, el promedio 2012-2025 de los valores que el modelo recibe
    en ese mes. Se respeta el mismo desfase que en la tabla de modelado: el
    mapa de agosto usa el clima y el NDVI de julio.
  - clase_mapbiomas y % de bosque: anio mas reciente disponible (2024, que es
    el anio t-1 de las filas 2025 de la tabla).
  - Elevacion, pendiente y distancias: estaticos.
Modelo: el algoritmo elegido como final en fase8_modelo.py, reentrenado con
toda la tabla 2012-2025 y los mismos hiperparametros.
Niveles: cortes naturales de Jenks (5 clases) calculados con todas las
celdas-mes del dominio del modelo (celdas con >= 20 % de bosque en 2024) de
los 12 meses juntos, para que un mismo color signifique lo mismo en todos
los meses.

Lee:
  data/processed/fase8_modelos_temporales.joblib, fase8_tabla_modelado.parquet
  data/processed/fase5_predictores_mensuales.csv, fase5_predictores_mensuales_2025_c4.csv
  data/processed/fase7_ndvi_mensual.csv, fase7_ndvi_mensual_2025_c4.csv
  data/processed/fase5_predictores_anuales_c4.csv, fase5_predictores_estaticos.csv
  data/processed/fase7_dist_carretera.csv, fase7_dist_centro_poblado.csv
  data/processed/grilla_1km.gpkg, area_estudio.gpkg
Genera:
  data/processed/fase8_modelo_mapas.joblib
  data/processed/fase8_susceptibilidad_mensual.csv
  resultados/cortes_niveles_susceptibilidad.csv, resultados/niveles_por_mes.csv
  resultados/fig_mapas_susceptibilidad_12meses.png
  visor/datos.js
"""
import json
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from config import DATA_PROCESSED_DIR, ROOT_DIR
from fase7_modelo_c4 import construir_modelos
from fase8_interpretacion import INDICADORES_POR_DIMENSION, NOMBRE_INDICADOR, agregar_por_indicador, calcular_shap
from fase8_modelo import NOMBRES, cargar_datos, peso_positivo

RESULTADOS_DIR = ROOT_DIR / "resultados"
VISOR_DIR = ROOT_DIR / "visor"
VISOR_DIR.mkdir(exist_ok=True)

ANIO_INICIO, ANIO_FIN = 2012, 2025
ANIO_USO_SUELO = 2024
UMBRAL_BOSQUE = 0.20
N_NIVELES = 5
NIVELES = ["Muy baja", "Baja", "Media", "Alta", "Muy alta"]
COLORES = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
# Nombres segun MapBiomasPeru_legend.csv (data/raw/mapbiomas_c4/), solo las clases presentes en la grilla
NOMBRE_CLASE = {3: "Bosque", 12: "Formación herbácea", 15: "Pasto", 21: "Mosaico agropecuario", 23: "Playa",
                24: "Infraestructura urbana", 33: "Río, lago u océano", 35: "Palma aceitera", 66: "Matorral y otros arbustales"}
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def climatologia_lag(df, columnas):
    """Promedio por celda y mes de mapa, usando el valor del mes anterior (mismo desfase que la tabla)."""
    df = df.copy()
    df["mes_mapa"] = df["mes"] % 12 + 1
    df["anio_mapa"] = np.where(df["mes"] == 12, df["anio"] + 1, df["anio"])
    df = df[(df["anio_mapa"] >= ANIO_INICIO) & (df["anio_mapa"] <= ANIO_FIN)]
    return df.groupby(["cell_id", "mes_mapa"])[columnas].mean().reset_index().rename(columns={"mes_mapa": "mes"})


def construir_predictores_mapa(cols_clase):
    clima = pd.concat([
        pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv", usecols=["cell_id", "anio", "mes", "precip_mm", "temp_max_c"]),
        pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales_2025_c4.csv", usecols=["cell_id", "anio", "mes", "precip_mm", "temp_max_c"]),
    ])
    ndvi = pd.concat([
        pd.read_csv(DATA_PROCESSED_DIR / "fase7_ndvi_mensual.csv"),
        pd.read_csv(DATA_PROCESSED_DIR / "fase7_ndvi_mensual_2025_c4.csv"),
    ])
    clim_clima = climatologia_lag(clima, ["precip_mm", "temp_max_c"])
    clim_ndvi = climatologia_lag(ndvi, ["ndvi"])

    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales_c4.csv")
    uso = anuales[anuales["anio"] == ANIO_USO_SUELO][["cell_id", "clase_mapbiomas", "pct_bosque_1km"]].copy()
    uso["clase_mapbiomas"] = uso["clase_mapbiomas"].round().astype(int)

    estaticos = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv")[["cell_id", "elevacion", "pendiente"]]
    carretera = pd.read_csv(DATA_PROCESSED_DIR / "fase7_dist_carretera.csv")
    residencial = pd.read_csv(DATA_PROCESSED_DIR / "fase7_dist_centro_poblado.csv")

    base = pd.DataFrame([(c, m) for c in estaticos["cell_id"] for m in range(1, 13)], columns=["cell_id", "mes"])
    base = (base.merge(clim_clima, on=["cell_id", "mes"], how="left").merge(clim_ndvi, on=["cell_id", "mes"], how="left")
            .merge(uso, on="cell_id", how="left").merge(estaticos, on="cell_id", how="left")
            .merge(carretera, on="cell_id", how="left").merge(residencial, on="cell_id", how="left"))

    sin_ndvi = base["ndvi"].isna()
    if sin_ndvi.any():
        base.loc[sin_ndvi, "ndvi"] = base.loc[sin_ndvi, "mes"].map(base.groupby("mes")["ndvi"].mean())
        print(f"Celdas-mes sin NDVI en ningun anio (se usa el promedio del mes): {int(sin_ndvi.sum())}")
    for col in cols_clase:
        base[col] = (base["clase_mapbiomas"] == int(col.split("_")[1])).astype(int)
    clases_no_vistas = sorted(set(base["clase_mapbiomas"]) - {int(c.split("_")[1]) for c in cols_clase})
    if clases_no_vistas:
        print(f"Clases de 2024 que no aparecen en la tabla de entrenamiento (todas sus indicadoras quedan en 0): {clases_no_vistas}")
    base["en_dominio"] = base["pct_bosque_1km"] >= UMBRAL_BOSQUE
    return base


def cortes_jenks(valores, k):
    """Cortes naturales de Jenks (Fisher) por programacion dinamica; devuelve los k-1 limites internos."""
    x = np.sort(np.asarray(valores, dtype=float))
    n = len(x)
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

    def ssd(i, j):  # suma de cuadrados de x[i:j]
        m = j - i
        return s2[j] - s2[i] - (s1[j] - s1[i]) ** 2 / m

    costo = np.full((k + 1, n + 1), np.inf)
    origen = np.zeros((k + 1, n + 1), dtype=int)
    costo[1, 1:] = ssd(np.zeros(n, dtype=int), np.arange(1, n + 1))
    for c in range(2, k + 1):
        for j in range(c, n + 1):
            i = np.arange(c - 1, j)
            total = costo[c - 1, i] + ssd(i, np.full_like(i, j))
            mejor = np.argmin(total)
            costo[c, j], origen[c, j] = total[mejor], i[mejor]
    limites, j = [], n
    for c in range(k, 1, -1):
        j = origen[c, j]
        limites.append(x[j - 1])
    return sorted(limites)


def main():
    paquete = joblib.load(DATA_PROCESSED_DIR / "fase8_modelos_temporales.joblib")
    cols_features, cols_clase, final = paquete["cols_features"], paquete["cols_clase"], paquete["modelo_final"]

    tabla, cols_check, _ = cargar_datos()
    assert cols_check == cols_features
    inicio = time.time()
    modelo = construir_modelos(peso_positivo(tabla["positivo"]))[final]
    modelo.fit(tabla[cols_features], tabla["positivo"])
    joblib.dump({"modelo": modelo, "cols_features": cols_features, "cols_clase": cols_clase, "algoritmo": final},
                DATA_PROCESSED_DIR / "fase8_modelo_mapas.joblib")
    print(f"Modelo de mapas ({final}) entrenado con 2012-2025 en {time.time() - inicio:.0f} s")

    base = construir_predictores_mapa(cols_clase)
    assert base[cols_features].notna().all().all(), base[cols_features].isna().sum()
    base["probabilidad"] = modelo.predict_proba(base[cols_features])[:, 1]
    print(f"Celdas-mes del mapa: {len(base)} ({base['cell_id'].nunique()} celdas); en dominio (>=20 % bosque 2024): {int(base['en_dominio'].sum())}")

    inicio = time.time()
    limites = cortes_jenks(base.loc[base["en_dominio"], "probabilidad"], N_NIVELES)
    print(f"Cortes de Jenks ({time.time() - inicio:.0f} s): {[round(v, 4) for v in limites]}")
    base["nivel"] = np.digitize(base["probabilidad"], limites, right=True)
    base["nivel_nombre"] = base["nivel"].map(dict(enumerate(NIVELES)))

    pd.DataFrame({"nivel": NIVELES,
                  "desde": [0.0] + limites, "hasta": limites + [1.0]}).to_csv(RESULTADOS_DIR / "cortes_niveles_susceptibilidad.csv", index=False)
    conteo = base.pivot_table(index="mes", columns="nivel_nombre", values="cell_id", aggfunc="count", fill_value=0)[NIVELES]
    conteo.index = MESES
    conteo.to_csv(RESULTADOS_DIR / "niveles_por_mes.csv")
    print("\nCeldas por nivel y mes:")
    print(conteo.to_string())

    # Cortes por cuantiles, solo como comparacion en el informe
    q = np.quantile(base.loc[base["en_dominio"], "probabilidad"], [0.2, 0.4, 0.6, 0.8])
    print(f"(Comparacion) cortes por quintiles: {[round(v, 4) for v in q]}")

    # --- SHAP por celda-mes para el visor ---
    inicio = time.time()
    valores = calcular_shap(modelo, base[cols_features])
    por_indicador = agregar_por_indicador(valores, cols_features, cols_clase)
    print(f"SHAP de {len(base)} celdas-mes calculado en {time.time() - inicio:.0f} s")
    indicadores = list(por_indicador.columns)
    abs_vals = por_indicador.abs().to_numpy()
    top3 = np.argsort(-abs_vals, axis=1)[:, :3]
    for r in range(3):
        base[f"factor{r + 1}"] = [indicadores[i] for i in top3[:, r]]
        base[f"shap{r + 1}"] = por_indicador.to_numpy()[np.arange(len(base)), top3[:, r]]

    historicos = tabla[tabla["positivo"] == 1].groupby(["cell_id", "mes"]).size().rename("incendios_historicos").reset_index()
    base = base.merge(historicos, on=["cell_id", "mes"], how="left")
    base["incendios_historicos"] = base["incendios_historicos"].fillna(0).astype(int)

    columnas_salida = ["cell_id", "mes", "probabilidad", "nivel", "nivel_nombre", "en_dominio", "pct_bosque_1km",
                       "incendios_historicos"] + [f"{p}{r}" for r in range(1, 4) for p in ("factor", "shap")] + \
                      ["precip_mm", "temp_max_c", "elevacion", "pendiente", "ndvi", "clase_mapbiomas", "dist_carretera_m", "dist_zona_residencial_m"]
    base[columnas_salida].to_csv(DATA_PROCESSED_DIR / "fase8_susceptibilidad_mensual.csv", index=False)

    # --- Figura con los 12 mapas ---
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio").to_crs(celdas.crs)
    cmap = ListedColormap(COLORES)
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    for mes, ax in zip(range(1, 13), axes.ravel()):
        datos = celdas.merge(base[base["mes"] == mes][["cell_id", "nivel", "en_dominio"]], on="cell_id")
        datos.plot(column="nivel", cmap=cmap, vmin=0, vmax=N_NIVELES - 1, ax=ax, linewidth=0)
        datos[~datos["en_dominio"]].plot(ax=ax, facecolor="none", edgecolor="#555555", hatch="////", linewidth=0)
        distrito.boundary.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(MESES[mes - 1])
        ax.set_axis_off()
    leyenda = [Patch(color=c, label=n) for c, n in zip(COLORES, NIVELES)] + \
              [Patch(facecolor="white", edgecolor="#555555", hatch="////", label="< 20 % de bosque (fuera del dominio)")]
    fig.legend(handles=leyenda, loc="lower center", ncol=6, fontsize=11)
    fig.suptitle(f"Susceptibilidad mensual a incendios forestales - Pólvora ({NOMBRES[final]}, climatología 2012-2025)", fontsize=15)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(RESULTADOS_DIR / "fig_mapas_susceptibilidad_12meses.png", dpi=130)

    # --- Datos del visor (archivo JS para poder abrir index.html sin servidor) ---
    celdas_4326 = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    geometria = {c: [[round(x, 5), round(y, 5)] for x, y in g.exterior.coords]
                 for c, g in zip(celdas_4326["cell_id"], celdas_4326.geometry)}
    distrito_4326 = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio").to_crs("EPSG:4326")
    base = base.sort_values(["cell_id", "mes"])
    registros = []
    for cell_id, grupo in base.groupby("cell_id", sort=True):
        g0 = grupo.iloc[0]
        registros.append({
            "id": cell_id, "g": geometria[cell_id], "dom": bool(g0["en_dominio"]),
            "bosque": round(float(g0["pct_bosque_1km"]) * 100, 1), "clase": int(g0["clase_mapbiomas"]),
            "elev": round(float(g0["elevacion"])), "pend": round(float(g0["pendiente"]), 1),
            "dcar": round(float(g0["dist_carretera_m"])), "dres": round(float(g0["dist_zona_residencial_m"])),
            "p": [round(float(v), 5) for v in grupo["probabilidad"]],
            "n": [int(v) for v in grupo["nivel"]],
            "h": [int(v) for v in grupo["incendios_historicos"]],
            "pr": [round(float(v), 1) for v in grupo["precip_mm"]],
            "tm": [round(float(v), 1) for v in grupo["temp_max_c"]],
            "nd": [round(float(v), 3) for v in grupo["ndvi"]],
            "f": [[[indicadores.index(fila[f"factor{r}"]), round(float(fila[f"shap{r}"]), 4)] for r in range(1, 4)]
                  for _, fila in grupo.iterrows()],
        })
    meta = {
        "modelo": NOMBRES[final], "escala_shap": "log-odds" if final == "xgboost" else "probabilidad",
        "niveles": NIVELES, "colores": COLORES, "cortes": [round(v, 5) for v in limites], "meses": MESES,
        "claves": indicadores,
        "indicadores": [NOMBRE_INDICADOR[i] for i in indicadores],
        "clases": NOMBRE_CLASE,
        "dimension": [next(d for d, cols in INDICADORES_POR_DIMENSION.items() if i in cols) for i in indicadores],
        "distrito": json.loads(distrito_4326[["geometry"]].to_json()),
    }
    with open(VISOR_DIR / "datos.js", "w", encoding="utf-8") as f:
        f.write("const META = " + json.dumps(meta, ensure_ascii=False) + ";\n")
        f.write("const CELDAS = " + json.dumps(registros, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"Guardado: {VISOR_DIR / 'datos.js'}")


if __name__ == "__main__":
    main()
