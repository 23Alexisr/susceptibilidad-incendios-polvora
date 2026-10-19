"""
Fase 7_c4 (paso 5) - Reconstruye la tabla de modelado de 8 indicadores con
MapBiomas Coleccion 4 y el periodo extendido a 2012-2025 (ANIO_FIN=2025,
antes 2024). Version paralela de fase7_construir_tabla.py: NO modifica
fase7_tabla_modelado.parquet (C3, 2012-2024), produce
fase7_tabla_modelado_c4.parquet aparte.

Fuentes que CAMBIAN respecto al pipeline C3 original de la Fase 7:
  - clase_mapbiomas, pct_bosque_1km: fase5_predictores_anuales_c4.csv
    (MapBiomas C4, anios 2011-2024, ver fase5_anuales_c4.py).
  - ndvi: fase7_ndvi_mensual.csv (2011-2024, sin cambios) +
    fase7_ndvi_mensual_2025_c4.csv (2025 nuevo, ver fase7_ndvi_mensual_c4.py).
  - precip_mm, temp_max_c: fase5_predictores_mensuales.csv (2000-2024, sin
    cambios) + fase5_predictores_mensuales_2025_c4.csv (2025 nuevo).
  - Etiquetas (eventos de incendio forestal): fase3b_eventos_c4.csv
    (reclasificado con C4, periodo de deteccion FIRMS hasta 2025-12-31).

Fuentes que NO cambian (estaticas, no dependen de MapBiomas ni del periodo):
  - elevacion, pendiente: fase5_predictores_estaticos.csv
  - dist_carretera_m: fase7_dist_carretera.csv (MTC, snapshot dic-2018)
  - dist_zona_residencial_m: fase7_dist_centro_poblado.csv (INEI, snapshot)

Imputacion de NDVI: IDENTICA metodologia a la decision del usuario para la
version C3 (climatologia celda+mes, calculada SOLO con los anios de
entrenamiento 2012-2021 -- el enunciado del encargo de C4 confirmo
explicitamente mantener esto sin cambios, aunque el conjunto de prueba
ahora sea 2022-2025 en vez de 2022-2024).
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from config import DATA_PROCESSED_DIR

ANIO_INICIO = 2012
ANIO_FIN = 2025  # antes 2024
UMBRAL_BOSQUE = 0.20
ANIO_CORTE_TEST = 2022  # sin cambios: train 2012-2021, test ahora 2022-2025


def construir_etiquetas_por_evento(
    categorias=("incendio_forestal_no_estricto", "incendio_forestal_estricto"),
) -> pd.DataFrame:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos_c4.csv", parse_dates=["fecha_inicio"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year
    eventos["mes"] = eventos["fecha_inicio"].dt.month

    extendida = eventos[eventos["categoria_dominante"].isin(categorias)].copy()
    con_viirs = extendida[extendida["sensores"].str.contains("viirs_snpp")]
    con_viirs = con_viirs[(con_viirs["anio"] >= ANIO_INICIO) & (con_viirs["anio"] <= ANIO_FIN)]
    print(f"Eventos extendidos (C4) con participacion VIIRS, {ANIO_INICIO}-{ANIO_FIN}: {len(con_viirs)} (de {len(extendida)} extendidos totales)")

    puntos = gpd.GeoDataFrame(
        con_viirs, geometry=gpd.points_from_xy(con_viirs["lon"], con_viirs["lat"]), crs="EPSG:4326"
    ).to_crs("EPSG:32718")

    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    union = gpd.sjoin(puntos, celdas, how="inner", predicate="within")
    print(f"Eventos que cayeron dentro de alguna celda: {len(union)}")

    etiquetas = (
        union.groupby(["cell_id", "anio", "mes"])
        .size()
        .reset_index(name="n_eventos_incendio_forestal")
    )
    etiquetas["positivo"] = 1
    return etiquetas


def main() -> None:
    print("--- Construyendo etiquetas por evento (C4, periodo 2012-2025) ---")
    etiquetas = construir_etiquetas_por_evento()
    print(f"Combinaciones celda-anio-mes positivas: {len(etiquetas)}")

    print("\n--- Cargando predictores (C4 + 2025) ---")
    estaticos = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv")[["cell_id", "elevacion", "pendiente"]]

    anuales_c4 = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales_c4.csv")[
        ["cell_id", "anio", "clase_mapbiomas", "pct_bosque_1km"]
    ].copy()
    anuales_c4["clase_mapbiomas"] = anuales_c4["clase_mapbiomas"].round().astype(int)
    print(f"Predictores anuales C4: anios {sorted(anuales_c4['anio'].unique())}")

    mensuales_2000_2024 = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv")[
        ["cell_id", "anio", "mes", "precip_mm", "temp_max_c"]
    ]
    mensuales_2025 = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales_2025_c4.csv")[
        ["cell_id", "anio", "mes", "precip_mm", "temp_max_c"]
    ]
    mensuales = pd.concat([mensuales_2000_2024, mensuales_2025], ignore_index=True)
    print(f"Predictores mensuales (clima) combinados: {mensuales['anio'].min()}-{mensuales['anio'].max()}")

    ndvi_2011_2024 = pd.read_csv(DATA_PROCESSED_DIR / "fase7_ndvi_mensual.csv")
    ndvi_2025 = pd.read_csv(DATA_PROCESSED_DIR / "fase7_ndvi_mensual_2025_c4.csv")
    ndvi = pd.concat([ndvi_2011_2024, ndvi_2025], ignore_index=True)
    print(f"NDVI combinado: {ndvi['anio'].min()}-{ndvi['anio'].max()}")

    dist_carretera = pd.read_csv(DATA_PROCESSED_DIR / "fase7_dist_carretera.csv")
    dist_residencial = pd.read_csv(DATA_PROCESSED_DIR / "fase7_dist_centro_poblado.csv")

    celdas_ids = estaticos["cell_id"].unique()
    filas_esqueleto = [
        {"cell_id": c, "anio": a, "mes": m}
        for c in celdas_ids
        for a in range(ANIO_INICIO, ANIO_FIN + 1)
        for m in range(1, 13)
    ]
    tabla = pd.DataFrame(filas_esqueleto)
    print(f"Esqueleto celda x anio x mes ({ANIO_INICIO}-{ANIO_FIN}): {len(tabla)} filas")

    tabla["anio_clima"] = np.where(tabla["mes"] == 1, tabla["anio"] - 1, tabla["anio"])
    tabla["mes_clima"] = np.where(tabla["mes"] == 1, 12, tabla["mes"] - 1)

    mensuales_ren = mensuales.rename(columns={"anio": "anio_clima", "mes": "mes_clima"})
    tabla = tabla.merge(
        mensuales_ren[["cell_id", "anio_clima", "mes_clima", "precip_mm", "temp_max_c"]],
        on=["cell_id", "anio_clima", "mes_clima"],
        how="left",
    )

    ndvi_ren = ndvi.rename(columns={"anio": "anio_clima", "mes": "mes_clima"})
    tabla = tabla.merge(
        ndvi_ren[["cell_id", "anio_clima", "mes_clima", "ndvi"]],
        on=["cell_id", "anio_clima", "mes_clima"],
        how="left",
    )

    tabla["anio_uso_suelo"] = tabla["anio"] - 1
    anuales_ren = anuales_c4.rename(columns={"anio": "anio_uso_suelo"})
    tabla = tabla.merge(
        anuales_ren[["cell_id", "anio_uso_suelo", "clase_mapbiomas", "pct_bosque_1km"]],
        on=["cell_id", "anio_uso_suelo"],
        how="left",
    )

    tabla = tabla.merge(estaticos, on="cell_id", how="left")
    tabla = tabla.merge(dist_carretera, on="cell_id", how="left")
    tabla = tabla.merge(dist_residencial, on="cell_id", how="left")

    tabla = tabla.merge(etiquetas[["cell_id", "anio", "mes", "positivo"]], on=["cell_id", "anio", "mes"], how="left")
    tabla["positivo"] = tabla["positivo"].fillna(0).astype(int)

    tabla = tabla.drop(columns=["anio_clima", "mes_clima", "anio_uso_suelo"])

    n_antes_filtro = len(tabla)
    pos_antes_filtro = tabla["positivo"].sum()

    tabla_filtrada = tabla[tabla["pct_bosque_1km"] >= UMBRAL_BOSQUE].copy()

    print(f"\nFilas antes del filtro de bosque >=20%: {n_antes_filtro} ({pos_antes_filtro} positivos)")
    print(f"Filas después del filtro:                {len(tabla_filtrada)} ({tabla_filtrada['positivo'].sum()} positivos)")
    print(f"Celdas-mes descartadas por tener <20% de bosque: {n_antes_filtro - len(tabla_filtrada)}")

    # --- Imputacion de ndvi: IDENTICA metodologia que la version C3 ---
    print("\n--- Imputación de ndvi (climatología celda+mes, solo años de entrenamiento 2012-2021) ---")
    tabla_filtrada["ndvi_imputado"] = tabla_filtrada["ndvi"].isna().astype(int)
    n_nulos_antes = int(tabla_filtrada["ndvi_imputado"].sum())
    n_pos_nulos_antes = int(tabla_filtrada.loc[tabla_filtrada["positivo"] == 1, "ndvi_imputado"].sum())
    print(f"Celdas-mes con ndvi faltante (antes de imputar): {n_nulos_antes} ({n_nulos_antes/len(tabla_filtrada):.4%} de la tabla)")
    print(f"De esas, con positivo=1 (incendio): {n_pos_nulos_antes} de {int(tabla_filtrada['positivo'].sum())} positivos totales")

    train_mask = tabla_filtrada["anio"] < ANIO_CORTE_TEST
    lookup_celda_mes = (
        tabla_filtrada.loc[train_mask & tabla_filtrada["ndvi"].notna()].groupby(["cell_id", "mes"])["ndvi"].median()
    )
    lookup_mes_global = (
        tabla_filtrada.loc[train_mask & tabla_filtrada["ndvi"].notna()].groupby("mes")["ndvi"].median()
    )

    idx_nulo = tabla_filtrada["ndvi"].isna()
    claves = pd.MultiIndex.from_frame(tabla_filtrada.loc[idx_nulo, ["cell_id", "mes"]])
    tabla_filtrada.loc[idx_nulo, "ndvi"] = claves.map(lookup_celda_mes).to_numpy()

    idx_nulo_respaldo = tabla_filtrada["ndvi"].isna()
    n_respaldo = int(idx_nulo_respaldo.sum())
    if n_respaldo > 0:
        print(f"Celdas-mes que usan respaldo de mediana del mes (todas las celdas, 2012-2021): {n_respaldo}")
        tabla_filtrada.loc[idx_nulo_respaldo, "ndvi"] = tabla_filtrada.loc[idx_nulo_respaldo, "mes"].map(lookup_mes_global).to_numpy()

    n_nulos_despues = int(tabla_filtrada["ndvi"].isna().sum())
    print(f"Nulos de ndvi después de imputar: {n_nulos_despues} (debe ser 0)")
    if n_nulos_despues > 0:
        raise SystemExit("AVISO: quedaron nulos de ndvi sin imputar. Revisar antes de continuar.")

    tabla_filtrada = tabla_filtrada.drop(columns=["pct_bosque_1km"])

    ruta = DATA_PROCESSED_DIR / "fase7_tabla_modelado_c4.parquet"
    tabla_filtrada.to_parquet(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    print("\n--- Resumen final (C4, 2012-2025) ---")
    print(f"Filas: {len(tabla_filtrada)}")
    print(f"Positivos: {tabla_filtrada['positivo'].sum()}")
    print(f"Proporción de positivos: {tabla_filtrada['positivo'].mean():.4%}")

    train = tabla_filtrada[tabla_filtrada["anio"] < ANIO_CORTE_TEST]
    test = tabla_filtrada[tabla_filtrada["anio"] >= ANIO_CORTE_TEST]
    print(f"\nTrain (2012-2021): {len(train)} filas, {train['positivo'].sum()} positivos")
    print(f"Test  (2022-2025): {len(test)} filas, {test['positivo'].sum()} positivos")

    cols_predictoras = ["precip_mm", "temp_max_c", "elevacion", "pendiente", "clase_mapbiomas", "ndvi", "dist_carretera_m", "dist_zona_residencial_m"]
    nulos = tabla_filtrada[cols_predictoras].isna().sum()
    print(f"\nNulos por columna predictora (de {len(tabla_filtrada)} filas):")
    print(nulos)
    for col in cols_predictoras:
        n = int(nulos[col])
        if n > 0:
            print(f"  *** {col}: {n} celdas-mes con dato faltante ({n/len(tabla_filtrada):.4%} de la tabla) ***")

    # --- Comparacion explicita con la poblacion C3 ---
    FILAS_C3, POS_C3 = 323_988, 1_300
    print(f"\n--- Comparación con C3 (fase7_tabla_modelado.parquet) ---")
    print(f"C3: {FILAS_C3} filas, {POS_C3} positivos")
    print(f"C4: {len(tabla_filtrada)} filas, {tabla_filtrada['positivo'].sum()} positivos")
    print(f"Diferencia: {len(tabla_filtrada) - FILAS_C3:+d} filas, {int(tabla_filtrada['positivo'].sum()) - POS_C3:+d} positivos")


if __name__ == "__main__":
    main()
