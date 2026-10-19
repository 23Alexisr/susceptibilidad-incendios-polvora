"""
Fase 6b (pasos 2-5) - Reconstruye la tabla de modelado con las correcciones
acordadas:

2. Etiqueta positiva por EVENTOS (Fase 3b), definicion EXTENDIDA
   (incendio_forestal_no_estricto + incendio_forestal_estricto), no por
   deteccion individual como en la Fase 5 original.
3. Filtro de grilla: solo filas con >= 20% de bosque en el año anterior
   (pct_bosque_1km, ya rezagado).
4. Predictores de accesibilidad: se eliminan dist_via_m y
   dist_centro_poblado_m (OSM). Se mantiene dist_rio_m (OSM, unico que se
   conserva de esa fuente). Se usan dist_agropecuario_m y
   dist_borde_bosque_m (distancia a "no bosque" de MapBiomas del año
   anterior) -- ambos YA estaban calculados en la Fase 5
   (fase5_predictores_anuales.csv), no hubo que volver a consultar Earth
   Engine para esto.
5. Solo VIIRS, 2012-2024: la tabla queda restringida a esos años, y la
   etiqueta positiva solo se construye a partir de eventos donde VIIRS
   participo (sensores == 'viirs_snpp' o 'modis,viirs_snpp'). Eventos
   detectados unicamente por MODIS puro no cuentan como positivo aqui
   (MODIS se reserva para el analisis de tendencia de la Fase 3b/6b, no
   para entrenar).
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from config import DATA_PROCESSED_DIR

ANIO_INICIO = 2012
ANIO_FIN = 2024
UMBRAL_BOSQUE = 0.20


def construir_etiquetas_por_evento(
    categorias=("incendio_forestal_no_estricto", "incendio_forestal_estricto"),
) -> pd.DataFrame:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year
    eventos["mes"] = eventos["fecha_inicio"].dt.month

    extendida = eventos[eventos["categoria_dominante"].isin(categorias)].copy()
    con_viirs = extendida[extendida["sensores"].str.contains("viirs_snpp")]
    con_viirs = con_viirs[(con_viirs["anio"] >= ANIO_INICIO) & (con_viirs["anio"] <= ANIO_FIN)]
    print(f"Eventos extendidos con participacion VIIRS, 2012-2024: {len(con_viirs)} (de {len(extendida)} extendidos totales)")

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


def main(categorias=("incendio_forestal_no_estricto", "incendio_forestal_estricto"), nombre_salida="fase6b_tabla_modelado.parquet") -> None:
    print(f"--- Construyendo etiquetas por evento ({categorias}, con VIIRS) ---")
    etiquetas = construir_etiquetas_por_evento(categorias)
    print(f"Combinaciones celda-año-mes positivas: {len(etiquetas)}")

    print("\n--- Cargando predictores ya calculados en la Fase 5 (sin volver a consultar Earth Engine) ---")
    estaticos = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv")
    osm = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_osm.csv")[["cell_id", "dist_rio_m"]]  # solo rios
    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv")
    anuales["clase_mapbiomas"] = anuales["clase_mapbiomas"].round().astype(int)
    mensuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv")

    # Reutilizamos el SPI ya calculado en la Fase 5 (misma metodologia,
    # calculado sobre 2000-2024 completo para tener referencia climatologica
    # larga) en vez de recalcularlo solo con 2012-2024, que daria una
    # referencia mas corta e inestable.
    tabla_previa = pd.read_parquet(DATA_PROCESSED_DIR / "tabla_modelado.parquet")
    spi_lookup = tabla_previa[["cell_id", "anio", "mes", "spi_1m", "spi_3m"]].drop_duplicates()

    celdas_ids = estaticos["cell_id"].unique()
    filas_esqueleto = [
        {"cell_id": c, "anio": a, "mes": m}
        for c in celdas_ids
        for a in range(ANIO_INICIO, ANIO_FIN + 1)
        for m in range(1, 13)
    ]
    tabla = pd.DataFrame(filas_esqueleto)
    print(f"Esqueleto celda x año x mes (2012-2024): {len(tabla)} filas")

    # Rezago de clima/NDMI: mes anterior.
    tabla["anio_clima"] = np.where(tabla["mes"] == 1, tabla["anio"] - 1, tabla["anio"])
    tabla["mes_clima"] = np.where(tabla["mes"] == 1, 12, tabla["mes"] - 1)
    cols_clima_base = ["precip_mm", "dias_secos_mes", "temp_max_c", "vpd_kpa", "ndmi"]
    mensuales_ren = mensuales.rename(columns={"anio": "anio_clima", "mes": "mes_clima"})
    tabla = tabla.merge(
        mensuales_ren[["cell_id", "anio_clima", "mes_clima"] + cols_clima_base],
        on=["cell_id", "anio_clima", "mes_clima"],
        how="left",
    )
    tabla = tabla.merge(spi_lookup, on=["cell_id", "anio", "mes"], how="left")

    # Rezago de uso de suelo: año anterior.
    tabla["anio_uso_suelo"] = tabla["anio"] - 1
    cols_uso_suelo = [
        "clase_mapbiomas",
        "pct_bosque_1km",
        "pct_perdida_bosque_ese_anio",
        "dist_borde_bosque_m",
        "dist_agropecuario_m",
    ]
    anuales_ren = anuales.rename(columns={"anio": "anio_uso_suelo"})
    tabla = tabla.merge(
        anuales_ren[["cell_id", "anio_uso_suelo"] + cols_uso_suelo], on=["cell_id", "anio_uso_suelo"], how="left"
    )

    # Fisicos + accesibilidad (solo rio de OSM; sin vias ni centros poblados).
    tabla = tabla.merge(estaticos, on="cell_id", how="left")
    tabla = tabla.merge(osm, on="cell_id", how="left")

    # Etiqueta.
    tabla = tabla.merge(etiquetas[["cell_id", "anio", "mes", "positivo"]], on=["cell_id", "anio", "mes"], how="left")
    tabla["positivo"] = tabla["positivo"].fillna(0).astype(int)

    tabla = tabla.drop(columns=["anio_clima", "mes_clima", "anio_uso_suelo"])

    n_antes_filtro = len(tabla)
    pos_antes_filtro = tabla["positivo"].sum()

    # Filtro de bosque >= 20% (año anterior).
    tabla_filtrada = tabla[tabla["pct_bosque_1km"] >= UMBRAL_BOSQUE].copy()

    print(f"\nFilas antes del filtro de bosque >=20%: {n_antes_filtro} ({pos_antes_filtro} positivos)")
    print(f"Filas después del filtro:                {len(tabla_filtrada)} ({tabla_filtrada['positivo'].sum()} positivos)")
    print(f"Celdas-mes descartadas por tener <20% de bosque: {n_antes_filtro - len(tabla_filtrada)}")

    ruta = DATA_PROCESSED_DIR / nombre_salida
    tabla_filtrada.to_parquet(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    print("\n--- Resumen final ---")
    print(f"Filas: {len(tabla_filtrada)}")
    print(f"Positivos: {tabla_filtrada['positivo'].sum()}")
    print(f"Proporción de positivos: {tabla_filtrada['positivo'].mean():.4%}")
    print(f"Nulos por columna:\n{tabla_filtrada.isna().sum()}")


if __name__ == "__main__":
    main()
