"""
Fase 7 (paso 4) - Construye la tabla de modelado realineada a los 8
indicadores de la matriz de operacionalizacion de la tesis (2 por
dimension), EN VEZ de las 16 variables del modelo piloto de la Fase 6b.

Este script NO modifica ni borra nada de la Fase 6b: el modelo de 16
variables queda intacto como referencia (src/fase6b_*.py,
data/processed/fase6b_tabla_modelado*.parquet). Este es un pipeline
paralelo (fase7_*) que parte de los MISMOS insumos ya calculados en la
Fase 5 para las 6 variables que se mantienen, y solo agrega las 2
variables nuevas (ndvi, dist_carretera_m, dist_zona_residencial_m -- en
realidad 3 nuevas: ndvi reemplaza a ndmi en la dimension ambiental, y las
2 de la dimension antropogenica son enteramente nuevas).

POBLACION Y FILTRO: IDENTICOS a la Fase 6b, a proposito (lo pidio
explicitamente la consigna) -- se reutiliza la MISMA funcion de etiquetado
por eventos (extendida + participacion VIIRS + 2012-2024) y el MISMO
filtro de >=20% de bosque en el anio anterior. pct_bosque_1km se usa aqui
SOLO para filtrar filas, no se incluye como columna predictora en la
tabla final (se elimina despues de filtrar).

Las 8 variables finales (predictoras) de esta tabla:
  Climatica:      precip_mm, temp_max_c              (de la Fase 5, sin cambios)
  Topografica:    elevacion, pendiente                (de la Fase 5, sin cambios)
  Ambiental:      clase_mapbiomas, ndvi               (clase: Fase 5 sin cambios; ndvi: NUEVO, fase7_ndvi_mensual.py)
  Antropogenica:  dist_carretera_m, dist_zona_residencial_m  (NUEVAS, fase7_vias_distancia.py y fase7_centros_poblados.py)

Rezagos (mismas reglas que la Fase 5/6b, sin fuga de datos):
  - Clima y NDVI: mes CALENDARIO ANTERIOR (M-1).
  - Uso de suelo (clase_mapbiomas, pct_bosque_1km para el filtro): ANO ANTERIOR (Y-1).
  - Fisicos (elevacion, pendiente) y las 2 distancias nuevas: estaticos, no varian en el tiempo.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from config import DATA_PROCESSED_DIR

ANIO_INICIO = 2012
ANIO_FIN = 2024
UMBRAL_BOSQUE = 0.20  # igual que fase6b_construir_tabla.py
ANIO_CORTE_TEST = 2022  # debe coincidir con fase7_modelo.py: train 2012-2021, test 2022-2024


def construir_etiquetas_por_evento(
    categorias=("incendio_forestal_no_estricto", "incendio_forestal_estricto"),
) -> pd.DataFrame:
    """Identica a la funcion homonima de fase6b_construir_tabla.py: eventos
    (Fase 3b) de definicion extendida, con participacion de VIIRS,
    2012-2024, agrupados por celda-anio-mes. Se reproduce aqui (no se
    importa desde fase6b_construir_tabla.py) para que este script sea
    autocontenido y no dependa de que el modulo de la Fase 6b permanezca
    sin cambios en el futuro.
    """
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year
    eventos["mes"] = eventos["fecha_inicio"].dt.month

    extendida = eventos[eventos["categoria_dominante"].isin(categorias)].copy()
    con_viirs = extendida[extendida["sensores"].str.contains("viirs_snpp")]
    con_viirs = con_viirs[(con_viirs["anio"] >= ANIO_INICIO) & (con_viirs["anio"] <= ANIO_FIN)]
    print(f"Eventos extendidos con participacion VIIRS, {ANIO_INICIO}-{ANIO_FIN}: {len(con_viirs)} (de {len(extendida)} extendidos totales)")

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
    print("--- Construyendo etiquetas por evento (identico a la Fase 6b) ---")
    etiquetas = construir_etiquetas_por_evento()
    print(f"Combinaciones celda-anio-mes positivas: {len(etiquetas)}")

    print("\n--- Cargando predictores (6 reutilizados de la Fase 5 + 2 nuevos de la Fase 7) ---")
    estaticos = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv")[
        ["cell_id", "elevacion", "pendiente"]
    ]
    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv")[
        ["cell_id", "anio", "clase_mapbiomas", "pct_bosque_1km"]
    ].copy()
    anuales["clase_mapbiomas"] = anuales["clase_mapbiomas"].round().astype(int)
    mensuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv")[
        ["cell_id", "anio", "mes", "precip_mm", "temp_max_c"]
    ]
    ndvi = pd.read_csv(DATA_PROCESSED_DIR / "fase7_ndvi_mensual.csv")
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

    # --- Rezago de clima + NDVI: mes anterior (M-1, con rollover de anio) ---
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

    # --- Rezago de uso de suelo: anio anterior (Y-1) ---
    tabla["anio_uso_suelo"] = tabla["anio"] - 1
    anuales_ren = anuales.rename(columns={"anio": "anio_uso_suelo"})
    tabla = tabla.merge(
        anuales_ren[["cell_id", "anio_uso_suelo", "clase_mapbiomas", "pct_bosque_1km"]],
        on=["cell_id", "anio_uso_suelo"],
        how="left",
    )

    # --- Estaticos (fisicos + accesibilidad nueva) ---
    tabla = tabla.merge(estaticos, on="cell_id", how="left")
    tabla = tabla.merge(dist_carretera, on="cell_id", how="left")
    tabla = tabla.merge(dist_residencial, on="cell_id", how="left")

    # --- Etiqueta ---
    tabla = tabla.merge(etiquetas[["cell_id", "anio", "mes", "positivo"]], on=["cell_id", "anio", "mes"], how="left")
    tabla["positivo"] = tabla["positivo"].fillna(0).astype(int)

    tabla = tabla.drop(columns=["anio_clima", "mes_clima", "anio_uso_suelo"])

    n_antes_filtro = len(tabla)
    pos_antes_filtro = tabla["positivo"].sum()

    # --- Filtro de bosque >=20% (ano anterior) -- IDENTICO al de la Fase 6b ---
    tabla_filtrada = tabla[tabla["pct_bosque_1km"] >= UMBRAL_BOSQUE].copy()

    print(f"\nFilas antes del filtro de bosque >=20%: {n_antes_filtro} ({pos_antes_filtro} positivos)")
    print(f"Filas después del filtro:                {len(tabla_filtrada)} ({tabla_filtrada['positivo'].sum()} positivos)")
    print(f"Celdas-mes descartadas por tener <20% de bosque: {n_antes_filtro - len(tabla_filtrada)}")

    # Control explicito pedido por la consigna: la poblacion NO debe cambiar
    # respecto al piloto de 16 variables (323,988 filas, 1,300 positivas).
    FILAS_ESPERADAS, POS_ESPERADOS = 323_988, 1_300
    if len(tabla_filtrada) != FILAS_ESPERADAS or tabla_filtrada["positivo"].sum() != POS_ESPERADOS:
        print(
            f"\n*** AVISO: la poblacion NO coincide con la esperada "
            f"({FILAS_ESPERADAS} filas, {POS_ESPERADOS} positivos). "
            f"Se obtuvo {len(tabla_filtrada)} filas, {tabla_filtrada['positivo'].sum()} positivos. "
            f"Revisar antes de continuar. ***"
        )
    else:
        print(f"\nControl de poblacion OK: coincide exactamente con el piloto de 16 variables ({FILAS_ESPERADAS} filas, {POS_ESPERADOS} positivos).")

    # --- Imputacion de ndvi (decision del usuario, 2026-10-09) ---
    # Climatologia celda+mes calculada SOLO con los anios de entrenamiento
    # (2012-2021, ANIO_CORTE_TEST=2022) y aplicada tanto a train como a
    # test, para no filtrar informacion del test hacia el entrenamiento.
    # Respaldo: si una celda+mes no tiene NINGUN dato en 2012-2021, se usa
    # la mediana de ese mes calendario entre TODAS las celdas (tambien
    # solo con 2012-2021). ndvi_imputado (0/1) queda como columna de
    # registro, no se usa como predictor.
    print("\n--- Imputación de ndvi (climatología celda+mes, solo años de entrenamiento 2012-2021) ---")
    tabla_filtrada["ndvi_imputado"] = tabla_filtrada["ndvi"].isna().astype(int)
    n_nulos_antes = int(tabla_filtrada["ndvi_imputado"].sum())
    n_pos_nulos_antes = int(tabla_filtrada.loc[tabla_filtrada["positivo"] == 1, "ndvi_imputado"].sum())
    print(f"Celdas-mes con ndvi faltante (antes de imputar): {n_nulos_antes} ({n_nulos_antes/len(tabla_filtrada):.4%} de la tabla)")
    print(f"De esas, con positivo=1 (incendio): {n_pos_nulos_antes} de {int(tabla_filtrada['positivo'].sum())} positivos totales")
    print("Distribución por mes calendario de TODAS las celdas-mes con ndvi faltante:")
    print(tabla_filtrada.loc[tabla_filtrada["ndvi_imputado"] == 1, "mes"].value_counts().sort_index())
    print("Distribución por mes calendario de las celdas-mes CON INCENDIO y ndvi faltante:")
    print(tabla_filtrada.loc[(tabla_filtrada["ndvi_imputado"] == 1) & (tabla_filtrada["positivo"] == 1), "mes"].value_counts().sort_index())

    train_mask = tabla_filtrada["anio"] < ANIO_CORTE_TEST
    lookup_celda_mes = (
        tabla_filtrada.loc[train_mask & tabla_filtrada["ndvi"].notna()]
        .groupby(["cell_id", "mes"])["ndvi"]
        .median()
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
        print(f"\nCeldas-mes sin dato de su propia celda+mes en 2012-2021 (usan respaldo de mediana del mes, todas las celdas): {n_respaldo}")
        tabla_filtrada.loc[idx_nulo_respaldo, "ndvi"] = tabla_filtrada.loc[idx_nulo_respaldo, "mes"].map(lookup_mes_global).to_numpy()

    n_nulos_despues = int(tabla_filtrada["ndvi"].isna().sum())
    print(f"\nNulos de ndvi después de imputar: {n_nulos_despues} (debe ser 0)")
    if n_nulos_despues > 0:
        raise SystemExit("AVISO: quedaron nulos de ndvi sin imputar incluso con el respaldo. Revisar antes de continuar.")

    # pct_bosque_1km se usa SOLO como filtro, no como predictor: se elimina
    # de la tabla final (si se necesita para analisis posteriores, esta en
    # fase5_predictores_anuales.csv).
    tabla_filtrada = tabla_filtrada.drop(columns=["pct_bosque_1km"])

    ruta = DATA_PROCESSED_DIR / "fase7_tabla_modelado.parquet"
    tabla_filtrada.to_parquet(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    print("\n--- Resumen final ---")
    print(f"Filas: {len(tabla_filtrada)}")
    print(f"Positivos: {tabla_filtrada['positivo'].sum()}")
    print(f"Proporción de positivos: {tabla_filtrada['positivo'].mean():.4%}")
    cols_predictoras = ["precip_mm", "temp_max_c", "elevacion", "pendiente", "clase_mapbiomas", "ndvi", "dist_carretera_m", "dist_zona_residencial_m"]
    print(f"\nNulos por columna predictora (de {len(tabla_filtrada)} filas):")
    nulos = tabla_filtrada[cols_predictoras].isna().sum()
    print(nulos)
    for col in cols_predictoras:
        n = int(nulos[col])
        if n > 0:
            print(f"  *** {col}: {n} celdas-mes con dato faltante ({n/len(tabla_filtrada):.4%} de la tabla) ***")


if __name__ == "__main__":
    main()
