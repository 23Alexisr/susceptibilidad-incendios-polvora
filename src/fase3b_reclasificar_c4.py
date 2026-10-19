"""
Fase 3b_c4 (pasos 1-2) - Version paralela de fase3b_reclasificar.py usando
MapBiomas Coleccion 4 (hasta 2025) en vez de Coleccion 3 (hasta 2024).

Como se tratan los anios t+1/t+2 que caen en o despues de 2026 (pedido
explicito del encargo, punto 2): igual que el script original con la C3,
se usa `.clip(upper=MAPBIOMAS_ANIO_MAX)`. Con MAPBIOMAS_ANIO_MAX=2025:
  - Deteccion de 2023: t+1=2024 (real), t+2=2025 (real) -- ya no hay que
    recortar NINGUNO de los dos (con C3, t+2=2025 SI se recortaba a 2024).
  - Deteccion de 2024: t+1=2025 (real, NUEVO respecto a la C3, donde se
    recortaba a 2024 = repetia t), t+2=2026 (no existe, se recorta a 2025
    = repite t+1).
  - Deteccion de 2025: t+1=2026 (no existe, se recorta a 2025 = repite t),
    t+2=2027 (no existe, se recorta a 2025 = repite t+1, que a su vez
    repite t). Es decir, para las detecciones del anio mas reciente (2025)
    la ventana de verificacion "hacia adelante" queda vacia, exactamente
    igual de limitada que como quedaban las de 2024 bajo la C3 -- el
    problema simplemente se desplaza un anio, no desaparece (es inherente
    a usar la ultima coleccion disponible, cualquiera que sea).
Se reutiliza la columna "anios_verificados_adelante" ya presente en el
script original (cuenta cuantos de t+1/t+2 son datos NUEVOS de verdad, no
un recorte que repite un anio ya usado) para cuantificar esto explicitamente
en el reporte final, sin necesidad de logica adicional.
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1"
MAPBIOMAS_ANIO_MAX = 2025  # C4: un anio mas que la C3
BOSQUE_CODES = [3, 4, 5, 6]
AGROPECUARIO_CODES = [9, 15, 21, 35, 40]  # identicos a la C3, verificado contra la leyenda oficial de la C4
RESOLUCION_SENSOR_M = {"modis": 1000, "viirs_snpp": 375}
VECINDAD_PIXELES = 256
TAMANO_LOTE = 800


def construir_fc_buffer(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in gdf.iterrows():
        lado = RESOLUCION_SENSOR_M[fila["sensor"]]
        punto = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        cuadrado = punto.buffer(lado / 2, 1).bounds()
        features.append(ee.Feature(cuadrado, {"id": fila["id"]}))
    return ee.FeatureCollection(features)


def construir_fc_puntos(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in gdf.iterrows():
        punto = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        features.append(ee.Feature(punto, {"id": fila["id"]}))
    return ee.FeatureCollection(features)


def moda_por_anio(gdf: gpd.GeoDataFrame, columna_anio: str, nombre_resultado: str) -> pd.DataFrame:
    imagen_base = ee.Image(MAPBIOMAS_ID)
    resultados = []
    for anio in sorted(gdf[columna_anio].unique()):
        subset = gdf[gdf[columna_anio] == anio]
        for inicio in range(0, len(subset), TAMANO_LOTE):
            lote = subset.iloc[inicio : inicio + TAMANO_LOTE]
            fc = construir_fc_buffer(lote)
            banda = imagen_base.select(f"classification_{anio}")
            reducido = banda.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
            for feat in reducido.getInfo()["features"]:
                p = feat["properties"]
                resultados.append({"id": p["id"], nombre_resultado: p.get("mode")})
        print(f"  [{nombre_resultado}] anio {anio}: {len(subset)} puntos")
    return pd.DataFrame(resultados)


def fraccion_bosque_y_distancia_agro(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    resultados = []
    for anio in sorted(gdf["anio_tm1"].unique()):
        subset = gdf[gdf["anio_tm1"] == anio]
        clasif = ee.Image(MAPBIOMAS_ID).select(f"classification_{anio}")
        bosque_mask = clasif.remap(BOSQUE_CODES, [1] * len(BOSQUE_CODES), 0).rename("es_bosque")
        agro_mask = clasif.remap(AGROPECUARIO_CODES, [1] * len(AGROPECUARIO_CODES), 0).rename("es_agro")
        dist_agro = agro_mask.fastDistanceTransform(VECINDAD_PIXELES).sqrt().multiply(30).rename("dist_agro_m")

        for inicio in range(0, len(subset), TAMANO_LOTE):
            lote = subset.iloc[inicio : inicio + TAMANO_LOTE]

            fc_buffer = construir_fc_buffer(lote)
            red_frac = bosque_mask.reduceRegions(collection=fc_buffer, reducer=ee.Reducer.mean(), scale=30)
            frac_por_id = {f["properties"]["id"]: f["properties"].get("mean") for f in red_frac.getInfo()["features"]}

            fc_puntos = construir_fc_puntos(lote)
            red_dist = dist_agro.reduceRegions(collection=fc_puntos, reducer=ee.Reducer.first(), scale=30)
            dist_por_id = {f["properties"]["id"]: f["properties"].get("first") for f in red_dist.getInfo()["features"]}

            for id_ in lote["id"]:
                resultados.append(
                    {
                        "id": id_,
                        "pct_bosque_pixel": frac_por_id.get(id_),
                        "dist_agropecuario_deteccion_m": dist_por_id.get(id_),
                    }
                )
        print(f"  [estricto] anio_tm1 {anio}: {len(subset)} puntos")
    return pd.DataFrame(resultados)


def main() -> None:
    detecciones = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado_c4.gpkg", layer="detecciones_clasificadas")
    incendios = detecciones[detecciones["categoria"] == "incendio_forestal"].copy().reset_index(drop=True)
    incendios["id"] = incendios.index.astype(str)
    print(f"Detecciones 'incendio_forestal' (C4) a revisar: {len(incendios)}")

    incendios["anio_t1_real"] = incendios["anio_deteccion"] + 1
    incendios["anio_t2_real"] = incendios["anio_deteccion"] + 2
    incendios["anio_t1"] = incendios["anio_t1_real"].clip(upper=MAPBIOMAS_ANIO_MAX)
    incendios["anio_t2"] = incendios["anio_t2_real"].clip(upper=MAPBIOMAS_ANIO_MAX)
    incendios["anios_verificados_adelante"] = (
        (incendios["anio_t1"] > incendios["anio_t"]).astype(int)
        + (incendios["anio_t2"] > incendios["anio_t1"]).astype(int)
    )

    print("\n--- Paso 1: clase en t+1 y t+2 ---")
    clase_t1 = moda_por_anio(incendios, "anio_t1", "clase_t1")
    clase_t2 = moda_por_anio(incendios, "anio_t2", "clase_t2")
    incendios = incendios.merge(clase_t1, on="id", how="left").merge(clase_t2, on="id", how="left")
    incendios["bosque_t1"] = incendios["clase_t1"].isin(BOSQUE_CODES)
    incendios["bosque_t2"] = incendios["clase_t2"].isin(BOSQUE_CODES)

    incendios["categoria_extendida"] = incendios.apply(
        lambda f: "incendio_forestal" if (f["bosque_t1"] and f["bosque_t2"]) else "quema_deforestacion_extendida",
        axis=1,
    )

    print("\n--- Paso 2: fraccion real de bosque y distancia a agropecuario (t-1) ---")
    extra = fraccion_bosque_y_distancia_agro(incendios)
    incendios = incendios.merge(extra, on="id", how="left")

    incendios["es_estricto"] = (
        (incendios["categoria_extendida"] == "incendio_forestal")
        & (incendios["pct_bosque_pixel"] >= 0.8)
        & (incendios["dist_agropecuario_deteccion_m"] > 500)
    )

    ruta = DATA_PROCESSED_DIR / "fase3b_incendios_depurados_c4.csv"
    incendios.drop(columns="geometry").to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")
    incendios.to_file(DATA_PROCESSED_DIR / "fase3b_incendios_depurados_c4.gpkg", layer="incendios", driver="GPKG")

    print("\n=== Comparacion de conteos (C4) ===")
    n_original = len(incendios)
    n_extendida = (incendios["categoria_extendida"] == "incendio_forestal").sum()
    n_estricto = incendios["es_estricto"].sum()
    print(f"Fase 3_c4 original (incendio_forestal, t-1 vs t):        {n_original}")
    print(f"Reclasificados a quema_deforestacion_extendida:       {n_original - n_extendida}")
    print(f"Fase 3b_c4 - version extendida (t, t+1, t+2 todo bosque): {n_extendida}  ({n_extendida/n_original:.1%} del original)")
    print(f"Fase 3b_c4 - version ESTRICTA: {n_estricto}  ({n_estricto/n_original:.1%} del original)")

    print("\nDistribucion de 'anios_verificados_adelante' (0 = no se pudo verificar ningun anio nuevo hacia adelante):")
    print(incendios["anios_verificados_adelante"].value_counts().sort_index())
    print("\nPor anio de deteccion, cuantos tienen 0/1/2 anios verificados hacia adelante:")
    print(incendios.groupby(["anio_deteccion", "anios_verificados_adelante"]).size().unstack(fill_value=0).tail(6))


if __name__ == "__main__":
    main()
