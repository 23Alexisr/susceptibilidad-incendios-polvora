"""
Fase 3b (pasos 1-2) - Depuracion de la clasificacion de incendios forestales.

Paso 1 - Reclasificacion con ventana t, t+1, t+2:
La Fase 3 original solo miraba t-1 vs t. Aqui se revisa ademas t+1 y t+2:
si una deteccion clasificada como "incendio_forestal" (bosque en t-1 Y en t)
resulta que el pixel deja de ser bosque en t+1 O en t+2, se reclasifica como
"quema_deforestacion_extendida" -- porque lo mas probable es que el fuego
haya sido el primer paso de un desmonte que se completo 1-2 anios despues
(patron muy documentado en la Amazonia: se quema, se deja "secar" lo
chamuscado, se termina de limpiar al anio siguiente).

Limitacion: MapBiomas llega hasta 2024. Para detecciones de 2023 solo hay
1 anio de ventana hacia adelante disponible (t+1=2024, t+2 se recorta a
2024 = repite t+1); para 2024 no hay NINGUN anio nuevo hacia adelante
(t+1 y t+2 se recortan a 2024 = el mismo t). Se guarda cuantos anios de
ventana futura realmente se pudieron verificar por deteccion, para ser
transparentes sobre que tan confiable es "incendio_forestal" en los
anios mas recientes.

Paso 2 - Version estricta:
En vez de "clase mayoritaria" (moda), se calcula la FRACCION real de
pixeles de 30 m que son bosque dentro de la huella del pixel del sensor,
y se exige >= 80%. Ademas se exige que el punto este a mas de 500 m de
cualquier pixel de "area agropecuaria" (codigos 9,15,21,35,40) en el
anio t-1. Esto descarta casos "moda=bosque pero mezclado" y casos muy
cerca del borde con zonas agricolas (donde el humo/geolocalizacion puede
confundir el origen real del fuego).
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1"
MAPBIOMAS_ANIO_MAX = 2024
BOSQUE_CODES = [3, 4, 5, 6]
AGROPECUARIO_CODES = [9, 15, 21, 35, 40]
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
    """Usa anio_tm1 (ya en el gdf) para calcular, por deteccion:
    - pct_bosque_pixel: fraccion real de pixeles de 30m que son bosque en la huella del sensor
    - dist_agropecuario_m: distancia (m) al pixel agropecuario mas cercano, en el punto exacto
    """
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
            # OJO: reduceRegions() de una imagen de UNA sola banda nombra la
            # propiedad de salida segun el reducer ("first"), no segun el
            # nombre de la banda -- por eso se lee "first" aqui y no
            # "dist_agro_m" (ese nombre si aplica cuando se reducen varias
            # bandas juntas, como en fase5_anuales.py).
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
    detecciones = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado.gpkg", layer="detecciones_clasificadas")
    incendios = detecciones[detecciones["categoria"] == "incendio_forestal"].copy().reset_index(drop=True)
    incendios["id"] = incendios.index.astype(str)
    print(f"Detecciones 'incendio_forestal' a revisar: {len(incendios)}")

    incendios["anio_t1_real"] = incendios["anio_deteccion"] + 1
    incendios["anio_t2_real"] = incendios["anio_deteccion"] + 2
    incendios["anio_t1"] = incendios["anio_t1_real"].clip(upper=MAPBIOMAS_ANIO_MAX)
    incendios["anio_t2"] = incendios["anio_t2_real"].clip(upper=MAPBIOMAS_ANIO_MAX)
    # Cuantos anios de ventana futura son datos NUEVOS de verdad (no un
    # recorte que repite un anio ya usado).
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

    ruta = DATA_PROCESSED_DIR / "fase3b_incendios_depurados.csv"
    incendios.drop(columns="geometry").to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    # También como GeoPackage, para poder mapearlos después.
    incendios.to_file(DATA_PROCESSED_DIR / "fase3b_incendios_depurados.gpkg", layer="incendios", driver="GPKG")

    print("\n=== Comparacion de conteos ===")
    n_original = len(incendios)
    n_extendida = (incendios["categoria_extendida"] == "incendio_forestal").sum()
    n_estricto = incendios["es_estricto"].sum()
    print(f"Fase 3 original (incendio_forestal, t-1 vs t):        {n_original}")
    print(f"Reclasificados a quema_deforestacion_extendida:       {n_original - n_extendida}")
    print(f"Fase 3b - version extendida (t, t+1, t+2 todo bosque): {n_extendida}  ({n_extendida/n_original:.1%} del original)")
    print(f"Fase 3b - version ESTRICTA (extendida + >=80% bosque + >500m de agro): {n_estricto}  ({n_estricto/n_original:.1%} del original)")

    print("\nDistribucion de 'anios_verificados_adelante' (0 = no se pudo verificar futuro, dato reciente):")
    print(incendios["anios_verificados_adelante"].value_counts().sort_index())


if __name__ == "__main__":
    main()
