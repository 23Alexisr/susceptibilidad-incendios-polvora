"""
Fase 5 (paso 4) - Predictores anuales de uso de suelo por celda: clase
MapBiomas mayoritaria, % de bosque en la celda (1 km2), distancia al borde
del bosque, distancia a area agropecuaria, y perdida de bosque detectada
ese mismo anio (Hansen).

Se calcula para los anios 2000-2023: estos son los anios que se van a usar
como "anio t-1" (rezagado) de los anios de estudio 2001-2024, para evitar
fuga de datos (regla del proyecto: los predictores de uso de suelo deben
ser del anio ANTERIOR al que se predice).

Bosque = codigos {3,4,5,6} (Nivel 1 "Formacion boscosa", ver Fase 3).
Area agropecuaria = codigos {9,15,21,35,40} (Nivel 1 "3. Area agropecuaria"
de la leyenda oficial de MapBiomas Peru Coleccion 3: 9-Plantacion forestal,
15-Pasto, 21-Mosaico agropecuario, 35-Palma aceitera, 40-Arroz).

Distancia al borde de bosque / a lo agropecuario: se calcula con
ee.Image.fastDistanceTransform() sobre la mascara binaria correspondiente,
muestreada en el CENTROIDE de cada celda (la distancia es, por definicion,
una medida puntual). Nota tecnica: se probo primero con ee.Image.distance()
+ ee.Kernel.euclidean(), pero esa combinacion dio resultados incorrectos
(casi todo nulo o 0) por un problema de escala/proyeccion; fastDistanceTransform
se probo contra una muestra de 300 celdas (0 nulos, distancias maximas
~2 km, muy por debajo del radio de busqueda de ~7.7 km) y da resultados
consistentes.
"""
import sys

import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1"
HANSEN_ID = "UMD/hansen/global_forest_change_2025_v1_13"
BOSQUE_CODES = [3, 4, 5, 6]
AGROPECUARIO_CODES = [9, 15, 21, 35, 40]
VECINDAD_PIXELES = 256  # radio de busqueda para fastDistanceTransform, en pixeles de 30 m (~7.7 km).
# Probado contra una muestra de 300 celdas: 0 nulos, distancia maxima encontrada ~2 km,
# muy por debajo del radio de busqueda, asi que 256 pixeles alcanza sin problema.
TAMANO_LOTE = 800


def construir_fc_poligonos(celdas: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in celdas.iterrows():
        coords = list(fila.geometry.exterior.coords)
        geom = ee.Geometry.Polygon([coords])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def construir_fc_puntos(centroides: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in centroides.iterrows():
        geom = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def procesar_anio(anio: int, celdas: gpd.GeoDataFrame, centroides: gpd.GeoDataFrame) -> pd.DataFrame:
    clasificacion = ee.Image(MAPBIOMAS_ID).select(f"classification_{anio}")
    bosque_mask = clasificacion.remap(BOSQUE_CODES, [1] * len(BOSQUE_CODES), 0).rename("es_bosque")
    agro_mask = clasificacion.remap(AGROPECUARIO_CODES, [1] * len(AGROPECUARIO_CODES), 0).rename("es_agro")

    hansen = ee.Image(HANSEN_ID)
    # OJO: la banda "lossyear" de Hansen viene ENMASCARADA (sin dato, no
    # en 0) en los pixeles que nunca perdieron bosque. Si se compara
    # directo con .eq(), la mascara se propaga y celdas de bosque 100%
    # intacto (sin ninguna perdida jamas) terminan con TODA la banda
    # enmascarada -> reduceRegions devuelve nulo en vez de 0. Se corrige
    # con unmask(0) antes de comparar: un pixel que nunca perdio bosque
    # pasa a valer 0, que nunca sera igual a (anio-2000) para anio>2000.
    perdida_ese_anio = hansen.select("lossyear").unmask(0).eq(anio - 2000).rename("perdida_bosque")

    # --- Parte 1: estadisticas de area (clase mayoritaria, % bosque, % perdida) ---
    # Dos pasadas porque Reducer.mode() (categorica, 1 banda) y Reducer.mean()
    # (las mascaras binarias, 2 bandas) no se pueden combinar directo con
    # Reducer.combine cuando el numero de bandas no calza con cada reducer.
    imagen_clase = clasificacion.rename("clase_mapbiomas")
    imagen_masks = bosque_mask.addBands(perdida_ese_anio)

    filas_area = []
    for inicio in range(0, len(celdas), TAMANO_LOTE):
        lote = celdas.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc_poligonos(lote)

        red_clase = imagen_clase.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
        red_masks = imagen_masks.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

        info_clase = {f["properties"]["cell_id"]: f["properties"].get("mode") for f in red_clase.getInfo()["features"]}
        info_masks = {f["properties"]["cell_id"]: f["properties"] for f in red_masks.getInfo()["features"]}

        for cell_id, props in info_masks.items():
            filas_area.append(
                {
                    "cell_id": cell_id,
                    "clase_mapbiomas": info_clase.get(cell_id),
                    "pct_bosque_1km": props.get("es_bosque"),
                    "pct_perdida_bosque_ese_anio": props.get("perdida_bosque"),
                }
            )

    # --- Parte 2: distancias (puntuales, en el centroide) ---
    # fastDistanceTransform devuelve, para cada pixel, la distancia en PIXELES
    # (al cuadrado, de ahi el sqrt) al pixel valido (=1) mas cercano en la
    # mascara de entrada. Se multiplica por 30 (resolucion nativa de
    # MapBiomas en metros) para convertir a metros.
    dist_bosque = (
        bosque_mask.Not().fastDistanceTransform(VECINDAD_PIXELES).sqrt().multiply(30).rename("dist_borde_bosque_m")
    )
    dist_agro = agro_mask.fastDistanceTransform(VECINDAD_PIXELES).sqrt().multiply(30).rename("dist_agropecuario_m")
    imagen_dist = dist_bosque.addBands(dist_agro)

    filas_dist = []
    for inicio in range(0, len(centroides), TAMANO_LOTE):
        lote = centroides.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc_puntos(lote)
        reducido = imagen_dist.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=30)
        info = reducido.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            filas_dist.append(
                {
                    "cell_id": p["cell_id"],
                    "dist_borde_bosque_m": p.get("dist_borde_bosque_m"),
                    "dist_agropecuario_m": p.get("dist_agropecuario_m"),
                }
            )

    df_area = pd.DataFrame(filas_area)
    df_dist = pd.DataFrame(filas_dist)
    df = df_area.merge(df_dist, on="cell_id", how="left")
    df["anio"] = anio
    return df


def main(anios) -> None:
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    print(f"Celdas: {len(celdas)}, anios a procesar: {anios}")

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv"
    primero = not ruta.exists()

    for anio in anios:
        df = procesar_anio(anio, celdas, centroides)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        print(f"anio {anio}: {len(df)} filas. pct_bosque medio={df['pct_bosque_1km'].mean():.3f}, "
              f"dist_borde_bosque medio={df['dist_borde_bosque_m'].mean():.0f}m, "
              f"nulos dist_bosque={df['dist_borde_bosque_m'].isna().sum()}")


if __name__ == "__main__":
    anios = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(2000, 2024))
    main(anios)
