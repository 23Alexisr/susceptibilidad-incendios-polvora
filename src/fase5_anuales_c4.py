"""
Fase 5_c4 (paso 4, version reducida) - Predictores anuales de uso de suelo
por celda usando MapBiomas Coleccion 4, SOLO las 2 variables que requiere
el modelo de 8 indicadores de la Fase 7 (clase_mapbiomas y
pct_bosque_1km) -- a diferencia de fase5_anuales.py (C3), que ademas
calculaba perdida de bosque (Hansen) y 2 distancias, no necesarias aqui.

Anios calculados: 2011-2024 (14 anios). Son los que se usan como "anio
t-1" (rezagado) de los anios de modelado 2012-2025 de la Fase 7_c4, para
evitar fuga de datos.

Bosque = codigos {3,4,5,6} (identico a la C3, ver fase3_filtro_bosque_c4.py).
"""
import sys

import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1"
BOSQUE_CODES = [3, 4, 5, 6]
TAMANO_LOTE = 800


def construir_fc_poligonos(celdas: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in celdas.iterrows():
        coords = list(fila.geometry.exterior.coords)
        geom = ee.Geometry.Polygon([coords])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def procesar_anio(anio: int, celdas: gpd.GeoDataFrame) -> pd.DataFrame:
    clasificacion = ee.Image(MAPBIOMAS_ID).select(f"classification_{anio}")
    bosque_mask = clasificacion.remap(BOSQUE_CODES, [1] * len(BOSQUE_CODES), 0).rename("es_bosque")

    imagen_clase = clasificacion.rename("clase_mapbiomas")

    filas = []
    for inicio in range(0, len(celdas), TAMANO_LOTE):
        lote = celdas.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc_poligonos(lote)

        red_clase = imagen_clase.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
        red_bosque = bosque_mask.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

        info_clase = {f["properties"]["cell_id"]: f["properties"].get("mode") for f in red_clase.getInfo()["features"]}
        info_bosque = {f["properties"]["cell_id"]: f["properties"].get("mean") for f in red_bosque.getInfo()["features"]}

        for cell_id in info_clase:
            filas.append(
                {
                    "cell_id": cell_id,
                    "clase_mapbiomas": info_clase.get(cell_id),
                    "pct_bosque_1km": info_bosque.get(cell_id),
                }
            )

    df = pd.DataFrame(filas)
    df["anio"] = anio
    return df


def main(anios) -> None:
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    print(f"Celdas: {len(celdas)}, anios a procesar (C4): {anios}")

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_anuales_c4.csv"
    primero = not ruta.exists()

    for anio in anios:
        df = procesar_anio(anio, celdas)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        print(f"anio {anio}: {len(df)} filas. pct_bosque medio={df['pct_bosque_1km'].mean():.3f}, nulos={df['pct_bosque_1km'].isna().sum()}")


if __name__ == "__main__":
    anios = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(2011, 2025))
    main(anios)
