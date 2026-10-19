"""
Fase 5 (paso 2) - Predictores estaticos por celda: elevacion, pendiente y
orientacion, calculados una sola vez (no cambian mes a mes ni año a año
en la escala de tiempo de este estudio).

Fuente: JAXA/ALOS/AW3D30/V4_1 (verificado en la Fase 4). Pendiente y
orientacion se derivan con ee.Terrain sobre el mosaico de elevacion.
Se usa el promedio del DEM de 30 m dentro de cada celda de 1 km
(reduceRegions), no un solo pixel central, para que el valor represente
mejor el terreno de toda la celda.
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

TAMANO_LOTE = 800


def construir_fc(celdas_4326: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in celdas_4326.iterrows():
        coords = list(fila.geometry.exterior.coords)
        geom = ee.Geometry.Polygon([coords])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def main() -> None:
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    print(f"Celdas: {len(celdas)}")

    # ee.Terrain.slope()/aspect() calculan el gradiente segun la proyeccion
    # "por defecto" de la imagen. Un mosaic() de una ImageCollection no
    # conserva de forma fiable la proyeccion nativa de cada tile (30 m),
    # asi que sin fijarla explicitamente, Earth Engine calculaba la
    # pendiente sobre una grilla mucho mas gruesa (se notaba en valores
    # de pendiente/orientacion repetidos en muchas celdas). Se fija la
    # proyeccion nativa real (EPSG:4326, ~30 m) antes de derivar pendiente
    # y orientacion.
    proyeccion_nativa = (
        ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1").select("DSM").first().projection()
    )
    dem = (
        ee.ImageCollection("JAXA/ALOS/AW3D30/V4_1")
        .select("DSM")
        .mosaic()
        .setDefaultProjection(proyeccion_nativa)
        .rename("elevacion")
    )
    pendiente = ee.Terrain.slope(dem).rename("pendiente")
    orientacion = ee.Terrain.aspect(dem).rename("orientacion")
    imagen = dem.addBands(pendiente).addBands(orientacion)

    filas = []
    for inicio in range(0, len(celdas), TAMANO_LOTE):
        lote = celdas.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc(lote)
        reducido = imagen.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
        info = reducido.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            filas.append(
                {
                    "cell_id": p["cell_id"],
                    "elevacion": p.get("elevacion"),
                    "pendiente": p.get("pendiente"),
                    "orientacion": p.get("orientacion"),
                }
            )
        print(f"  lote {inicio}-{inicio+len(lote)} OK")

    df = pd.DataFrame(filas)
    print(df.describe())

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv"
    df.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
