"""
Fase 7 (paso 1) - NDVI mensual por celda: predictor NUEVO de la dimension
"Ambiental" de la matriz de operacionalizacion (variable 6 de 8), en
reemplazo de NDMI (que SI formaba parte del modelo de 16 variables pero
no esta en la lista de 8 indicadores finales).

Fuente: MOD13A3.061 (Terra Vegetation Indices Monthly Global 1km),
banda NDVI, ya agregada a compuesto MENSUAL por el proveedor (no hay que
construir el compuesto como en fase5_mensuales.py con CHIRPS/ERA5, que son
diarios/sub-mensuales).

Factor de escala: la banda NDVI de MOD13A3 viene como entero escalado
(rango tipico -2000 a 10000); el valor real es ese entero * 0.0001
(documentado en la pagina del producto en GEE Data Catalog). Se aplica
ese factor antes de agregar.

Mascara de calidad: se usa la banda SummaryQA (clasificacion estandar de
los productos MOD13 de NASA): 0 = buena, 1 = marginal (usable), 2 =
nieve/hielo, 3 = nuboso. Se enmascaran los pixeles con SummaryQA > 1
(nieve/hielo y nuboso) antes de promediar -- en el distrito de Polvora
(selva baja, sin nieve) esto practicamente solo filtra pixeles nublados.

Agregacion espacial: promedio del NDVI (ya enmascarado) dentro del
POLIGONO de cada celda de 1 km2 (reduceRegions sobre celdas_4326, reducer
mean), no un muestreo puntual en el centroide -- a diferencia de
precip_mm/temp_max_c (CHIRPS ~5.5km y ERA5-Land ~11km, mucho mas gruesos
que la celda, donde promediar o muestrear el centroide da lo mismo), la
resolucion nativa de MOD13A3 (~926 m) es comparable al tamano de la celda
(1000 m), asi que promediar el area captura mejor variacion intra-celda.

Periodo calculado: 2011-01 a 2024-12 (2011 incluido porque el predictor de
NDVI de enero-2012, primer mes de la tabla de modelado de la Fase 7, usa
el mes CALENDARIO ANTERIOR = diciembre-2011; ver regla de rezago en
fase7_construir_tabla.py).
"""
import datetime as dt

import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MOD13A3_ID = "MODIS/061/MOD13A3"
FACTOR_ESCALA_NDVI = 0.0001
QA_MAXIMO_ACEPTABLE = 1  # SummaryQA: 0=buena, 1=marginal (se aceptan); 2=nieve/hielo, 3=nuboso (se descartan)
TAMANO_LOTE = 800


def construir_fc_poligonos(celdas_4326: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in celdas_4326.iterrows():
        coords = list(fila.geometry.exterior.coords)
        geom = ee.Geometry.Polygon([coords])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def imagen_ndvi_del_mes(anio: int, mes: int) -> ee.Image:
    inicio = ee.Date.fromYMD(anio, mes, 1)
    fin = inicio.advance(1, "month")
    img = ee.ImageCollection(MOD13A3_ID).filterDate(inicio, fin).first()

    ndvi_crudo = img.select("NDVI")
    qa = img.select("SummaryQA")
    mascara_buena = qa.lte(QA_MAXIMO_ACEPTABLE)

    ndvi = ndvi_crudo.multiply(FACTOR_ESCALA_NDVI).updateMask(mascara_buena).rename("ndvi")
    return ndvi


def procesar_mes(anio: int, mes: int, celdas_4326: gpd.GeoDataFrame) -> pd.DataFrame:
    imagen = imagen_ndvi_del_mes(anio, mes)
    filas = []
    for inicio in range(0, len(celdas_4326), TAMANO_LOTE):
        lote = celdas_4326.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc_poligonos(lote)
        # scale=250: igual criterio que fase5_mensuales.py (nivel de piramide
        # de lectura); la resolucion real del pixel sigue siendo ~926 m.
        reducido = imagen.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=250)
        info = reducido.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            filas.append({"cell_id": p["cell_id"], "ndvi": p.get("mean")})
    df = pd.DataFrame(filas)
    df["anio"] = anio
    df["mes"] = mes
    return df


def generar_meses(inicio: tuple, fin: tuple):
    actual = dt.date(inicio[0], inicio[1], 1)
    ultimo = dt.date(fin[0], fin[1], 1)
    while actual <= ultimo:
        yield actual.year, actual.month
        if actual.month == 12:
            actual = dt.date(actual.year + 1, 1, 1)
        else:
            actual = dt.date(actual.year, actual.month + 1, 1)


def main(meses) -> None:
    celdas_4326 = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    print(f"Celdas: {len(celdas_4326)}, meses a procesar: {len(meses)} ({meses[0]} a {meses[-1]})")

    ruta = DATA_PROCESSED_DIR / "fase7_ndvi_mensual.csv"
    primero = not ruta.exists()

    for i, (anio, mes) in enumerate(meses, start=1):
        df = procesar_mes(anio, mes, celdas_4326)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        n_nulos = df["ndvi"].isna().sum()
        print(
            f"[{i}/{len(meses)}] {anio}-{mes:02d}: {len(df)} filas, "
            f"ndvi_medio={df['ndvi'].mean():.4f}, nulos={n_nulos}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        meses = [(2020, 8)]
    else:
        meses = list(generar_meses((2011, 1), (2024, 12)))
    main(meses)
