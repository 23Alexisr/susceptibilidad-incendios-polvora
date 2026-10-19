"""
Fase 5 (paso 5) - Predictores mensuales de clima y vegetacion por celda:
precipitacion (CHIRPS), dias secos en el mes (CHIRPS), temperatura maxima
y deficit de presion de vapor -VPD- (ERA5-Land), y NDMI (MODIS).

Se calcula para TODOS los meses entre 2000-01 y 2024-12 (25 anios): el mes
extra 2000 es necesario porque, por la regla de "sin fuga de datos", el
predictor climatico del mes de enero 2001 usa el mes de DICIEMBRE 2000
(mes anterior), no enero 2001 mismo.

VPD (deficit de presion de vapor) se deriva con la formula de Tetens/Magnus
a partir de temperature_2m y dewpoint_temperature_2m (temperatura media y
punto de rocio mensual), ambas convertidas de Kelvin a Celsius.

"Dias secos en el mes" es una SIMPLIFICACION de "dias secos consecutivos":
se cuenta el numero de dias del mes con precipitacion < 1 mm, sin verificar
que sean consecutivos NI encadenar con el mes anterior. Calcular la racha
maxima real de dias secos consecutivos exigiria extraer la serie DIARIA
completa por celda (miles de consultas extra a Earth Engine), lo cual excede
el alcance de esta prueba de viabilidad. Queda anotado como limitacion.
"""
import datetime as dt

import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

TAMANO_LOTE = 800


def construir_fc_puntos(centroides: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in centroides.iterrows():
        geom = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        features.append(ee.Feature(geom, {"cell_id": fila["cell_id"]}))
    return ee.FeatureCollection(features)


def imagen_del_mes(anio: int, mes: int) -> ee.Image:
    inicio = ee.Date.fromYMD(anio, mes, 1)
    fin = inicio.advance(1, "month")

    chirps_mes = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(inicio, fin)
    precip_mm = chirps_mes.sum().rename("precip_mm")
    dias_secos = chirps_mes.map(lambda img: img.lt(1)).sum().rename("dias_secos_mes")

    era5_mes = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(inicio, fin).first()
    temp_max_c = era5_mes.select("temperature_2m_max").subtract(273.15).rename("temp_max_c")
    t_c = era5_mes.select("temperature_2m").subtract(273.15)
    td_c = era5_mes.select("dewpoint_temperature_2m").subtract(273.15)
    es = t_c.expression("0.6108 * exp(17.27*T/(T+237.3))", {"T": t_c}).rename("es_kpa")
    ea = td_c.expression("0.6108 * exp(17.27*Td/(Td+237.3))", {"Td": td_c}).rename("ea_kpa")
    vpd_kpa = es.subtract(ea).rename("vpd_kpa")

    mod_mes = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(inicio, fin)
    ndmi = mod_mes.map(
        lambda img: img.normalizedDifference(["sur_refl_b02", "sur_refl_b06"]).rename("ndmi")
    ).mean()

    return ee.Image.cat([precip_mm, dias_secos, temp_max_c, vpd_kpa, ndmi])


def procesar_mes(anio: int, mes: int, centroides: gpd.GeoDataFrame) -> pd.DataFrame:
    imagen = imagen_del_mes(anio, mes)
    filas = []
    for inicio in range(0, len(centroides), TAMANO_LOTE):
        lote = centroides.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc_puntos(lote)
        # CHIRPS ~5.5 km y ERA5 ~11 km son mas gruesos que la celda de 1 km:
        # el "scale" aqui solo fija el nivel de piramide de lectura, el
        # valor real sigue siendo el del pixel climatico que cae en el punto.
        # NDMI (500 m) se muestrea igual, en el centroide.
        reducido = imagen.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=250)
        info = reducido.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            filas.append(
                {
                    "cell_id": p["cell_id"],
                    "precip_mm": p.get("precip_mm"),
                    "dias_secos_mes": p.get("dias_secos_mes"),
                    "temp_max_c": p.get("temp_max_c"),
                    "vpd_kpa": p.get("vpd_kpa"),
                    "ndmi": p.get("ndmi"),
                }
            )
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
    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    print(f"Celdas: {len(centroides)}, meses a procesar: {len(meses)} ({meses[0]} a {meses[-1]})")

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv"
    primero = not ruta.exists()

    for i, (anio, mes) in enumerate(meses, start=1):
        df = procesar_mes(anio, mes, centroides)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        print(
            f"[{i}/{len(meses)}] {anio}-{mes:02d}: {len(df)} filas, "
            f"precip_media={df['precip_mm'].mean():.1f}mm, temp_max_media={df['temp_max_c'].mean():.1f}C, "
            f"nulos_ndmi={df['ndmi'].isna().sum()}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        meses = [(2020, 8)]
    else:
        meses = list(generar_meses((2000, 1), (2024, 12)))
    main(meses)
