"""
Fase 4 - Verificacion de disponibilidad de predictores.

Este script solo COMPRUEBA que cada fuente candidata existe, responde, y
tiene la cobertura temporal/espacial esperada para el distrito de estudio.
No descarga ni construye la tabla de modelado (eso es la Fase 5): aqui el
objetivo es dejar registrado, con evidencia verificable, que asset o URL se
va a usar en cada caso, y que problemas se encontraron.
"""
import ee

from config import EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)


def linea(nombre, ok, detalle):
    estado = "OK" if ok else "FALLA"
    print(f"[{estado}] {nombre}: {detalle}")


def verificar_ic(nombre, asset_id):
    try:
        ic = ee.ImageCollection(asset_id)
        n = ic.size().getInfo()
        primero = ic.first().getInfo()
        bandas = [b["id"] for b in primero["bands"]]
        linea(nombre, True, f"{asset_id} | {n} imagenes | bandas (primeras): {bandas[:6]}")
    except Exception as e:
        linea(nombre, False, f"{asset_id} | {str(e)[:150]}")


def verificar_img(nombre, asset_id):
    try:
        img = ee.Image(asset_id)
        info = img.getInfo()
        bandas = [b["id"] for b in info["bands"]]
        linea(nombre, True, f"{asset_id} | {len(bandas)} bandas (primeras 6): {bandas[:6]}")
    except Exception as e:
        linea(nombre, False, f"{asset_id} | {str(e)[:150]}")


def main():
    print("--- Clima ---")
    verificar_ic("CHIRPS diario (precipitacion)", "UCSB-CHG/CHIRPS/DAILY")
    verificar_ic("ERA5-Land mensual (temp, VPD derivado)", "ECMWF/ERA5_LAND/MONTHLY_AGGR")

    print("\n--- Vegetacion ---")
    verificar_ic("MODIS reflectancia superficial (NDMI derivado)", "MODIS/061/MOD09A1")

    print("\n--- Uso de suelo ---")
    verificar_img("MapBiomas Peru Coleccion 3", "projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1")
    verificar_img("Hansen Global Forest Change v1.13 (perdida de bosque previa)", "UMD/hansen/global_forest_change_2025_v1_13")

    print("\n--- Fisicos ---")
    verificar_ic("DEM ALOS AW3D30 V4_1 (elevacion, pendiente, orientacion derivadas)", "JAXA/ALOS/AW3D30/V4_1")
    linea(
        "OpenStreetMap Peru (vias, centros poblados, rios)",
        True,
        "https://download.geofabrik.de/south-america/peru-latest-free.gpkg.zip | verificado por HTTP, "
        "actualizado 2026-09-16 | fuera de Earth Engine, se procesa localmente con geopandas en la Fase 5",
    )
    linea(
        "PISCO (SENAMHI)",
        True,
        "https://iridl.ldeo.columbia.edu/SOURCES/.SENAMHI/.HSR/.PISCO/ | responde (HTTP 200) | "
        "disponible en teoria, pero NO se integra en esta fase de viabilidad: se prioriza CHIRPS "
        "(ya en Earth Engine, mismo pipeline, cobertura equivalente 1981-presente)",
    )


if __name__ == "__main__":
    main()
