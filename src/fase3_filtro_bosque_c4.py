"""
Fase 3_c4 - Version paralela de fase3_filtro_bosque.py usando MapBiomas
Peru Coleccion 4 (en vez de Coleccion 3) como fuente de uso de suelo, para
el experimento de migracion + extension del periodo a 2012-2025. No
modifica nada del pipeline original: produce firms_clasificado_c4.gpkg
aparte.

Fuente: MapBiomas Peru Coleccion 4, asset de cobertura/uso de suelo
  projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1
Verificado en Earth Engine (2026-10-18): 41 bandas, classification_1985 a
classification_2025 (un anio mas que la Coleccion 3, que llegaba a 2024).
Nota sobre el nombre del asset: a diferencia de la C3 (que usa el sufijo
"_integration_v1"), el asset publico de la C4 que SI es accesible usa el
sufijo "_coverage_v1"; se intento tambien listar la carpeta
"peru/collection4" para explorar que otros assets hay, pero esa operacion
de listado especifica esta bloqueada por permisos (a diferencia de C1/C2/C3,
cuyas carpetas si se pueden listar) -- el asset de cobertura en si SI es
accesible por ID directo, que es lo unico que se necesita aqui.

Leyenda verificada contra el archivo oficial MapBiomasPeru_legend.csv
(descargado de https://peru.mapbiomas.org/descargas/mapas-de-las-colecciones/,
2026-10-18, guardado en data/raw/mapbiomas_c4/): los codigos de "Formacion
boscosa" (3=Bosque, 4=Bosque seco, 5=Manglar, 6=Bosque inundable) y de
"Plantacion forestal" (9, agrupado bajo "Area agropecuaria", NO bosque)
son IDENTICOS a los de la Coleccion 3. No hubo que cambiar ningun codigo.

MAPBIOMAS_ANIO_MAX = 2025 (antes 2024): como la C4 ya cubre 2025, las
detecciones de 2025 (t=2025) ya NO necesitan el ajuste "se usa el ultimo
anio disponible" que si aplicaba en la Coleccion 3 -- esto reduce el
numero de detecciones con anio_ajustado=True respecto al pipeline C3.
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1"
MAPBIOMAS_ANIO_MIN = 1985
MAPBIOMAS_ANIO_MAX = 2025  # C4: un anio mas que la C3 (2024)
BOSQUE_CODES = [3, 4, 5, 6]  # identico a la C3, verificado contra la leyenda oficial de la C4

RESOLUCION_SENSOR_M = {"modis": 1000, "viirs_snpp": 375}


def cargar_detecciones() -> gpd.GeoDataFrame:
    capas = []
    for sensor in ["modis", "viirs_snpp"]:
        gdf = gpd.read_file(DATA_PROCESSED_DIR / "firms_distrito_c4.gpkg", layer=sensor)
        gdf["sensor"] = sensor
        capas.append(gdf[["sensor", "acq_date", "geometry"]])
    todas = pd.concat(capas, ignore_index=True)
    todas["acq_date"] = pd.to_datetime(todas["acq_date"])
    todas["anio_deteccion"] = todas["acq_date"].dt.year
    todas["id"] = todas.index.astype(str)

    todas["anio_ajustado"] = todas["anio_deteccion"] > MAPBIOMAS_ANIO_MAX
    todas["anio_t"] = todas["anio_deteccion"].clip(upper=MAPBIOMAS_ANIO_MAX)
    todas["anio_tm1"] = (todas["anio_deteccion"] - 1).clip(upper=MAPBIOMAS_ANIO_MAX)
    assert (todas["anio_tm1"] >= MAPBIOMAS_ANIO_MIN).all()

    return todas


def construir_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in gdf.iterrows():
        radio_lado = RESOLUCION_SENSOR_M[fila["sensor"]]
        punto = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        cuadrado = punto.buffer(radio_lado / 2, 1).bounds()
        features.append(ee.Feature(cuadrado, {"id": fila["id"]}))
    return ee.FeatureCollection(features)


def extraer_clase_mayoritaria(gdf: gpd.GeoDataFrame, columna_anio: str, nombre_resultado: str) -> pd.DataFrame:
    imagen = ee.Image(MAPBIOMAS_ID)
    resultados = []

    anios = sorted(gdf[columna_anio].unique())
    print(f"  Extrayendo '{nombre_resultado}' para {len(anios)} anios distintos: {anios}")

    for anio in anios:
        subset = gdf[gdf[columna_anio] == anio]
        fc = construir_fc(subset)
        banda = imagen.select(f"classification_{anio}")
        reducido = banda.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
        info = reducido.getInfo()
        for feat in info["features"]:
            resultados.append({"id": feat["properties"]["id"], nombre_resultado: feat["properties"].get("mode")})
        print(f"    anio {anio}: {len(subset)} puntos procesados")

    return pd.DataFrame(resultados)


def main() -> None:
    gdf = cargar_detecciones()
    print(f"Total de detecciones a clasificar: {len(gdf)}")
    print(f"Detecciones con anio fuera del rango de MapBiomas C4 (usan {MAPBIOMAS_ANIO_MAX}): {gdf['anio_ajustado'].sum()}")

    print("\nExtrayendo clase de MapBiomas C4 en el anio t (anio de la deteccion)...")
    clase_t = extraer_clase_mayoritaria(gdf, "anio_t", "clase_t")

    print("\nExtrayendo clase de MapBiomas C4 en el anio t-1 (anio anterior)...")
    clase_tm1 = extraer_clase_mayoritaria(gdf, "anio_tm1", "clase_tm1")

    gdf = gdf.merge(clase_t, on="id", how="left").merge(clase_tm1, on="id", how="left")

    gdf["bosque_t"] = gdf["clase_t"].isin(BOSQUE_CODES)
    gdf["bosque_tm1"] = gdf["clase_tm1"].isin(BOSQUE_CODES)

    def clasificar(fila):
        if not fila["bosque_tm1"]:
            return "quema_agricola_u_otra"
        return "incendio_forestal" if fila["bosque_t"] else "quema_deforestacion"

    gdf["categoria"] = gdf.apply(clasificar, axis=1)

    ruta_salida = DATA_PROCESSED_DIR / "firms_clasificado_c4.gpkg"
    gdf.drop(columns=["acq_date"]).assign(acq_date=gdf["acq_date"].astype(str)).to_file(
        ruta_salida, layer="detecciones_clasificadas", driver="GPKG"
    )
    print(f"\nGuardado: {ruta_salida}")

    print("\n--- Conteo por categoria ---")
    print(gdf["categoria"].value_counts())
    print("\n--- Conteo por categoria y anio de deteccion (ultimos anios) ---")
    tabla = gdf.groupby(["anio_deteccion", "categoria"]).size().unstack(fill_value=0)
    print(tabla.tail(6))
    tabla.to_csv(DATA_PROCESSED_DIR / "fase3_conteo_categoria_anio_c4.csv")

    print(f"\nFilas con clasificacion nula (revisar): {gdf['categoria'].isna().sum()}")
    print(f"Filas sin clase_t o clase_tm1 (fuera de imagen?): {gdf['clase_t'].isna().sum()} / {gdf['clase_tm1'].isna().sum()}")


if __name__ == "__main__":
    main()
