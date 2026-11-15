"""
Fase 3 - Filtro de bosque con MapBiomas Peru (Coleccion 4).

Clasifica cada foco de calor de FIRMS que cae en Polvora segun lo que habia
en el terreno el anio anterior y el mismo anio:
  - incendio forestal: habia bosque antes y sigue habiendo bosque
  - quema por deforestacion: habia bosque antes y ya no
  - quema agricola u otra: antes no habia bosque

Lee:    data/processed/firms_distrito_c4.gpkg (focos ya recortados a Polvora)
Genera: data/processed/firms_clasificado_c4.gpkg
        data/processed/fase3_conteo_categoria_anio_c4.csv
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

# Conexion a Google Earth Engine con mi proyecto
ee.Initialize(project=EE_PROJECT_ID)

# Mapa de cobertura de MapBiomas Peru, coleccion 4 (tiene un mapa por anio)
MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1"
MAPBIOMAS_ANIO_MIN = 1985
MAPBIOMAS_ANIO_MAX = 2025  # C4: un anio mas que la C3 (2024)
BOSQUE_CODES = [3, 4, 5, 6]  # codigos de bosque en la leyenda de MapBiomas

# Tamanio del pixel que ve cada satelite (en metros)
RESOLUCION_SENSOR_M = {"modis": 1000, "viirs_snpp": 375}


# 1) Cargo los focos de calor de los dos sensores y les pongo el anio
def cargar_detecciones() -> gpd.GeoDataFrame:
    capas = []
    for sensor in ["modis", "viirs_snpp"]:
        gdf = gpd.read_file(DATA_PROCESSED_DIR / "firms_distrito_c4.gpkg", layer=sensor)
        gdf["sensor"] = sensor
        capas.append(gdf[["sensor", "acq_date", "geometry"]])
    # junto MODIS y VIIRS en una sola tabla
    todas = pd.concat(capas, ignore_index=True)
    todas["acq_date"] = pd.to_datetime(todas["acq_date"])
    todas["anio_deteccion"] = todas["acq_date"].dt.year
    todas["id"] = todas.index.astype(str)  # id para unir los resultados despues

    # anio_t = anio del foco, anio_tm1 = anio anterior (t-1)
    # si el foco es de un anio que MapBiomas todavia no tiene, uso el ultimo (2025)
    todas["anio_ajustado"] = todas["anio_deteccion"] > MAPBIOMAS_ANIO_MAX
    todas["anio_t"] = todas["anio_deteccion"].clip(upper=MAPBIOMAS_ANIO_MAX)
    todas["anio_tm1"] = (todas["anio_deteccion"] - 1).clip(upper=MAPBIOMAS_ANIO_MAX)
    assert (todas["anio_tm1"] >= MAPBIOMAS_ANIO_MIN).all()

    return todas


# 2) Cada foco es un punto, pero el satelite ve un cuadrado (375 m VIIRS, 1 km MODIS).
#    Convierto cada punto en ese cuadrado para mirar todo lo que habia alrededor.
def construir_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in gdf.iterrows():
        radio_lado = RESOLUCION_SENSOR_M[fila["sensor"]]
        punto = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        cuadrado = punto.buffer(radio_lado / 2, 1).bounds()  # buffer = circulo, bounds = el cuadrado que lo encierra
        features.append(ee.Feature(cuadrado, {"id": fila["id"]}))
    return ee.FeatureCollection(features)


# 3) Para cada cuadrado le pregunto a MapBiomas que cobertura habia ese anio.
#    Uso la moda: la clase que mas se repite entre los pixeles de 30 m del cuadrado.
def extraer_clase_mayoritaria(gdf: gpd.GeoDataFrame, columna_anio: str, nombre_resultado: str) -> pd.DataFrame:
    imagen = ee.Image(MAPBIOMAS_ID)
    resultados = []

    anios = sorted(gdf[columna_anio].unique())
    print(f"  Extrayendo '{nombre_resultado}' para {len(anios)} anios distintos: {anios}")

    # voy anio por anio porque cada anio tiene su propio mapa (banda)
    for anio in anios:
        subset = gdf[gdf[columna_anio] == anio]
        fc = construir_fc(subset)
        banda = imagen.select(f"classification_{anio}")
        reducido = banda.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=30)
        info = reducido.getInfo()  # trae el resultado de Earth Engine a mi PC
        for feat in info["features"]:
            resultados.append({"id": feat["properties"]["id"], nombre_resultado: feat["properties"].get("mode")})
        print(f"    anio {anio}: {len(subset)} puntos procesados")

    return pd.DataFrame(resultados)


def main() -> None:
    gdf = cargar_detecciones()
    print(f"Total de detecciones a clasificar: {len(gdf)}")
    print(f"Detecciones con anio fuera del rango de MapBiomas C4 (usan {MAPBIOMAS_ANIO_MAX}): {gdf['anio_ajustado'].sum()}")

    # cobertura en el anio del foco
    print("\nExtrayendo clase de MapBiomas C4 en el anio t (anio de la deteccion)...")
    clase_t = extraer_clase_mayoritaria(gdf, "anio_t", "clase_t")

    # cobertura en el anio anterior (lo que habia ANTES del fuego)
    print("\nExtrayendo clase de MapBiomas C4 en el anio t-1 (anio anterior)...")
    clase_tm1 = extraer_clase_mayoritaria(gdf, "anio_tm1", "clase_tm1")

    # pego las dos clases a cada foco usando el id
    gdf = gdf.merge(clase_t, on="id", how="left").merge(clase_tm1, on="id", how="left")

    # True si la clase es de bosque
    gdf["bosque_t"] = gdf["clase_t"].isin(BOSQUE_CODES)
    gdf["bosque_tm1"] = gdf["clase_tm1"].isin(BOSQUE_CODES)

    # 4) Regla para clasificar cada foco
    #    ej. foco en 2020: miro 2019 (antes) y 2020 (despues)
    def clasificar(fila):
        if not fila["bosque_tm1"]:
            return "quema_agricola_u_otra"   # antes no era bosque (chacra, pasto, etc.)
        return "incendio_forestal" if fila["bosque_t"] else "quema_deforestacion"
        # era bosque y sigue siendolo -> incendio forestal
        # era bosque y ya no -> quemaron para deforestar

    gdf["categoria"] = gdf.apply(clasificar, axis=1)

    # guardo los focos ya clasificados (la fecha como texto para que el gpkg no de problemas)
    ruta_salida = DATA_PROCESSED_DIR / "firms_clasificado_c4.gpkg"
    gdf.drop(columns=["acq_date"]).assign(acq_date=gdf["acq_date"].astype(str)).to_file(
        ruta_salida, layer="detecciones_clasificadas", driver="GPKG"
    )
    print(f"\nGuardado: {ruta_salida}")

    # resumen: cuantos focos hay de cada tipo, en total y por anio
    print("\n--- Conteo por categoria ---")
    print(gdf["categoria"].value_counts())
    print("\n--- Conteo por categoria y anio de deteccion (ultimos anios) ---")
    tabla = gdf.groupby(["anio_deteccion", "categoria"]).size().unstack(fill_value=0)
    print(tabla.tail(6))
    tabla.to_csv(DATA_PROCESSED_DIR / "fase3_conteo_categoria_anio_c4.csv")

    # control: no deberia haber focos sin clase
    print(f"\nFilas con clasificacion nula (revisar): {gdf['categoria'].isna().sum()}")
    print(f"Filas sin clase_t o clase_tm1 (fuera de imagen?): {gdf['clase_t'].isna().sum()} / {gdf['clase_tm1'].isna().sum()}")


if __name__ == "__main__":
    main()
