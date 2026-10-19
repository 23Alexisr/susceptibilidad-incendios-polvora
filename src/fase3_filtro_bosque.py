"""
Fase 3 - Filtro de bosque en pie: clasifica cada deteccion de calor segun
si ocurrio en bosque que se mantuvo como bosque, en bosque que se convirtio
a otra cobertura (deforestacion), o en un area que ya no era bosque.

Fuente de cobertura de suelo: MapBiomas Peru, Coleccion 3 (verificada en
Earth Engine, ver Fase 3 mas abajo -> unica coleccion publicada como asset
de EE; se probaron rutas candidatas para una Coleccion 4 y no existen).
Asset: projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1
Cobertura temporal: 1985-2024 (banda classification_{anio} por cada anio).

Codigos de "Bosque" (Nivel 1 = "Formacion boscosa"), tomados de la leyenda
oficial de la Coleccion 3 (PDF "Descripcion de la leyenda - MapBiomas Peru
Coleccion 3", descargado y verificado):
    3 - Bosque (bosque humedo, por bioma: Amazonia/Andes/Bosque seco ecuatorial)
    4 - Bosque seco
    5 - Manglar
    6 - Bosque inundable
IMPORTANTE: el codigo 9 "Plantacion forestal" NO se incluye como bosque:
en la leyenda oficial esta clasificado dentro de "3. Area agropecuaria",
no dentro de "1. Formacion boscosa". Esto coincide con la definicion de
la tesis (un incendio en una plantacion forestal no es un incendio en
"bosque que sigue siendo bosque" en el sentido ecologico que interesa).

Criterio para el desajuste de resolucion (pixel FIRMS >> pixel MapBiomas):
Cada deteccion FIRMS es un punto (el centroide reportado del pixel del
sensor), no el pixel completo. Para aproximar el area real del pixel:
  - VIIRS S-NPP: resolucion nominal 375 m -> se construye un cuadrado de
    375 m de lado centrado en el punto.
  - MODIS: resolucion nominal ~1 km -> cuadrado de 1000 m de lado.
Dentro de ese cuadrado se extrae la CLASE MAYORITARIA (moda) de los
pixeles de 30 m de MapBiomas. Se usa moda (no solo el pixel central)
porque un incendio real puede detectarse con el centroide reportado
ligeramente desplazado del borde real del bosque, y la moda es mas
robusta a ese error de geolocalizacion que un solo pixel de 30 m.

Regla de clasificacion (t = anio de la deteccion, t-1 = anio anterior):
  (a) bosque en t-1 Y bosque en t         -> incendio_forestal
  (b) bosque en t-1 Y NO bosque en t      -> quema_deforestacion
  (c) NO bosque en t-1 (sea cual sea t)   -> quema_agricola_u_otra

Si el anio de la deteccion (t) o el anio anterior (t-1) caen fuera del
rango cubierto por MapBiomas (1985-2024) -- esto pasa con las detecciones
de 2025 y 2026 -- se usa el ultimo anio disponible (2024) y se marca la
fila con anio_ajustado=True, tal como pide la consigna.
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MAPBIOMAS_ID = "projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1"
MAPBIOMAS_ANIO_MIN = 1985
MAPBIOMAS_ANIO_MAX = 2024
BOSQUE_CODES = [3, 4, 5, 6]  # Nivel 1 "Formacion boscosa" (ver docstring)

RESOLUCION_SENSOR_M = {"modis": 1000, "viirs_snpp": 375}


def cargar_detecciones() -> gpd.GeoDataFrame:
    capas = []
    for sensor in ["modis", "viirs_snpp"]:
        gdf = gpd.read_file(DATA_PROCESSED_DIR / "firms_distrito.gpkg", layer=sensor)
        gdf["sensor"] = sensor
        capas.append(gdf[["sensor", "acq_date", "geometry"]])
    todas = pd.concat(capas, ignore_index=True)
    todas["acq_date"] = pd.to_datetime(todas["acq_date"])
    todas["anio_deteccion"] = todas["acq_date"].dt.year
    todas["id"] = todas.index.astype(str)

    todas["anio_ajustado"] = todas["anio_deteccion"] > MAPBIOMAS_ANIO_MAX
    todas["anio_t"] = todas["anio_deteccion"].clip(upper=MAPBIOMAS_ANIO_MAX)
    todas["anio_tm1"] = (todas["anio_deteccion"] - 1).clip(upper=MAPBIOMAS_ANIO_MAX)
    # No deberia ocurrir (MODIS empieza en 2001, muy por encima de 1985),
    # pero se deja el control por si acaso.
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
    """Para cada anio distinto en columna_anio, extrae con Earth Engine la
    clase mayoritaria de MapBiomas dentro del cuadrado-pixel de cada punto.
    Se agrupa por anio porque cada anio es una banda distinta de la imagen.
    """
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
            resultados.append(
                {
                    "id": feat["properties"]["id"],
                    nombre_resultado: feat["properties"].get("mode"),
                }
            )
        print(f"    anio {anio}: {len(subset)} puntos procesados")

    return pd.DataFrame(resultados)


def main() -> None:
    gdf = cargar_detecciones()
    print(f"Total de detecciones a clasificar: {len(gdf)}")
    print(f"Detecciones con anio fuera del rango de MapBiomas (usan 2024): {gdf['anio_ajustado'].sum()}")

    print("\nExtrayendo clase de MapBiomas en el anio t (anio de la deteccion)...")
    clase_t = extraer_clase_mayoritaria(gdf, "anio_t", "clase_t")

    print("\nExtrayendo clase de MapBiomas en el anio t-1 (anio anterior)...")
    clase_tm1 = extraer_clase_mayoritaria(gdf, "anio_tm1", "clase_tm1")

    gdf = gdf.merge(clase_t, on="id", how="left").merge(clase_tm1, on="id", how="left")

    gdf["bosque_t"] = gdf["clase_t"].isin(BOSQUE_CODES)
    gdf["bosque_tm1"] = gdf["clase_tm1"].isin(BOSQUE_CODES)

    def clasificar(fila):
        if not fila["bosque_tm1"]:
            return "quema_agricola_u_otra"
        return "incendio_forestal" if fila["bosque_t"] else "quema_deforestacion"

    gdf["categoria"] = gdf.apply(clasificar, axis=1)

    ruta_salida = DATA_PROCESSED_DIR / "firms_clasificado.gpkg"
    gdf.drop(columns=["acq_date"]).assign(acq_date=gdf["acq_date"].astype(str)).to_file(
        ruta_salida, layer="detecciones_clasificadas", driver="GPKG"
    )
    print(f"\nGuardado: {ruta_salida}")

    print("\n--- Conteo por categoria ---")
    print(gdf["categoria"].value_counts())

    print("\n--- Conteo por categoria y anio de deteccion ---")
    tabla = gdf.groupby(["anio_deteccion", "categoria"]).size().unstack(fill_value=0)
    print(tabla)
    tabla.to_csv(DATA_PROCESSED_DIR / "fase3_conteo_categoria_anio.csv")

    print("\n--- Conteo por categoria y sensor ---")
    print(gdf.groupby(["sensor", "categoria"]).size().unstack(fill_value=0))

    print(f"\nFilas con clasificacion nula (revisar): {gdf['categoria'].isna().sum()}")
    print(f"Filas sin clase_t o clase_tm1 (fuera de imagen?): {gdf['clase_t'].isna().sum()} / {gdf['clase_tm1'].isna().sum()}")


if __name__ == "__main__":
    main()
