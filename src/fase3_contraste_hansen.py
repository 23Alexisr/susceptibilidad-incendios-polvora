"""
Fase 3 (contraste) - Compara la clasificacion basada en MapBiomas (Coleccion 3)
contra un indicador independiente de bosque/perdida de bosque construido con
Hansen Global Forest Change v1.13 (2000-2025).

Asset verificado en Earth Engine: UMD/hansen/global_forest_change_2025_v1_13
(bandas: treecover2000, loss, lossyear, gain, datamask).

Por que Hansen es solo un contraste y no la fuente principal: Hansen no
clasifica tipos de cobertura (no distingue bosque de "no bosque" en el
sentido de uso de suelo, solo mide porcentaje de dosel arboreo en el
anio 2000 y el anio en que se perdio ese dosel). No puede decir por si
solo si un area no boscosa es agricultura, pasto u otra cosa. Por eso
sirve para validar el eje "bosque -> no bosque" (deforestacion), pero no
reemplaza a MapBiomas para las 3 categorias completas.

Criterio de "bosque" con Hansen para un anio dado Y:
  treecover2000 >= 30%  Y  (loss == 0  O  lossyear > (Y - 2000))
  Es decir: tenia cobertura arborea en 2000 Y, si hubo perdida registrada,
  esa perdida ocurrio despues del anio Y (todavia no habia pasado).
  El umbral de 30% es el estandar mas usado en la literatura (p.ej. Global
  Forest Watch) para definir "bosque" a partir de este producto.
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

HANSEN_ID = "UMD/hansen/global_forest_change_2025_v1_13"
UMBRAL_COBERTURA = 30


def construir_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    features = []
    for _, fila in gdf.iterrows():
        radio_lado = 375 if fila["sensor"] == "viirs_snpp" else 1000
        punto = ee.Geometry.Point([fila.geometry.x, fila.geometry.y])
        cuadrado = punto.buffer(radio_lado / 2, 1).bounds()
        features.append(ee.Feature(cuadrado, {"id": fila["id"]}))
    return ee.FeatureCollection(features)


def main() -> None:
    gdf = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado.gpkg", layer="detecciones_clasificadas")
    print(f"Detecciones a contrastar: {len(gdf)}")

    hansen = ee.Image(HANSEN_ID)

    # Earth Engine corta las consultas de FeatureCollection en 5000
    # elementos: se procesa en lotes para quedar bien por debajo de ese
    # limite (ademas, lotes mas chicos son mas faciles de reintentar si
    # alguno falla por timeout de red).
    TAMANO_LOTE = 1000
    filas = []
    for inicio in range(0, len(gdf), TAMANO_LOTE):
        lote = gdf.iloc[inicio : inicio + TAMANO_LOTE]
        fc = construir_fc(lote)
        reducido = hansen.select(["treecover2000", "loss", "lossyear"]).reduceRegions(
            collection=fc, reducer=ee.Reducer.mode(), scale=30
        )
        info = reducido.getInfo()
        for feat in info["features"]:
            p = feat["properties"]
            filas.append(
                {
                    "id": p["id"],
                    "treecover2000": p.get("treecover2000"),
                    "loss": p.get("loss"),
                    "lossyear": p.get("lossyear"),
                }
            )
        print(f"  lote {inicio}-{inicio + len(lote)}: {len(lote)} puntos procesados")

    hansen_df = pd.DataFrame(filas)
    print(f"Puntos con datos de Hansen: {len(hansen_df)}")

    gdf = gdf.merge(hansen_df, on="id", how="left")

    def bosque_hansen(fila, anio_col):
        if pd.isna(fila["treecover2000"]):
            return None
        if fila["treecover2000"] < UMBRAL_COBERTURA:
            return False
        if fila["loss"] == 0:
            return True
        anio_objetivo = fila[anio_col] - 2000
        return fila["lossyear"] > anio_objetivo

    gdf["hansen_bosque_tm1"] = gdf.apply(lambda f: bosque_hansen(f, "anio_tm1"), axis=1)
    gdf["hansen_bosque_t"] = gdf.apply(lambda f: bosque_hansen(f, "anio_t"), axis=1)

    def clasificar_hansen(fila):
        if fila["hansen_bosque_tm1"] is None:
            return None
        if not fila["hansen_bosque_tm1"]:
            return "quema_agricola_u_otra"
        return "incendio_forestal" if fila["hansen_bosque_t"] else "quema_deforestacion"

    gdf["categoria_hansen"] = gdf.apply(clasificar_hansen, axis=1)

    print("\n--- Acuerdo general MapBiomas vs Hansen (bosque en t-1) ---")
    coincide_tm1 = (gdf["bosque_tm1"] == gdf["hansen_bosque_tm1"]).mean()
    print(f"Coinciden en si habia bosque en t-1: {coincide_tm1:.1%}")

    print("\n--- Acuerdo general MapBiomas vs Hansen (bosque en t) ---")
    coincide_t = (gdf["bosque_t"] == gdf["hansen_bosque_t"]).mean()
    print(f"Coinciden en si habia bosque en t: {coincide_t:.1%}")

    print("\n--- Matriz de confusion: categoria (MapBiomas) vs categoria_hansen ---")
    matriz = pd.crosstab(gdf["categoria"], gdf["categoria_hansen"])
    print(matriz)

    print("\n--- Foco en 'quema_deforestacion' (MapBiomas): que dice Hansen de esos mismos puntos ---")
    deforestacion_mb = gdf[gdf["categoria"] == "quema_deforestacion"]
    print(deforestacion_mb["categoria_hansen"].value_counts(dropna=False))

    ruta_salida = DATA_PROCESSED_DIR / "fase3_contraste_hansen.csv"
    gdf.drop(columns="geometry").to_csv(ruta_salida, index=False)
    print(f"\nGuardado: {ruta_salida}")


if __name__ == "__main__":
    main()
