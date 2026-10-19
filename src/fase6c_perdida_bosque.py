"""
Fase 6c (insumo 1) - Perdida anual de bosque en el distrito completo
(no promediada desde la grilla de 1 km como en la Fase 5/7), en hectareas,
2001-2024, usando Hansen Global Forest Change v1.13 (mismo asset ya
verificado en la Fase 3).

Se calcula directo sobre el poligono del distrito para tener el numero
mas preciso posible (suma real de pixeles de 30 m dentro del limite
oficial), en vez de reusar el promedio por celda de la Fase 5 (que fue
pensado como predictor rezagado, no como serie de tendencia).
"""
import ee
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

HANSEN_ID = "UMD/hansen/global_forest_change_2025_v1_13"
ANIO_INICIO = 2001
ANIO_FIN = 2024


def main() -> None:
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")
    geom = distrito.geometry.iloc[0]
    coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else list(geom.geoms[0].exterior.coords)
    ee_geom = ee.Geometry.Polygon([coords])

    hansen = ee.Image(HANSEN_ID)
    # Igual que en la Fase 5: lossyear viene enmascarado (no en 0) donde
    # nunca hubo perdida, hay que des-enmascarar antes de comparar.
    lossyear = hansen.select("lossyear").unmask(0)
    area_pixel_ha = ee.Image.pixelArea().divide(10000)

    filas = []
    for anio in range(ANIO_INICIO, ANIO_FIN + 1):
        mascara_anio = lossyear.eq(anio - 2000)
        area_ha = (
            mascara_anio.multiply(area_pixel_ha)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom, scale=30, maxPixels=1e9)
            .getInfo()
        )
        valor = area_ha.get("lossyear", 0)
        filas.append({"anio": anio, "perdida_bosque_ha": valor})
        print(f"  {anio}: {valor:.1f} ha")

    df = pd.DataFrame(filas)
    ruta = DATA_PROCESSED_DIR / "fase6c_perdida_bosque_ha.csv"
    df.to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")
    print(f"Total 2001-2024: {df['perdida_bosque_ha'].sum():.0f} ha")


if __name__ == "__main__":
    main()
