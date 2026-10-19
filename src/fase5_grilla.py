"""
Fase 5 (paso 1) - Construye la grilla de celdas de 1 km x 1 km sobre el
distrito de estudio (Polvora).

Por que UTM 18S (EPSG:32718): para que las celdas sean realmente de 1 km x
1 km necesitamos trabajar en un sistema de coordenadas metrico (proyectado),
no en grados (EPSG:4326, donde 1 grado no equivale a una distancia fija).
Polvora esta centrado alrededor de 76.6°W, dentro de la zona UTM 18S
(72°W-78°W), la misma zona que ya usamos en la Fase 1 para calcular el area
del distrito.

Criterio de inclusion de celdas: se conserva una celda si su CENTROIDE cae
dentro del poligono del distrito. Es el criterio estandar para grillas de
analisis (evita celdas fantasma que solo tocan una esquina del distrito).
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from config import DATA_PROCESSED_DIR

CRS_METRICO = "EPSG:32718"
TAMANO_CELDA_M = 1000


def main() -> None:
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")
    distrito_utm = distrito.to_crs(CRS_METRICO)
    geom_distrito = distrito_utm.union_all()

    minx, miny, maxx, maxy = geom_distrito.bounds
    # Alineamos el origen de la grilla a multiplos de 1000 m para que sea
    # reproducible (no depende del bounding box exacto del distrito).
    minx = np.floor(minx / TAMANO_CELDA_M) * TAMANO_CELDA_M
    miny = np.floor(miny / TAMANO_CELDA_M) * TAMANO_CELDA_M

    xs = np.arange(minx, maxx, TAMANO_CELDA_M)
    ys = np.arange(miny, maxy, TAMANO_CELDA_M)
    print(f"Bounding box UTM: x=[{minx},{maxx}], y=[{miny},{maxy}]")
    print(f"Candidatas antes de filtrar: {len(xs) * len(ys)}")

    celdas = []
    for x in xs:
        for y in ys:
            celdas.append(box(x, y, x + TAMANO_CELDA_M, y + TAMANO_CELDA_M))

    grilla = gpd.GeoDataFrame({"geometry": celdas}, crs=CRS_METRICO)
    grilla["centroide"] = grilla.geometry.centroid
    dentro = grilla["centroide"].within(geom_distrito)
    grilla = grilla[dentro].reset_index(drop=True)
    grilla["cell_id"] = [f"c{idx:05d}" for idx in range(len(grilla))]

    print(f"Celdas cuyo centroide cae dentro del distrito: {len(grilla)}")
    print(f"Area cubierta por la grilla: {len(grilla) * 1.0:.1f} km2 (area real del distrito: {distrito_utm.geometry.area.sum()/1e6:.1f} km2)")

    # Guardamos las celdas (poligonos, para calcular % de bosque, etc.) y
    # los centroides por separado en EPSG:4326 (lat/lon), que es lo que
    # exigen la mayoria de las consultas a Earth Engine.
    celdas_gdf = grilla[["cell_id", "geometry"]].copy()
    centroides_gdf = grilla[["cell_id", "centroide"]].set_geometry("centroide").rename_geometry("geometry")
    centroides_gdf = centroides_gdf.set_crs(CRS_METRICO)

    ruta = DATA_PROCESSED_DIR / "grilla_1km.gpkg"
    celdas_gdf.to_file(ruta, layer="celdas_utm", driver="GPKG")
    centroides_gdf.to_crs("EPSG:4326").to_file(ruta, layer="centroides_4326", driver="GPKG")
    celdas_gdf.to_crs("EPSG:4326").to_file(ruta, layer="celdas_4326", driver="GPKG")
    print(f"Guardado: {ruta} (capas: celdas_utm, celdas_4326, centroides_4326)")


if __name__ == "__main__":
    main()
