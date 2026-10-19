"""
Fase 5 (paso 3) - Distancia de cada celda a la via, centro poblado y rio
mas cercano, usando el extracto de OpenStreetMap de Peru (Geofabrik,
verificado en la Fase 4).

Se filtra por bounding box al leer los shapefiles (con un margen de ~20 km
alrededor del distrito) para no cargar el Peru completo en memoria: estos
shapefiles nacionales son grandes (cientos de miles de features).

Limitacion ya documentada en la Fase 4: OSM es un snapshot del mapa ACTUAL
(2026-09-16), no un historico. Las distancias se tratan como constantes en
todo el periodo de estudio (2001-2024): es una simplificacion aceptable
para una prueba de viabilidad, no para un estudio que necesite la red vial
de un anio especifico del pasado.
"""
import geopandas as gpd

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

CRS_METRICO = "EPSG:32718"
CARPETA_OSM = DATA_RAW_DIR / "osm"


def main() -> None:
    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    minx, miny, maxx, maxy = centroides.total_bounds
    margen = 0.2  # grados (~20 km), margen para no perder la via/rio mas cercano si esta justo afuera del distrito
    bbox = (minx - margen, miny - margen, maxx + margen, maxy + margen)
    print(f"Bounding box de lectura (con margen): {bbox}")

    centroides_utm = centroides.to_crs(CRS_METRICO)

    print("\nLeyendo vias...")
    vias = gpd.read_file(CARPETA_OSM / "gis_osm_roads_free_1.shp", bbox=bbox)
    print(f"  {len(vias)} vias en el area de interes")
    vias_utm = vias.to_crs(CRS_METRICO)

    print("Leyendo centros poblados...")
    poblados = gpd.read_file(CARPETA_OSM / "gis_osm_places_free_1.shp", bbox=bbox)
    print(f"  {len(poblados)} centros poblados en el area de interes")
    poblados_utm = poblados.to_crs(CRS_METRICO)

    print("Leyendo rios...")
    rios = gpd.read_file(CARPETA_OSM / "gis_osm_waterways_free_1.shp", bbox=bbox)
    print(f"  {len(rios)} tramos de rio/quebrada en el area de interes")
    rios_utm = rios.to_crs(CRS_METRICO)

    print("\nCalculando distancias (sjoin_nearest)...")
    centroides_utm = centroides_utm.reset_index(drop=True)

    def distancia_a_capa(capa_utm, nombre_col):
        if len(capa_utm) == 0:
            print(f"  AVISO: capa vacia para '{nombre_col}', se deja como nulo.")
            centroides_utm[nombre_col] = None
            return
        cercano = gpd.sjoin_nearest(
            centroides_utm[["cell_id", "geometry"]], capa_utm[["geometry"]], distance_col=nombre_col
        )
        cercano = cercano.drop_duplicates(subset="cell_id")
        centroides_utm[nombre_col] = centroides_utm["cell_id"].map(
            cercano.set_index("cell_id")[nombre_col]
        )

    distancia_a_capa(vias_utm, "dist_via_m")
    distancia_a_capa(poblados_utm, "dist_centro_poblado_m")
    distancia_a_capa(rios_utm, "dist_rio_m")

    df = centroides_utm.drop(columns="geometry")
    print("\n--- Resumen ---")
    print(df[["dist_via_m", "dist_centro_poblado_m", "dist_rio_m"]].describe())

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_osm.csv"
    df.to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
