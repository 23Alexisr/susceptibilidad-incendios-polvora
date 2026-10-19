"""
Fase 7 (paso 2) - Distancia de cada celda a la Carretera Marginal de la
Selva (ruta PE-5N): predictor NUEVO de la dimension "Antropogenica" de la
matriz de operacionalizacion (variable 7 de 8).

Fuente PREFERIDA usada (MTC, Red Vial Nacional): se obtuvo, no se
sustituyo por OpenStreetMap. Verificado paso a paso antes de usarla:
  1. Pagina oficial de descargas del MTC (https://portal.mtc.gob.pe/
     estadisticas/descarga.html) publica el shapefile completo de la Red
     Vial Nacional; el corte mas reciente disponible es DICIEMBRE 2018
     ("red_vial_nacional_dic18.zip", Last-Modified 2019-06-17 segun el
     servidor -- no hay un corte mas nuevo publicado en ese portal al
     2026-10-08). Descargado a data/raw/mtc/red_vial_nacional_dic18/.
  2. El campo CODRUTA del shapefile tiene 275 segmentos con el codigo
     exacto "PE-5N". El campo TRAYECTO de esos segmentos describe la ruta
     como "Pte. Reither (PE-5S) - Villa Rica - ... - Tocache - Tarapoto
     (PE-5N B) - Moyobamba - ..." -- es decir, PE-5N es la Carretera
     Marginal de la Selva, tramo norte, y pasa explicitamente por Tocache
     (la provincia del distrito de estudio).
  3. Verificado geometricamente: la distancia minima entre la geometria de
     PE-5N y el poligono del distrito de Polvora es 0 m (la via SI cruza
     el distrito), confirmando que es la via correcta a usar.

Limitacion a documentar en la tesis: el shapefile del MTC es un corte
estatico de diciembre de 2018, no una serie historica. El enunciado pidio
usar "solo la Marginal, porque existe durante todo el periodo 2012-2024"
-- es razonable asumir su trazado no cambio sustancialmente en Polvora
entre 2012 y 2024 (es la via nacional troncal de la zona, no una via
vecinal nueva), pero a diferencia de los predictores de Earth Engine (que
sí tienen una imagen por cada año/mes real), esta distancia se trata como
ESTATICA en todo el periodo -- misma limitacion ya documentada para las
vias de OpenStreetMap en la Fase 5 (ver fase5_osm_distancias.py).
"""
import geopandas as gpd

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

CRS_METRICO = "EPSG:32718"
RUTA_SHP_MTC = DATA_RAW_DIR / "mtc" / "red_vial_nacional_dic18" / "red_vial_nacional_dic18.shp"
CODIGO_RUTA = "PE-5N"  # Carretera Marginal de la Selva, tramo norte (ver docstring)


def main() -> None:
    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    centroides_utm = centroides.to_crs(CRS_METRICO)

    print(f"Leyendo Red Vial Nacional (MTC, dic-2018): {RUTA_SHP_MTC}")
    vias = gpd.read_file(RUTA_SHP_MTC, encoding="latin1")
    print(f"  Segmentos totales en el shapefile nacional: {len(vias)}")

    marginal = vias[vias["CODRUTA"] == CODIGO_RUTA].copy()
    print(f"  Segmentos con CODRUTA == '{CODIGO_RUTA}' (Marginal de la Selva): {len(marginal)}")
    if len(marginal) == 0:
        raise SystemExit(f"No se encontraron segmentos con CODRUTA == '{CODIGO_RUTA}'. Revisar el shapefile.")

    trayectos = marginal["TRAYECTO"].unique()
    print(f"  Descripcion de trayecto(s) encontrados ({len(trayectos)} distintas):")
    for t in trayectos:
        print(f"    - {t}")

    marginal_utm = marginal.to_crs(CRS_METRICO)
    geom_marginal = marginal_utm.union_all()

    print("\nCalculando distancia euclidiana de cada centroide a la Marginal (sjoin_nearest)...")
    centroides_utm = centroides_utm.reset_index(drop=True)
    cercano = gpd.sjoin_nearest(
        centroides_utm[["cell_id", "geometry"]],
        gpd.GeoDataFrame({"geometry": [geom_marginal]}, crs=CRS_METRICO),
        distance_col="dist_carretera_m",
    )
    cercano = cercano.drop_duplicates(subset="cell_id")[["cell_id", "dist_carretera_m"]]

    print(f"\nResumen dist_carretera_m (metros):")
    print(cercano["dist_carretera_m"].describe())
    print(f"Celdas sin dato: {cercano['dist_carretera_m'].isna().sum()}")

    ruta_salida = DATA_PROCESSED_DIR / "fase7_dist_carretera.csv"
    cercano.to_csv(ruta_salida, index=False)
    print(f"\nGuardado: {ruta_salida}")


if __name__ == "__main__":
    main()
