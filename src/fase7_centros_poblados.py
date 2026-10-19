"""
Fase 7 (paso 3) - Distancia de cada celda al centro poblado mas cercano:
predictor NUEVO de la dimension "Antropogenica" de la matriz de
operacionalizacion (variable 8 de 8), como proxy de zona residencial.

Fuente PREFERIDA usada (INEI, geoespacial.inei.gob.pe): se obtuvo, no se
sustituyo por OpenStreetMap. El enunciado mencionaba sige.inei.gob.pe/
test/atlas/ o GeoIDEP como punto de partida: ese visor web (SIGE/Atlas) es
una aplicacion Javascript antigua (OpenLayers + jQuery) sin un enlace de
descarga directa localizable; en vez de eso se encontro el servicio real
a traves del catalogo oficial de GeoIDEP (catalogo-nacional-de-servicios-
web/servicio-de-descarga-wfs), que lista el servicio WFS que el propio
INEI publica en su servidor geoespacial:

    https://geoespacial.inei.gob.pe/Interoperabilidad/ig_centropoblado/wfs

Capa: "Interoperabilidad:ig_centropoblado" (GeoServer del INEI). Cada
feature trae, entre otros campos, "fuente": "INEI - CPV RESULTADOS" (Censos
de Poblacion y Vivienda), confirmando que es informacion censal oficial
del INEI, no un mirror de terceros.

Se descargan solo los centros poblados dentro del bounding box del
distrito + margen de ~0.3 grados (~33 km) por WFS (filtro BBOX), no la
capa nacional completa, para no descargar datos innecesarios. Con ese
margen el total es 753 centros poblados (ver impresion del script),
suficiente para que el "mas cercano" de cualquier celda del distrito este
garantizado dentro del margen (el distrito mide ~56 x 65 km).
"""
import geopandas as gpd
import pandas as pd
import requests

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

CRS_METRICO = "EPSG:32718"
WFS_URL = "https://geoespacial.inei.gob.pe/Interoperabilidad/ig_centropoblado/wfs"
MARGEN_GRADOS = 0.3
CARPETA_SALIDA = DATA_RAW_DIR / "inei"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def descargar_centros_poblados(bbox: tuple) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "Interoperabilidad:ig_centropoblado",
        "outputFormat": "application/json",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
    }
    print(f"Consultando WFS del INEI (bbox={bbox})...")
    r = requests.get(WFS_URL, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    gdf = gpd.read_file(r.text, driver="GeoJSON")
    gdf = gdf.set_crs("EPSG:4326")
    print(f"  Centros poblados recibidos: {len(gdf)}")
    return gdf


def main() -> None:
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")
    minx, miny, maxx, maxy = distrito.total_bounds
    bbox = (minx - MARGEN_GRADOS, miny - MARGEN_GRADOS, maxx + MARGEN_GRADOS, maxy + MARGEN_GRADOS)

    centros = descargar_centros_poblados(bbox)

    ruta_raw = CARPETA_SALIDA / "centros_poblados_polvora_bbox.gpkg"
    centros.to_file(ruta_raw, driver="GPKG")
    print(f"Guardado (crudo, fuente INEI): {ruta_raw}")

    print("\nMuestra de campos disponibles:", [c for c in centros.columns if c != "geometry"])
    if "ambito" in centros.columns:
        print("Distribucion de 'ambito' (1=urbano, 2=rural, segun INEI):")
        print(centros["ambito"].value_counts())

    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    centroides_utm = centroides.to_crs(CRS_METRICO).reset_index(drop=True)
    centros_utm = centros.to_crs(CRS_METRICO)

    print("\nCalculando distancia euclidiana de cada centroide al centro poblado mas cercano (sjoin_nearest)...")
    cercano = gpd.sjoin_nearest(
        centroides_utm[["cell_id", "geometry"]],
        centros_utm[["geometry"]],
        distance_col="dist_zona_residencial_m",
    )
    cercano = cercano.drop_duplicates(subset="cell_id")[["cell_id", "dist_zona_residencial_m"]]

    print(f"\nResumen dist_zona_residencial_m (metros):")
    print(cercano["dist_zona_residencial_m"].describe())
    print(f"Celdas sin dato: {cercano['dist_zona_residencial_m'].isna().sum()}")

    # Control: si el centro poblado mas cercano a alguna celda cae MUY cerca
    # del borde del bbox usado para la consulta WFS, el margen podria no ser
    # suficiente (el "mas cercano real" podria estar fuera del bbox
    # descargado). Se avisa explicitamente en vez de asumir que esta bien.
    centroides_4326 = centroides.reset_index(drop=True)
    dist_a_borde_bbox = pd.concat(
        [
            (centroides_4326.geometry.x - bbox[0]).abs(),
            (bbox[2] - centroides_4326.geometry.x).abs(),
            (centroides_4326.geometry.y - bbox[1]).abs(),
            (bbox[3] - centroides_4326.geometry.y).abs(),
        ],
        axis=1,
    ).min(axis=1)
    # dist_a_borde_bbox esta en grados; ~1 grado ~ 111 km en latitud.
    margen_m_aprox = dist_a_borde_bbox.min() * 111_000
    print(f"\nControl de margen: distancia minima de cualquier centroide al borde del bbox consultado ~{margen_m_aprox:.0f} m")
    print("(si esta cifra fuera MENOR que la distancia maxima calculada abajo, el margen de 0.3 grados seria insuficiente)")
    print(f"Distancia maxima calculada a un centro poblado: {cercano['dist_zona_residencial_m'].max():.0f} m")

    ruta_salida = DATA_PROCESSED_DIR / "fase7_dist_centro_poblado.csv"
    cercano.to_csv(ruta_salida, index=False)
    print(f"\nGuardado: {ruta_salida}")


if __name__ == "__main__":
    main()
