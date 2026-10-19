"""
Fase 2_c4 (parte 2) - Version paralela de fase2_procesar_firms.py para el
experimento de migracion a MapBiomas Coleccion 4 + extension del periodo a
2012-2025. No modifica nada del pipeline original (C3): produce
firms_distrito_c4.gpkg aparte.

Los CSV crudos de FIRMS (data/raw/firms/*_peru_bruto.csv) YA cubren hasta
2026-06-30 (se verifico antes de escribir este script, no hizo falta
volver a descargar nada de la API de FIRMS). Este script solo:
  1. Recorta al distrito de Polvora (igual que el original).
  2. Recorta ademas a acq_date <= FECHA_CORTE (2025-12-31), para que el
     periodo de este experimento sea exactamente el pedido (2012-2025) y
     no se cuelen detecciones de 2026 para las que de todas formas no hay
     mapa de uso de suelo (MapBiomas C4 llega hasta 2025).
"""
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR

CARPETA_FIRMS = DATA_RAW_DIR / "firms"
FECHA_CORTE = "2025-12-31"

ARCHIVOS = {
    "modis": CARPETA_FIRMS / "modis_peru_bruto.csv",
    "viirs_snpp": CARPETA_FIRMS / "viirs_snpp_peru_bruto.csv",
}


def cargar_y_recortar(nombre: str, ruta_csv, distrito: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = pd.read_csv(ruta_csv)
    print(f"[{nombre}] filas crudas (Peru completo, hasta la fecha de la ultima descarga): {len(df)}")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df[df["acq_date"] <= FECHA_CORTE]
    print(f"[{nombre}] filas tras recortar a <= {FECHA_CORTE}: {len(df)}")

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")

    recorte = gpd.sjoin(gdf, distrito[["geometry"]], how="inner", predicate="within").drop(columns="index_right")
    recorte["sensor"] = nombre
    recorte["anio"] = recorte["acq_date"].dt.year
    recorte["mes"] = recorte["acq_date"].dt.month

    print(f"[{nombre}] filas dentro del distrito de estudio: {len(recorte)}")
    return recorte


def main() -> None:
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")

    recortes = []
    for nombre, ruta in ARCHIVOS.items():
        recortes.append(cargar_y_recortar(nombre, ruta, distrito))

    todas = pd.concat(recortes, ignore_index=True)

    ruta_gpkg = DATA_PROCESSED_DIR / "firms_distrito_c4.gpkg"
    for nombre in ARCHIVOS:
        sub = todas[todas["sensor"] == nombre]
        if len(sub) > 0:
            gpd.GeoDataFrame(sub, geometry="geometry", crs="EPSG:4326").to_file(ruta_gpkg, layer=nombre, driver="GPKG")
    print(f"\nGuardado: {ruta_gpkg} (una capa por sensor)")

    print("\n--- Total de detecciones por sensor (dentro del distrito, hasta 2025-12-31) ---")
    print(todas.groupby("sensor").size())
    print("\n--- Conteo por anio y sensor (ultimos anios) ---")
    print(todas.groupby(["anio", "sensor"]).size().unstack(fill_value=0).tail(6))


if __name__ == "__main__":
    main()
