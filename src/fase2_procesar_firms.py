"""
Fase 2 (parte 2) - Recorta las detecciones FIRMS (ya descargadas para todo
Peru) al distrito de estudio, y calcula las entregas pedidas:
  - total de detecciones por sensor
  - conteo por anio y por mes
  - distribucion de confianza (MODIS es numerica 0-100; VIIRS es categorica
    l/n/h = low/nominal/high, asi que se reportan por separado)

Requiere haber corrido antes src/fase2_firms.py (deja los CSV crudos en
data/raw/firms/).
"""
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FIGURES_DIR

CARPETA_FIRMS = DATA_RAW_DIR / "firms"

ARCHIVOS = {
    "modis": CARPETA_FIRMS / "modis_peru_bruto.csv",
    "viirs_snpp": CARPETA_FIRMS / "viirs_snpp_peru_bruto.csv",
}


def cargar_y_recortar(nombre: str, ruta_csv, distrito: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = pd.read_csv(ruta_csv)
    print(f"[{nombre}] filas crudas (Peru completo): {len(df)}")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # sjoin en vez de clip: para puntos es mas directo y mas rapido, y
    # conserva todas las columnas originales sin partir geometrias.
    recorte = gpd.sjoin(gdf, distrito[["geometry"]], how="inner", predicate="within").drop(
        columns="index_right"
    )
    recorte["sensor"] = nombre
    recorte["acq_date"] = pd.to_datetime(recorte["acq_date"])
    recorte["anio"] = recorte["acq_date"].dt.year
    recorte["mes"] = recorte["acq_date"].dt.month

    print(f"[{nombre}] filas dentro del distrito de estudio: {len(recorte)}")
    return recorte


def main() -> None:
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")

    recortes = []
    for nombre, ruta in ARCHIVOS.items():
        if not ruta.exists():
            print(f"AVISO: no se encontro {ruta}, se omite el sensor '{nombre}'.")
            continue
        recortes.append(cargar_y_recortar(nombre, ruta, distrito))

    todas = pd.concat(recortes, ignore_index=True)

    # Guardamos el recorte completo (todas las columnas originales de FIRMS)
    # como GeoPackage, una capa por sensor para no mezclar columnas de
    # confianza que tienen tipos distintos (numerica vs categorica).
    ruta_gpkg = DATA_PROCESSED_DIR / "firms_distrito.gpkg"
    for nombre in ARCHIVOS:
        sub = todas[todas["sensor"] == nombre]
        if len(sub) > 0:
            gpd.GeoDataFrame(sub, geometry="geometry", crs="EPSG:4326").to_file(
                ruta_gpkg, layer=nombre, driver="GPKG"
            )
    print(f"\nGuardado: {ruta_gpkg} (una capa por sensor)")

    print("\n--- Total de detecciones por sensor (dentro del distrito) ---")
    print(todas.groupby("sensor").size())

    print("\n--- Conteo por anio y sensor ---")
    tabla_anio = todas.groupby(["anio", "sensor"]).size().unstack(fill_value=0)
    print(tabla_anio)
    tabla_anio.to_csv(DATA_PROCESSED_DIR / "firms_conteo_por_anio.csv")

    print("\n--- Conteo por mes (agregado, todos los anios) y sensor ---")
    tabla_mes = todas.groupby(["mes", "sensor"]).size().unstack(fill_value=0)
    print(tabla_mes)
    tabla_mes.to_csv(DATA_PROCESSED_DIR / "firms_conteo_por_mes.csv")

    print("\n--- Distribucion de confianza: MODIS (numerica 0-100) ---")
    modis = todas[todas["sensor"] == "modis"]
    if len(modis) > 0:
        # Al concatenar con VIIRS (que usa codigos de letra l/n/h en la misma
        # columna "confidence"), pandas guarda toda la columna como texto.
        # Para MODIS la convertimos de vuelta a numero antes de resumir.
        confianza_modis = pd.to_numeric(modis["confidence"], errors="coerce")
        print(confianza_modis.describe())

    print("\n--- Distribucion de confianza: VIIRS S-NPP (categorica l/n/h) ---")
    viirs = todas[todas["sensor"] == "viirs_snpp"]
    if len(viirs) > 0:
        print(viirs["confidence"].value_counts())

    # Figura simple para el informe de viabilidad (Fase 7): detecciones por
    # anio y por sensor, para ver la tendencia y comparar cobertura temporal.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    tabla_anio.plot(kind="bar", ax=ax)
    ax.set_title("Detecciones FIRMS por anio y sensor - distrito de estudio")
    ax.set_xlabel("Anio")
    ax.set_ylabel("Numero de detecciones")
    fig.tight_layout()
    ruta_fig = FIGURES_DIR / "fase2_detecciones_por_anio.png"
    fig.savefig(ruta_fig, dpi=150)
    print(f"\nFigura guardada en: {ruta_fig}")


if __name__ == "__main__":
    main()
