"""
Fase 7_c4 (extension) - Agrega el anio 2025 (12 meses) al NDVI mensual por
celda, reutilizando EXACTAMENTE la misma funcion de extraccion que
fase7_ndvi_mensual.py (MOD13A3.061, factor de escala 0.0001, mascara
SummaryQA<=1, promedio por poligono de celda). NO modifica
fase7_ndvi_mensual.csv (2011-2024), escribe un archivo nuevo aparte que
luego se concatena en fase7_construir_tabla_c4.py.
"""
import geopandas as gpd

from fase7_ndvi_mensual import generar_meses, procesar_mes
from config import DATA_PROCESSED_DIR

if __name__ == "__main__":
    celdas_4326 = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_4326")
    meses = list(generar_meses((2025, 1), (2025, 12)))
    print(f"Celdas: {len(celdas_4326)}, meses a procesar: {len(meses)} ({meses[0]} a {meses[-1]})")

    ruta = DATA_PROCESSED_DIR / "fase7_ndvi_mensual_2025_c4.csv"
    primero = True
    for i, (anio, mes) in enumerate(meses, start=1):
        df = procesar_mes(anio, mes, celdas_4326)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        n_nulos = df["ndvi"].isna().sum()
        print(f"[{i}/{len(meses)}] {anio}-{mes:02d}: {len(df)} filas, ndvi_medio={df['ndvi'].mean():.4f}, nulos={n_nulos}")
    print(f"\nGuardado: {ruta}")
