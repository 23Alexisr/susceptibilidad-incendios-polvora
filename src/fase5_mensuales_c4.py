"""
Fase 5_c4 (paso 5, extension) - Agrega el anio 2025 (12 meses) a los
predictores climaticos mensuales, para la extension del periodo de la
Fase 7 hasta 2025. Reutiliza EXACTAMENTE la misma funcion de extraccion
que fase5_mensuales.py (mismas fuentes: CHIRPS Daily, ERA5-Land Monthly
Aggr, MODIS MOD09A1 para NDMI -- esta ultima no se usa en el modelo de 8
indicadores, pero se calcula igual para no tener un CSV con columnas
distintas al original) -- NO modifica fase5_predictores_mensuales.csv
(2000-2024), escribe un archivo nuevo aparte que luego se concatena en
fase7_construir_tabla_c4.py.
"""
from fase5_mensuales import generar_meses, procesar_mes
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR

if __name__ == "__main__":
    centroides = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="centroides_4326")
    meses = list(generar_meses((2025, 1), (2025, 12)))
    print(f"Celdas: {len(centroides)}, meses a procesar: {len(meses)} ({meses[0]} a {meses[-1]})")

    ruta = DATA_PROCESSED_DIR / "fase5_predictores_mensuales_2025_c4.csv"
    primero = True
    for i, (anio, mes) in enumerate(meses, start=1):
        df = procesar_mes(anio, mes, centroides)
        df.to_csv(ruta, mode="w" if primero else "a", header=primero, index=False)
        primero = False
        print(
            f"[{i}/{len(meses)}] {anio}-{mes:02d}: {len(df)} filas, "
            f"precip_media={df['precip_mm'].mean():.1f}mm, temp_max_media={df['temp_max_c'].mean():.1f}C, "
            f"nulos_precip={df['precip_mm'].isna().sum()}, nulos_temp={df['temp_max_c'].isna().sum()}"
        )
    print(f"\nGuardado: {ruta}")
