"""
Fase 8 (paso 1) - Tabla final de modelado y analisis descriptivo.

Parte de fase7_tabla_modelado_c4.parquet (MapBiomas C4, 2012-2025, celdas-mes
con bosque >= 20 % en el anio anterior) y cambia solo la imputacion del NDVI:
la metodologia pide el PROMEDIO historico de la misma celda y mes calculado
con 2012-2021, mientras que fase7_construir_tabla_c4.py usaba la mediana.
Como esa tabla conserva la marca ndvi_imputado, los valores no imputados son
los originales y se puede reimputar sin volver a consultar Earth Engine.

Lee:
  data/processed/fase7_tabla_modelado_c4.parquet
Genera:
  data/processed/fase8_tabla_modelado.parquet
  resultados/descriptivo_indicadores.csv
  resultados/incendios_por_anio.csv, resultados/incendios_por_mes.csv
  resultados/fig_incendios_por_anio.png, resultados/fig_incendios_por_mes.png
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED_DIR, ROOT_DIR

RESULTADOS_DIR = ROOT_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

ANIO_CORTE_TEST = 2022
FILAS_ESPERADAS = 351_732
POSITIVOS_ESPERADOS = 1_455
INDICADORES = [
    "precip_mm", "temp_max_c", "elevacion", "pendiente",
    "ndvi", "clase_mapbiomas", "dist_carretera_m", "dist_zona_residencial_m",
]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def reimputar_ndvi(tabla: pd.DataFrame) -> pd.DataFrame:
    tabla = tabla.copy()
    faltante = tabla["ndvi_imputado"] == 1
    tabla.loc[faltante, "ndvi"] = float("nan")

    base = tabla[(tabla["anio"] < ANIO_CORTE_TEST) & ~faltante]
    promedio_celda_mes = base.groupby(["cell_id", "mes"])["ndvi"].mean()
    promedio_mes = base.groupby("mes")["ndvi"].mean()

    claves = pd.MultiIndex.from_frame(tabla.loc[faltante, ["cell_id", "mes"]])
    tabla.loc[faltante, "ndvi"] = claves.map(promedio_celda_mes).to_numpy()

    # Respaldo para celdas que nunca tuvieron NDVI valido en ese mes entre 2012 y 2021
    sin_valor = tabla["ndvi"].isna()
    tabla.loc[sin_valor, "ndvi"] = tabla.loc[sin_valor, "mes"].map(promedio_mes).to_numpy()
    print(f"NDVI imputado con promedio celda+mes: {int(faltante.sum() - sin_valor.sum())}")
    print(f"NDVI imputado con promedio del mes (respaldo): {int(sin_valor.sum())}")
    assert tabla["ndvi"].notna().all()
    return tabla


def main() -> None:
    original = pd.read_parquet(DATA_PROCESSED_DIR / "fase7_tabla_modelado_c4.parquet")
    n_filas, n_pos = len(original), int(original["positivo"].sum())
    print(f"Filas: {n_filas} (esperado {FILAS_ESPERADAS}) | positivos: {n_pos} (esperado {POSITIVOS_ESPERADOS})")
    if n_filas != FILAS_ESPERADAS or n_pos != POSITIVOS_ESPERADOS:
        raise SystemExit("La tabla de partida no coincide con los conteos esperados.")

    tabla = reimputar_ndvi(original)
    diferencia = (tabla["ndvi"] - original["ndvi"]).abs()
    print(f"Cambio en NDVI por usar promedio en vez de mediana: media {diferencia[tabla['ndvi_imputado'] == 1].mean():.4f}, maximo {diferencia.max():.4f}")

    ruta = DATA_PROCESSED_DIR / "fase8_tabla_modelado.parquet"
    tabla.to_parquet(ruta, index=False)
    print(f"Guardado: {ruta}")

    # --- Descriptivo de los 8 indicadores ---
    descriptivo = tabla[INDICADORES].agg(["mean", "std", "min", "max"]).T
    descriptivo.columns = ["media", "desv_estandar", "minimo", "maximo"]
    descriptivo["n"] = len(tabla)
    descriptivo.to_csv(RESULTADOS_DIR / "descriptivo_indicadores.csv")
    print("\nDescriptivo de los 8 indicadores:")
    print(descriptivo.round(3).to_string())

    frecuencia_clase = tabla["clase_mapbiomas"].value_counts().rename("celdas_mes")
    frecuencia_clase.to_csv(RESULTADOS_DIR / "frecuencia_clase_mapbiomas.csv")
    print("\nFrecuencia de clase_mapbiomas (variable categorica):")
    print(frecuencia_clase.to_string())

    # --- Frecuencia de celdas-mes con incendio ---
    por_anio = tabla.groupby("anio").agg(celdas_mes=("positivo", "size"), con_incendio=("positivo", "sum"))
    por_anio["pct_con_incendio"] = por_anio["con_incendio"] / por_anio["celdas_mes"] * 100
    por_mes = tabla.groupby("mes").agg(celdas_mes=("positivo", "size"), con_incendio=("positivo", "sum"))
    por_mes["pct_del_total_incendios"] = por_mes["con_incendio"] / n_pos * 100
    por_mes.insert(0, "nombre_mes", MESES)
    por_anio.to_csv(RESULTADOS_DIR / "incendios_por_anio.csv")
    por_mes.to_csv(RESULTADOS_DIR / "incendios_por_mes.csv")
    print("\nCeldas-mes con incendio por anio:")
    print(por_anio.round(3).to_string())
    print("\nCeldas-mes con incendio por mes:")
    print(por_mes.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(por_anio.index.astype(str), por_anio["con_incendio"], color="#c0504d")
    ax.set_xlabel("Año")
    ax.set_ylabel("Celdas-mes con incendio")
    ax.set_title("Celdas-mes con incendio forestal por año (2012-2025)")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "fig_incendios_por_anio.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(MESES, por_mes["con_incendio"], color="#c0504d")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Celdas-mes con incendio")
    ax.set_title("Celdas-mes con incendio forestal por mes (suma 2012-2025)")
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / "fig_incendios_por_mes.png", dpi=150)
    print("\nFiguras guardadas en resultados/")


if __name__ == "__main__":
    main()
