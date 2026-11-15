"""
Fase 8 (apoyo) - Grafico de las 4 424 detecciones VIIRS S-NPP dentro del
distrito de Polvora, 2012-2025, por anio y por mes. Solo VIIRS: MODIS y las
detecciones de 2026 no forman parte del estudio.

Lee:
  data/processed/firms_distrito_c4.gpkg (capa viirs_snpp)
Genera:
  resultados/fig_detecciones_viirs.png
  resultados/detecciones_viirs_anio_mes.csv
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED_DIR, ROOT_DIR

RESULTADOS_DIR = ROOT_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

DETECCIONES_ESPERADAS = 4_424
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
COLOR_BARRA = "#2a78d6"
COLOR_FONDO = "#fcfcfb"
COLOR_TEXTO = "#3d3d3a"
COLOR_GRILLA = "#e4e3df"


def barras(ax, etiquetas, valores, titulo):
    ax.bar(etiquetas, valores, color=COLOR_BARRA, width=0.7, zorder=2)
    for x, v in zip(etiquetas, valores):
        ax.annotate(f"{v:,}".replace(",", " "), (x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=COLOR_TEXTO)
    ax.set_title(titulo, loc="left", fontsize=11, color=COLOR_TEXTO)
    ax.set_ylabel("Detecciones", color=COLOR_TEXTO)
    ax.set_ylim(0, max(valores) * 1.12)
    ax.grid(axis="y", color=COLOR_GRILLA, lw=0.8, zorder=0)
    ax.set_facecolor(COLOR_FONDO)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRILLA)
    ax.tick_params(colors=COLOR_TEXTO, length=0)


def main():
    viirs = gpd.read_file(DATA_PROCESSED_DIR / "firms_distrito_c4.gpkg", layer="viirs_snpp")
    print(f"Detecciones VIIRS: {len(viirs)} (esperado {DETECCIONES_ESPERADAS})")
    if len(viirs) != DETECCIONES_ESPERADAS:
        raise SystemExit("La capa viirs_snpp no tiene el numero esperado de detecciones.")

    por_anio = viirs.groupby("anio").size()
    por_mes = viirs.groupby("mes").size().reindex(range(1, 13), fill_value=0)

    tabla = pd.crosstab(viirs["anio"], viirs["mes"]).reindex(columns=range(1, 13), fill_value=0)
    tabla.columns = MESES
    tabla["Total"] = tabla.sum(axis=1)
    tabla.loc["Total"] = tabla.sum()
    tabla.to_csv(RESULTADOS_DIR / "detecciones_viirs_anio_mes.csv")
    print(tabla.to_string())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor=COLOR_FONDO)
    barras(ax1, por_anio.index.astype(str), por_anio.values, "Por año")
    barras(ax2, MESES, por_mes.values, "Por mes (suma 2012-2025)")
    n = f"{len(viirs):,}".replace(",", " ")
    fig.suptitle(f"Detecciones VIIRS en el distrito de Pólvora, 2012-2025 (n = {n})",
                 x=0.06, ha="left", fontsize=13, color=COLOR_TEXTO)
    fig.text(0.06, 0.01, "Fuente: NASA FIRMS, VIIRS S-NPP 375 m (VIIRS_SNPP_SP). Todas las detecciones dentro del distrito, "
             "antes de filtrar por bosque.", fontsize=8, color="#73726c")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(RESULTADOS_DIR / "fig_detecciones_viirs.png", dpi=150)
    print("Figura guardada en resultados/fig_detecciones_viirs.png")


if __name__ == "__main__":
    main()
