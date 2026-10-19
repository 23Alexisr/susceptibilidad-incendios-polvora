"""
Fase 7 - Genera la figura de detecciones por anio junto a la perdida anual
de bosque, y el informe final reports/VIABILIDAD.md.
"""
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR


def figura_detecciones_y_perdida():
    cat = pd.read_csv(DATA_PROCESSED_DIR / "fase3_conteo_categoria_anio.csv", index_col=0)
    cat.index.name = "anio"

    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv")
    # pct_perdida_bosque_ese_anio ya esta en fraccion de la celda (0-1);
    # el promedio de todas las celdas x el area total del distrito
    # (2263.7 km2, Fase 1) aproxima el area de bosque perdida ese anio.
    AREA_DISTRITO_KM2 = 2263.7
    perdida = anuales.groupby("anio")["pct_perdida_bosque_ese_anio"].mean() * AREA_DISTRITO_KM2

    anios = list(range(2001, 2025))
    incendios = cat.loc[anios, "incendio_forestal"]
    perdida_alineada = perdida.reindex(anios)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(anios, incendios, color="#c44e52", label="Detecciones de incendio forestal (FIRMS)")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("N° detecciones de incendio forestal", color="#c44e52")
    ax1.tick_params(axis="y", labelcolor="#c44e52")

    ax2 = ax1.twinx()
    ax2.plot(anios, perdida_alineada, color="#4c72b0", marker="o", linewidth=2, label="Pérdida de bosque (Hansen)")
    ax2.set_ylabel("Pérdida de bosque estimada (km²/año)", color="#4c72b0")
    ax2.tick_params(axis="y", labelcolor="#4c72b0")

    ax1.set_title("Detecciones de incendio forestal vs. pérdida anual de bosque\nDistrito de Pólvora, 2001-2024")
    fig.tight_layout()
    ruta = FIGURES_DIR / "fase7_detecciones_vs_perdida_bosque.png"
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada: {ruta}")
    return anios, incendios, perdida_alineada


if __name__ == "__main__":
    figura_detecciones_y_perdida()
