"""
Fase 3b (paso 5) - Eventos de incendio forestal por anio, SOLO con MODIS
(2001-2024, para tener una serie larga y comparable, sin el salto de 2012
cuando entra VIIRS), version normal (extendida) vs. version estricta.
"""
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_PROCESSED_DIR, FIGURES_DIR


def main() -> None:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year

    modis = eventos[eventos["sensores"] == "modis"].copy()
    print(f"Eventos solo-MODIS: {len(modis)} de {len(eventos)} eventos totales")

    normal = modis[modis["categoria_dominante"].isin(["incendio_forestal_no_estricto", "incendio_forestal_estricto"])]
    estricta = modis[modis["categoria_dominante"] == "incendio_forestal_estricto"]

    anios = list(range(2001, 2025))
    conteo_normal = normal.groupby("anio").size().reindex(anios, fill_value=0)
    conteo_estricta = estricta.groupby("anio").size().reindex(anios, fill_value=0)

    print("\nConteo por anio (normal vs estricta):")
    print(pd.DataFrame({"normal": conteo_normal, "estricta": conteo_estricta}))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ancho = 0.4
    ax.bar([a - ancho / 2 for a in anios], conteo_normal, width=ancho, label="Versión normal (extendida)", color="#e07b39")
    ax.bar([a + ancho / 2 for a in anios], conteo_estricta, width=ancho, label="Versión estricta", color="#7a1f1f")
    ax.set_xlabel("Año")
    ax.set_ylabel("N° de eventos de incendio forestal")
    ax.set_title("Eventos de incendio forestal por año — solo MODIS (2001-2024)\nVersión normal vs. versión estricta")
    ax.legend()
    ax.set_xticks(anios)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    ruta = FIGURES_DIR / "fase3b_eventos_por_anio_modis.png"
    fig.savefig(ruta, dpi=150)
    print(f"\nGuardado: {ruta}")

    # Correlacion de tendencia simple (pendiente de una regresion lineal)
    import numpy as np

    for nombre, serie in [("normal", conteo_normal), ("estricta", conteo_estricta)]:
        pendiente = np.polyfit(anios, serie.values, 1)[0]
        print(f"Tendencia lineal ({nombre}): {pendiente:+.2f} eventos/año")


if __name__ == "__main__":
    main()
