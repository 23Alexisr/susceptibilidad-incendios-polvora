"""
Fase 6b (paso 1) - Eventos de incendio forestal por anio, SOLO con VIIRS
(2012-2024), version extendida vs estricta, con pendiente de tendencia e
intervalo de confianza al 95%.

Analogo al grafico solo-MODIS de la Fase 3b (que sirve para la serie
larga 2001-2024); este usa VIIRS porque es el sensor con el que se
entrena el modelo en la Fase 6b (mejor resolucion espacial, 375 m).

"Solo VIIRS" en este grafico significa: eventos cuyo conjunto de
sensores es EXACTAMENTE 'viirs_snpp' (se excluyen eventos mixtos
modis+viirs para que la serie sea de un solo sensor, comparable entre
anios).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_PROCESSED_DIR, FIGURES_DIR


def tendencia_con_ic(anios, conteo, nombre):
    anios = np.array(anios, dtype=float)
    conteo = np.array(conteo, dtype=float)
    resultado = stats.linregress(anios, conteo)
    n = len(anios)
    t_critico = stats.t.ppf(0.975, df=n - 2)
    margen_pendiente = t_critico * resultado.stderr
    ic_bajo, ic_alto = resultado.slope - margen_pendiente, resultado.slope + margen_pendiente
    print(
        f"  [{nombre}] pendiente = {resultado.slope:+.3f} eventos/año "
        f"(IC95%: [{ic_bajo:+.3f}, {ic_alto:+.3f}]), "
        f"p-valor = {resultado.pvalue:.3f}, r² = {resultado.rvalue**2:.3f}"
    )

    # Banda de confianza al 95% de la RECTA ajustada (no solo de la
    # pendiente): formula estandar de regresion lineal simple.
    y_ajustado = resultado.intercept + resultado.slope * anios
    residuos = conteo - y_ajustado
    s = np.sqrt(np.sum(residuos**2) / (n - 2))
    x_media = anios.mean()
    sxx = np.sum((anios - x_media) ** 2)
    se_linea = s * np.sqrt(1 / n + (anios - x_media) ** 2 / sxx)
    banda_baja = y_ajustado - t_critico * se_linea
    banda_alta = y_ajustado + t_critico * se_linea

    return resultado, ic_bajo, ic_alto, y_ajustado, banda_baja, banda_alta


def main() -> None:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year

    viirs = eventos[eventos["sensores"] == "viirs_snpp"].copy()
    viirs = viirs[(viirs["anio"] >= 2012) & (viirs["anio"] <= 2024)]
    print(f"Eventos solo-VIIRS (2012-2024): {len(viirs)}")

    extendida = viirs[viirs["categoria_dominante"].isin(["incendio_forestal_no_estricto", "incendio_forestal_estricto"])]
    estricta = viirs[viirs["categoria_dominante"] == "incendio_forestal_estricto"]

    anios = list(range(2012, 2025))
    conteo_extendida = extendida.groupby("anio").size().reindex(anios, fill_value=0)
    conteo_estricta = estricta.groupby("anio").size().reindex(anios, fill_value=0)

    print("\nConteo por año:")
    print(pd.DataFrame({"extendida": conteo_extendida, "estricta": conteo_estricta}))

    print("\nTendencia lineal:")
    res_ext, ic_ext_lo, ic_ext_hi, y_ext, banda_lo_ext, banda_hi_ext = tendencia_con_ic(
        anios, conteo_extendida.values, "extendida"
    )
    res_est, ic_est_lo, ic_est_hi, y_est, banda_lo_est, banda_hi_est = tendencia_con_ic(
        anios, conteo_estricta.values, "estricta"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, conteo, res, y_ajustado, banda_lo, banda_hi, titulo, color in [
        (ax1, conteo_extendida, res_ext, y_ext, banda_lo_ext, banda_hi_ext, "Versión extendida", "#e07b39"),
        (ax2, conteo_estricta, res_est, y_est, banda_lo_est, banda_hi_est, "Versión estricta", "#7a1f1f"),
    ]:
        ax.bar(anios, conteo, color=color, alpha=0.7, label="Eventos observados")
        ax.plot(anios, y_ajustado, color="black", linewidth=2, label=f"Tendencia: {res.slope:+.2f}/año")
        ax.fill_between(anios, banda_lo, banda_hi, alpha=0.15, color="black", label="IC 95% de la recta")
        ax.set_title(f"{titulo}\np={res.pvalue:.3f}, r²={res.rvalue**2:.3f}")
        ax.set_xlabel("Año")
        ax.set_ylabel("N° de eventos de incendio forestal")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Eventos de incendio forestal por año — solo VIIRS (2012-2024)")
    fig.tight_layout()
    ruta = FIGURES_DIR / "fase6b_tendencia_viirs.png"
    fig.savefig(ruta, dpi=150)
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
