"""
Fase 6c - Tendencias de las condiciones climaticas y de bosque,
2001-2024, distrito de Polvora. Prueba de Mann-Kendall + pendiente de Sen
para cada una de las 4 series:
  1. Perdida anual de bosque (Hansen), hectareas -- ya calculada en
     fase6c_perdida_bosque.py (promedio directo sobre el distrito).
  2. Temperatura maxima promedio de la epoca seca (junio-septiembre),
     ERA5-Land -- promedio espacial (todas las celdas) y temporal
     (los 4 meses) por anio.
  3. Precipitacion acumulada de la epoca seca (junio-septiembre), CHIRPS
     -- suma de los 4 meses por celda, promediada entre celdas, por anio.
  4. Numero de meses por anio con SPI-3 < -1 (sequia) -- se recalcula el
     SPI-3 desde cero sobre la serie MENSUAL SIN REZAGO (Fase 5 guarda el
     SPI ya rezagado un mes para usarlo como predictor, no sirve aqui
     directamente), promediado espacialmente por mes, y se cuenta cuantos
     meses de cada anio caen por debajo de -1.

Por que Mann-Kendall + Sen y no una regresion lineal simple (como en las
Fases 3b/6b): Mann-Kendall es una prueba NO PARAMETRICA -- no asume que
los datos sigan una distribucion normal ni que la tendencia sea
perfectamente lineal con residuos homogeneos, y es mucho menos sensible a
valores atipicos (un solo anio con perdida de bosque anormalmente alta no
distorsiona la prueba tanto como distorsionaria una regresion lineal por
minimos cuadrados). Es el estandar en climatologia e hidrologia para
detectar tendencias en series de tiempo ambientales cortas y ruidosas
como estas (24 anios). La pendiente de Sen es el equivalente robusto a la
pendiente de una regresion: la mediana de todas las pendientes posibles
entre cada par de puntos de la serie, tambien resistente a atipicos.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import gamma, norm

from config import DATA_PROCESSED_DIR, FIGURES_DIR

ANIO_INICIO = 2001
ANIO_FIN = 2024
MESES_EPOCA_SECA = [6, 7, 8, 9]


def recalcular_spi3_sin_rezago(mensuales: pd.DataFrame) -> pd.DataFrame:
    """Igual metodologia que fase5_ensamblar.py::calcular_spi, pero
    aplicada aqui de forma independiente sobre la serie mensual cruda
    (sin el rezago de un mes que tiene la version guardada en la tabla
    de modelado), para poder contar meses secos del ANIO CALENDARIO real.
    """
    df = mensuales.sort_values(["cell_id", "anio", "mes"]).copy()
    df["precip_3m"] = (
        df.groupby("cell_id")["precip_mm"].rolling(3, min_periods=3).sum().reset_index(level=0, drop=True)
    )

    resultados = []
    for (cell_id, mes), grupo in df.dropna(subset=["precip_3m"]).groupby(["cell_id", "mes"]):
        valores = grupo["precip_3m"].values
        q = (valores == 0).mean()
        no_cero = valores[valores > 0]
        if len(no_cero) < 4:
            spi = np.full(len(valores), np.nan)
        else:
            forma, _, escala = gamma.fit(no_cero, floc=0)
            prob = np.where(valores == 0, q, q + (1 - q) * gamma.cdf(valores, forma, loc=0, scale=escala))
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spi = norm.ppf(prob)
        out = grupo[["cell_id", "anio", "mes"]].copy()
        out["spi_3m"] = spi
        resultados.append(out)

    return pd.concat(resultados, ignore_index=True)


def reportar_mk(nombre: str, anios: np.ndarray, valores: np.ndarray) -> dict:
    resultado = mk.original_test(valores)
    magnitud_decada = resultado.slope * 10
    print(f"\n--- {nombre} ---")
    print(f"  Tendencia detectada: {resultado.trend}")
    print(f"  Tau de Kendall: {resultado.Tau:.3f}")
    print(f"  p-valor: {resultado.p:.4f}  {'(SIGNIFICATIVA, p<0.05)' if resultado.p < 0.05 else '(NO significativa, p>=0.05)'}")
    print(f"  Pendiente de Sen: {resultado.slope:+.4f} por año  ->  {magnitud_decada:+.3f} por década")
    return {
        "serie": nombre,
        "tendencia": resultado.trend,
        "tau_kendall": resultado.Tau,
        "p_valor": resultado.p,
        "significativa_p<0.05": resultado.p < 0.05,
        "pendiente_sen_por_anio": resultado.slope,
        "pendiente_sen_por_decada": magnitud_decada,
        "intercepto_sen": resultado.intercept,
    }


def graficar(anios, valores, resultado_mk, titulo, ylabel, ruta, color):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(anios, valores, marker="o", color=color, label="Observado")
    linea_sen = resultado_mk["intercepto_sen"] + resultado_mk["pendiente_sen_por_anio"] * (anios - anios[0])
    estilo = "-" if resultado_mk["significativa_p<0.05"] else "--"
    etiqueta_sig = "significativa" if resultado_mk["significativa_p<0.05"] else "no significativa"
    ax.plot(
        anios, linea_sen, color="black", linestyle=estilo,
        label=f"Pendiente de Sen: {resultado_mk['pendiente_sen_por_decada']:+.2f}/década ({etiqueta_sig}, p={resultado_mk['p_valor']:.3f})",
    )
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


def main() -> None:
    print("Cargando datos mensuales (Fase 5)...")
    mensuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv")
    mensuales = mensuales[(mensuales["anio"] >= ANIO_INICIO) & (mensuales["anio"] <= ANIO_FIN)]

    perdida = pd.read_csv(DATA_PROCESSED_DIR / "fase6c_perdida_bosque_ha.csv")

    # --- Serie 2: temperatura maxima media de la epoca seca ---
    seca = mensuales[mensuales["mes"].isin(MESES_EPOCA_SECA)]
    temp_seca = seca.groupby("anio")["temp_max_c"].mean().reindex(range(ANIO_INICIO, ANIO_FIN + 1))

    # --- Serie 3: precipitacion acumulada de la epoca seca ---
    precip_por_celda_anio = seca.groupby(["cell_id", "anio"])["precip_mm"].sum().reset_index()
    precip_seca = precip_por_celda_anio.groupby("anio")["precip_mm"].mean().reindex(range(ANIO_INICIO, ANIO_FIN + 1))

    # --- Serie 4: meses con SPI-3 < -1 (recalculado sin rezago) ---
    print("Recalculando SPI-3 sin rezago (para contar meses secos del año calendario real)...")
    spi3 = recalcular_spi3_sin_rezago(mensuales)
    spi3_distrito = spi3.groupby(["anio", "mes"])["spi_3m"].mean().reset_index()
    spi3_distrito["mes_seco"] = spi3_distrito["spi_3m"] < -1
    meses_secos = spi3_distrito.groupby("anio")["mes_seco"].sum().reindex(range(ANIO_INICIO, ANIO_FIN + 1), fill_value=0)

    anios = np.array(range(ANIO_INICIO, ANIO_FIN + 1))

    print("\n" + "=" * 70)
    print("PRUEBAS DE MANN-KENDALL Y PENDIENTE DE SEN")
    print("=" * 70)

    resultados = []
    r1 = reportar_mk("Pérdida de bosque (ha/año)", anios, perdida["perdida_bosque_ha"].values)
    resultados.append(r1)
    graficar(anios, perdida["perdida_bosque_ha"].values, r1, "Pérdida anual de bosque (Hansen)", "Hectáreas",
              FIGURES_DIR / "fase6c_tendencia_perdida_bosque.png", "#c44e52")

    r2 = reportar_mk("Temperatura máxima época seca (°C)", anios, temp_seca.values)
    resultados.append(r2)
    graficar(anios, temp_seca.values, r2, "Temperatura máxima promedio, época seca (jun-sep)", "°C",
              FIGURES_DIR / "fase6c_tendencia_temperatura.png", "#e07b39")

    r3 = reportar_mk("Precipitación acumulada época seca (mm)", anios, precip_seca.values)
    resultados.append(r3)
    graficar(anios, precip_seca.values, r3, "Precipitación acumulada, época seca (jun-sep)", "mm",
              FIGURES_DIR / "fase6c_tendencia_precipitacion.png", "#4c72b0")

    r4 = reportar_mk("N° de meses/año con SPI-3 < -1", anios, meses_secos.values.astype(float))
    resultados.append(r4)
    graficar(anios, meses_secos.values, r4, "Meses por año con SPI-3 < -1 (sequía)", "N° de meses",
              FIGURES_DIR / "fase6c_tendencia_meses_secos.png", "#8172b3")

    df_resultados = pd.DataFrame(resultados)
    ruta = DATA_PROCESSED_DIR / "fase6c_tendencias_mann_kendall.csv"
    df_resultados.to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    # Panel combinado con las 4 series
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    datos_panel = [
        (perdida["perdida_bosque_ha"].values, r1, "Pérdida de bosque", "ha", "#c44e52"),
        (temp_seca.values, r2, "Temp. máx. época seca", "°C", "#e07b39"),
        (precip_seca.values, r3, "Precip. época seca", "mm", "#4c72b0"),
        (meses_secos.values, r4, "Meses secos (SPI-3<-1)", "N° meses", "#8172b3"),
    ]
    for ax, (valores, resultado, titulo, ylabel, color) in zip(axes.flat, datos_panel):
        ax.plot(anios, valores, marker="o", color=color)
        linea_sen = resultado["intercepto_sen"] + resultado["pendiente_sen_por_anio"] * (anios - anios[0])
        estilo = "-" if resultado["significativa_p<0.05"] else "--"
        ax.plot(anios, linea_sen, color="black", linestyle=estilo, linewidth=1.5)
        sig = "p<0.05" if resultado["significativa_p<0.05"] else "n.s."
        ax.set_title(f"{titulo}\n{resultado['pendiente_sen_por_decada']:+.2f}/década ({sig})", fontsize=10)
        ax.set_ylabel(ylabel)
    fig.suptitle("Tendencias 2001-2024 — distrito de Pólvora (Mann-Kendall + pendiente de Sen)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fase6c_panel_tendencias.png", dpi=150)
    print(f"Guardado: {FIGURES_DIR / 'fase6c_panel_tendencias.png'}")


if __name__ == "__main__":
    main()
