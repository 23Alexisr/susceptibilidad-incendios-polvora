"""
Fase 5 (paso 7) - Ensambla la tabla de modelado final: una fila por
celda-mes, con todos los predictores YA REZAGADOS para evitar fuga de
datos, y la etiqueta de incendio forestal.

Rezagos aplicados (regla del proyecto: predictores de periodos ANTERIORES
al mes que se predice):
  - Clima y NDMI: se usa el mes CALENDARIO ANTERIOR (M-1). Por ejemplo,
    para predecir la celda-mes de agosto 2015, el precip_mm/temp_max_c/
    vpd_kpa/ndmi/dias_secos_mes vienen de JULIO 2015.
  - SPI (1 y 3 meses): calculado sobre la serie de precipitacion que
    termina en el mes M-1 (coherente con el resto de variables de clima).
  - Uso de suelo (clase MapBiomas, % bosque, distancias, perdida de
    bosque): se usa el AÑO ANTERIOR (Y-1) completo, sin importar el mes.
    Por ejemplo, para cualquier mes del 2015, el uso de suelo viene del
    mapa de 2014.
  - Fisicos (DEM, pendiente, orientacion) y accesibilidad (distancias a
    vias/centros poblados/rios): no varian en el tiempo en este estudio,
    se usan tal cual.

Periodo de la tabla final: anios 2001-2024 (2001 porque MODIS, el sensor
FIRMS mas antiguo, empieza ese anio; 2024 porque es el ultimo anio
cubierto por MapBiomas Coleccion 3, la fuente de uso de suelo).
"""
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from config import DATA_PROCESSED_DIR, FIGURES_DIR

ANIO_INICIO = 2001
ANIO_FIN = 2024


def calcular_spi(mensual: pd.DataFrame, ventana: int, nombre_col: str) -> pd.DataFrame:
    """SPI por celda: para cada celda y mes calendario, ajusta una gamma
    (con correccion por meses de precipitacion cero, metodo estandar de
    McKee et al. 1993) a la precipitacion acumulada en una ventana de
    `ventana` meses, usando TODA la serie disponible (2000-2024) como
    referencia climatologica -- cuantos mas anios de referencia, mas
    estable es el ajuste.
    """
    mensual = mensual.sort_values(["cell_id", "anio", "mes"]).copy()
    mensual["precip_ventana"] = (
        mensual.groupby("cell_id")["precip_mm"]
        .rolling(ventana, min_periods=ventana)
        .sum()
        .reset_index(level=0, drop=True)
    )

    resultados = []
    for (cell_id, mes), grupo in mensual.dropna(subset=["precip_ventana"]).groupby(["cell_id", "mes"]):
        valores = grupo["precip_ventana"].values
        q = (valores == 0).mean()
        no_cero = valores[valores > 0]

        if len(no_cero) < 4:
            # Muy pocos datos para ajustar una gamma de forma confiable;
            # se deja SPI nulo para ese grupo (deberia ser raro).
            spi = np.full(len(valores), np.nan)
        else:
            forma, _, escala = gamma.fit(no_cero, floc=0)
            prob = np.where(
                valores == 0,
                q,
                q + (1 - q) * gamma.cdf(valores, forma, loc=0, scale=escala),
            )
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spi = norm.ppf(prob)

        out = grupo[["cell_id", "anio", "mes"]].copy()
        out[nombre_col] = spi
        resultados.append(out)

    return pd.concat(resultados, ignore_index=True)


def main() -> None:
    print("Cargando insumos...")
    estaticos = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_estaticos.csv")
    osm = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_osm.csv")
    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv")
    # ee.Reducer.mode() sobre una banda categorica devolvio ruido de punto
    # flotante (ej. 2.999999999999995 en vez de 3.0): son codigos de clase
    # discretos, se redondea antes de usar.
    anuales["clase_mapbiomas"] = anuales["clase_mapbiomas"].round().astype(int)
    mensuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_mensuales.csv")
    etiquetas = pd.read_csv(DATA_PROCESSED_DIR / "fase5_etiquetas.csv")

    print("Calculando SPI-1 y SPI-3 (esto ajusta miles de distribuciones gamma, toma un momento)...")
    spi1 = calcular_spi(mensuales, 1, "spi_1m")
    spi3 = calcular_spi(mensuales, 3, "spi_3m")
    mensuales = mensuales.merge(spi1, on=["cell_id", "anio", "mes"], how="left")
    mensuales = mensuales.merge(spi3, on=["cell_id", "anio", "mes"], how="left")

    # --- Construir el esqueleto celda x anio x mes del periodo de estudio ---
    celdas = estaticos["cell_id"].unique()
    filas_esqueleto = [
        {"cell_id": c, "anio": a, "mes": m}
        for c in celdas
        for a in range(ANIO_INICIO, ANIO_FIN + 1)
        for m in range(1, 13)
    ]
    tabla = pd.DataFrame(filas_esqueleto)
    print(f"Esqueleto celda x anio x mes: {len(tabla)} filas")

    # --- Rezago de clima/NDMI: mes anterior (M-1, con rollover de anio) ---
    tabla["anio_clima"] = np.where(tabla["mes"] == 1, tabla["anio"] - 1, tabla["anio"])
    tabla["mes_clima"] = np.where(tabla["mes"] == 1, 12, tabla["mes"] - 1)

    cols_clima = ["precip_mm", "dias_secos_mes", "temp_max_c", "vpd_kpa", "ndmi", "spi_1m", "spi_3m"]
    mensuales_ren = mensuales.rename(columns={"anio": "anio_clima", "mes": "mes_clima"})
    tabla = tabla.merge(
        mensuales_ren[["cell_id", "anio_clima", "mes_clima"] + cols_clima],
        on=["cell_id", "anio_clima", "mes_clima"],
        how="left",
    )

    # --- Rezago de uso de suelo: anio anterior (Y-1) ---
    tabla["anio_uso_suelo"] = tabla["anio"] - 1
    cols_uso_suelo = [
        "clase_mapbiomas",
        "pct_bosque_1km",
        "pct_perdida_bosque_ese_anio",
        "dist_borde_bosque_m",
        "dist_agropecuario_m",
    ]
    anuales_ren = anuales.rename(columns={"anio": "anio_uso_suelo"})
    tabla = tabla.merge(
        anuales_ren[["cell_id", "anio_uso_suelo"] + cols_uso_suelo],
        on=["cell_id", "anio_uso_suelo"],
        how="left",
    )

    # --- Estaticos (fisicos + accesibilidad OSM) ---
    tabla = tabla.merge(estaticos, on="cell_id", how="left")
    tabla = tabla.merge(osm, on="cell_id", how="left")

    # --- Etiqueta ---
    tabla = tabla.merge(
        etiquetas[["cell_id", "anio", "mes", "positivo"]], on=["cell_id", "anio", "mes"], how="left"
    )
    tabla["positivo"] = tabla["positivo"].fillna(0).astype(int)

    tabla = tabla.drop(columns=["anio_clima", "mes_clima", "anio_uso_suelo"])

    ruta = DATA_PROCESSED_DIR / "tabla_modelado.parquet"
    tabla.to_parquet(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    # --- Reporte de la Fase 5 ---
    n_filas = len(tabla)
    n_positivos = tabla["positivo"].sum()
    proporcion = n_positivos / n_filas
    print(f"\nFilas totales: {n_filas}")
    print(f"Positivos: {n_positivos}")
    print(f"Proporcion de positivos: {proporcion:.4%}  (desbalance 1:{(1/proporcion):.0f} aprox.)")

    print("\nNulos por columna (de los predictores, no del esqueleto):")
    cols_predictoras = cols_clima + cols_uso_suelo + ["elevacion", "pendiente", "orientacion", "dist_via_m", "dist_centro_poblado_m", "dist_rio_m"]
    print(tabla[cols_predictoras].isna().sum())

    # --- Matriz de correlacion (solo variables numericas continuas) ---
    numericas = [c for c in cols_predictoras if c != "clase_mapbiomas"]
    corr = tabla[numericas].corr()
    corr.to_csv(DATA_PROCESSED_DIR / "fase5_matriz_correlacion.csv")

    print("\nPares con |r| > 0.8:")
    pares_altos = []
    for i, a in enumerate(numericas):
        for b in numericas[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.8:
                pares_altos.append((a, b, r))
                print(f"  {a} <-> {b}: r = {r:.3f}")
    if not pares_altos:
        print("  (ninguno)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numericas)))
    ax.set_xticklabels(numericas, rotation=90)
    ax.set_yticks(range(len(numericas)))
    ax.set_yticklabels(numericas)
    fig.colorbar(im, label="r de Pearson")
    ax.set_title("Matriz de correlacion - predictores de la tabla de modelado")
    fig.tight_layout()
    ruta_fig = FIGURES_DIR / "fase5_matriz_correlacion.png"
    fig.savefig(ruta_fig, dpi=150)
    print(f"\nFigura guardada en: {ruta_fig}")


if __name__ == "__main__":
    main()
