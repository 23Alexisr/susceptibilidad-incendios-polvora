"""
Ayuda visual (no es parte del pipeline principal, no genera ningun csv/parquet
usado por fases posteriores): mapa de la grilla de 1 km x 1 km del distrito de
Polvora, coloreando cada celda segun si cumple o no el criterio de bosque
(>=20% de bosque, el mismo umbral UMBRAL_BOSQUE de src/fase6b_construir_tabla.py).

Criterio EXACTO (igual al de la tabla de modelado fase6b_tabla_modelado.parquet,
ver src/fase6b_construir_tabla.py:106-117,133): para cada mes evaluado entre
2012 y 2024, el predictor de bosque usa el uso de suelo del AÑO ANTERIOR
(anio_uso_suelo = anio_mes - 1). Por lo tanto los anios de mapa de uso de
suelo realmente evaluados son 2011-2023 (no 2012-2023: ese fue un error en
la version anterior de este script, que olvidaba incluir 2011 como insumo
del rezago de los meses de 2012). Una celda "cumple" si alcanza >=20% de
bosque en ALGUN anio_uso_suelo de 2011-2023 (es decir, pudo haber sido
positiva-de-bosque para prediccion en al menos un mes del periodo).

Verificado por comparacion directa: el conjunto de celdas que cumple con
este criterio coincide EXACTAMENTE (0 diferencias) con el conjunto de
cell_id presentes en data/processed/fase6b_tabla_modelado.parquet
(2,098 celdas) -- ver verificacion en la conversacion/commit que corrigio
este script.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from config import DATA_PROCESSED_DIR, FIGURES_DIR

CRS_METRICO = "EPSG:32718"
UMBRAL_BOSQUE = 0.20  # igual que src/fase6b_construir_tabla.py
ANIO_MES_INICIO, ANIO_MES_FIN = 2012, 2024  # periodo de meses evaluados por el modelo

COLOR_CUMPLE = "#2b8a3e"
COLOR_NO_CUMPLE = "#d9a93e"
COLOR_LIMITE = "#1b1b1b"


def formato_miles(n: int) -> str:
    """Separador de miles con espacio (2 263), no coma (2,263)."""
    return f"{n:,}".replace(",", " ")


def main() -> None:
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio").to_crs(
        CRS_METRICO
    )

    anuales = pd.read_csv(DATA_PROCESSED_DIR / "fase5_predictores_anuales.csv")

    # anio_uso_suelo = anio_mes - 1, para anio_mes en [2012, 2024] => 2011-2023.
    anios_uso_suelo = sorted({a - 1 for a in range(ANIO_MES_INICIO, ANIO_MES_FIN + 1)})
    faltantes = [a for a in anios_uso_suelo if a not in set(anuales["anio"].unique())]
    print(f"Meses evaluados: {ANIO_MES_INICIO}-{ANIO_MES_FIN} -> años de uso de suelo (rezagados): {anios_uso_suelo}")
    if faltantes:
        print(f"AVISO: faltan años de uso de suelo en fase5_predictores_anuales.csv: {faltantes}")

    subset = anuales[anuales["anio"].isin(anios_uso_suelo)]
    max_bosque = subset.groupby("cell_id")["pct_bosque_1km"].max().rename("pct_bosque_max")

    celdas = celdas.merge(max_bosque, on="cell_id", how="left")
    celdas["cumple"] = celdas["pct_bosque_max"] >= UMBRAL_BOSQUE
    celdas["cumple"] = celdas["cumple"].fillna(False)  # celdas sin dato = no cumple, no se ocultan

    n_total = len(celdas)
    n_cumple = int(celdas["cumple"].sum())
    n_no_cumple = int((~celdas["cumple"]).sum())
    n_sin_dato = int(celdas["pct_bosque_max"].isna().sum())
    print(f"Celdas que cumplen (>= {UMBRAL_BOSQUE:.0%} bosque en algún año de uso de suelo {anios_uso_suelo[0]}-{anios_uso_suelo[-1]}): {n_cumple}")
    print(f"Celdas que NO cumplen: {n_no_cumple} (de las cuales sin dato de bosque: {n_sin_dato})")
    print(f"Total de celdas: {n_total}")

    # Verificacion cruzada contra la tabla de modelado real (si existe).
    ruta_tabla = DATA_PROCESSED_DIR / "fase6b_tabla_modelado.parquet"
    if ruta_tabla.exists():
        celdas_tabla = set(pd.read_parquet(ruta_tabla, columns=["cell_id"])["cell_id"].unique())
        celdas_cumple = set(celdas.loc[celdas["cumple"], "cell_id"])
        print(f"Verificación cruzada con {ruta_tabla.name}: coincide exactamente = {celdas_tabla == celdas_cumple}")

    # --- Figura ---
    fig, ax = plt.subplots(figsize=(9, 10))

    celdas[~celdas["cumple"]].plot(ax=ax, color=COLOR_NO_CUMPLE, edgecolor="white", linewidth=0.1)
    celdas[celdas["cumple"]].plot(ax=ax, color=COLOR_CUMPLE, edgecolor="white", linewidth=0.1)
    distrito.boundary.plot(ax=ax, color=COLOR_LIMITE, linewidth=1.3)

    ax.set_axis_off()  # sin título: se omite a pedido explícito

    # --- Leyenda (con conteos, separador de miles con espacio) ---
    elementos_leyenda = [
        Patch(facecolor=COLOR_CUMPLE, edgecolor="white", label=f"Cumple criterio de bosque (n = {formato_miles(n_cumple)})"),
        Patch(facecolor=COLOR_NO_CUMPLE, edgecolor="white", label=f"No cumple criterio de bosque (n = {formato_miles(n_no_cumple)})"),
        Line2D([0], [0], color=COLOR_LIMITE, linewidth=1.3, label="Límite del distrito de Pólvora"),
    ]
    ax.legend(
        handles=elementos_leyenda,
        loc="lower left",
        fontsize=9.5,
        frameon=True,
        framealpha=1.0,
        title=f"Total de celdas: {formato_miles(n_total)}",
        title_fontsize=9.5,
    )

    minx, miny, maxx, maxy = celdas.total_bounds
    ancho = maxx - minx
    alto = maxy - miny
    ax.set_xlim(minx - 0.05 * ancho, maxx + 0.05 * ancho)
    ax.set_ylim(miny - 0.05 * alto, maxy + 0.06 * alto)

    # --- Panel de escala + norte (coordenadas de EJES, no de datos) ---
    # El polígono del distrito es muy irregular. Antes de fijar esta
    # posición se verificó por intersección geométrica real (shapely,
    # con margen de seguridad de ~1% del ancho) que este rectángulo NO
    # toca el polígono del distrito en ningún punto -- a diferencia de
    # la esquina superior derecha usada en la version anterior, que sí
    # lo tocaba (la "bota" del distrito llega hasta ahí). La esquina
    # superior IZQUIERDA, en cambio, cae sobre la bahía/entrante vacía
    # que tiene el distrito en su extremo norte.
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.78), 0.36, 0.20,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            transform=ax.transAxes, facecolor="white", edgecolor="#888888",
            linewidth=0.8, zorder=10, clip_on=False,
        )
    )

    # Flecha de norte: a la IZQUIERDA del panel, separada de la barra de
    # escala por un espacio en blanco explícito (no comparten columna).
    ax.annotate(
        "", xy=(0.09, 0.945), xytext=(0.09, 0.80),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(facecolor="black", edgecolor="black", width=4.5, headwidth=14, headlength=14),
        zorder=11,
    )
    ax.text(
        0.09, 0.955, "N", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12.5, fontweight="bold", zorder=11,
    )

    # Separador visual explícito entre la flecha y la escala.
    ax.plot([0.175, 0.175], [0.81, 0.93], transform=ax.transAxes, color="#c9c9c9", linewidth=1.0, zorder=11)

    # Barra de escala: a la DERECHA del panel. El ancho en fracción de eje
    # se calcula a partir del rango de datos (metros) realmente mostrado
    # (xlim ya fijado arriba), para que la distancia sea exacta.
    rango_x_datos_m = ax.get_xlim()[1] - ax.get_xlim()[0]
    longitudes_km = [0, 5, 10]
    frac_x0 = 0.225
    frac_y_barra = 0.865
    alto_barra_frac = 0.018

    for i in range(len(longitudes_km) - 1):
        f0 = frac_x0 + (longitudes_km[i] * 1000) / rango_x_datos_m
        f1 = frac_x0 + (longitudes_km[i + 1] * 1000) / rango_x_datos_m
        color = "black" if i % 2 == 0 else "white"
        ax.add_patch(
            plt.Rectangle(
                (f0, frac_y_barra), f1 - f0, alto_barra_frac,
                transform=ax.transAxes, facecolor=color, edgecolor="black",
                linewidth=0.6, zorder=11, clip_on=False,
            )
        )
    for km in longitudes_km:
        f = frac_x0 + (km * 1000) / rango_x_datos_m
        ax.text(
            f, frac_y_barra - 0.014, f"{km}", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, zorder=11,
        )
    f_centro = frac_x0 + (longitudes_km[-1] * 1000 / 2) / rango_x_datos_m
    ax.text(
        f_centro, frac_y_barra - 0.055, "Escala (km)", transform=ax.transAxes,
        ha="center", va="top", fontsize=8, zorder=11,
    )

    fig.tight_layout()
    ruta = FIGURES_DIR / "mapa_grilla_bosque20.png"
    fig.savefig(ruta, dpi=300, facecolor="white")
    print(f"\nGuardado: {ruta} (300 dpi)")


if __name__ == "__main__":
    main()
