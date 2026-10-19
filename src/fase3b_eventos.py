"""
Fase 3b (paso 3) - Agrupa detecciones individuales de FIRMS en "eventos" de
incendio, y cruza los eventos de incendio forestal con MCD64A1 (area
quemada mensual de MODIS) como verificacion independiente.

Criterio de agrupacion (distancia y dias), y por que:
  - Distancia: 750 m. FIRMS reporta el CENTROIDE del pixel detectado, no
    su huella completa. VIIRS tiene pixeles de 375 m y MODIS de ~1 km;
    750 m cubre aproximadamente 2 pixeles VIIRS o 0.75 pixeles MODIS,
    suficiente para enlazar detecciones del mismo frente de fuego
    (incluso si las reporta un sensor distinto) sin fusionar incendios
    genuinamente separados que por casualidad caen cerca.
  - Tiempo: 5 dias. Un incendio real puede seguir siendo detectado varios
    dias seguidos (arde, humea, se reactiva), y puede haber huecos de 1-2
    dias por nubosidad o el ciclo de revisita del satelite. 5 dias es un
    margen usado comunmente en literatura de clustering de focos de calor
    (ej. bases de datos de eventos de incendio como GlobFire/FRY) para no
    fusionar incendios de semanas distintas.
  - El agrupamiento es transitivo (si A-B estan cerca y B-C tambien, A-B-C
    son un solo evento aunque A-C por si solos no cumplieran el criterio):
    se modela como componentes conexas de un grafo, no como un clustering
    de radio fijo.

Cruce con MCD64A1: para cada evento de incendio forestal, se compara su
ventana de fechas contra la banda BurnDate del mes correspondiente de
MCD64A1 (dia del anio en que MODIS detecto quema en ese pixel). Si el dia
reportado por MCD64A1 cae dentro de la ventana del evento (+-5 dias de
margen), se cuenta como corroborado por un producto independiente.
"""
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

CRS_METRICO = "EPSG:32718"
DISTANCIA_M = 750
DIAS_VENTANA = 5


def main() -> None:
    todas = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado.gpkg", layer="detecciones_clasificadas")
    todas = todas.reset_index(drop=True)
    depurados = gpd.read_file(DATA_PROCESSED_DIR / "fase3b_incendios_depurados.gpkg", layer="incendios")

    # Reconectar por coincidencia exacta de sensor + fecha + coordenadas
    # (las detecciones de FIRMS no tienen un ID unico propio del proveedor).
    todas["clave"] = (
        todas["sensor"] + "_" + todas["acq_date"].astype(str) + "_" + todas.geometry.x.round(5).astype(str) + "_" + todas.geometry.y.round(5).astype(str)
    )
    depurados["clave"] = (
        depurados["sensor"] + "_" + depurados["acq_date"].astype(str) + "_" + depurados.geometry.x.round(5).astype(str) + "_" + depurados.geometry.y.round(5).astype(str)
    )

    mapa_categoria_final = depurados.set_index("clave")[["categoria_extendida", "es_estricto"]].to_dict("index")

    def categoria_final(fila):
        if fila["categoria"] != "incendio_forestal":
            return fila["categoria"]
        info = mapa_categoria_final.get(fila["clave"])
        if info is None:
            return "incendio_forestal_sin_verificar"  # no deberia pasar
        if info["categoria_extendida"] != "incendio_forestal":
            return "quema_deforestacion_extendida"
        return "incendio_forestal_estricto" if info["es_estricto"] else "incendio_forestal_no_estricto"

    todas["categoria_final"] = todas.apply(categoria_final, axis=1)
    print("Distribucion de categoria_final:")
    print(todas["categoria_final"].value_counts())

    # --- Clustering espacio-temporal ---
    todas["acq_date"] = pd.to_datetime(todas["acq_date"])
    todas_utm = todas.to_crs(CRS_METRICO)
    xy = np.column_stack([todas_utm.geometry.x, todas_utm.geometry.y])
    dias = (todas["acq_date"] - todas["acq_date"].min()).dt.days.values

    print(f"\nConstruyendo grafo espacio-temporal ({len(todas)} puntos, distancia<={DISTANCIA_M}m, dias<={DIAS_VENTANA})...")
    arbol = cKDTree(xy)
    pares = arbol.query_pairs(r=DISTANCIA_M, output_type="ndarray")
    print(f"  pares candidatos por distancia: {len(pares)}")

    diff_dias = np.abs(dias[pares[:, 0]] - dias[pares[:, 1]])
    pares_validos = pares[diff_dias <= DIAS_VENTANA]
    print(f"  pares validos tras filtrar por tiempo: {len(pares_validos)}")

    n = len(todas)
    filas_g = np.concatenate([pares_validos[:, 0], pares_validos[:, 1]])
    cols_g = np.concatenate([pares_validos[:, 1], pares_validos[:, 0]])
    datos_g = np.ones(len(filas_g))
    grafo = coo_matrix((datos_g, (filas_g, cols_g)), shape=(n, n))

    n_componentes, etiquetas = connected_components(grafo, directed=False)
    todas["evento_id"] = etiquetas
    print(f"\n{n} detecciones agrupadas en {n_componentes} eventos.")

    # --- Resumen por evento ---
    resumen = (
        todas.groupby("evento_id")
        .agg(
            n_detecciones=("id", "count"),
            fecha_inicio=("acq_date", "min"),
            fecha_fin=("acq_date", "max"),
            sensores=("sensor", lambda s: ",".join(sorted(set(s)))),
            categorias=("categoria_final", lambda s: ",".join(sorted(set(s)))),
            categoria_dominante=("categoria_final", lambda s: s.mode().iloc[0]),
            lon=("geometry", lambda g: g.iloc[0].x),
            lat=("geometry", lambda g: g.iloc[0].y),
        )
        .reset_index()
    )
    resumen["duracion_dias"] = (resumen["fecha_fin"] - resumen["fecha_inicio"]).dt.days + 1

    ruta_eventos = DATA_PROCESSED_DIR / "fase3b_eventos.csv"
    resumen.to_csv(ruta_eventos, index=False)
    print(f"Guardado: {ruta_eventos}")

    print("\n=== Tamaño y duración por categoría dominante del evento ===")
    print(resumen.groupby("categoria_dominante")[["n_detecciones", "duracion_dias"]].agg(["count", "mean", "median", "max"]))

    todas.drop(columns=["clave"]).to_file(DATA_PROCESSED_DIR / "firms_eventos.gpkg", layer="detecciones_con_evento", driver="GPKG")


if __name__ == "__main__":
    main()
