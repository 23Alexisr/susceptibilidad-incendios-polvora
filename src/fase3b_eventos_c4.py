"""
Fase 3b_c4 (paso 3) - Version paralela de fase3b_eventos.py: agrupa las
detecciones clasificadas con MapBiomas C4 (firms_clasificado_c4.gpkg) en
eventos de incendio, usando el mismo criterio de agrupacion (<=750 m,
<=5 dias, componentes conexas) que el pipeline original con la C3. No
depende de Earth Engine (la agrupacion es puramente geometrica/temporal).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from config import DATA_PROCESSED_DIR

CRS_METRICO = "EPSG:32718"
DISTANCIA_M = 750
DIAS_VENTANA = 5


def main() -> None:
    todas = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado_c4.gpkg", layer="detecciones_clasificadas")
    todas = todas.reset_index(drop=True)
    depurados = gpd.read_file(DATA_PROCESSED_DIR / "fase3b_incendios_depurados_c4.gpkg", layer="incendios")

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
            return "incendio_forestal_sin_verificar"
        if info["categoria_extendida"] != "incendio_forestal":
            return "quema_deforestacion_extendida"
        return "incendio_forestal_estricto" if info["es_estricto"] else "incendio_forestal_no_estricto"

    todas["categoria_final"] = todas.apply(categoria_final, axis=1)
    print("Distribucion de categoria_final (C4):")
    print(todas["categoria_final"].value_counts())

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

    ruta_eventos = DATA_PROCESSED_DIR / "fase3b_eventos_c4.csv"
    resumen.to_csv(ruta_eventos, index=False)
    print(f"Guardado: {ruta_eventos}")

    print("\n=== Tamaño y duración por categoría dominante del evento (C4) ===")
    print(resumen.groupby("categoria_dominante")[["n_detecciones", "duracion_dias"]].agg(["count", "mean", "median", "max"]))

    todas.drop(columns=["clave"]).to_file(DATA_PROCESSED_DIR / "firms_eventos_c4.gpkg", layer="detecciones_con_evento", driver="GPKG")


if __name__ == "__main__":
    main()
