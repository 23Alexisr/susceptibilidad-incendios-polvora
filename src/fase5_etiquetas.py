"""
Fase 5 (paso 6) - Etiqueta de la tabla de modelado: marca como positiva
cada celda-mes que tuvo al menos una deteccion FIRMS clasificada como
"incendio_forestal" (categoria (a) de la Fase 3) dentro de sus limites.
"""
import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED_DIR

CRS_METRICO = "EPSG:32718"


def main() -> None:
    celdas = gpd.read_file(DATA_PROCESSED_DIR / "grilla_1km.gpkg", layer="celdas_utm")
    detecciones = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado.gpkg", layer="detecciones_clasificadas")
    detecciones = detecciones.to_crs(CRS_METRICO)

    print(f"Celdas: {len(celdas)}, detecciones totales: {len(detecciones)}")

    union = gpd.sjoin(detecciones, celdas, how="inner", predicate="within")
    print(f"Detecciones que cayeron dentro de alguna celda: {len(union)} (deberia ser ~igual al total)")

    incendios = union[union["categoria"] == "incendio_forestal"].copy()
    incendios["anio_deteccion"] = incendios["anio_deteccion"].astype(int)
    print(f"Detecciones de incendio_forestal: {len(incendios)}")

    # anio_deteccion/mes ya estan en el gpkg (de la Fase 3); si por algun
    # motivo faltara "mes", se recalcula desde acq_date.
    if "mes" not in incendios.columns:
        incendios["mes"] = pd.to_datetime(incendios["acq_date"]).dt.month

    etiquetas = (
        incendios.groupby(["cell_id", "anio_deteccion", "mes"])
        .size()
        .reset_index(name="n_detecciones_incendio_forestal")
    )
    etiquetas = etiquetas.rename(columns={"anio_deteccion": "anio"})
    etiquetas["positivo"] = 1

    print(f"\nCombinaciones celda-anio-mes con al menos 1 incendio forestal: {len(etiquetas)}")
    print(f"Suma de detecciones (control, debe ser <= {len(incendios)}): {etiquetas['n_detecciones_incendio_forestal'].sum()}")

    ruta = DATA_PROCESSED_DIR / "fase5_etiquetas.csv"
    etiquetas.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
