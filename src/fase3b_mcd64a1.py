"""
Fase 3b (paso 3, cruce con MCD64A1) - Para cada evento de incendio forestal
(estricto o no estricto), consulta el producto MCD64A1 (area quemada
mensual de MODIS, 500 m) en el mes de inicio del evento, y compara el dia
de quema que reporta contra la ventana de fechas del evento (FIRMS).

MCD64A1 es un producto MENSUAL: cada imagen cubre un mes calendario y su
banda "BurnDate" da el dia-del-anio en que se detecto la quema en ese
pixel (0 = sin quema ese mes). Se compara ese dia contra fecha_inicio/
fecha_fin del evento con +-5 dias de margen (mismo margen que se uso para
agrupar detecciones en eventos).
"""
import datetime as dt

import ee
import pandas as pd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID

ee.Initialize(project=EE_PROJECT_ID)

MCD64A1_ID = "MODIS/061/MCD64A1"
MARGEN_DIAS = 5
TAMANO_LOTE = 800


def main() -> None:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio", "fecha_fin"])
    incendios = eventos[
        eventos["categoria_dominante"].isin(["incendio_forestal_estricto", "incendio_forestal_no_estricto"])
    ].copy()
    print(f"Eventos de incendio forestal a cruzar con MCD64A1: {len(incendios)}")

    incendios["anio_mes_inicio"] = incendios["fecha_inicio"].dt.to_period("M").astype(str)

    resultados = []
    grupos = sorted(incendios["anio_mes_inicio"].unique())
    for i, ym in enumerate(grupos, start=1):
        anio, mes = map(int, ym.split("-"))
        subset = incendios[incendios["anio_mes_inicio"] == ym]

        inicio_mes = ee.Date.fromYMD(anio, mes, 1)
        fin_mes = inicio_mes.advance(1, "month")
        coleccion = ee.ImageCollection(MCD64A1_ID).filterDate(inicio_mes, fin_mes)
        n_img = coleccion.size().getInfo()
        if n_img == 0:
            for _, fila in subset.iterrows():
                resultados.append({"evento_id": fila["evento_id"], "burn_date_dia_anio": None, "corroborado_mcd64a1": False})
            continue
        imagen = coleccion.first().select("BurnDate")

        for lote_inicio in range(0, len(subset), TAMANO_LOTE):
            lote = subset.iloc[lote_inicio : lote_inicio + TAMANO_LOTE]
            feats = [
                ee.Feature(ee.Geometry.Point([f["lon"], f["lat"]]), {"evento_id": int(f["evento_id"])})
                for _, f in lote.iterrows()
            ]
            fc = ee.FeatureCollection(feats)
            red = imagen.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=500)
            info = {f["properties"]["evento_id"]: f["properties"].get("first") for f in red.getInfo()["features"]}

            for _, fila in lote.iterrows():
                burn_day = info.get(int(fila["evento_id"]))
                corroborado = False
                if burn_day is not None and burn_day > 0:
                    fecha_quema = dt.date(anio, 1, 1) + dt.timedelta(days=int(burn_day) - 1)
                    delta_inicio = abs((fecha_quema - fila["fecha_inicio"].date()).days)
                    delta_fin = abs((fecha_quema - fila["fecha_fin"].date()).days)
                    corroborado = min(delta_inicio, delta_fin) <= MARGEN_DIAS
                resultados.append(
                    {"evento_id": fila["evento_id"], "burn_date_dia_anio": burn_day, "corroborado_mcd64a1": corroborado}
                )
        print(f"  [{i}/{len(grupos)}] {ym}: {len(subset)} eventos")

    df_resultado = pd.DataFrame(resultados)
    incendios = incendios.merge(df_resultado, on="evento_id", how="left")

    ruta = DATA_PROCESSED_DIR / "fase3b_eventos_mcd64a1.csv"
    incendios.to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta}")

    print("\n=== Corroboracion por MCD64A1 ===")
    print(incendios.groupby("categoria_dominante")["corroborado_mcd64a1"].agg(["sum", "count", "mean"]))


if __name__ == "__main__":
    main()
