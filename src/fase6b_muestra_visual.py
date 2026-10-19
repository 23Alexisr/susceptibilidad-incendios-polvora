"""
Fase 6b (paso 9) - Muestra aleatoria de 50 eventos de incendio forestal
(definicion EXTENDIDA, los mismos que alimentan el modelo: con
participacion de VIIRS, 2012-2024) con imagenes Sentinel-2 de ANTES y
DESPUES, exportadas como PNG para revision visual.

Por que Sentinel-2 y no Planet NICFI: NICFI sigue sin estar disponible en
el proyecto de Earth Engine del usuario (verificado de nuevo al inicio de
esta fase). Sentinel-2 esta disponible sin registro adicional.

Por que desde 2018: se verifico empiricamente que Sentinel-2 SR
Harmonized (Level-2A, la coleccion usada aqui) recien tiene cobertura
real sobre este punto del distrito desde el 2017-07-23 (no desde 2015
como se podria asumir por el lanzamiento del satelite -- el
procesamiento atmosferico de Nivel 2A se extendio a Sudamerica mas
tarde). Se deja un margen y se restringe la muestra a eventos desde 2018
para asegurar imagen "antes" de calidad con varios meses de cobertura
previa disponible.

Composite "antes"/"despues": mediana de Sentinel-2 SR armonizado,
filtrado a <30% de nubosidad, en una ventana de 2 meses ANTES del inicio
del evento y de 2 a 4 meses DESPUES del fin del evento (se deja un margen
de 2 meses tras el fuego para que el humo se disipe, antes de que
empiece la regeneracion a borrar la cicatriz). Se usa una combinacion de
bandas falso color (B12-SWIR2, B8-NIR, B4-Rojo) porque resalta cicatrices
de quema mucho mejor que el color verdadero: la vegetacion sana se ve
verde brillante y el area quemada se ve marron/rojiza oscura.
"""
import io
import random

import ee
import pandas as pd
import requests
from PIL import Image

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID, FIGURES_DIR, SEED

ee.Initialize(project=EE_PROJECT_ID)

N_MUESTRA = 50
ANIO_MINIMO = 2018
LADO_BUFFER_M = 800
DIMENSION_PX = 400

CARPETA_SALIDA = FIGURES_DIR / "fase6b_muestra_visual"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


def composite_sentinel2(centro, fecha_inicio, fecha_fin):
    coleccion = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(centro)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )
    return coleccion.median().select(["B12", "B8", "B4"])


def contar_imagenes(centro, fecha_inicio, fecha_fin):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(centro)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .size()
        .getInfo()
    )


def descargar_thumbnail(imagen, region, ruta_salida):
    url = imagen.getThumbURL(
        {"region": region, "dimensions": DIMENSION_PX, "min": 0, "max": 3000, "format": "png"}
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    Image.open(io.BytesIO(resp.content)).save(ruta_salida)


def main() -> None:
    eventos = pd.read_csv(DATA_PROCESSED_DIR / "fase3b_eventos.csv", parse_dates=["fecha_inicio", "fecha_fin"])
    eventos["anio"] = eventos["fecha_inicio"].dt.year

    candidatos = eventos[
        eventos["categoria_dominante"].isin(["incendio_forestal_no_estricto", "incendio_forestal_estricto"])
        & eventos["sensores"].str.contains("viirs_snpp")
        & (eventos["anio"] >= ANIO_MINIMO)
    ]
    print(f"Eventos candidatos (extendida, con VIIRS, >={ANIO_MINIMO}): {len(candidatos)}")

    muestra = candidatos.sample(n=min(N_MUESTRA, len(candidatos)), random_state=SEED).reset_index(drop=True)
    print(f"Muestra seleccionada: {len(muestra)} eventos")

    metadatos = []
    for i, fila in muestra.iterrows():
        centro = ee.Geometry.Point([fila["lon"], fila["lat"]])
        region = centro.buffer(LADO_BUFFER_M / 2).bounds()

        fecha_ini = fila["fecha_inicio"]
        fecha_fin = fila["fecha_fin"]
        id_evento = int(fila["evento_id"])
        nombre_base = f"evento_{id_evento:05d}"

        exito_antes = exito_despues = False

        # Ventana "antes": empieza en 60 dias antes del evento y se
        # ensancha (120, 180, 270) si no hay suficientes imagenes sin
        # nubes -- la cobertura de Sentinel-2 en la zona es irregular.
        for dias_atras in (60, 120, 180, 270):
            antes_ini = (fecha_ini - pd.Timedelta(days=dias_atras)).strftime("%Y-%m-%d")
            antes_fin = fecha_ini.strftime("%Y-%m-%d")
            if contar_imagenes(centro, antes_ini, antes_fin) > 0:
                break
        try:
            img_antes = composite_sentinel2(centro, antes_ini, antes_fin)
            descargar_thumbnail(img_antes, region, CARPETA_SALIDA / f"{nombre_base}_antes.png")
            exito_antes = True
        except Exception as e:
            print(f"  [{i+1}/{len(muestra)}] evento {id_evento}: FALLO imagen 'antes' ({str(e)[:100]})")

        for dias_adelante_fin in (120, 180, 270, 365):
            despues_ini = (fecha_fin + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            despues_fin = (fecha_fin + pd.Timedelta(days=dias_adelante_fin)).strftime("%Y-%m-%d")
            if contar_imagenes(centro, despues_ini, despues_fin) > 0:
                break
        try:
            img_despues = composite_sentinel2(centro, despues_ini, despues_fin)
            descargar_thumbnail(img_despues, region, CARPETA_SALIDA / f"{nombre_base}_despues.png")
            exito_despues = True
        except Exception as e:
            print(f"  [{i+1}/{len(muestra)}] evento {id_evento}: FALLO imagen 'despues' ({str(e)[:100]})")

        metadatos.append(
            {
                "evento_id": id_evento,
                "lon": fila["lon"],
                "lat": fila["lat"],
                "fecha_inicio": fecha_ini.date(),
                "fecha_fin": fecha_fin.date(),
                "n_detecciones": fila["n_detecciones"],
                "categoria_dominante": fila["categoria_dominante"],
                "sensores": fila["sensores"],
                "imagen_antes_ok": exito_antes,
                "imagen_despues_ok": exito_despues,
            }
        )
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(muestra)}] procesados")

    df_meta = pd.DataFrame(metadatos)
    df_meta.to_csv(CARPETA_SALIDA / "metadatos.csv", index=False)
    print(f"\nGuardado: {CARPETA_SALIDA / 'metadatos.csv'}")
    print(f"Imagenes 'antes' exitosas: {df_meta['imagen_antes_ok'].sum()}/{len(df_meta)}")
    print(f"Imagenes 'despues' exitosas: {df_meta['imagen_despues_ok'].sum()}/{len(df_meta)}")


if __name__ == "__main__":
    main()
