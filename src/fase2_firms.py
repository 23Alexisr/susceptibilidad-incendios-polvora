"""
Fase 2 - Detecciones de calor NASA FIRMS (VIIRS S-NPP y MODIS) para Peru.

Por que el endpoint "country" no se uso:
El plan original era usar los archivos anuales por pais (endpoint
/api/country/ o la descarga empaquetada en firms.modaps.eosdis.nasa.gov/country/).
Al verificarlo en el sitio oficial de FIRMS (https://firms.modaps.eosdis.nasa.gov/api/)
ese servicio aparece marcado explicitamente como "Feature currently not
available" (con enlace a su pagina de interrupciones). Confirmado tambien
al llamarlo directamente: devuelve "Invalid API call" (HTTP 400).

Alternativa usada: la "Area API" (/api/area/csv/...), que si esta activa.
Documentacion verificada en https://firms.modaps.eosdis.nasa.gov/api/area/:
  - Formato: /api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]
  - AREA_COORDINATES = "west,south,east,north"
  - DAY_RANGE: 1..5 (maximo 5 dias por peticion, limite duro de la API)
  - SOURCE para datos historicos de calidad cientifica (no near-real-time):
      MODIS_SP (MODIS Standard Processing)
      VIIRS_SNPP_SP (VIIRS Suomi-NPP Standard Processing)

Como no se puede pedir un rango de anios en una sola llamada, se pide en
bloques de 5 dias (el maximo permitido) desde la fecha de inicio de cada
sensor hasta hoy. Se pide la bounding box de todo el Peru (no cuesta mas
que pedir solo el distrito: el "peso" de la peticion en la API depende de
celdas de 60x60 grados, y tanto Peru completo como un distrito caben en una
sola celda) y despues se recorta al distrito de forma local, tal como pide
la consigna de la Fase 2.

Cobertura temporal pedida (segun lo indicado en la tarea, verificado ademas
contra la pagina de descargas de FIRMS que confirma el inicio real de cada
sensor):
  - MODIS Collection 6.1: datos reales desde 2000-11-01; se pide desde
    2001-01-01 porque asi lo pidio la tarea explicitamente.
  - VIIRS S-NPP 375m: datos desde 2012-01-20 (fecha real de inicio del
    sensor operacional), coincide con lo pedido en la tarea ("2012 en
    adelante").
"""
import time
from datetime import date, timedelta

import requests

from config import DATA_RAW_DIR, FIRMS_MAP_KEY

# Bounding box de Peru (west, south, east, north), calculado con
# geopandas.total_bounds() sobre el limite oficial ADM0 del IGN/HDX ya
# descargado en la Fase 1 (data/raw/per_admin_boundaries/per_admin0.geojson).
# No se inventa: sale directo de ese archivo.
PERU_BBOX = "-81.32823,-18.35093,-68.65228,-0.03861"

DAY_RANGE = 5  # maximo permitido por la Area API de FIRMS (verificado en su documentacion)
HOY = date.today()

SENSORES = {
    "modis": {"source": "MODIS_SP", "inicio": date(2001, 1, 1)},
    "viirs_snpp": {"source": "VIIRS_SNPP_SP", "inicio": date(2012, 1, 20)},
}

CARPETA_SALIDA = DATA_RAW_DIR / "firms"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


def generar_fechas_inicio_bloque(inicio: date, fin: date, paso: int = DAY_RANGE):
    actual = inicio
    while actual <= fin:
        yield actual
        actual += timedelta(days=paso)


def descargar_sensor(nombre: str, source: str, inicio: date, fin: date) -> tuple[int, list[date]]:
    """Descarga un sensor completo en bloques de DAY_RANGE dias.

    Escribe directamente a disco (linea por linea) en vez de acumular todo
    en memoria, para poder manejar millones de filas sin problemas de RAM
    y para no perder el progreso si el proceso se interrumpe a mitad de
    camino.
    """
    url_base = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/{source}/{PERU_BBOX}/{DAY_RANGE}"
    bloques = list(generar_fechas_inicio_bloque(inicio, fin))
    total_bloques = len(bloques)
    ruta_salida = CARPETA_SALIDA / f"{nombre}_peru_bruto.csv"

    print(f"[{nombre}] {total_bloques} peticiones de {DAY_RANGE} dias, desde {inicio} hasta {fin}")

    encabezado_escrito = False
    filas_totales = 0
    bloques_fallidos: list[date] = []

    with open(ruta_salida, "w", encoding="utf-8", newline="") as f:
        for i, fecha_inicio in enumerate(bloques, start=1):
            url = f"{url_base}/{fecha_inicio.isoformat()}"
            texto = None
            for intento in range(5):
                try:
                    r = requests.get(url, timeout=60)
                    # Las respuestas de error de FIRMS son paginas HTML, no CSV.
                    if r.status_code == 200 and not r.text.lstrip().startswith("<"):
                        texto = r.text
                        break
                except requests.RequestException:
                    pass
                time.sleep(2 * (intento + 1))

            if texto is None:
                bloques_fallidos.append(fecha_inicio)
                print(f"  AVISO: fallo definitivo en bloque que inicia {fecha_inicio}, se omite.")
                time.sleep(0.4)
                continue

            lineas = texto.strip("\n").split("\n")
            if lineas and lineas[0]:
                if not encabezado_escrito:
                    f.write(lineas[0] + "\n")
                    encabezado_escrito = True
                datos = lineas[1:]
                for linea in datos:
                    f.write(linea + "\n")
                filas_totales += len(datos)

            if i % 100 == 0 or i == total_bloques:
                print(f"  [{nombre}] {i}/{total_bloques} peticiones, {filas_totales} filas acumuladas")

            time.sleep(0.4)  # ritmo conservador para no exceder la cuota de la MAP_KEY

    print(f"[{nombre}] Completado: {filas_totales} filas guardadas en {ruta_salida}")
    if bloques_fallidos:
        print(f"[{nombre}] {len(bloques_fallidos)} bloques de 5 dias fallaron y se omitieron: {bloques_fallidos}")
    return filas_totales, bloques_fallidos


def main() -> None:
    if not FIRMS_MAP_KEY:
        raise SystemExit("Falta FIRMS_MAP_KEY en el archivo .env")

    resumen = {}
    for nombre, cfg in SENSORES.items():
        filas, fallidos = descargar_sensor(nombre, cfg["source"], cfg["inicio"], HOY)
        resumen[nombre] = {"filas": filas, "bloques_fallidos": len(fallidos)}

    print("\n--- Resumen de descarga ---")
    for nombre, info in resumen.items():
        print(f"{nombre}: {info['filas']} filas, {info['bloques_fallidos']} bloques fallidos")


if __name__ == "__main__":
    main()
