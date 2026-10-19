"""
Fase 1 - Area de estudio: limite del distrito de Polvora.

Ambito cambiado de Tocache (ubigeo 221001) a Polvora (ubigeo 221003) por
pedido explicito del usuario. Ambos distritos pertenecen a la misma
provincia (Tocache) y region (San Martin).

Fuente de datos: dataset "Peru - Subnational Administrative Boundaries"
(COD-AB), publicado en el Humanitarian Data Exchange (HDX) por OCHA ROLAC,
cuya fuente original es el Instituto Geografico Nacional del Peru (IGN).
No usamos geoBoundaries porque su release "gbOpen" para PER (verificado en
su API/repositorio de GitHub) solo llega hasta el nivel ADM2 (provincia);
no publica ADM3 (distrito) para Peru, asi que no sirve para este caso.

URL del dataset (verificada manualmente, no inventada):
https://data.humdata.org/dataset/cod-ab-per
Recurso descargado: per_admin_boundaries.geojson.zip
  -> archivo per_admin3.geojson (1873 distritos, nivel ADM3)

Verificacion del ubigeo: el codigo INEI 221003 (departamento 22 = San Martin,
provincia 10 = Tocache, distrito 03 = Polvora) es unico en todo el pais (un
solo registro con ese pcode, sin homonimos con nombre parecido) y aparece
junto a los otros 4 distritos de la provincia Tocache (Tocache 221001,
Nuevo Progreso 221002, Polvora 221003, Shunte 221004, Uchiza 221005), lo que
coincide con el listado oficial de distritos de la provincia.
"""
import geopandas as gpd
import matplotlib.pyplot as plt

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, FIGURES_DIR

RUTA_ADM3 = DATA_RAW_DIR / "per_admin_boundaries" / "per_admin3.geojson"
RUTA_ADM2 = DATA_RAW_DIR / "per_admin_boundaries" / "per_admin2.geojson"
UBIGEO_DISTRITO = "PE221003"
NOMBRE_DISTRITO = "Polvora"

# CRS proyectado para Peru central/oriental (UTM zona 18S). Lo usamos solo
# para calcular el area en km2 correctamente: en grados (EPSG:4326) el area
# calculada directamente esta distorsionada porque los grados no son unidades
# de distancia constante.
CRS_METRICO = "EPSG:32718"


def main() -> None:
    gdf = gpd.read_file(RUTA_ADM3)
    print(f"Distritos totales en el archivo nacional: {len(gdf)}")

    distrito = gdf[gdf["adm3_pcode"] == UBIGEO_DISTRITO].copy()
    if len(distrito) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 distrito con pcode {UBIGEO_DISTRITO}, "
            f"se encontraron {len(distrito)}. Revisar manualmente."
        )

    # Chequeo extra pedido para este cambio de ambito: confirmar que no hay
    # otro distrito en el pais con nombre parecido que se pueda confundir
    # con el que buscamos (evita, por ejemplo, tomar un distrito homonimo
    # de otra region por error).
    nombre_encontrado = distrito.iloc[0]["adm3_name"]
    parecidos = gdf[
        gdf["adm3_name"].str.contains(nombre_encontrado, case=False, na=False)
        & (gdf["adm3_pcode"] != UBIGEO_DISTRITO)
    ]
    if len(parecidos) > 0:
        print(
            f"AVISO: existen otros {len(parecidos)} distrito(s) con nombre "
            f"parecido a '{nombre_encontrado}':"
        )
        print(parecidos[["adm3_name", "adm3_pcode", "adm2_name", "adm1_name"]])
    else:
        print(f"Verificado: '{nombre_encontrado}' no tiene homonimos en el pais.")

    fila = distrito.iloc[0]
    print("\n--- Distrito encontrado ---")
    print(f"Nombre: {fila['adm3_name']}")
    print(f"Provincia: {fila['adm2_name']} (pcode {fila['adm2_pcode']})")
    print(f"Departamento/Region: {fila['adm1_name']} (pcode {fila['adm1_pcode']})")
    print(f"Ubigeo (adm3_pcode): {fila['adm3_pcode']}")
    print(f"CRS original: {distrito.crs}")

    # Area de dos formas: la que trae el atributo del dataset, y la que
    # calculamos nosotros reproyectando a un CRS metrico, como control cruzado.
    area_atributo = fila["area_sqkm"]
    area_calculada = distrito.to_crs(CRS_METRICO).geometry.area.iloc[0] / 1e6
    print(f"Area segun atributo del dataset: {area_atributo:.2f} km2")
    print(f"Area recalculada (reproyectado a {CRS_METRICO}): {area_calculada:.2f} km2")

    # Guardamos en GeoPackage, formato recomendado (reemplaza al shapefile:
    # un solo archivo, sin limite de 10 caracteres en nombres de columna,
    # soporta bien tipos de datos y multiples capas).
    ruta_salida = DATA_PROCESSED_DIR / "area_estudio.gpkg"
    distrito.to_file(ruta_salida, layer="distrito_estudio", driver="GPKG")
    print(f"\nGuardado en: {ruta_salida} (capa 'distrito_estudio')")

    # Mapa simple de verificacion visual: el distrito, con la provincia
    # como contexto para confirmar que no confundimos distrito con provincia.
    provincias = gpd.read_file(RUTA_ADM2)
    provincia_tocache = provincias[provincias["adm2_pcode"] == "PE2210"]

    fig, ax = plt.subplots(figsize=(7, 7))
    if len(provincia_tocache) == 1:
        provincia_tocache.boundary.plot(ax=ax, color="gray", linewidth=1, linestyle="--")
    distrito.plot(ax=ax, color="#c86428", edgecolor="black", linewidth=1.2, alpha=0.7)
    ax.set_title(
        f"Distrito de {NOMBRE_DISTRITO} (San Martin, Peru)\n"
        "linea punteada = provincia de Tocache"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.tight_layout()

    ruta_figura = FIGURES_DIR / "fase1_mapa_area_estudio.png"
    fig.savefig(ruta_figura, dpi=150)
    print(f"Mapa guardado en: {ruta_figura}")


if __name__ == "__main__":
    main()
