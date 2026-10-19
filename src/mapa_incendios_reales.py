"""
Ayuda visual (no es parte del pipeline principal): mapa interactivo con los
incendios forestales REALES (categoria (a) de la Fase 3) sobre imagen
satelital, coloreados por periodo, para que se entienda donde ha estado
ocurriendo el fenomeno dentro del distrito -- a diferencia del mapa de la
Fase 1, que solo mostraba el contorno del distrito.
"""
import ee
import geemap
import geopandas as gpd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID, FIGURES_DIR

ee.Initialize(project=EE_PROJECT_ID)

detecciones = gpd.read_file(DATA_PROCESSED_DIR / "firms_clasificado.gpkg", layer="detecciones_clasificadas")
incendios = detecciones[detecciones["categoria"] == "incendio_forestal"].copy()
print(f"Incendios forestales a graficar: {len(incendios)}")

# Periodos para colorear (year solo no da buen contraste visual con tantos
# puntos; se agrupan en 4 tramos).
def periodo(anio):
    if anio <= 2011:
        return "2001-2011 (solo MODIS)"
    if anio <= 2016:
        return "2012-2016"
    if anio <= 2021:
        return "2017-2021"
    return "2022-2026"

incendios["periodo"] = incendios["anio_deteccion"].apply(periodo)

colores = {
    "2001-2011 (solo MODIS)": "#2166ac",
    "2012-2016": "#67a9cf",
    "2017-2021": "#fdae61",
    "2022-2026": "#d73027",
}

distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")
distrito_ee = geemap.geopandas_to_ee(distrito)

mapa = geemap.Map(basemap="HYBRID")
mapa.centerObject(distrito_ee, 11)
mapa.addLayer(
    distrito_ee.style(color="ffffff", fillColor="00000000", width=2), {}, "Limite distrito Polvora"
)

for periodo_nombre, color in colores.items():
    sub = incendios[incendios["periodo"] == periodo_nombre]
    if len(sub) == 0:
        continue
    fc = geemap.geopandas_to_ee(sub)
    mapa.addLayer(fc.style(color=color, pointSize=4, pointShape="circle"), {}, f"Incendios {periodo_nombre} (n={len(sub)})")

ruta_html = FIGURES_DIR / "mapa_incendios_forestales_reales.html"
mapa.to_html(str(ruta_html))
print(f"Mapa guardado en: {ruta_html}")
