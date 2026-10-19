"""
Ayuda visual (no es parte del pipeline principal): genera un mapa
interactivo en HTML con imagen satelital real (Sentinel-2) y el limite
del distrito de Tocache superpuesto, para verificar visualmente que el
poligono de la Fase 1 cae en el lugar correcto.

Uso: .venv\\Scripts\\python src\\inspeccionar_area_estudio.py
Abre el archivo HTML resultante en el navegador.
"""
import ee
import geemap
import geopandas as gpd

from config import DATA_PROCESSED_DIR, EE_PROJECT_ID, FIGURES_DIR

ee.Initialize(project=EE_PROJECT_ID)

distrito = gpd.read_file(DATA_PROCESSED_DIR / "area_estudio.gpkg", layer="distrito_estudio")
distrito_ee = geemap.geopandas_to_ee(distrito)

# Verificamos que la coleccion exista y responda antes de usarla (regla del
# proyecto: nunca asumir un asset de Earth Engine sin comprobarlo).
coleccion = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
primera = coleccion.first().getInfo()
assert primera is not None, "La coleccion COPERNICUS/S2_SR_HARMONIZED no respondio."

# Mediana de 2025 con poca nubosidad: una sola imagen podria tener nubes
# sobre el distrito, la mediana de varias fechas las filtra en la practica.
s2_mediana = (
    coleccion.filterBounds(distrito_ee)
    .filterDate("2025-01-01", "2025-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .median()
)

vis_rgb = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}

mapa = geemap.Map(basemap="HYBRID")
mapa.centerObject(distrito_ee, 11)
mapa.addLayer(s2_mediana, vis_rgb, "Sentinel-2 mediana 2025 (RGB real)")
mapa.addLayer(
    distrito_ee.style(color="ff0000", fillColor="00000000", width=2),
    {},
    "Limite distrito Polvora",
)

ruta_html = FIGURES_DIR / "fase1_inspeccion_satelital.html"
mapa.to_html(str(ruta_html))
print(f"Mapa interactivo guardado en: {ruta_html}")
