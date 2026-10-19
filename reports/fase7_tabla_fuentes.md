# Fase 7 — Tabla de fuentes (anexo de tesis)

Los 8 indicadores de la matriz de operacionalización, con su fuente exacta.
Verificado contra el servidor/API real el **2026-10-08** (salvo las 4 fuentes
reutilizadas sin cambios de la Fase 5, verificadas originalmente el
2026-09-16 — ver `reports/fase4_disponibilidad_predictores.md`). No se copió
ningún dato de memoria: cada fila fue comprobada con una consulta real (ver
columna "Verificado en").

**Actualización 2026-10-18:** la fila 5 (`clase_mapbiomas`) y la nota de
`pct_bosque_1km` se actualizaron para reflejar la migración de MapBiomas
Perú Colección 3 a **Colección 4** y la extensión del periodo de modelado a
2012-2025 (ver `reports/VIABILIDAD.md`, sección "Migración a Colección 4").
El resto de la tabla (filas 1-4 y 6-8) no cambió.

| # | Variable | Dimensión | Producto y versión | Resolución espacial original | Resolución temporal | URL de acceso | Fecha de acceso | Verificado en |
|---|---|---|---|---|---|---|---|---|
| 1 | `precip_mm` | Climática | CHIRPS Daily (UCSB-CHG) | ~0.05° (~5.5 km) | Diaria, agregada a suma mensual | https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY (asset GEE: `UCSB-CHG/CHIRPS/DAILY`) | 2026-09-16 | `src/fase5_mensuales.py:47-48` |
| 2 | `temp_max_c` | Climática | ERA5-Land Monthly Aggregated, banda `temperature_2m_max` | ~0.1° (~11 km) | Mensual (ya agregado por el proveedor) | https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR (asset GEE: `ECMWF/ERA5_LAND/MONTHLY_AGGR`) | 2026-09-16 | `src/fase5_mensuales.py:51-52` |
| 3 | `elevacion` | Topográfica | ALOS AW3D30 **V4.1**, banda `DSM` | 30 m | Estático (mosaico global, sin serie temporal) | https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1 (asset GEE: `JAXA/ALOS/AW3D30/V4_1`) | 2026-09-16 | `src/fase5_estaticos.py:45-53` |
| 4 | `pendiente` | Topográfica | Derivado de ALOS AW3D30 V4.1 (`ee.Terrain.slope()`) | 30 m | Estático | mismo asset que `elevacion` | 2026-09-16 | `src/fase5_estaticos.py:54` |
| 5 | `clase_mapbiomas` | Ambiental | MapBiomas Perú **Colección 4** (`mapbiomas_peru_collection4_coverage_v1`, 1985-2025) — antes Colección 3 (1985-2024), ver nota abajo | 30 m | 1 imagen/año; se usa el año **t−1** respecto al mes predicho | https://peru.mapbiomas.org/descargas/mapas-de-las-colecciones/ (asset GEE: `projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1`) | **2026-10-18** | `src/fase5_anuales_c4.py` |
| 6 | `ndvi` | Ambiental | **MOD13A3.061** (Terra Vegetation Indices Monthly Global 1 km), banda `NDVI`, factor de escala 0.0001, enmascarado con `SummaryQA` (se descartan nieve/hielo y nuboso, códigos 2 y 3) | ~926.6 m (nativo ~1 km) | Mensual (compuesto nativo del producto, no se construyó a partir de un producto diario) | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3 (asset GEE: `MODIS/061/MOD13A3`) | 2026-10-08 | `src/fase7_ndvi_mensual.py:46-58` |
| 7 | `dist_carretera_m` | Antropogénica | Ministerio de Transportes y Comunicaciones (MTC), **Red Vial Nacional**, corte **diciembre 2018** (última publicada en el portal al 2026-10-08). Filtrado a la ruta `CODRUTA = PE-5N` (Carretera Marginal de la Selva, tramo norte — el campo `TRAYECTO` del propio shapefile la describe pasando por Tocache) | Vectorial (líneas; sin resolución raster) | Estático — corte único dic-2018, tratado como constante en 2012-2024 (misma limitación ya documentada para vías de OSM en la Fase 5) | https://portal.mtc.gob.pe/estadisticas/files/shapefiles/red_vial_nacional_dic18.zip | 2026-10-08 | `src/fase7_vias_distancia.py` |
| 8 | `dist_zona_residencial_m` | Antropogénica | Instituto Nacional de Estadística e Informática (INEI), capa de **Centros Poblados** (`ig_centropoblado`), servicio WFS propio del INEI (atributo `fuente = "INEI - CPV RESULTADOS"`, Censos de Población y Vivienda) | Vectorial (puntos; sin resolución raster) | Estático — snapshot del servicio al momento de la consulta | https://geoespacial.inei.gob.pe/Interoperabilidad/ig_centropoblado/wfs (capa `Interoperabilidad:ig_centropoblado`) | 2026-10-08 | `src/fase7_centros_poblados.py` |

## Nota — variable de filtro (no es uno de los 8 indicadores)

`pct_bosque_1km` (MapBiomas Perú Colección 4, mismo asset que `clase_mapbiomas`,
30 m, año t−1) se usa **únicamente** para filtrar la población a celdas-mes con
≥20% de bosque; no entra a la tabla final como columna predictora
(`src/fase7_construir_tabla_c4.py`, filtro `UMBRAL_BOSQUE`). La versión C3
(`src/fase7_construir_tabla.py`) sigue intacta como referencia, sin cambios.

## Nota — migración de Colección 3 a Colección 4 (MapBiomas)

- **Acceso al asset**: la carpeta `projects/mapbiomas-public/assets/peru/collection4`
  existe en Earth Engine pero su *listado* está bloqueado por permisos (a
  diferencia de las carpetas de C1/C2/C3, que sí se pueden listar). El asset de
  cobertura en sí **sí es accesible por ID directo**:
  `projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1`
  (41 bandas, `classification_1985` a `classification_2025`), confirmado el 2026-10-18.
- **Leyenda verificada**: se descargó el archivo oficial `MapBiomasPeru_legend.csv`
  (https://peru.mapbiomas.org/wp-content/uploads/sites/6/2026/08/MapBiomasPeru_legend.csv,
  2026-10-18, guardado en `data/raw/mapbiomas_c4/`). Los códigos de bosque
  (3=Bosque, 4=Bosque seco, 5=Manglar, 6=Bosque inundable) y de plantación forestal
  (9, bajo "Área agropecuaria") son **idénticos** a los de la Colección 3 — no
  cambió ningún código de clasificación.

## Cómo se encontraron las fuentes 7 y 8 (trazabilidad para la tesis)

- **MTC**: la página de descargas oficial (`portal.mtc.gob.pe/estadisticas/descarga.html`)
  publica un enlace directo al shapefile de la Red Vial Nacional. Antes de usarlo se
  verificó, por dos vías independientes, que el código `PE-5N` corresponde a la
  Marginal: (a) el campo `TRAYECTO` del shapefile describe la ruta pasando por
  "Tocache"; (b) la distancia geométrica mínima entre la geometría de `PE-5N` y el
  polígono del distrito de Pólvora es 0 m (la vía cruza el distrito).
- **INEI**: `sige.inei.gob.pe/test/atlas/` (sugerido en el encargo) es un visor web
  antiguo (OpenLayers + jQuery) sin enlace de descarga directa localizable en un
  tiempo razonable. En su lugar se usó el catálogo oficial de **GeoIDEP**
  (`geoidep.gob.pe/catalogo-nacional-de-servicios-web/servicio-de-descarga-wfs`),
  que lista el servicio WFS que el propio INEI publica en su servidor
  (`geoespacial.inei.gob.pe`) — así que la fuente final sigue siendo el INEI
  directamente, no un intermediario.
