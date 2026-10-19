# Fase 4 — Disponibilidad de predictores

Verificado el 2026-09-16 contra el distrito de Pólvora (provincia Tocache, San Martín).
Todos los asset ID de Earth Engine fueron comprobados con `ee.Image(...).getInfo()` o
`ee.ImageCollection(...).size().getInfo()` (ver `src/fase4_verificar_predictores.py`),
no se copiaron de memoria sin probar.

## Clima

| Predictor | Fuente | Asset / URL | Resolución | Periodo cubierto | Problemas encontrados |
|---|---|---|---|---|---|
| Precipitación | CHIRPS diario | `UCSB-CHG/CHIRPS/DAILY` | ~0.05° (~5.5 km) | 1981-01-01 a 2026-08-31 | Ninguno. |
| Temperatura máxima | ERA5-Land mensual | `ECMWF/ERA5_LAND/MONTHLY_AGGR`, banda `temperature_2m_max` | ~0.1° (~11 km) | 1950-02 a 2026-08 | La colección `ECMWF/ERA5_LAND/MONTHLY` (sin `_AGGR`) está **deprecada** por Google (aviso explícito en la API) y solo llega a 2023-04; se usa la versión `_AGGR` vigente. |
| Déficit de presión de vapor (VPD) | ERA5-Land mensual (derivado) | mismo asset, bandas `temperature_2m` y `dewpoint_temperature_2m` | ~0.1° (~11 km) | 1950-02 a 2026-08 | No es una banda directa: se calcula con la fórmula de Tetens/Magnus a partir de temperatura y punto de rocío (ambas en Kelvin, se convierten a °C). Se implementa en la Fase 5. |
| SPI (1 y 3 meses) | Derivado de CHIRPS | — (cálculo propio) | igual que CHIRPS | igual que CHIRPS | No es un asset: se calcula ajustando una distribución gamma a la precipitación mensual acumulada por celda, estandarizando el percentil resultante. Se implementa en la Fase 5. |
| Días secos consecutivos | Derivado de CHIRPS | — (cálculo propio) | igual que CHIRPS | igual que CHIRPS | Igual que SPI: se calcula, no es un asset. |
| PISCO (SENAMHI) | IRI Data Library, Columbia University | `https://iridl.ldeo.columbia.edu/SOURCES/.SENAMHI/.HSR/.PISCO/` | ~0.1° | 1981–2025 (precipitación) | Responde (HTTP 200), **sí está disponible**, pero se decidió **no integrarlo** en esta fase de viabilidad: exigiría un segundo pipeline de datos fuera de Earth Engine para una ganancia marginal frente a CHIRPS (misma resolución aproximada, cobertura equivalente, y CHIRPS ya corre en el mismo entorno que todo lo demás). Queda documentado como alternativa disponible si el jurado pide validar con datos de estación. |

## Vegetación

| Predictor | Fuente | Asset / URL | Resolución | Periodo cubierto | Problemas encontrados |
|---|---|---|---|---|---|
| NDMI (humedad de la vegetación) | MODIS Surface Reflectance 8-day | `MODIS/061/MOD09A1`, bandas `sur_refl_b02` (NIR) y `sur_refl_b06` (SWIR) | 500 m, compuesto de 8 días | 2000-02-18 a 2026-08-29 | No es una banda directa: NDMI = (NIR−SWIR)/(NIR+SWIR), se calcula en la Fase 5. |

## Uso de suelo

| Predictor | Fuente | Asset / URL | Resolución | Periodo cubierto | Problemas encontrados |
|---|---|---|---|---|---|
| Clase MapBiomas | MapBiomas Perú Colección 3 | `projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1` | 30 m, 1 imagen/año | 1985–2024 | Ya usado y verificado en la Fase 3. No cubre 2025-2026 (se usa 2024 como último año disponible para esos casos, tal como se hizo en la Fase 3). |
| Pérdida de bosque previa (año t-1) | Hansen Global Forest Change v1.13 | `UMD/hansen/global_forest_change_2025_v1_13`, banda `lossyear` | 30 m | pérdida detectada 2001–2025 | Ninguno (ya verificado en el contraste de la Fase 3). |
| % de bosque en 1 km, distancia al borde del bosque, distancia a agricultura/pastos | Derivado de MapBiomas Colección 3 | mismo asset que "Clase MapBiomas" | 30 m (agregado a ventanas de 1 km) | 1985–2024 | No son assets aparte: se calculan con estadística focal (`reduceNeighborhood`) y transformadas de distancia (`fastDistanceTransform`) sobre la clasificación anual de MapBiomas. Se implementan en la Fase 5. |

## Físicos y accesibilidad

| Predictor | Fuente | Asset / URL | Resolución | Periodo cubierto | Problemas encontrados |
|---|---|---|---|---|---|
| Elevación (DEM) | ALOS AW3D30 | `JAXA/ALOS/AW3D30/V4_1`, banda `DSM` | 30 m | Estático (sin serie temporal; mosaico global) | La versión `V3_2` está **deprecada** por Google en favor de `V4_1`; se usa la vigente. |
| Pendiente y orientación | Derivado del DEM | mismo asset | 30 m | Estático | No son bandas directas: se calculan con `ee.Terrain.slope()` / `ee.Terrain.aspect()` sobre el DSM. |
| Distancia a vías, centros poblados y ríos | OpenStreetMap (extracto Perú) | `https://download.geofabrik.de/south-america/peru-latest-free.gpkg.zip` | Datos vectoriales (sin resolución raster fija) | Snapshot actualizado al 2026-09-16 (OSM no tiene series históricas fiables) | Está **fuera de Earth Engine**: es un GeoPackage de ~600 MB que se descarga y procesa localmente con geopandas (filtrar por distrito, calcular distancias). OSM refleja el estado actual del mapeo, no el histórico exacto de cada año pasado — limitación a documentar: las distancias a vías/centros poblados se tratan como estáticas en todo el periodo, lo cual es una simplificación razonable para una prueba de viabilidad pero no sería válido para un estudio que necesite la red vial de, por ejemplo, 2005. |

## Resumen de decisiones

- **Todos los predictores requeridos tienen una fuente verificada y accesible** para el periodo común aproximado 2001–2024 (limitado por MapBiomas, que es el más corto de todos).
- Dos assets de Earth Engine que se iban a usar por defecto están **deprecados** y fueron reemplazados por su sucesor vigente: `ECMWF/ERA5_LAND/MONTHLY` → `ECMWF/ERA5_LAND/MONTHLY_AGGR`; `JAXA/ALOS/AW3D30/V3_2` → `V4_1`.
- **PISCO se deja fuera** de esta fase por decisión de simplicidad (no por falta de acceso), documentado arriba.
- **OSM es un snapshot estático actual**, no histórico — limitación explícita para la sección de limitaciones de la tesis.
