# Bitácora de trabajo

Registro cronológico de los pasos ejecutados para completar el sistema (MapBiomas C4, 2012-2025).
Las duraciones son las medidas al correr cada script. Las horas marcadas con ~ son aproximadas (scripts corridos en segundo plano).
Las salidas completas de cada ejecución están en `resultados/log_*.txt`.

| Fecha y hora | Paso | Qué se corrió | Duración | Resultado |
|---|---|---|---|---|
| 2026-10-21 21:40 | Estadísticas VIIRS (sesión previa) | script auxiliar sobre `firms_distrito_c4.gpkg` | ~1 min | 4 424 detecciones VIIRS 2012-2025; 1 455 celdas-mes con incendio; guardado `data/processed/estadisticas_viirs_polvora.csv` |
| 2026-10-21 22:21 | Revisión del repositorio | inspección de `src/`, `data/processed/`, `reports/` | ~5 min | Pipeline C4 corrido hasta `fase7_modelo_c4.py`; faltaban mapas mensuales y visor. Diferencias con la metodología anotadas en RESULTADOS_PRELIMINARES.md, sección 1 |
| 2026-10-21 22:22 | Parte 1: tabla final y descriptivo | `python src/fase8_tabla_final.py` | 3 s | Verificado: 351 732 celdas-mes, 1 455 con incendio. NDVI reimputado con promedio celda+mes 2012-2021 (51 733) y respaldo promedio del mes (3 482) |
| ~2026-10-21 22:23 | Parte 2: modelado | `python src/fase8_modelo.py` | 95 s | Prueba temporal: RF AUC-ROC 0.8506 / AUC-PR 0.0177; XGB 0.9003 / 0.0223; azar 0.5 / 0.0027. Espacial (media ± desv.): RF 0.8912 ± 0.0199 / 0.0392 ± 0.0041; XGB 0.9063 ± 0.0204 / 0.0396 ± 0.0058; azar 0.5 / 0.0041 ± 0.0010. Modelo final: XGBoost |
| ~2026-10-21 22:25 | Parte 3: interpretación e hipótesis | `python src/fase8_interpretacion.py` | 268 s | HG: AUC-ROC 0.9003, IC95 [0.8883, 0.9122]. SHAP: topográfica 47.3 %, climática 23.8 %, antropogénica 20.0 %, ambiental 9.0 %. Permutación (XGBoost): HE1-HE4 con p = 0.0099; con RF, HE3 no significativa (p = 0.73) |
| ~2026-10-21 22:26 | Ajuste de figura | regeneración de `fig_tratamiento_vs_control.png` desde su CSV | 5 s | Barras de error solo en la validación espacial; leyenda debajo |
| ~2026-10-21 22:27 | Parte 5: visor (primera versión) | escritura de `visor/index.html`; copia local de Leaflet 1.9.4 en `visor/lib/` | — | Visor sin servidor (se abre con doble clic) |
| ~2026-10-21 22:31 | Parte 4: mapas (intento 1) | `python src/fase8_mapas.py` | 73 s | Error al exportar el límite del distrito a JSON (columna de fecha no serializable). No se escribió `datos.js` |
| 2026-10-21 23:09 | Parte 4: mapas (intento 2) | `python src/fase8_mapas.py` tras exportar solo la geometría del distrito | 81 s | 12 mapas; cortes de Jenks 0.0760 / 0.2531 / 0.4838 / 0.7217; `visor/datos.js` (2,4 MB) |
| 2026-10-21 23:10 | Prueba del visor | Edge sin interfaz: captura del visor y de un clic simulado en la celda c01053 | ~1 min | Mapa, leyenda, selector de mes y panel con los 3 factores SHAP funcionan. Se agregaron los íconos de Leaflet que faltaban, tildes en las dimensiones y una nota sobre la escala de la probabilidad |
| 2026-10-21 23:12 | Partes 6 y 7: documentación | redacción de RESULTADOS_PRELIMINARES.md y GUIA_EXPLICADA.md | — | Ver esos archivos |
| 2026-10-23 19:20 | Corrección del mapa de fondo del visor | edición de `visor/index.html`; prueba en Edge sin interfaz | ~5 min | OpenStreetMap devolvía 403 al abrir el visor como archivo local y CARTO pedía clave. Se cambió a capas de Esri (topográfico por defecto, calles e imagen satelital); verificado que cargan |
| 2026-10-23 19:40 | Rediseño del visor | reescritura de `visor/index.html`; prueba en Edge sin interfaz | ~20 min | Mapa a pantalla completa con paneles flotantes; línea de tiempo de 12 meses con reproducción automática; indicadores del mes; leyenda con barra de proporciones; selector de fondo (relieve, calles, satélite); control de opacidad; panel de celda con barras SHAP divergentes y evolución anual |
