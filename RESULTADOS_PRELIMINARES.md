# Resultados preliminares: susceptibilidad a incendios forestales en el distrito de Pólvora

MapBiomas Perú Colección 4 · periodo 2012-2025 · grilla de 1 km (2 263 celdas) · semilla 42
Fecha de cierre: 2026-10-21

Contenido:
1. Estado del repositorio al iniciar
2. Lo que se hizo en esta sesión
3. Datos y análisis descriptivo
4. Tratamiento vs. control (desempeño de los modelos)
5. Interpretación (SHAP y permutación) y decisión sobre las hipótesis
6. Mapas de susceptibilidad
7. Visor web: cómo abrirlo
8. Gráficos principales
9. Errores y problemas encontrados
10. Lo que convendría pulir

---

## 1. Estado del repositorio al iniciar (2026-10-21 22:21)

### Ya estaba hecho

| Componente | Script | Salida | Estado |
|---|---|---|---|
| Área de estudio (Pólvora) | `fase1_area_estudio.py` | `area_estudio.gpkg` | Hecho |
| Descarga FIRMS (MODIS_SP, VIIRS_SNPP_SP) | `fase2_firms.py` | `data/raw/firms/*_peru_bruto.csv` (hasta 2026-06-30) | Hecho |
| Recorte al distrito, corte 2025-12-31 | `fase2_procesar_firms_c4.py` | `firms_distrito_c4.gpkg` (4 424 VIIRS) | Hecho |
| Clasificación bosque t-1 / t con MapBiomas C4 | `fase3_filtro_bosque_c4.py` | `firms_clasificado_c4.gpkg` | Hecho |
| Verificación t+1, t+2 y versión estricta | `fase3b_reclasificar_c4.py` | `fase3b_incendios_depurados_c4.*` | Hecho |
| Agrupación en eventos (750 m, 5 días) | `fase3b_eventos_c4.py` | `fase3b_eventos_c4.csv` | Hecho |
| Grilla 1 km (2 263 celdas) | `fase5_grilla.py` | `grilla_1km.gpkg` | Hecho |
| Predictores anuales C4 (clase, % bosque) | `fase5_anuales_c4.py` | `fase5_predictores_anuales_c4.csv` (2011-2024) | Hecho |
| Clima 2025 (CHIRPS, ERA5-Land) | `fase5_mensuales_c4.py` | `fase5_predictores_mensuales_2025_c4.csv` | Hecho |
| NDVI 2025 | `fase7_ndvi_mensual_c4.py` | `fase7_ndvi_mensual_2025_c4.csv` | Hecho |
| Tabla de modelado (8 indicadores, bosque >= 20 %) | `fase7_construir_tabla_c4.py` | `fase7_tabla_modelado_c4.parquet`: 351 732 filas, 1 455 positivos | Hecho |
| RF y XGBoost, validación temporal y espacial, SHAP, permutación | `fase7_modelo_c4.py` | `fase7_resumen_metricas_c4.csv` y otros `fase7_*_c4.csv` | Hecho, con diferencias respecto a la metodología (ver abajo) |

### Diferencias entre lo que había y la metodología

1. **Imputación de NDVI:** `fase7_construir_tabla_c4.py` usaba la **mediana** histórica de la celda y el mes (2012-2021); la metodología pide el **promedio**.
2. **Validación espacial:** `fase7_modelo_c4.py` reportaba una sola métrica, calculada sobre todas las predicciones fuera de pliegue juntas; la metodología pide la media y la desviación estándar de los 5 pliegues.
3. **SHAP:** se calculaba con un XGBoost entrenado con **toda** la tabla y sobre una muestra de toda la tabla; la metodología pide SHAP del **modelo final** sobre 15 000 registros de **prueba**.
4. **Modelo final:** no había una regla de selección; la metodología pide el de mayor AUC-PR en la validación espacial.
5. **No existían:** la tabla tratamiento vs. control, la prueba de la HG, la importancia SHAP por indicador (sumando las variables indicadoras de la cobertura), los mapas mensuales y el visor.

---

## 2. Lo que se hizo en esta sesión

No se modificó ni se borró ningún script ni dato existente. Todo lo nuevo está en scripts `fase8_*.py`.

| Script nuevo | Qué hace | Por qué es una copia y no una edición |
|---|---|---|
| `src/fase8_tabla_final.py` | Verifica 351 732 / 1 455, reimputa el NDVI con el **promedio** celda+mes 2012-2021 y calcula el análisis descriptivo | Reemplaza la imputación por mediana de `fase7_construir_tabla_c4.py`. Parte de su parquet, que conserva la marca `ndvi_imputado`, así que no hace falta volver a consultar Earth Engine |
| `src/fase8_modelo.py` | RF y XGBoost vs. azar, prueba temporal, 5 pliegues espaciales con media y desviación, elección del modelo final | Corrige los puntos 2 y 4. Importa `construir_modelos` de `fase7_modelo_c4.py`, así que los hiperparámetros son exactamente los mismos |
| `src/fase8_interpretacion.py` | SHAP del modelo final sobre 15 000 registros de prueba, permutación agrupada (100 iteraciones), prueba de la HG | Corrige el punto 3 y agrega la HG |
| `src/fase8_mapas.py` | 12 mapas mensuales, 5 niveles (Jenks), SHAP por celda y datos del visor | Nuevo |
| `visor/index.html` | Visor web | Nuevo |

Otros cambios de detalle respecto a `fase7_modelo_c4.py`:
- En la validación espacial, el peso de clase de cada pliegue se calcula solo con los datos de entrenamiento de ese pliegue. Antes se calculaba con toda la tabla, lo que deja pasar un poco de información de la parte de prueba.
- La permutación agrupada también reporta la caída del AUC-PR, además de la del AUC-ROC.
- Se agregó una referencia de azar empírica, con puntajes aleatorios uniformes, para comprobar la teórica.

Supuesto que conviene confirmar: la correspondencia **HE1 = climática, HE2 = topográfica, HE3 = ambiental, HE4 = antropogénica** sigue el orden en que se listan las dimensiones en el encargo; no encontré las hipótesis redactadas en el repositorio.

---

## 3. Datos y análisis descriptivo

### 3.1 Población de estudio

| Concepto | Valor |
|---|---|
| Celdas de la grilla | 2 263 (1 km × 1 km) |
| Celdas-mes con bosque >= 20 % en el año anterior (2012-2025) | **351 732** ✔ |
| Celdas-mes con incendio forestal | **1 455** ✔ (0,414 %) |
| Entrenamiento 2012-2021 | 251 244 celdas-mes, 1 185 con incendio |
| Prueba 2022-2025 | 100 488 celdas-mes, 270 con incendio (0,269 %) |

### 3.2 Imputación del NDVI

- Hay 55 215 celdas-mes sin NDVI (15,7 %), casi todas en la época de lluvias por la nubosidad: de febrero a abril faltan unas 10 000 por mes, y de agosto a septiembre menos de 500.
- 51 733 se imputaron con el promedio de la misma celda y el mismo mes en 2012-2021.
- 3 482 no tenían ningún valor válido de esa celda y ese mes en 2012-2021; para ellas se usó el promedio de ese mes en todas las celdas.
- Frente a la mediana usada antes, el valor cambia en promedio 0,015 de NDVI (máximo 0,13).
- Ejemplo: la celda c01053 en marzo de 2012 no tiene NDVI. Sus valores de marzo entre 2013 y 2021 son 0,859; 0,840; 0,860; 0,833; 0,880; 0,860; 0,734; 0,857 y 0,814. El promedio es **0,8375** (la mediana era 0,8565).

### 3.3 Los 8 indicadores (351 732 celdas-mes)

| Dimensión | Indicador | Media | Desv. estándar | Mínimo | Máximo |
|---|---|---|---|---|---|
| Climática | Precipitación del mes anterior (mm) | 133,9 | 85,1 | 11,0 | 554,8 |
| Climática | Temperatura máxima del mes anterior (°C) | 25,3 | 4,2 | 13,8 | 43,8 |
| Topográfica | Elevación (m) | 1 445,3 | 742,5 | 420,5 | 3 658,4 |
| Topográfica | Pendiente (°) | 26,4 | 7,7 | 3,7 | 43,7 |
| Ambiental | NDVI del mes anterior | 0,822 | 0,072 | 0,115 | 0,995 |
| Ambiental | Cobertura MapBiomas (año anterior) | categórica (ver abajo) | | | |
| Antropogénica | Distancia a carretera (m) | 18 522,9 | 12 558,1 | 0,6 | 45 863,9 |
| Antropogénica | Distancia a centro poblado (m) | 7 465,9 | 5 474,6 | 128,2 | 24 327,8 |

La cobertura es una variable categórica, así que su media y su desviación no tienen sentido. Su frecuencia es: bosque (3) 334 536 (95,1 %); mosaico agropecuario (21) 12 576; pasto (15) 2 112; formación herbácea (12) 1 572; palma aceitera (35) 348; río (33) 324; playa (23) 168; matorral (66) 96.

### 3.4 Celdas-mes con incendio por año y por mes

| Año | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Con incendio | 169 | 102 | 96 | 148 | 155 | 78 | 106 | 119 | 149 | 63 | 80 | 60 | 71 | 59 |
| % de celdas-mes | 0,67 | 0,40 | 0,38 | 0,59 | 0,62 | 0,31 | 0,42 | 0,47 | 0,59 | 0,25 | 0,32 | 0,24 | 0,28 | 0,23 |

| Mes | Ene | Feb | Mar | Abr | May | Jun | Jul | Ago | Sep | Oct | Nov | Dic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Con incendio | 20 | 10 | 5 | 16 | 11 | 54 | 119 | **529** | **443** | 162 | 66 | 20 |
| % del total | 1,4 | 0,7 | 0,3 | 1,1 | 0,8 | 3,7 | 8,2 | **36,4** | **30,4** | 11,1 | 4,5 | 1,4 |

- Agosto y septiembre concentran el 66,8 % de las celdas-mes con incendio; de julio a octubre, el 86,1 %.
- Según CHIRPS, julio (42 mm) y agosto (48 mm) son los meses más secos (ver `data/processed/estadisticas_viirs_polvora.csv`).
- Los años del periodo de prueba (2022-2025) tienen menos incendios que el promedio de entrenamiento: 0,27 % frente a 0,47 %. Esto explica en parte por qué el AUC-PR es menor en la prueba temporal (sección 4).

---

## 4. Tratamiento vs. control

Las condiciones son las mismas para los tres: la misma tabla, la misma partición temporal, los mismos 116 bloques espaciales de 5 × 5 km repartidos en 5 pliegues (GroupKFold) y la semilla 42. Los modelos usan pesos de clase (RF `class_weight="balanced"`; XGBoost `scale_pos_weight` = negativos/positivos) y no se optimizan sus hiperparámetros.

**Referencias del azar (control):**
- AUC-ROC = 0,5.
- AUC-PR = proporción de positivos (prevalencia): **0,0027** en la prueba 2022-2025 y **0,0041 ± 0,0010** en promedio en los pliegues espaciales. En toda la tabla es 0,00414.
- Un clasificador con puntajes aleatorios dio 0,504 y 0,0028 en la prueba, prácticamente igual a la referencia teórica.

| Tratamiento | Métrica | Prueba temporal 2022-2025 | Validación espacial (media ± desv., 5 pliegues) | Mejora sobre el azar (prueba / espacial) |
|---|---|---|---|---|
| Random Forest | AUC-ROC | 0,8506 | 0,8912 ± 0,0199 | +0,35 / +0,39 |
| Random Forest | AUC-PR | 0,0177 | 0,0392 ± 0,0041 | 6,6× / 9,6× |
| **XGBoost** | AUC-ROC | **0,9003** | **0,9063 ± 0,0204** | +0,40 / +0,41 |
| **XGBoost** | AUC-PR | **0,0223** | **0,0396 ± 0,0058** | 8,3× / 9,7× |
| Azar (control) | AUC-ROC | 0,5000 | 0,5000 | — |
| Azar (control) | AUC-PR | 0,0027 | 0,0041 ± 0,0010 | — |

AUC-ROC y AUC-PR de cada pliegue espacial:

| Pliegue | Positivos | RF AUC-ROC | XGB AUC-ROC | RF AUC-PR | XGB AUC-PR | Azar AUC-PR |
|---|---|---|---|---|---|---|
| 1 | 234 | 0,9062 | 0,9271 | 0,0448 | 0,0485 | 0,0033 |
| 2 | 375 | 0,8635 | 0,8910 | 0,0356 | 0,0353 | 0,0053 |
| 3 | 344 | 0,8813 | 0,8869 | 0,0393 | 0,0377 | 0,0049 |
| 4 | 216 | 0,9134 | 0,9296 | 0,0414 | 0,0420 | 0,0031 |
| 5 | 286 | 0,8913 | 0,8969 | 0,0351 | 0,0346 | 0,0041 |

**Modelo final: XGBoost**, por tener el mayor AUC-PR medio en la validación espacial (0,0396 frente a 0,0392). La diferencia con Random Forest es mínima y está muy por debajo de la desviación entre pliegues: en AUC-PR espacial los dos modelos están prácticamente empatados. XGBoost sí es claramente mejor en la prueba temporal (AUC-ROC 0,900 frente a 0,851; AUC-PR 0,022 frente a 0,018) y en AUC-ROC espacial, así que la elección es coherente con todas las métricas.

**Cómo leer el AUC-PR:** un valor de 0,022 parece bajo, pero el azar obtiene 0,0027, así que el modelo es unas 8 veces más preciso que elegir celdas al azar. Con incendios tan raros (1 de cada 370 celdas-mes en la prueba) no se espera un AUC-PR alto: el modelo sirve para **ordenar** las celdas según su riesgo, no para predecir con certeza cuál se quemará.

---

## 5. Interpretación y decisión sobre las hipótesis

### 5.1 SHAP (XGBoost, 15 000 registros de prueba elegidos al azar con semilla 42; 46 con incendio)

| Puesto | Indicador | Dimensión | SHAP medio (abs.) | % |
|---|---|---|---|---|
| 1 | Elevación | Topográfica | 3,551 | 37,4 |
| 2 | Precipitación | Climática | 1,567 | 16,5 |
| 3 | Distancia a centro poblado | Antropogénica | 1,387 | 14,6 |
| 4 | Pendiente | Topográfica | 0,934 | 9,8 |
| 5 | NDVI | Ambiental | 0,763 | 8,0 |
| 6 | Temperatura máxima | Climática | 0,689 | 7,3 |
| 7 | Distancia a carretera | Antropogénica | 0,507 | 5,3 |
| 8 | Cobertura MapBiomas | Ambiental | 0,088 | 0,9 |

| Dimensión | % de la importancia SHAP |
|---|---|
| Topográfica | 47,3 |
| Climática | 23,8 |
| Antropogénica | 20,0 |
| Ambiental | 9,0 |

Lo que muestra el gráfico de puntos (`resultados/fig_shap_beeswarm.png`):
- A **menor elevación**, mayor susceptibilidad. Los 1 455 incendios ocurrieron entre 420 y 1 656 m (mediana 659 m). Ninguno ocurrió por encima de 2 000 m, aunque ahí está el 23 % de las celdas-mes (las zonas altas del oeste).
- Un **mes anterior con poca lluvia** aumenta la susceptibilidad.
- **Estar cerca de centros poblados** aumenta la susceptibilidad.
- Las **pendientes suaves** aumentan la susceptibilidad.

Los valores SHAP de XGBoost están en escala logit, es decir, el logaritmo de las probabilidades a favor; un valor positivo empuja la predicción hacia "incendio".

### 5.2 Permutación agrupada por dimensión (prueba 2022-2025, 100 iteraciones)

Para cada dimensión se desordenan juntas todas sus columnas y se mide cuánto cae el AUC. El valor p es (1 + número de permutaciones con AUC >= AUC original) / 101; el mínimo posible con 100 iteraciones es **0,0099**.

**Modelo final (XGBoost), AUC-ROC original 0,9003:**

| Hipótesis | Dimensión | Reducción del AUC-ROC (media ± desv.) | p | Reducción del AUC-PR | p (AUC-PR) | Decisión (α = 0,05) |
|---|---|---|---|---|---|---|
| HE1 | Climática | 0,1052 ± 0,0074 | 0,0099 | 0,0142 | 0,0099 | Se acepta |
| HE2 | Topográfica | 0,2396 ± 0,0136 | 0,0099 | 0,0139 | 0,0099 | Se acepta |
| HE3 | Ambiental | 0,0160 ± 0,0033 | 0,0099 | 0,0041 | 0,0198 | Se acepta |
| HE4 | Antropogénica | 0,0670 ± 0,0078 | 0,0099 | 0,0075 | 0,0099 | Se acepta |

**Comprobación con Random Forest** (no es el modelo final, pero muestra si el resultado es robusto):

| Hipótesis | Reducción AUC-ROC | p |
|---|---|---|
| HE1 climática | 0,1244 | 0,0099 |
| HE2 topográfica | 0,1323 | 0,0099 |
| HE3 ambiental | −0,0042 | **0,7327** (no significativa) |
| HE4 antropogénica | 0,0307 | 0,0099 |

### 5.3 Hipótesis general (HG)

| Contraste | Resultado | Decisión |
|---|---|---|
| AUC-ROC del modelo final en la prueba 2022-2025 > 0,5 | 0,9003; IC 95 % bootstrap (1 000 remuestreos) [0,8883; 0,9122]; prueba U de Mann-Whitney unilateral p < 0,001 | **Se acepta la HG**: el modelo supera claramente al azar |

### 5.4 Síntesis para el alcance explicativo

- **La dimensión que más explica la susceptibilidad es la topográfica.** Lo dicen las dos técnicas: SHAP le asigna el 47 % y es la que más reduce el AUC al permutarse (−0,24). Dentro de ella pesa sobre todo la **elevación**: los incendios se concentran en las zonas bajas del este del distrito, cerca del valle del Huallaga.
- **La dimensión climática es la segunda** (24 % en SHAP; −0,11 de AUC). Su indicador principal es la **precipitación del mes anterior**, y es la que produce la variación **estacional**: un mismo lugar pasa de "muy baja" en marzo a "muy alta" en agosto.
- **La dimensión antropogénica es la tercera** (20 %; −0,07). Pesa más la **cercanía a centros poblados** que la distancia a carreteras, lo que es coherente con quemas que se escapan cerca de las zonas habitadas.
- **La dimensión ambiental es la más débil** (9 %; −0,016). Su efecto es estadísticamente significativo con XGBoost, pero pequeño y **no robusto**: con Random Forest no es significativo. Una razón probable es que el filtro de bosque >= 20 % deja una cobertura casi constante (95 % de las celdas-mes son bosque) y un NDVI poco variable (0,82 ± 0,07). Al haber poca variación, estos indicadores tienen poco que aportar.
- La caída del AUC al permutar y la importancia SHAP ordenan las cuatro dimensiones igual (topográfica > climática > antropogénica > ambiental). Que dos métodos distintos coincidan respalda la conclusión.

---

## 6. Mapas de susceptibilidad

- **Datos de entrada:** un mapa por mes (enero a diciembre) para las 2 263 celdas. En cada mes se usa la climatología de los predictores: el promedio 2012-2025 de la precipitación, la temperatura máxima y el NDVI **del mes anterior**, con el mismo desfase que en la tabla de modelado. La cobertura y el % de bosque son de 2024, y los demás indicadores son fijos.
- **Modelo:** XGBoost reentrenado con toda la tabla 2012-2025 y los mismos hiperparámetros. El modelo evaluado en la sección 4 se entrenó solo con 2012-2021; para el producto final conviene aprovechar todos los años.
- **Celdas fuera del dominio:** 168 celdas tienen menos de 20 % de bosque en 2024. El modelo no se entrenó con ese tipo de celdas, así que su valor es una extrapolación. Se muestran rayadas y no se usaron para calcular los cortes.
- **Clases de cobertura sin ejemplos:** una clase de 2024 (24, infraestructura urbana) no aparece en la tabla de entrenamiento; esas celdas quedan con todas las indicadoras de cobertura en 0.

**Método de cortes: cortes naturales de Jenks**, 5 clases, calculados **una sola vez** con las 25 140 celdas-mes del dominio de los 12 meses juntos.

| Nivel | Probabilidad del modelo |
|---|---|
| Muy baja | < 0,076 |
| Baja | 0,076 – 0,253 |
| Media | 0,253 – 0,484 |
| Alta | 0,484 – 0,722 |
| Muy alta | > 0,722 |

Por qué Jenks:
- Agrupa los valores en las clases más homogéneas posibles por dentro y más distintas entre sí. Es el método más usado en mapas de susceptibilidad.
- La alternativa de quintiles (20 % de celdas en cada nivel) **no funciona con estos datos**: más del 40 % de las celdas-mes tienen probabilidad prácticamente 0, así que los dos primeros cortes por quintiles salían iguales (0,000 y 0,000).
- Los intervalos iguales tampoco sirven, porque casi todas las celdas caerían en el primero.
- Se usan los mismos cortes para los 12 meses, de modo que "muy alta" significa lo mismo en marzo que en agosto y se puede comparar la estacionalidad.

**Importante:** como el modelo se entrena con pesos de clase, la "probabilidad" está inflada (en agosto llega a 0,95) y no es la probabilidad real de que ocurra un incendio, que en promedio es 0,4 %. Sirve para **ordenar y comparar** celdas y meses.

Celdas por nivel y mes (incluye las 168 celdas fuera del dominio):

| Mes | Muy baja | Baja | Media | Alta | Muy alta |
|---|---|---|---|---|---|
| Enero | 2 027 | 194 | 35 | 7 | 0 |
| Febrero | 1 965 | 205 | 67 | 24 | 2 |
| Marzo | 1 988 | 210 | 55 | 9 | 1 |
| Abril | 2 200 | 57 | 6 | 0 | 0 |
| Mayo | 1 975 | 213 | 55 | 18 | 2 |
| Junio | 1 512 | 167 | 185 | 236 | 163 |
| Julio | 1 439 | 229 | 163 | 181 | 251 |
| Agosto | 1 424 | 199 | 127 | 133 | **380** |
| Septiembre | 1 436 | 187 | 170 | 191 | 279 |
| Octubre | 1 531 | 189 | 143 | 196 | 204 |
| Noviembre | 1 895 | 220 | 94 | 46 | 8 |
| Diciembre | 2 025 | 191 | 37 | 6 | 4 |

**Concordancia con los incendios registrados** (es descriptiva, no una validación: el modelo del mapa vio esos años). En las celdas-mes del dominio, la proporción con al menos un incendio en 2012-2025 crece con el nivel: muy baja 0,3 %, baja 6,3 %, media 11,9 %, alta 23,9 %, muy alta 70,5 %. El nivel "muy alta" reúne 907 de los 1 451 incendios del dominio (63 %) en solo el 5 % de las celdas-mes.

---

## 7. Visor web: cómo abrirlo en tu PC

1. Abre la carpeta `C:\Users\alexi\Documents\proyecto_prediccion\visor\`.
2. Haz doble clic en **`index.html`**. Se abre en el navegador (Edge, Chrome o Firefox) y no necesita servidor ni instalar nada.
3. Uso:
   - **Panel izquierdo:** muestra el mes elegido. Para cambiarlo, haz clic en las barras de la línea de tiempo (su altura es el número de celdas en riesgo alto o muy alto), usa los botones ◀ ▶▶ o las flechas ← → del teclado. El botón ▶, o la barra espaciadora, recorre los 12 meses automáticamente.
   - **Indicadores del mes:** celdas en riesgo alto o muy alto, su porcentaje y las celdas-mes con incendio registrado.
   - **Leyenda:** los 5 niveles con sus rangos, cuántas celdas hay en cada uno y una barra con sus proporciones.
   - **Capas:** fondo de relieve, calles o satélite; control de opacidad de las celdas para ver el terreno debajo; casilla para resaltar en morado las celdas que se quemaron en ese mes; casilla para mostrar u ocultar las celdas fuera del dominio.
   - **Clic en una celda:** se abre el panel derecho con su índice y su nivel, los **3 factores SHAP** como barras (hacia la derecha si aumentan el riesgo, hacia la izquierda si lo reducen), su índice en cada mes del año (con un punto morado en los meses con incendio registrado) y sus datos. Esc lo cierra.
4. Leaflet, la biblioteca del mapa, está copiada en `visor/lib/`, así que el visor funciona sin internet. Lo único que necesita conexión es el **mapa de fondo**; sin internet, las celdas se ven sobre fondo gris.
5. Para regenerar los datos del visor: `python src/fase8_mapas.py`, que reescribe `visor/datos.js`.
6. Para llevarlo a otra PC, basta copiar la carpeta `visor/` completa.

---

## 8. Gráficos principales (carpeta `resultados/`)

| Archivo | Contenido |
|---|---|
| `fig_incendios_por_anio.png` | Celdas-mes con incendio por año |
| `fig_incendios_por_mes.png` | Celdas-mes con incendio por mes |
| `fig_tratamiento_vs_control.png` | AUC-ROC y AUC-PR de RF, XGBoost y el azar |
| `fig_curvas_roc_pr_prueba.png` | Curvas ROC y precisión-exhaustividad en la prueba 2022-2025 |
| `fig_shap_indicador.png` | Importancia SHAP de los 8 indicadores |
| `fig_shap_dimension.png` | Importancia SHAP por dimensión |
| `fig_shap_beeswarm.png` | Valores SHAP de cada registro (dirección del efecto) |
| `fig_permutacion_dimension.png` | Reducción del AUC al permutar cada dimensión (HE1-HE4) |
| `fig_mapas_susceptibilidad_12meses.png` | Los 12 mapas mensuales |

Las tablas en CSV están en la misma carpeta: `descriptivo_indicadores.csv`, `incendios_por_anio.csv`, `incendios_por_mes.csv`, `tabla_tratamiento_vs_control.csv`, `metricas_pliegues_espaciales.csv`, `shap_importancia_*.csv`, `permutacion_dimension.csv`, `hipotesis.csv`, `cortes_niveles_susceptibilidad.csv` y `niveles_por_mes.csv`. Las salidas completas de consola están en `log_*.txt`.

![Tratamiento vs. control](resultados/fig_tratamiento_vs_control.png)
![SHAP por indicador](resultados/fig_shap_indicador.png)
![SHAP beeswarm](resultados/fig_shap_beeswarm.png)
![Permutación por dimensión](resultados/fig_permutacion_dimension.png)
![Mapas mensuales](resultados/fig_mapas_susceptibilidad_12meses.png)

---

## 9. Errores y problemas encontrados

1. **Error al exportar el límite del distrito al visor** (primer intento de `fase8_mapas.py`): la capa del distrito tiene una columna de fecha que no se puede convertir a JSON. Se resolvió exportando solo la geometría y se volvió a correr sin problemas.
2. **Los íconos de Leaflet no se descargaron al principio:** el botón de capas salía en blanco. Se resolvió descargando la carpeta `images/`.
3. **Random Forest asigna probabilidad 0 a muchas celdas-mes:** se ve como un tramo recto en su curva ROC, porque alrededor del 15 % de los incendios de prueba quedan empatados en 0. Esto le quita AUC-ROC en la prueba temporal (0,851). No es un error, sino una consecuencia de usar árboles sin límite de profundidad sin optimizar hiperparámetros.
4. **HE3 (ambiental) no es robusta:** es significativa con XGBoost (p = 0,0099), pero no con Random Forest (p = 0,73). Conviene decirlo en la tesis.
5. **Valor p mínimo de la permutación:** con 100 iteraciones, el p más pequeño posible es 0,0099. Todas las dimensiones significativas llegan a ese mínimo, así que el valor p no permite ordenarlas; para eso se usa el tamaño de la reducción del AUC.
6. **Probabilidades infladas:** los pesos de clase hacen que las probabilidades no estén calibradas (ver sección 6). En la sustentación hay que presentarlas como un índice relativo.
7. **El modelo final se eligió por un margen muy estrecho** (AUC-PR 0,0396 frente a 0,0392). Si el jurado pregunta, la elección se sostiene también por la prueba temporal y el AUC-ROC.
8. **El mapa de fondo de OpenStreetMap salía bloqueado ("Access blocked", error 403) al abrir el visor con doble clic** (lo detectó el usuario el 2026-10-23). OpenStreetMap rechaza las páginas abiertas como archivo local, y CARTO tampoco sirve porque pide una clave. Se cambió a los mapas de Esri (topográfico, calles e imagen satelital), que sí cargan.
9. **Las etiquetas de los ejes del gráfico SHAP de puntos están en inglés** ("SHAP value", "Feature value", "High/Low"), porque las pone la biblioteca `shap`.

---

## 10. Lo que convendría pulir

- [ ] Confirmar que la correspondencia HE1-HE4 con las dimensiones coincide con la redacción de la tesis.
- [ ] Traducir las etiquetas del gráfico SHAP de puntos (`fig_shap_beeswarm.png`).
- [ ] Si el jurado lo pide, calibrar las probabilidades (por ejemplo con regresión isotónica) para mostrar una probabilidad real además del nivel relativo.
- [ ] Aumentar la permutación a 1 000 iteraciones para tener valores p más finos (mínimo 0,001). Con XGBoost tarda unos 12 minutos.
- [ ] Hacer un análisis de sensibilidad de la muestra SHAP: solo tiene 46 incendios. Repetirlo con otra semilla o con todos los positivos de la prueba mostraría si el orden de los indicadores se mantiene.
- [ ] Evaluar si la cobertura MapBiomas aporta algo dentro de un dominio que es 95 % bosque, o si conviene presentarla como una variable de control.
- [ ] Los mapas usan la cobertura de 2024. Cuando salga la de 2025, regenerarlos con `fase8_mapas.py` cambiando `ANIO_USO_SUELO`.
- [ ] Agregar al visor la opción de mostrar la probabilidad continua (degradado) además de los 5 niveles.
- [ ] Repetir el experimento con la definición **estricta** de incendio (`es_estricto`) como análisis de robustez.
