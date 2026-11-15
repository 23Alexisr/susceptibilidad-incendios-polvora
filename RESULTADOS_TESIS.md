# Resultados para la tesis: susceptibilidad a incendios forestales en Pólvora

Guía de estudio de la Fase 9 · corrida del 2026-11-09 · semilla 42 en todo lo aleatorio

Este documento parte de lo que ya estaba hecho en la Fase 8 (ver `RESULTADOS_PRELIMINARES.md` y `GUIA_EXPLICADA.md`) y agrega cinco pasos que faltaban o había que reforzar:

1. Una **regresión logística** como modelo de referencia.
2. La **hipótesis general** contrastada con el azar y con la regresión logística.
3. El **ajuste de hiperparámetros** de Random Forest y XGBoost.
4. La **calibración isotónica** de las probabilidades.
5. La **prueba de permutación con 1 000 repeticiones**.

Todos los resultados nuevos están en `resultados/fase9/`.

---

## 0. Resumen en pocas frases

- Los tres modelos (regresión logística, Random Forest y XGBoost) **predicen mucho mejor que el azar**: AUC-ROC de 0,90 a 0,91 en 2022-2025, frente a 0,5 del azar.
- La **regresión logística, que es el modelo más simple, rinde casi igual** que los modelos de aprendizaje automático. En la prueba 2022-2025, ningún modelo le gana con una diferencia estadísticamente significativa.
- El ajuste de hiperparámetros mejoró a Random Forest y lo dejó como **modelo final**, pero por un margen muy pequeño frente a XGBoost.
- La calibración convierte el "índice" del modelo en una **probabilidad realista** (por ejemplo, 0,90 pasa a 0,05). Sin embargo, como en 2022-2025 hubo menos incendios que antes, esa probabilidad **sobreestima el riesgo** alrededor del doble.
- La permutación con 1 000 repeticiones muestra que **las cuatro dimensiones aportan de forma significativa** (p < 0,05). El orden de importancia es: **topográfica > climática > antropogénica > ambiental**. La ambiental aporta muy poco.

---

## 1. Qué hace cada script y en qué orden

### Lo que ya existía (Fase 8; no se modificó)

| Orden | Script | Qué hace | Tiempo aprox. |
|---|---|---|---|
| 1 | `src/fase8_tabla_final.py` | Verifica la tabla (351 732 celdas-mes, 1 455 con incendio), imputa el NDVI con el promedio y hace el análisis descriptivo | 1 min |
| 2 | `src/fase8_modelo.py` | RF y XGBoost frente al azar, con la prueba temporal y los 5 pliegues espaciales; elige el modelo final de la Fase 8 (XGBoost) | 2 min |
| 3 | `src/fase8_interpretacion.py` | SHAP, permutación con 100 repeticiones e HG de la Fase 8 | 5 min |
| 4 | `src/fase8_mapas.py` | Mapas mensuales y datos del visor | 2 min |
| — | `src/fase8_grafico_viirs.py` | Gráfico de apoyo con las 4 424 detecciones VIIRS (creado el 2026-11-02) | segundos |

### Lo nuevo (Fase 9)

Los scripts **reutilizan** funciones de la Fase 8 y la Fase 7 en lugar de copiarlas: `cargar_datos` y `peso_positivo` (de `fase8_modelo.py`), `construir_modelos` (de `fase7_modelo_c4.py`) y las dimensiones `INDICADORES_POR_DIMENSION` (de `fase8_interpretacion.py`). Así todos usan exactamente los mismos datos, columnas y bloques.

| Orden para volver a correrlos | Script | Qué hace | Tiempo aprox. |
|---|---|---|---|
| — | `src/fase9_comun.py` | No se ejecuta solo: define la carpeta `resultados/fase9/`, separa 2012-2021 de 2022-2025 y arma los 5 pliegues por bloques | — |
| 1 | `src/fase9_logistica.py` | Paso 1: regresión logística y tabla comparativa | 1 min |
| 2 | `src/fase9_ajuste_hiperparametros.py` | Paso 3: búsqueda de hiperparámetros solo con 2012-2021; guarda el modelo final ajustado | 15 min |
| 3 | `src/fase9_calibracion.py` | Paso 4: calibración isotónica; guarda las predicciones del modelo final en 2022-2025 | 1 min |
| 4 | `src/fase9_hipotesis_general.py` | Paso 2: HG con bootstrap. Si ya existen las predicciones del paso 3, también evalúa el modelo ajustado | 2 min |
| 5 | `src/fase9_permutacion.py` | Paso 5: permutación con 1 000 repeticiones; guarda el avance por dimensión | 30 min |

**Nota sobre el orden.** Los pasos se pidieron en el orden 1-2-3-4-5, y así se ejecutaron. La HG (paso 2) se corrió primero con el modelo final de la Fase 8 y **se volvió a correr** después de la calibración para agregar el modelo ajustado. Si repites todo desde cero, usa el orden de la tabla: así la HG sale completa en una sola corrida.

Para correr cualquiera, desde la carpeta del proyecto:

```
.venv\Scripts\python.exe src\fase9_logistica.py
```

La salida de consola de cada uno quedó en `resultados/fase9/log_fase9_*.txt`.

---

## 2. Paso 1: regresión logística de referencia

### Qué hace y por qué

- Una **regresión logística** es el modelo estadístico clásico para una respuesta sí/no. Suma los indicadores, cada uno multiplicado por un peso (coeficiente), y convierte esa suma en una probabilidad.
- Sirve de **referencia**: si un modelo complejo (RF o XGBoost) no supera a uno simple, la complejidad no se justifica por desempeño, aunque puede justificarse por otras razones, como capturar relaciones no lineales o interacciones.
- Usa los **mismos 8 indicadores**:
  - Las 7 variables numéricas se **estandarizan**: a cada una se le resta su media y se divide entre su desviación estándar. Así todas quedan en la misma escala y sus coeficientes se pueden comparar.
  - La cobertura MapBiomas entra como **variables indicadoras** 0/1, como en RF y XGBoost.
- **Pesos de clase balanceados**: cada incendio "pesa" lo mismo que unas 211 celdas sin incendio, igual que en los otros modelos.
- Se evalúa igual que RF y XGBoost: entrenada con 2012-2021 y probada en 2022-2025, y con los mismos 5 pliegues espaciales de la Fase 8.

### Ejemplo

El coeficiente de la elevación es −2,69. Como la elevación está estandarizada, una desviación estándar más de altura (unos 742 m) multiplica las "chances" de incendio por e^(−2,69) = **0,068**, es decir, las reduce unas 15 veces. En la celda c01053 en agosto de 2023, la logística da 0,78 y Random Forest 0,90. Los dos la ven como de riesgo alto.

### Resultados

| Modelo | AUC-ROC prueba 2022-2025 | AUC-PR prueba 2022-2025 | AUC-ROC espacial (media ± desv.) | AUC-PR espacial (media ± desv.) |
|---|---|---|---|---|
| **Regresión logística** | **0,9048** | **0,0228** | **0,913 ± 0,027** | **0,041 ± 0,014** |
| XGBoost (Fase 8) | 0,9003 | 0,0223 | 0,906 ± 0,020 | 0,040 ± 0,006 |
| Random Forest (Fase 8) | 0,8506 | 0,0177 | 0,891 ± 0,020 | 0,039 ± 0,004 |
| Azar | 0,5000 | 0,0027 | 0,500 | 0,004 ± 0,001 |

Por pliegue espacial:

| Pliegue | Logística AUC-ROC | RF AUC-ROC | XGB AUC-ROC | Logística AUC-PR | RF AUC-PR | XGB AUC-PR |
|---|---|---|---|---|---|---|
| 1 | 0,945 | 0,906 | 0,927 | 0,066 | 0,045 | 0,049 |
| 2 | 0,881 | 0,864 | 0,891 | 0,031 | 0,036 | 0,035 |
| 3 | 0,893 | 0,881 | 0,887 | 0,033 | 0,039 | 0,038 |
| 4 | 0,933 | 0,913 | 0,930 | 0,038 | 0,042 | 0,042 |
| 5 | 0,914 | 0,891 | 0,897 | 0,037 | 0,035 | 0,035 |

Coeficientes de la logística, en las variables numéricas estandarizadas (`logistica_coeficientes.csv`):

| Indicador | Coeficiente | Qué significa |
|---|---|---|
| Elevación | −2,69 | A mayor altura, mucho menos riesgo (el efecto más fuerte) |
| Precipitación del mes anterior | −1,47 | Un mes anterior lluvioso reduce el riesgo |
| Distancia a centro poblado | −1,21 | Estar lejos de un poblado reduce el riesgo |
| Distancia a carretera | +0,56 | Estar **lejos** de la carretera aumenta un poco el riesgo (ver nota) |
| Temperatura máxima | +0,22 | Más calor, algo más de riesgo |
| Pendiente | −0,16 | Efecto pequeño |
| NDVI | +0,06 | Casi sin efecto |

Nota: el signo de la distancia a carretera va contra la intuición. Probablemente se debe a que la carretera Marginal pasa por el valle, donde también está la mayoría de los poblados. Una vez que el modelo ya considera la distancia a los poblados y la elevación, la carretera aporta poco y su coeficiente puede cambiar de signo. Es una asociación, no una causa.

### Qué significa

- La regresión logística **iguala o supera ligeramente** a RF y XGBoost en la prueba 2022-2025 y en la validación espacial de la Fase 8.
- Esto indica que la relación entre los indicadores y los incendios es, en buena parte, **simple y monótona**: menos altura, menos lluvia y más cercanía a poblados significan más riesgo. Para eso no hace falta un modelo complejo.
- Es un resultado **honesto e importante** para la tesis. Conviene presentarlo como hallazgo, no esconderlo.
- En la validación espacial **dentro de 2012-2021** (la misma que se usó para ajustar hiperparámetros, paso 3), la logística obtiene un AUC-PR de **0,049**, por debajo de RF ajustado (0,057) y XGBoost ajustado (0,057). Es decir, ajustados, los modelos de árboles sí ordenan algo mejor a los positivos en zonas nuevas. En la prueba 2022-2025, en cambio, esa ventaja desaparece (sección 4).

Archivos: `comparacion_modelos.csv`, `comparacion_pliegues.csv`, `logistica_coeficientes.csv`, `logistica_pliegues.csv`, `logistica_pliegues_2012_2021.csv`, `predicciones_prueba_logistica.parquet`.

---

## 3. Paso 2: hipótesis general (HG)

### Qué hace y por qué

- **HG:** el mejor modelo predice la ocurrencia de incendios mejor que el azar.
- Se compara el **AUC-ROC en 2022-2025** del mejor modelo, es decir, el de mayor AUC-PR espacial:
  - contra el **azar** (0,5);
  - contra la **regresión logística**, para saber si el aprendizaje automático aporta algo sobre el modelo clásico.
- **Bootstrap pareado y estratificado, con 1 000 remuestreos:**
  - *Bootstrap:* se vuelve a sortear la muestra de prueba, con reemplazo, 1 000 veces, y en cada una se recalcula el AUC. La variación entre remuestreos muestra cuánta incertidumbre tiene el AUC.
  - *Estratificado:* se sortean por separado las 270 celdas-mes con incendio y las 100 218 sin incendio. Así todos los remuestreos tienen 270 positivos.
  - *Pareado:* los dos modelos se evalúan sobre el mismo remuestreo, de modo que se puede calcular un intervalo para su **diferencia**.
- **Intervalo de confianza del 95 %:** va del percentil 2,5 al 97,5 de los 1 000 valores.
- **Valor p unilateral:** (1 + número de remuestreos en que la diferencia es ≤ 0) / 1 001.

### Ejemplo

Ilustración: si en un remuestreo Random Forest ajustado obtiene un AUC de 0,909 y la logística 0,906, la diferencia es +0,003. En otro remuestreo, la diferencia puede ser negativa. En la corrida real, de las 1 000 diferencias, 378 fueron ≤ 0. Por eso p = 379/1 001 = 0,38: no hay evidencia de que RF sea mejor que la logística.

### Resultados

| Mejor modelo | AUC-ROC 2022-2025 (IC 95 %) | Contraste | Diferencia (IC 95 %) | p | Decisión (α = 0,05) |
|---|---|---|---|---|---|
| XGBoost, Fase 8 (sin ajustar) | 0,9003 [0,887; 0,912] | vs. azar | +0,400 [+0,387; +0,412] | 0,001 | **Supera al azar** |
| | | vs. logística (0,9048) | −0,005 [−0,016; +0,006] | 0,82 | No es mejor |
| **Random Forest, Fase 9 (ajustado, modelo final)** | **0,9063 [0,895; 0,917]** | vs. azar | +0,406 [+0,395; +0,417] | 0,001 | **Supera al azar** |
| | | vs. logística (0,9048) | +0,002 [−0,007; +0,010] | 0,38 | No es mejor |

El p = 0,001 es el mínimo posible con 1 000 remuestreos: ninguno de ellos tuvo un AUC ≤ 0,5.

### Qué significa

- **Se acepta la HG:** el modelo final predice mucho mejor que el azar, y todo el intervalo está muy lejos de 0,5.
- **El modelo de aprendizaje automático no supera de forma significativa a la regresión logística.** Las dos diferencias tienen intervalos que incluyen el 0.
- Para la redacción: "el modelo supera ampliamente al azar; frente a un modelo estadístico clásico, su desempeño es equivalente".

Archivos: `hipotesis_general.csv`, `fig_hipotesis_general.png`.

---

## 4. Paso 3: ajuste de hiperparámetros

### Qué hace y por qué

- Los **hiperparámetros** son las "perillas" del algoritmo: qué tan profundos son los árboles, cuántos datos debe tener cada hoja, la velocidad de aprendizaje, etc. En la Fase 8 se dejaron en valores estándar.
- Aquí se prueban **8 combinaciones por algoritmo**. Cada rejilla incluye la configuración de la Fase 8, así que el ajuste solo puede mantenerla o mejorarla.
  - Random Forest: profundidad máxima {sin límite, 12} × datos mínimos por hoja {1, 10} × fracción de variables por división {raíz cuadrada, 50 %}.
  - XGBoost: profundidad {3, 6} × tasa de aprendizaje {0,05, 0,1} × peso mínimo por hoja {1, 10}.
  - El resto queda fijo: 300 árboles, pesos de clase y semilla 42.
- **Solo se usan los años 2012-2021.** Los años 2022-2025 no se leen en este paso (el script lo comprueba con `assert`). Si se eligieran los hiperparámetros mirando la prueba, la prueba dejaría de ser una evaluación independiente.
- Cada combinación se evalúa con **GroupKFold de 5 pliegues por bloques de 5 × 5 km** dentro de 2012-2021, y se mide con el **AUC-PR medio**.
- El **modelo final** es la combinación con mayor AUC-PR. Después se reentrena con todo 2012-2021.

### Ejemplo

En Random Forest, permitir hojas con un solo dato (configuración de la Fase 8) hace que los árboles "memoricen" casos sueltos. Al exigir al menos 10 datos por hoja y limitar la profundidad a 12, los árboles son más generales, y el AUC-PR espacial sube de 0,048 a 0,057.

### Resultados (AUC-PR medio de los 5 pliegues, solo 2012-2021)

| Puesto | Algoritmo | Combinación | AUC-PR (media ± desv.) | AUC-ROC media |
|---|---|---|---|---|
| **1** | **Random Forest** | **profundidad 12, hoja ≥ 10, 50 % de variables** | **0,0573 ± 0,0125** | 0,920 |
| 2 | Random Forest | profundidad 12, hoja ≥ 1, 50 % de variables | 0,0569 ± 0,0123 | 0,918 |
| 3 | XGBoost | profundidad 3, tasa 0,05, peso mín. 10 | 0,0565 ± 0,0076 | 0,927 |
| 4 | Random Forest | sin límite, hoja ≥ 10, raíz cuadrada | 0,0563 ± 0,0113 | 0,925 |
| … | … | … | … | … |
| 13 | XGBoost (config. Fase 8) | profundidad 6, tasa 0,1, peso mín. 1 | 0,0513 ± 0,0097 | 0,912 |
| 15 | Random Forest (config. Fase 8) | sin límite, hoja ≥ 1, raíz cuadrada | 0,0481 ± 0,0075 | 0,904 |
| ref. | Regresión logística | — | 0,0490 ± 0,0179 | — |

La tabla completa con las 16 combinaciones y el AUC-PR de cada pliegue está en `ajuste_hiperparametros.csv`.

**Desempeño del modelo final en 2022-2025** (calculado después, en el paso 4): AUC-ROC **0,9063**, AUC-PR **0,0213**. Para comparar: XGBoost de la Fase 8 obtuvo 0,9003 / 0,0223, y la logística 0,9048 / 0,0228.

### Qué significa

- El ajuste **mejoró a los dos algoritmos** en la validación espacial: RF pasó de 0,048 a 0,057 y XGBoost de 0,051 a 0,057.
- El modelo final cambia de XGBoost (Fase 8) a **Random Forest ajustado**. La diferencia con XGBoost ajustado (0,0573 frente a 0,0565) es mucho menor que la variación entre pliegues (± 0,012), así que en la práctica están **empatados**.
- En la prueba 2022-2025, el ajuste sube el AUC-ROC de Random Forest de 0,851 a 0,906. El AUC-PR, en cambio, no supera al de XGBoost de la Fase 8 ni al de la logística.
- Los valores de AUC-PR de este paso (≈ 0,057) **no se comparan directamente** con los de la Fase 8 (≈ 0,040). Usan datos distintos: aquí solo 2012-2021, donde hay más incendios (0,47 % de las celdas-mes); allá, 2012-2025 (0,41 %).

Archivos: `ajuste_hiperparametros.csv`, `modelo_final_ajustado.joblib`, `predicciones_fuera_de_pliegue_2012_2021.parquet`.

---

## 5. Paso 4: calibración isotónica

### Qué hace y por qué

- Por los pesos de clase, el "puntaje" del modelo está **inflado**: una celda-mes puede recibir 0,90 aunque en realidad solo se quema el 0,3 % de las celdas-mes. Ese puntaje sirve para **ordenar** las celdas, pero no es una probabilidad real.
- La **regresión isotónica** aprende una función **creciente** que convierte el puntaje en una probabilidad realista. Que sea creciente significa que no cambia el orden: si una celda tenía más puntaje que otra, sigue teniendo más probabilidad. Por eso el AUC casi no cambia.
- **Se ajusta solo con 2012-2021**, usando las predicciones "fuera de pliegue" del paso 3: cada celda-mes de 2012-2021 se predijo con un modelo que no la vio, porque su bloque de 5 km estaba fuera del entrenamiento. No se usan las predicciones del modelo sobre sus propios datos de entrenamiento, porque estarían sobreajustadas.
- Después se aplica, sin cambios, a 2022-2025.
- **Brier score:** el promedio de (probabilidad − realidad)², donde la realidad es 1 si hubo incendio y 0 si no. 0 es perfecto; mientras más bajo, mejor. La **referencia** es predecir siempre la prevalencia de 2012-2021 (0,47 %).
- **Curva de calibración:** las celdas-mes se ordenan por su probabilidad y se forman grupos del mismo tamaño. En cada grupo se compara la probabilidad media predicha con la proporción real de incendios. Si el modelo está bien calibrado, los puntos caen sobre la diagonal.

### Ejemplo: la celda c01053

| Mes | Puntaje sin calibrar | Probabilidad calibrada | Logística |
|---|---|---|---|
| Abril 2023 | 0,048 | 0,0019 (1 en 530) | 0,056 |
| Agosto 2023 | 0,900 | 0,0475 (1 en 21) | 0,782 |

El orden se mantiene: agosto sigue siendo unas 25 veces más riesgoso que abril. Pero ahora el número sí se lee como probabilidad. Para todas las celdas-mes que sí se quemaron en 2022-2025, la mediana pasa de 0,76 (sin calibrar) a 0,022 (calibrada).

### Resultados (2022-2025)

| Versión | Brier | Probabilidad media predicha | Proporción real | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Sin calibrar | 0,07118 | 0,1190 | 0,0027 | 0,9063 | 0,0213 |
| **Calibrada (isotónica)** | **0,00272** | 0,0052 | 0,0027 | 0,9061 | 0,0204 |
| Referencia: siempre 0,47 % | 0,00268 | 0,0047 | 0,0027 | 0,5 | 0,0027 |

Curva de calibración (`fig_calibracion.png`, `calibracion_curva.csv`):

| Grupo | Sin calibrar: predicha → real | Calibrada: predicha → real |
|---|---|---|
| Más bajo | 0,000 → 0,000 | 0,00003 → 0,0000 |
| | 0,003 → 0,000 | 0,0004 → 0,0003 |
| | 0,026 → 0,002 | 0,0016 → 0,0018 |
| | 0,106 → 0,003 | 0,0041 → 0,0026 |
| | 0,285 → 0,004 | 0,0089 → 0,0042 |
| Más alto | 0,771 → 0,019 | 0,0400 → 0,0197 |

Muchas celdas-mes tienen exactamente el mismo puntaje (sobre todo 0), así que algunos de los 10 grupos se fusionan y quedan 6.

### Qué significa

- **Antes de calibrar**, el modelo dice 0,77 donde la realidad es 0,019: sobreestima unas 40 veces. El Brier es alto (0,071).
- **Después de calibrar**, el Brier baja **26 veces** (a 0,0027) y el AUC no cambia. Las probabilidades quedan en una escala realista.
- **Pero no son perfectas:** en los grupos de riesgo alto el modelo calibrado todavía dice el doble de lo que ocurrió (0,040 frente a 0,020). Y su Brier queda **apenas peor** que predecir siempre la prevalencia (0,00272 frente a 0,00268).
  - La causa es un **cambio de prevalencia**: en 2012-2021 se quemó el 0,47 % de las celdas-mes, pero en 2022-2025 solo el 0,27 %. La calibración aprendió la frecuencia de los años anteriores.
  - Además, el Brier premia mucho acertar los cientos de miles de "no incendio". Por eso una constante baja es difícil de superar con eventos tan raros, aunque no sirva para ordenar (su AUC es 0,5).
- **Cómo usarlo en la tesis:** el puntaje sin calibrar sigue siendo el mejor **índice para ordenar y mapear**. La probabilidad calibrada da el **orden de magnitud** del riesgo real ("en agosto, esta celda tiene alrededor de 5 % de probabilidad"), con la advertencia de que en años de menor actividad lo sobreestima.

Archivos: `calibrador_isotonico.joblib`, `calibracion_metricas.csv`, `calibracion_curva.csv`, `predicciones_prueba_modelo_final.parquet`, `fig_calibracion.png`.

---

## 6. Paso 5: prueba de permutación con 1 000 repeticiones (HE1 a HE4)

### Qué hace y por qué

- **La idea:** si una dimensión es importante, el modelo debe empeorar al "romper" su información.
- **Cómo se rompe:** en los 100 488 registros de 2022-2025 se desordenan al azar las filas de **todas las columnas de una dimensión a la vez, con la misma permutación**. Por ejemplo, la lluvia y la temperatura de la celda A pasan juntas a la celda B. Las otras tres dimensiones no se tocan.
  - Para la dimensión ambiental se mueven juntas 9 columnas: el NDVI y las 8 indicadoras de la cobertura.
- **El modelo no se reentrena:** es el Random Forest final, entrenado una sola vez con 2012-2021. En cada repetición solo se le pide una nueva predicción.
- Se mide la **caída del AUC-ROC**: AUC original (0,9063) menos AUC con la dimensión desordenada.
- Se repite **1 000 veces** con desórdenes distintos. Cada dimensión usa un generador nuevo con semilla 42.
- **Valor p** = (1 + repeticiones en que la caída fue ≤ 0) / 1 001. Si nunca fue ≤ 0, p = 1/1 001 = **0,000999**, el mínimo posible. Con 100 repeticiones (Fase 8), el mínimo era 0,0099.
- **Significativa si p < 0,05.** Además de la significancia, se exige que la caída media sea positiva.

### Ejemplo

Al desordenar la topografía, la celda c01053 (838 m de altura) puede recibir la elevación de una celda a 2 500 m. El modelo ya no sabe que está en zona baja, así que baja su riesgo, aunque en realidad sí está en zona de incendios. Al repetir esto en todas las celdas, el AUC-ROC cae de 0,906 a unos 0,744. La caída es de 0,16.

### Resultados (Random Forest ajustado, 2022-2025, AUC-ROC original 0,9063)

| Hipótesis | Dimensión | Caída del AUC-ROC (media ± desv.) | 95 % de las caídas entre | Repeticiones con caída ≤ 0 | p | Caída del AUC-PR (p) | Decisión |
|---|---|---|---|---|---|---|---|
| HE1 | Climática | 0,0884 ± 0,0067 | 0,075 y 0,102 | 0 | 0,000999 | 0,0124 (0,000999) | **Se acepta** |
| HE2 | Topográfica | **0,1618** ± 0,0103 | 0,141 y 0,182 | 0 | 0,000999 | 0,0117 (0,000999) | **Se acepta** |
| HE3 | Ambiental | 0,0068 ± 0,0025 | 0,002 y 0,012 | 1 | 0,0020 | 0,0027 (0,0030) | **Se acepta** (efecto muy pequeño) |
| HE4 | Antropogénica | 0,0546 ± 0,0069 | 0,041 y 0,068 | 0 | 0,000999 | 0,0050 (0,0110) | **Se acepta** |

Gráfico: `fig_permutacion_1000.png`. Detalle de cada repetición: `permutacion_avance.csv`.

### Qué significa

- **Las cuatro dimensiones contribuyen a la predicción** con el modelo final: en todas, p < 0,05.
- **Orden de importancia:** topográfica (0,16) > climática (0,09) > antropogénica (0,05) > ambiental (0,007). Como tres dimensiones tienen el mismo p mínimo, el valor p no sirve para ordenarlas: se ordenan por el **tamaño de la caída**.
- Es el mismo orden que dieron SHAP y la permutación de la Fase 8 con XGBoost. Que métodos y modelos distintos coincidan respalda la conclusión.
- **La dimensión ambiental es significativa pero casi irrelevante:** su caída (0,007) es unas 24 veces menor que la de la topográfica. En la Fase 8, con el Random Forest sin ajustar, no fue significativa (p = 0,73). Su aporte es **pequeño y depende del modelo**, y conviene decirlo así. La razón probable es que, dentro del bosque analizado, el NDVI y la cobertura varían muy poco (el 95 % de las celdas-mes son bosque).

---

## 7. Decisión sobre las hipótesis

Modelo final: **Random Forest ajustado** (Fase 9), α = 0,05.

| Hipótesis | Contraste | Resultado | Decisión |
|---|---|---|---|
| **HG** | AUC-ROC del modelo final en 2022-2025 > 0,5 | 0,9063, IC 95 % [0,895; 0,917], p = 0,001 | **Se acepta** |
| (complementario) | Modelo final > regresión logística | diferencia +0,002, IC 95 % [−0,007; +0,010], p = 0,38 | No se demuestra |
| **HE1** | La dimensión climática contribuye | caída 0,088, p = 0,000999 | **Se acepta** |
| **HE2** | La dimensión topográfica contribuye | caída 0,162, p = 0,000999 | **Se acepta** (la más importante) |
| **HE3** | La dimensión ambiental contribuye | caída 0,007, p = 0,002 | **Se acepta**, con un efecto muy pequeño y no robusto entre modelos |
| **HE4** | La dimensión antropogénica contribuye | caída 0,055, p = 0,000999 | **Se acepta** |

**En una frase:** el modelo predice la susceptibilidad mucho mejor que el azar, y la explican sobre todo la topografía (zonas bajas) y el clima (meses secos); le siguen la cercanía a poblados y, con un aporte mínimo, la vegetación. Un modelo estadístico clásico consigue un desempeño equivalente.

---

## 8. Limitaciones

1. **Hueco en los datos de VIIRS entre el 26 de julio y el 11 de agosto de 2022.** En esos 16 días, el archivo de FIRMS no tiene **ninguna** detección VIIRS en todo el Perú. Lo más probable es que se deba a una interrupción del satélite Suomi-NPP, no a la ausencia de fuegos. Cae en plena temporada de incendios (agosto concentra el 35 % de las detecciones) y **dentro del periodo de prueba**. Algunos incendios de esas semanas pudieron quedar sin registrar, así que se cuentan como "sin incendio". Esto puede bajar un poco el AUC medido en 2022-2025 y la prevalencia de ese año. Hubo otros huecos más cortos, de 6 y 7 días, en mayo-junio y julio de 2024.
2. **Las etiquetas de 2024 y 2025 se verificaron con menos años posteriores.** Un incendio forestal exige que el lugar siga siendo bosque en t+1 y t+2, pero MapBiomas solo llega a 2025. Para las detecciones de 2024 y 2025, el año faltante se reemplaza por el último disponible. Estos años son parte de la prueba.
3. **Cambio de prevalencia entre periodos:** 0,47 % en 2012-2021 y 0,27 % en 2022-2025. Afecta sobre todo al AUC-PR y a la calibración (sección 5).
4. **La regresión logística rinde casi igual que los modelos de árboles.** No es un error, pero limita la afirmación de que el aprendizaje automático "mejora" la predicción.
5. **El bootstrap y la permutación tratan cada celda-mes como independiente.** En realidad, las celdas vecinas y los meses seguidos se parecen (autocorrelación), así que los intervalos de confianza y los valores p pueden ser algo **optimistas**, es decir, más estrechos o más pequeños de lo real.
6. **La elección del modelo final de la Fase 8 usó los 5 pliegues espaciales sobre toda la tabla 2012-2025**, que incluye los años de prueba. En la Fase 9 esto se corrigió: el ajuste y la elección usan solo 2012-2021.
7. **La búsqueda de hiperparámetros es pequeña** (8 combinaciones por algoritmo). Con una búsqueda mayor, el resultado podría cambiar un poco. La diferencia entre RF y XGBoost ajustados está dentro del ruido.
8. **Indicadores estáticos:** la red vial (MTC, corte de diciembre de 2018) y los centros poblados (INEI) son una sola foto para todo el periodo.
9. **El NDVI falta en el 15,7 % de las celdas-mes** (por nubes) y se imputó con el promedio de la misma celda y mes en 2012-2021.
10. **La correspondencia HE1 = climática, HE2 = topográfica, HE3 = ambiental, HE4 = antropogénica es un supuesto.** Hay que confirmarla con la redacción de la tesis.
11. **Los mapas y el visor siguen usando XGBoost de la Fase 8**, no el Random Forest ajustado. Ver "pendientes".

---

## 9. Registro de cambios

| Fecha | Cambio | Archivos |
|---|---|---|
| 2026-11-02 | Gráfico y cuadro de las 4 424 detecciones VIIRS por año y mes | nuevo `src/fase8_grafico_viirs.py`; `resultados/fig_detecciones_viirs.png`, `resultados/detecciones_viirs_anio_mes.csv` |
| 2026-11-09 | Módulo común de la Fase 9 | nuevo `src/fase9_comun.py` |
| 2026-11-09 | Paso 1: regresión logística y comparación con RF y XGBoost; después se agregó su validación espacial dentro de 2012-2021 para compararla con el ajuste | nuevo `src/fase9_logistica.py` |
| 2026-11-09 | Paso 2: HG con bootstrap pareado y estratificado. Primero con XGBoost de la Fase 8; después se volvió a correr para incluir RF ajustado | nuevo `src/fase9_hipotesis_general.py` |
| 2026-11-09 | Paso 3: ajuste de hiperparámetros solo con 2012-2021. El modelo final pasa de XGBoost (Fase 8) a Random Forest ajustado | nuevo `src/fase9_ajuste_hiperparametros.py` |
| 2026-11-09 | Paso 4: calibración isotónica ajustada con las predicciones fuera de pliegue de 2012-2021 | nuevo `src/fase9_calibracion.py` |
| 2026-11-09 | Paso 5: permutación con 1 000 repeticiones y guardado del avance | nuevo `src/fase9_permutacion.py` |
| 2026-11-09 | Este documento | nuevo `RESULTADOS_TESIS.md` |

**No se modificó ni se borró ningún script ni dato que ya existiera**, ni el archivo `.env`. No se hicieron commits. Los únicos ajustes posteriores fueron sobre los scripts nuevos de la Fase 9:
- en `fase9_hipotesis_general.py`, `vert=False` se cambió por `orientation="horizontal"` para evitar un aviso de matplotlib;
- en `fase9_calibracion.py` se corrigió el título de la figura;
- a `fase9_logistica.py` se le agregó la validación dentro de 2012-2021.

### Diferencias de método respecto a la Fase 8

| Tema | Fase 8 | Fase 9 |
|---|---|---|
| Modelo final | XGBoost con hiperparámetros estándar, elegido con pliegues de 2012-2025 | Random Forest ajustado, elegido solo con 2012-2021 |
| Bootstrap de la HG | Simple (no estratificado), solo contra 0,5 | Estratificado y pareado, contra 0,5 y contra la logística |
| Permutación | 100 repeticiones (p mínimo 0,0099) | 1 000 repeticiones (p mínimo 0,000999) |
| Probabilidades | Sin calibrar (índice relativo) | Calibración isotónica disponible |

---

## 10. Pendientes sugeridos

- [ ] Decidir qué modelo usar en los mapas y el visor. Hoy usan XGBoost de la Fase 8. Para ser coherente con la Fase 9, habría que regenerarlos con el Random Forest ajustado (y, si se quiere, mostrar también la probabilidad calibrada).
- [ ] Confirmar la correspondencia HE1-HE4 con la redacción de la tesis.
- [ ] Explicar en la tesis que la regresión logística rinde igual que los modelos de árboles, y por qué igual se usan RF/XGBoost (relaciones no lineales, SHAP, robustez).
- [ ] Opcional: repetir la evaluación excluyendo agosto de 2022 para medir el efecto del hueco de VIIRS.
- [ ] Opcional: calcular SHAP del modelo final ajustado (hoy SHAP es de XGBoost de la Fase 8).

---

## 11. Glosario rápido

| Término | Significado sencillo |
|---|---|
| Regresión logística | Modelo estadístico clásico: suma ponderada de los indicadores convertida en probabilidad |
| Estandarizar | Restar la media y dividir entre la desviación estándar, para que todas las variables tengan la misma escala |
| Hiperparámetro | "Perilla" del algoritmo que se fija antes de entrenar (profundidad de los árboles, etc.) |
| Predicción fuera de pliegue | Predicción de un dato hecha por un modelo que no lo vio al entrenar |
| Bootstrap | Volver a sortear la muestra muchas veces para medir la incertidumbre de un resultado |
| Intervalo de confianza 95 % | Rango donde probablemente está el valor real |
| Calibración | Convertir un puntaje en una probabilidad que coincida con la frecuencia real |
| Brier score | Error medio de una probabilidad: (probabilidad − realidad)²; menor es mejor |
| Permutación | Desordenar una dimensión para ver cuánto empeora el modelo sin su información |
| Valor p | Probabilidad de ver un resultado así de extremo si no hubiera efecto; < 0,05 se considera significativo |
| Prevalencia | Proporción de casos positivos (celdas-mes con incendio) |
