# Guía explicada del sistema, para preparar la sustentación

Esta guía recorre cada parte del trabajo en el orden en que se ejecuta. En cada parte encontrarás:
- **Qué hace y por qué**, con un ejemplo de tus propios datos.
- **Dónde está en el código**: el script y sus líneas más importantes.
- **Cómo leer el resultado.**
- **Preguntas que podría hacer el jurado**, con una respuesta sugerida.

Los números son los de la corrida del 2026-10-21; el detalle completo está en `RESULTADOS_PRELIMINARES.md`.

---

## 0. La idea general en un párrafo

Dividimos el distrito de Pólvora en 2 263 cuadrados de 1 km × 1 km ("celdas"). Para cada celda y cada mes de 2012 a 2025 anotamos dos cosas:
- si hubo un incendio forestal (sí/no);
- 8 características de la celda en ese momento (lluvia, temperatura, altura, pendiente, verdor de la vegetación, tipo de cobertura, distancia a una carretera y distancia a un centro poblado).

Con esos ejemplos entrenamos dos algoritmos de aprendizaje automático (Random Forest y XGBoost) para que aprendan qué combinaciones de características se asocian a los incendios. Después:
- comprobamos que acierten más que el azar en años y zonas que no vieron;
- usamos SHAP y la permutación para explicar **qué** características pesan más;
- dibujamos mapas mensuales de susceptibilidad.

Esto es un **experimento computacional**:
- **Tratamientos** (lo que se compara): los dos algoritmos.
- **Control** (la referencia mínima): el azar.
- **Condiciones controladas**: los mismos datos, la misma partición de los datos y la misma semilla aleatoria (42), para que la única diferencia entre tratamientos sea el algoritmo.

---

## 1. Datos: de las detecciones de calor a la tabla de celdas-mes

### Qué hace y por qué

1. **Detecciones de calor:** el satélite VIIRS (NASA FIRMS) registra puntos de calor de 375 m. En Pólvora hubo 4 424 entre 2012 y 2025.
2. **¿Era bosque?** No todo punto de calor es un incendio forestal: muchos son quemas agrícolas. Por eso se consulta MapBiomas Colección 4, un mapa anual de cobertura del suelo. Una detección cuenta como incendio forestal solo si el lugar era bosque el año anterior (t−1), el mismo año (t) y los dos años siguientes (t+1, t+2). Así se descartan las quemas para deforestar.
3. **Eventos:** un mismo incendio genera varios puntos. Los puntos a menos de 750 m y con menos de 5 días de diferencia se agrupan en un solo evento.
4. **Celdas-mes:** cada evento se asigna a la celda y al mes donde empezó. Una celda-mes es "positiva" (1) si tuvo al menos un evento, y "negativa" (0) si no tuvo ninguno.
5. **Filtro de bosque:** solo se analizan las celdas-mes con al menos 20 % de bosque el año anterior. Una celda sin bosque no puede tener un incendio *forestal*, y meterla en el análisis solo agregaría ceros fáciles de acertar.

**Resultado:** 351 732 celdas-mes, de las cuales **1 455** tuvieron incendio (0,41 %, es decir, 1 de cada 242).

**Desfase temporal (importante):** para predecir agosto, el modelo usa la lluvia, la temperatura y el NDVI de **julio**, y la cobertura del **año anterior**. Así el modelo solo usa información que ya existía antes del incendio. Si usáramos la lluvia de agosto, esa lluvia podría estar influida por el propio periodo del incendio, y además no serviría para anticipar nada.

**Ejemplo:** la celda c01053 (838 m de altura, 84 % de bosque) tuvo incendio en 3 de los 14 agostos del periodo.

### Dónde está en el código

- `src/fase3b_eventos_c4.py`, líneas 18-19: `DISTANCIA_M = 750` y `DIAS_VENTANA = 5`, el criterio para agrupar puntos en eventos.
- `src/fase7_construir_tabla_c4.py`:
  - Línea 49: solo cuentan los eventos con participación de VIIRS.
  - Líneas 111-112: el desfase de un mes para el clima.
  - Línea 128: la cobertura se toma del año anterior (`anio - 1`).
  - Línea 148: el filtro `pct_bosque_1km >= 0.20`.
- `src/fase8_tabla_final.py`, línea 70: verifica que la tabla tenga exactamente 351 732 filas y 1 455 positivos; si no, el script se detiene.
- `src/fase8_tabla_final.py`, línea 82: calcula la media, la desviación estándar, el mínimo y el máximo de los indicadores.

### Cómo leer el resultado

- La **media** es el valor típico y la **desviación estándar** cuánto se alejan los valores de esa media. Por ejemplo, la precipitación del mes anterior tiene media 134 mm y desviación 85 mm: varía mucho entre meses secos y lluviosos.
- La elevación va de 420 m a 3 658 m, pero **todos** los incendios ocurrieron por debajo de 1 656 m.
- Agosto (529) y septiembre (443) reúnen dos tercios de todas las celdas-mes con incendio. Coinciden con el final de la época seca: julio y agosto son los meses de menos lluvia según CHIRPS.

### Preguntas del jurado

**¿Por qué celdas de 1 km y no puntos?**
Un punto de calor de VIIRS mide 375 m y su ubicación tiene error. La celda de 1 km es mayor que ese error y coincide con la escala de los predictores más finos. Además, la unidad de decisión de un mapa de susceptibilidad es un área, no un punto.

**¿Por qué solo 1 455 incendios si hubo 4 424 detecciones?**
Porque (a) muchas detecciones eran quemas agrícolas o para deforestar, no incendios forestales; (b) varias detecciones de un mismo incendio se agrupan en un solo evento; y (c) varios eventos en la misma celda y el mismo mes cuentan como una sola celda-mes positiva.

**¿Por qué el umbral de 20 % de bosque?**
Para que la población de estudio sean lugares donde *puede* ocurrir un incendio forestal. Es un umbral bajo a propósito: incluye los bordes del bosque, que es justamente donde ocurre la mayoría de los incendios.

---

## 2. Imputación del NDVI

### Qué hace y por qué

- El **NDVI** (índice de vegetación de diferencia normalizada) mide el verdor de la vegetación con imágenes de satélite. Va de −1 a 1; un bosque sano está cerca de 0,8-0,9.
- Cuando hay nubes, el satélite no ve el suelo y el dato falta. En Pólvora faltan **55 215** celdas-mes (15,7 %), casi todas en la época de lluvias: de febrero a abril faltan unas 10 000 por mes, y en agosto menos de 500.
- **Imputar** significa rellenar un dato faltante con una estimación razonable. Aquí se usa el **promedio de la misma celda en el mismo mes**, calculado con los años 2012-2021. Así se respetan dos cosas: cada lugar tiene su propio verdor típico, y ese verdor cambia con la estación.
- **Por qué solo 2012-2021:** son los años de entrenamiento. Si usáramos también 2022-2025, información de los años de prueba se colaría en los datos de entrenamiento. A eso se le llama **fuga de información**, y haría que el modelo pareciera mejor de lo que es.

**Ejemplo:** la celda c01053 no tiene NDVI en marzo de 2012. Sus marzos de 2013 a 2021 valen 0,859; 0,840; 0,860; 0,833; 0,880; 0,860; 0,734; 0,857 y 0,814. El promedio es **0,8375**, y ese es el valor que se usa.

Para 3 482 celdas-mes no hubo **ningún** marzo (o el mes que fuera) válido entre 2012 y 2021. En esos casos se usa el promedio de ese mes en todas las celdas.

### Dónde está en el código

`src/fase8_tabla_final.py`:
- Línea 51: `promedio_celda_mes = base.groupby(["cell_id", "mes"])["ndvi"].mean()`, donde `base` solo contiene años anteriores a 2022 y valores reales, no imputados.
- Línea 55: rellena los faltantes con ese promedio.
- Líneas 52 y 58-59: el respaldo con el promedio del mes.

### Cómo leer el resultado

La columna `ndvi_imputado` (0/1) marca qué valores fueron rellenados. El cambio frente a la versión anterior, que usaba la mediana, es pequeño (0,015 de NDVI en promedio).

### Preguntas del jurado

**¿La imputación no inventa datos?**
Rellena con el valor típico de esa celda en esa época del año, que es la mejor estimación disponible. Además, los faltantes se concentran en la época de lluvias, cuando casi no hay incendios (marzo tiene 5 de 1 455), así que influyen poco en lo que aprende el modelo.

**¿Por qué promedio y no mediana?**
Ambas son razonables. La mediana es menos sensible a valores raros; el promedio es el estándar y el que define la metodología. La diferencia entre ambas fue de 0,015 de NDVI en promedio, demasiado pequeña para cambiar las conclusiones.

**¿Por qué no eliminar las filas sin NDVI?**
Se perdería el 15,7 % de la tabla, y de forma desigual: casi todo de la época de lluvias. El modelo vería una estación distorsionada.

---

## 3. Partición temporal y validación espacial

### Qué hace y por qué

Un modelo siempre acierta más en los datos con los que aprendió. Para medir su desempeño real hay que evaluarlo con datos que **no vio**. Aquí se hace de dos maneras.

**a) Partición temporal**
- Se entrena con 2012-2021 (251 244 celdas-mes, 1 185 incendios) y se evalúa con 2022-2025 (100 488 celdas-mes, 270 incendios).
- Responde a la pregunta: *¿el modelo funciona en años futuros?* Es la situación real de uso.

**b) Validación espacial por bloques**
- El distrito se divide en **116 bloques de 5 × 5 km**, y los bloques se reparten en **5 grupos ("pliegues")**.
- Se entrena con 4 pliegues y se evalúa en el quinto, y se repite 5 veces, cambiando el pliegue de evaluación.
- Responde a la pregunta: *¿el modelo funciona en lugares que no vio?*
- **Por qué bloques y no celdas sueltas:** celdas vecinas se parecen mucho (misma altura, misma lluvia). A esto se le llama **autocorrelación espacial**. Si una celda está en entrenamiento y su vecina en evaluación, el modelo "copia" la respuesta de la vecina y parece mejor de lo que es. Con bloques de 5 km, toda una zona queda fuera del entrenamiento.
- **GroupKFold** es la función que reparte los bloques en pliegues sin que un bloque quede partido entre entrenamiento y evaluación.

### Dónde está en el código

`src/fase8_modelo.py`:
- Línea 95: `train = tabla[tabla["anio"] < 2022]`, la partición temporal.
- Líneas 61-63: el número de bloque de cada celda. Es la división entera de sus coordenadas UTM entre 5 000 m; por ejemplo, x = 352 400 m da el bloque 70.
- Línea 124: `GroupKFold(n_splits=5).split(X, y, groups=grupos)`.

### Cómo leer el resultado

Cada pliegue da un AUC distinto, por ejemplo XGBoost obtiene AUC-ROC entre 0,887 y 0,930. Se reporta la **media** (0,906) y la **desviación estándar** (0,020):
- la media es el desempeño esperado en una zona nueva;
- la desviación indica cuánto varía de una zona a otra.

Una desviación pequeña significa que el modelo es estable en todo el distrito.

### Preguntas del jurado

**¿Por qué no una partición aleatoria del 70/30?**
Porque mezclaría años y celdas vecinas entre entrenamiento y prueba, y el resultado saldría optimista. Las particiones temporal y espacial son más exigentes y más honestas.

**¿Por qué bloques de 5 km?**
Es varias veces el tamaño de la celda y abarca la distancia a la que los predictores siguen pareciéndose entre celdas vecinas. Un bloque más grande daría muy pocos bloques (con 116 bloques, cada pliegue tiene unos 23).

**¿Por qué el AUC-PR es más alto en la validación espacial (0,040) que en la temporal (0,022)?**
Por la prevalencia. En 2022-2025 hubo menos incendios (0,27 % de las celdas-mes) que en el promedio del periodo (0,41 %), y el AUC-PR depende de cuántos positivos hay. Por eso siempre se compara con la referencia del azar de cada conjunto: en los dos casos el modelo la multiplica por 8 a 10.

---

## 4. Los modelos: Random Forest y XGBoost

### Qué hace y por qué

- **Árbol de decisión:** una serie de preguntas del tipo "¿la elevación es menor que 900 m? → ¿llovió menos de 60 mm el mes anterior? → ...", que termina en una predicción.
- **Random Forest (bosque aleatorio):** entrena **300 árboles**, cada uno con una muestra distinta de los datos y de las variables, y **promedia** sus respuestas. Promediar muchos árboles distintos reduce los errores de cada uno.
- **XGBoost (gradient boosting):** también usa 300 árboles, pero los construye **uno tras otro**. Cada árbol nuevo intenta corregir los errores de los anteriores.
- **Pesos de clase:** solo 1 de cada 242 celdas-mes es incendio. Sin ajuste, el modelo aprendería que "nunca hay incendio" y acertaría el 99,6 % de las veces sin servir para nada. Con pesos de clase, cada incendio "vale" tanto como unas 211 celdas sin incendio en el entrenamiento (250 059 / 1 185). Así el modelo presta atención a los incendios.
- **Sin optimizar hiperparámetros:** los **hiperparámetros** son las "perillas" del algoritmo, como el número de árboles o su profundidad. Se dejaron en valores estándar e iguales para todas las corridas. Así la comparación es justa, y se evita ajustar el modelo a los datos de prueba.
- **Cobertura MapBiomas:** es una categoría (bosque, pasto, palma...), no un número. Se convierte en varias columnas de 0/1 llamadas **variables indicadoras**: una por clase. Por eso el modelo recibe 15 columnas a partir de 8 indicadores.

### Dónde está en el código

- `src/fase7_modelo_c4.py`:
  - Línea 88: Random Forest con `n_estimators=300, class_weight="balanced"`.
  - Línea 91: XGBoost con `n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=peso`.
- `src/fase8_modelo.py`:
  - Importa esa misma función, de modo que los hiperparámetros son idénticos.
  - Líneas 78-79: el peso se calcula como negativos / positivos.
  - Línea 212: el modelo final es `max(media_pr)`, es decir, el de mayor AUC-PR medio en la validación espacial.

### Cómo leer el resultado

XGBoost fue el modelo final (AUC-PR espacial 0,0396 frente a 0,0392). La diferencia es muy pequeña, así que en AUC-PR espacial **los dos modelos están prácticamente empatados**. XGBoost sí es mejor con claridad en la prueba temporal (AUC-ROC 0,900 frente a 0,851).

### Preguntas del jurado

**¿Por qué estos dos algoritmos y no una red neuronal?**
Son los más usados y exitosos con datos en forma de tabla. Funcionan bien con pocos positivos y permiten calcular SHAP de forma exacta y rápida (TreeExplainer). Una red neuronal necesitaría muchos más datos y sería más difícil de explicar.

**¿No optimizar hiperparámetros no los perjudica?**
Puede que se pierda algo de desempeño, pero se gana en una comparación justa y reproducible. Aun sin optimizar, los dos superan ampliamente al azar.

**Si la diferencia es tan pequeña, ¿por qué elegir XGBoost?**
Porque la regla se fijó antes de ver los resultados: mayor AUC-PR espacial. Además, XGBoost también gana en todas las demás métricas, así que la elección es coherente.

---

## 5. Métricas: AUC-ROC y AUC-PR

### Qué hace y por qué

El modelo da a cada celda-mes un puntaje entre 0 y 1. Las métricas miden si los incendios reales reciben puntajes más altos que las celdas sin incendio.

- **AUC-ROC** (área bajo la curva ROC): es la probabilidad de que, si tomas al azar una celda-mes con incendio y otra sin incendio, el modelo le dé el puntaje más alto a la que tuvo incendio.
  - 0,5 = azar, como lanzar una moneda.
  - 1,0 = perfecto.
  - El modelo final obtuvo **0,900**: acierta el orden en 9 de cada 10 pares.
- **AUC-PR** (área bajo la curva de precisión-exhaustividad, también llamada *precisión promedio*):
  - **Precisión:** de las celdas que el modelo marca como riesgosas, qué fracción realmente se quemó.
  - **Exhaustividad (recall):** de los incendios reales, qué fracción encontró el modelo.
  - El valor del azar es la **prevalencia**, es decir, la proporción de positivos: 0,0027 en la prueba.
  - El modelo obtuvo **0,0223**, unas **8 veces** el azar.
- **Por qué las dos:** con incendios tan raros, el AUC-ROC puede verse alto con facilidad, porque hay muchísimos negativos fáciles de acertar. El AUC-PR se fija solo en qué tan bien se encuentran los positivos y es más exigente. Reportar ambas da una imagen honesta.

### Dónde está en el código

`src/fase8_modelo.py`:
- Línea 108: `roc_auc_score` y `average_precision_score`.
- Línea 88: la referencia del azar para el AUC-PR, `y.mean()`, que es la prevalencia.

### Cómo leer el resultado

| | AUC-ROC prueba | AUC-PR prueba | AUC-ROC espacial | AUC-PR espacial |
|---|---|---|---|---|
| XGBoost | 0,900 | 0,022 | 0,906 ± 0,020 | 0,040 ± 0,006 |
| Random Forest | 0,851 | 0,018 | 0,891 ± 0,020 | 0,039 ± 0,004 |
| Azar | 0,500 | 0,0027 | 0,500 | 0,0041 ± 0,0010 |

**¿Un AUC-PR de 0,022 es malo?** No, si se compara con lo que corresponde:
- Si eliges celdas al azar, solo el 0,27 % se quema.
- Si eliges las que el modelo marca como más riesgosas, la proporción es unas 8 veces mayor.

Para incendios, que son raros y dependen de que alguien prenda fuego, no se espera predecir con certeza dónde ocurrirá cada uno, sino **dónde el riesgo es mayor**.

En la figura `fig_curvas_roc_pr_prueba.png`:
- **Curva ROC:** cuanto más se acerca a la esquina superior izquierda, mejor; la diagonal es el azar.
- **Curva precisión-exhaustividad:** la línea punteada es el azar, y las curvas de los modelos están siempre por encima.

### Preguntas del jurado

**¿Por qué no reportar la exactitud (accuracy)?**
Porque un modelo que dice "nunca hay incendio" tendría 99,7 % de exactitud y sería inútil. Con clases tan desbalanceadas, la exactitud engaña.

**¿Qué significa que la prevalencia sea la referencia del AUC-PR?**
Un modelo que asigna puntajes al azar tiene, en cualquier umbral, una precisión igual a la proporción de positivos. Por eso su AUC-PR es esa proporción. Lo comprobamos con puntajes aleatorios reales: 0,0028 frente a 0,0027 teórico.

**¿Por qué el AUC-ROC de Random Forest es menor en la prueba temporal?**
Random Forest da probabilidad 0 a muchas celdas-mes, y alrededor del 15 % de los incendios de prueba quedaron empatados en 0. Esos incendios no se pueden ordenar, lo que se ve como un tramo recto en su curva ROC. XGBoost reparte mejor los puntajes.

---

## 6. SHAP: qué indicadores explican cada predicción

### Qué hace y por qué

- **SHAP** (SHapley Additive exPlanations) reparte la predicción de cada celda-mes entre los indicadores, como se repartiría el premio de un equipo entre sus jugadores según cuánto aportó cada uno. Viene de la teoría de juegos (valores de Shapley).
- Para cada registro, la suma de los valores SHAP de los 8 indicadores, más un valor base, da la predicción del modelo.
  - Un valor SHAP **positivo** empuja la predicción hacia "incendio".
  - Un valor **negativo** la aleja.
- **TreeExplainer** es la versión de SHAP para modelos de árboles: calcula los valores exactos y rápido (10 segundos para 15 000 registros).
- Se aplicó al modelo final, entrenado con 2012-2021, sobre **15 000 registros de prueba** (2022-2025) elegidos al azar con semilla 42. Así la explicación se hace sobre datos que el modelo no vio.
- La **importancia** de un indicador es el promedio del valor absoluto de su SHAP: cuánto mueve, en promedio, la predicción, sin importar hacia qué lado.
- Las 8 columnas indicadoras de la cobertura se **suman** para obtener el aporte de la cobertura como un solo indicador. Se puede hacer porque los valores SHAP se suman.

**Ejemplo con la celda c01053:**
- En **agosto**, su probabilidad es 0,945 (muy alta). Los tres factores que más la explican:
  - elevación de 838 m: +1,02 (estar en zona baja sube el riesgo);
  - lluvia de julio de solo 43 mm: +0,72;
  - distancia de 6,3 km a un centro poblado: −0,60 (estar algo lejos lo baja).
- En **abril**, la misma celda tiene probabilidad casi 0. El factor dominante es la lluvia de marzo: −4,63, porque llueve mucho.

### Dónde está en el código

`src/fase8_interpretacion.py`:
- Línea 94: `test.sample(n=15000, random_state=42)`.
- Línea 81: `shap.TreeExplainer(modelo)`.
- Línea 101: la importancia como `abs().mean()`.
- Función `agregar_por_indicador`: suma las columnas de la cobertura.

### Cómo leer el resultado

| Dimensión | Importancia SHAP |
|---|---|
| Topográfica | 47 % |
| Climática | 24 % |
| Antropogénica | 20 % |
| Ambiental | 9 % |

Por indicador: elevación (37 %), precipitación (17 %), distancia a centro poblado (15 %), pendiente (10 %), NDVI (8 %), temperatura (7 %), distancia a carretera (5 %), cobertura (1 %).

**Cómo leer el gráfico de puntos** (`fig_shap_beeswarm.png`):
- Cada punto es un registro.
- La posición horizontal es su valor SHAP: a la derecha sube el riesgo, a la izquierda lo baja.
- El color es el valor del indicador: rojo = alto, azul = bajo.

Ejemplos:
- En **Elevación**, los puntos rojos (alta elevación) están muy a la izquierda: la altura reduce mucho el riesgo.
- En **Precipitación**, los rojos (mucha lluvia) también van a la izquierda: un mes anterior lluvioso reduce el riesgo.
- En **Distancia a centro poblado**, los rojos (lejos) van a la izquierda: estar cerca de poblados aumenta el riesgo.

### Preguntas del jurado

**¿SHAP demuestra que la elevación causa los incendios?**
No. SHAP explica **cómo usa el modelo** cada indicador, no relaciones de causa y efecto. La elevación probablemente representa otras cosas que van juntas: en las zonas bajas hay más gente, más agricultura y más calor. Por eso el alcance es explicativo del modelo y de las asociaciones, no causal.

**¿Por qué la muestra de 15 000 y no toda la prueba?**
Por costo de cálculo y por ser el tamaño definido en la metodología. Con 15 000 registros el promedio de los valores SHAP es estable. Una advertencia: la muestra solo incluye 46 incendios, y convendría repetirla con otra semilla para confirmar el orden de los indicadores.

**¿Por qué la cobertura importa tan poco?**
Porque el filtro de 20 % de bosque hace que el 95 % de las celdas-mes analizadas sean bosque. Una variable que casi no varía no puede explicar mucho.

---

## 7. Permutación agrupada y pruebas de hipótesis

### Qué hace y por qué

- **La idea:** si una dimensión es importante, al "romper" su información el modelo debe empeorar.
- **Cómo se rompe:** se desordenan al azar los valores de **todas las columnas de esa dimensión a la vez**, entre todos los registros de la prueba. Por ejemplo, la lluvia y la temperatura de la celda A pasan a la celda B, y las de B a la C. Las demás dimensiones quedan intactas. Luego se mide cuánto cae el AUC.
- Se desordenan juntas las columnas de una dimensión para conservar la relación entre ellas y medir el aporte de la dimensión completa. Por ejemplo, la lluvia y la temperatura están correlacionadas; si se desordenara solo una, la otra compensaría y se subestimaría la dimensión.
- Se repite **100 veces** con desórdenes distintos.
- **Valor p:** es la proporción de esas 100 repeticiones en que el AUC desordenado fue igual o mejor que el original. Si nunca lo fue, p = 1/101 = **0,0099**, el mínimo posible con 100 repeticiones.
  - **p < 0,05** significa que, si la dimensión no aportara nada, sería muy improbable (menos de 5 %) ver una caída así solo por azar. Por eso se concluye que su aporte es real (estadísticamente significativo).

**Hipótesis específicas:**
- HE1: la dimensión climática contribuye.
- HE2: la topográfica contribuye.
- HE3: la ambiental contribuye.
- HE4: la antropogénica contribuye.

**Hipótesis general (HG):** el modelo predice mejor que el azar (AUC-ROC > 0,5).

### Dónde está en el código

`src/fase8_interpretacion.py`:
- Línea 151: `rng.permutation(...)`, el desorden.
- Línea 154: se aplica la **misma** permutación a todas las columnas de la dimensión.
- Línea 157: el valor p.
- Líneas 184-185: para la HG, el intervalo de confianza bootstrap y la prueba U de Mann-Whitney.

### Cómo leer el resultado

| | Dimensión | Caída del AUC-ROC | p | Decisión |
|---|---|---|---|---|
| HE1 | Climática | 0,105 | 0,0099 | Se acepta |
| HE2 | Topográfica | 0,240 | 0,0099 | Se acepta |
| HE3 | Ambiental | 0,016 | 0,0099 | Se acepta (efecto pequeño) |
| HE4 | Antropogénica | 0,067 | 0,0099 | Se acepta |

- Ejemplo de lectura: al desordenar la topografía, el AUC-ROC baja de 0,900 a 0,661 (−0,24). Es la mayor caída, así que es la dimensión de la que más depende el modelo.
- **HG:** el AUC-ROC es 0,900 con un intervalo de confianza del 95 % de [0,888; 0,912]. Todo el intervalo está muy por encima de 0,5, y la prueba U de Mann-Whitney da p < 0,001. **Se acepta la HG.**
  - El **intervalo de confianza bootstrap** se obtiene repitiendo el cálculo del AUC 1 000 veces con remuestreos de la prueba: el 95 % de esos AUC cae dentro del intervalo.
  - La **prueba U de Mann-Whitney** compara los puntajes de las celdas con incendio contra los de las celdas sin incendio. Es matemáticamente equivalente a preguntar si el AUC-ROC es mayor que 0,5.
- **La advertencia importante:** con Random Forest, la dimensión ambiental **no** fue significativa (p = 0,73). Es decir, HE3 se acepta con el modelo final, pero el efecto es pequeño y depende del modelo. Es mejor decirlo antes de que lo pregunten.

### Preguntas del jurado

**Todas las HE tienen el mismo p (0,0099); ¿cómo sabes cuál pesa más?**
Ese p es el mínimo posible con 100 repeticiones: indica que ninguna repetición igualó el AUC original. Para ordenar las dimensiones se usa el **tamaño** de la caída del AUC, no el p: topográfica (0,24) > climática (0,11) > antropogénica (0,07) > ambiental (0,02).

**¿Por qué SHAP y permutación, si las dos miden importancia?**
Miden cosas distintas:
- SHAP: cuánto usa el modelo cada indicador para formar sus predicciones.
- Permutación: cuánto empeora el modelo sin ese indicador.

Que ambas den el mismo orden (topográfica > climática > antropogénica > ambiental) refuerza la conclusión. Además, la permutación da un valor p para contrastar las hipótesis.

**¿Por qué HE3 sale significativa con un modelo y no con el otro?**
Porque su aporte es pequeño (caída de 0,016) y está cerca del límite de lo detectable. El NDVI y la cobertura casi no varían dentro del bosque analizado. La conclusión honesta es que la dimensión ambiental tiene una contribución débil y no robusta.

---

## 8. Mapas mensuales de susceptibilidad

### Qué hace y por qué

- Se hace **un mapa por mes** (enero a diciembre) para las 2 263 celdas.
- **Climatología mensual:** un mapa de "agosto" no debe depender de un agosto particular. Por eso se usa, para cada celda, el **promedio 2012-2025** de los valores que el modelo recibe en agosto: la lluvia, la temperatura y el NDVI de **julio**, con el mismo desfase que en el entrenamiento. La cobertura es la de 2024, la más reciente, y la altura, la pendiente y las distancias no cambian.
- **Modelo:** XGBoost reentrenado con **todos** los años 2012-2025, con los mismos hiperparámetros. El modelo evaluado usó solo 2012-2021 para poder medirlo de forma honesta; para el producto final conviene aprovechar todos los datos.
- **Cinco niveles con cortes naturales de Jenks:**
  - El método busca los límites que forman grupos lo más parecidos posible por dentro y lo más distintos posible entre sí.
  - Los cortes se calcularon **una vez para los 12 meses juntos**, así un color significa lo mismo en todos los meses.
  - Cortes: 0,076 / 0,253 / 0,484 / 0,722.
  - Los quintiles (20 % de celdas por nivel) no sirven aquí: más del 40 % de las celdas-mes tienen probabilidad cercana a 0, y dos cortes salían iguales.
- **Celdas fuera del dominio:** 168 celdas tenían menos de 20 % de bosque en 2024. El modelo no se entrenó con ese tipo de celdas, así que se muestran rayadas, como una extrapolación.

### Dónde está en el código

`src/fase8_mapas.py`:
- Líneas 74-81: función `climatologia_lag`, el promedio por celda y mes con el desfase de un mes.
- Línea 62: `ANIO_USO_SUELO = 2024`.
- Línea 155: el reentrenamiento con toda la tabla.
- Línea 167: el cálculo de los cortes de Jenks.
- Línea 169: la asignación de niveles.

### Cómo leer el resultado

- **De enero a mayo**, casi todo el distrito está en "muy baja".
- **En junio** el riesgo empieza a subir en el este.
- **En agosto** llega al máximo: 380 celdas en "muy alta".
- **En noviembre** vuelve a bajar.
- Las zonas de riesgo alto están siempre en la parte **baja y oriental** del distrito (hacia el valle del Huallaga y Tocache). El oeste montañoso permanece en "muy baja".

Comprobación con los incendios observados:
- En las celdas-mes "muy alta", el 70 % tuvo al menos un incendio entre 2012 y 2025.
- En las "muy baja", solo el 0,3 %.

Esta comprobación es descriptiva, porque el modelo del mapa vio esos años; la validación real es la de la sección 3.

**Importante:** la "probabilidad" del mapa (por ejemplo 0,945) **no** es la probabilidad real de que se queme esa celda. Los pesos de clase la inflan. Es un **índice para comparar** celdas y meses.

### Preguntas del jurado

**¿Por qué usar la climatología y no el clima de un año concreto?**
Porque un mapa de susceptibilidad describe la propensión típica de cada lugar en cada época del año, no un pronóstico de un año específico. Con datos de un mes concreto, el mismo modelo podría usarse como alerta temprana.

**¿Por qué 5 niveles?**
Es el estándar en mapas de susceptibilidad y riesgo (muy baja a muy alta). Es fácil de leer para los usuarios, por ejemplo las autoridades locales o los bomberos forestales.

**¿Si una celda está en "muy alta", se va a quemar?**
No necesariamente. Significa que reúne las condiciones de los lugares que más se han quemado. Que se queme depende además de que haya una fuente de ignición, casi siempre humana.

---

## 9. El visor web

### Qué hace y por qué

Es una página que se abre con doble clic en `visor/index.html`. Muestra:
- el mapa del mes elegido;
- la leyenda de los 5 niveles con cuántas celdas hay en cada uno;
- al hacer clic en una celda: su probabilidad, su nivel y los **3 factores SHAP** que más explican esa predicción, con el valor del indicador y si aumentan o reducen la susceptibilidad.

Sirve para mostrar el sistema funcionando y para que un usuario sin conocimientos de programación lo explore.

### Dónde está en el código

- `src/fase8_mapas.py`, línea 191: elige los 3 factores con mayor valor SHAP absoluto (`np.argsort`) para cada celda-mes y los guarda en `visor/datos.js`.
- `visor/index.html`:
  - Línea 92: lo que ocurre al hacer clic en una celda.
  - Línea 144: función `actualizar()`, que colorea las celdas según el mes.
  - Línea 172: arma la lista de los 3 factores.

### Cómo leer el resultado

En el panel de una celda:
- "▲ aumenta" (en rojo) significa que ese indicador empuja el riesgo hacia arriba en ese mes.
- "▼ reduce" (en azul) significa lo contrario.

El número SHAP indica cuánto empuja, en la escala interna del modelo (logit).

### Preguntas del jurado

**¿Necesita internet?**
La biblioteca del mapa (Leaflet) está incluida en la carpeta. Solo el mapa de fondo de Esri (topográfico, calles o imagen satelital) necesita conexión; sin ella, las celdas se ven igual sobre fondo gris.

**¿Cómo se actualiza con datos nuevos?**
Se corre `python src/fase8_mapas.py`, que vuelve a generar `visor/datos.js`.

---

## 10. Glosario rápido

| Término | Significado sencillo |
|---|---|
| Celda-mes | Un cuadrado de 1 km en un mes concreto; es la unidad de análisis |
| Prevalencia | Proporción de casos positivos (0,41 % de las celdas-mes tuvieron incendio) |
| Imputación | Rellenar un dato faltante con una estimación |
| Fuga de información | Que datos de la prueba se cuelen en el entrenamiento y hagan ver al modelo mejor de lo que es |
| Autocorrelación espacial | Que los lugares cercanos se parezcan entre sí |
| Pliegue | Cada uno de los 5 grupos de bloques que se turnan como conjunto de evaluación |
| Hiperparámetro | Ajuste del algoritmo que se fija antes de entrenar (por ejemplo, el número de árboles) |
| Peso de clase | Hacer que cada incendio "cuente más" al entrenar, para compensar que son pocos |
| AUC-ROC | Probabilidad de que el modelo ordene bien un par incendio / no incendio (0,5 = azar) |
| AUC-PR | Qué tan bien encuentra el modelo los incendios cuidando no dar falsas alarmas (azar = prevalencia) |
| SHAP | Reparto de cada predicción entre los indicadores |
| Permutación | Desordenar una variable para ver cuánto empeora el modelo sin ella |
| Valor p | Probabilidad de ver un resultado así de extremo si en realidad no hubiera efecto; < 0,05 se considera significativo |
| Intervalo de confianza | Rango donde probablemente está el valor real (95 % de confianza) |
| Jenks | Método para elegir los límites de las clases de un mapa según cómo se agrupan los datos |
| Logit | Escala interna de XGBoost (logaritmo de las probabilidades a favor); un SHAP de +1 en logit multiplica las probabilidades a favor por 2,7 |
