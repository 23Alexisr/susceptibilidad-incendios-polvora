"""
Fase 9 (paso 4) - Calibracion isotonica del modelo final.

Por que: los pesos de clase hacen que las "probabilidades" del modelo esten
infladas (pueden llegar a 0.9 aunque solo el 0.4 % de las celdas-mes se
quema). La regresion isotonica aprende una funcion creciente que convierte
el puntaje del modelo en una probabilidad realista.

Se ajusta SOLO con 2012-2021: con las predicciones fuera de pliegue del
modelo final (validacion por bloques de 5 km dentro de 2012-2021, ver
fase9_ajuste_hiperparametros.py). No se usan las predicciones del modelo
sobre sus propios datos de entrenamiento, porque estarian sobreajustadas.
Despues se aplica, sin cambios, a las predicciones de 2022-2025.

Se compara antes y despues sobre 2022-2025:
  - Brier score: error cuadratico medio de la probabilidad (0 = perfecto).
    Referencia: predecir siempre la prevalencia de 2012-2021.
  - Curva de calibracion: 10 grupos con el mismo numero de celdas-mes
    (cuantiles); en cada grupo, probabilidad media predicha vs. proporcion
    real de incendios.

Lee:
  resultados/fase9/modelo_final_ajustado.joblib
  resultados/fase9/predicciones_fuera_de_pliegue_2012_2021.parquet
Genera (en resultados/fase9/):
  calibrador_isotonico.joblib, calibracion_metricas.csv, calibracion_curva.csv,
  predicciones_prueba_modelo_final.parquet, fig_calibracion.png
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fase8_modelo import NOMBRES
from fase9_comun import CARPETA_FASE9, datos_separados

N_GRUPOS = 10


def main():
    paquete = joblib.load(CARPETA_FASE9 / "modelo_final_ajustado.joblib")
    modelo, cols_features = paquete["modelo"], paquete["cols_features"]
    nombre = NOMBRES[paquete["algoritmo"]]
    print(f"Modelo final: {nombre} {paquete['params']}")

    # --- Ajuste de la calibracion: solo 2012-2021 ---
    oof = pd.read_parquet(CARPETA_FASE9 / "predicciones_fuera_de_pliegue_2012_2021.parquet")
    assert oof["anio"].max() <= 2021
    calibrador = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrador.fit(oof["prob_fuera_de_pliegue"], oof["positivo"])
    joblib.dump(calibrador, CARPETA_FASE9 / "calibrador_isotonico.joblib")
    prevalencia_entrenamiento = oof["positivo"].mean()

    # --- Evaluacion en 2022-2025 ---
    _, _, prueba, _, _ = datos_separados()
    y = prueba["positivo"].to_numpy()
    prob_sin = modelo.predict_proba(prueba[cols_features])[:, 1]
    prob_cal = calibrador.predict(prob_sin)

    pred = prueba[["cell_id", "anio", "mes", "positivo"]].copy()
    pred["modelo"] = nombre
    pred["prob_sin_calibrar"] = prob_sin
    pred["prob_calibrada"] = prob_cal
    pred.to_parquet(CARPETA_FASE9 / "predicciones_prueba_modelo_final.parquet", index=False)

    filas = []
    for version, prob in [("sin calibrar", prob_sin), ("calibrada (isotónica)", prob_cal),
                          ("referencia: prevalencia 2012-2021", np.full(len(y), prevalencia_entrenamiento))]:
        filas.append({"version": version, "brier": brier_score_loss(y, prob),
                      "prob_media_predicha": prob.mean(), "proporcion_real": y.mean(),
                      "auc_roc": roc_auc_score(y, prob), "auc_pr": average_precision_score(y, prob)})
    metricas = pd.DataFrame(filas)
    metricas.to_csv(CARPETA_FASE9 / "calibracion_metricas.csv", index=False)
    print(metricas.round(5).to_string(index=False))

    curvas = []
    for version, prob in [("sin calibrar", prob_sin), ("calibrada", prob_cal)]:
        real, predicha = calibration_curve(y, prob, n_bins=N_GRUPOS, strategy="quantile")
        curvas.append(pd.DataFrame({"version": version, "prob_media_predicha": predicha, "proporcion_real": real}))
    curvas = pd.concat(curvas, ignore_index=True)
    curvas.to_csv(CARPETA_FASE9 / "calibracion_curva.csv", index=False)
    print(curvas.round(5).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, version, color in zip(axes, ["sin calibrar", "calibrada"], ["#c44e52", "#4c72b0"]):
        sub = curvas[curvas["version"] == version]
        limite = max(sub["prob_media_predicha"].max(), sub["proporcion_real"].max()) * 1.05
        ax.plot([0, limite], [0, limite], "k--", lw=1, label="Calibración perfecta")
        ax.plot(sub["prob_media_predicha"], sub["proporcion_real"], "o-", color=color, label=nombre)
        brier = metricas.loc[metricas["version"].str.startswith(version[:8]), "brier"].iloc[0]
        ax.set(xlim=(0, limite), ylim=(0, limite), xlabel="Probabilidad media predicha",
               ylabel="Proporción real de celdas-mes con incendio", title=f"{version.capitalize()} (Brier = {brier:.5f})")
        ax.legend(loc="upper left")
    # con muchos empates en 0 los cuantiles se repiten y quedan menos de 10 grupos
    fig.suptitle("Curva de calibración en 2022-2025 (grupos por cuantiles; los empates reducen su número)")
    fig.tight_layout()
    fig.savefig(CARPETA_FASE9 / "fig_calibracion.png", dpi=150)
    print("Guardado: fig_calibracion.png")


if __name__ == "__main__":
    main()
