"""
Fase 9 (paso 2) - Hipotesis general (HG): el AUC-ROC del mejor modelo en la
prueba 2022-2025 frente al azar (0.5) y frente a la regresion logistica.

Bootstrap estratificado de 1 000 remuestreos (semilla 42): en cada
remuestreo se sortean con reemplazo, por separado, las celdas-mes con
incendio y las sin incendio, asi cada remuestreo conserva los 270
positivos. Los dos modelos se evaluan sobre el MISMO remuestreo (bootstrap
pareado), lo que permite dar un intervalo para la DIFERENCIA de AUC.

  - Intervalo de confianza del 95 %: percentiles 2.5 y 97.5.
  - Valor p unilateral = (1 + remuestreos en que la diferencia es <= 0) / (1 000 + 1).

"Mejor modelo" = el de mayor AUC-PR medio en la validacion espacial:
  - modelo final de la Fase 8 (siempre);
  - modelo final ajustado de la Fase 9, si ya existen sus predicciones
    (las genera fase9_calibracion.py).

Lee:
  data/processed/fase8_predicciones_prueba.parquet, data/processed/fase8_modelos_temporales.joblib
  resultados/fase9/predicciones_prueba_logistica.parquet
  resultados/fase9/predicciones_prueba_modelo_final.parquet (opcional)
Genera:
  resultados/fase9/hipotesis_general.csv, resultados/fase9/fig_hipotesis_general.png
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED_DIR, SEED
from fase8_modelo import NOMBRES
from fase9_comun import CARPETA_FASE9

N_BOOTSTRAP = 1_000
ALFA = 0.05
CLAVES = ["cell_id", "anio", "mes"]


def bootstrap_pareado(y, prob_mejor, prob_logistica):
    rng = np.random.default_rng(SEED)
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    auc_mejor, auc_log = np.empty(N_BOOTSTRAP), np.empty(N_BOOTSTRAP)
    for r in range(N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(positivos, len(positivos)), rng.choice(negativos, len(negativos))])
        auc_mejor[r] = roc_auc_score(y[idx], prob_mejor[idx])
        auc_log[r] = roc_auc_score(y[idx], prob_logistica[idx])
    return auc_mejor, auc_log


def contrastes(nombre_modelo, y, prob_mejor, prob_logistica):
    auc_mejor, auc_log = roc_auc_score(y, prob_mejor), roc_auc_score(y, prob_logistica)
    boot_mejor, boot_log = bootstrap_pareado(y, prob_mejor, prob_logistica)
    diferencia = boot_mejor - boot_log
    filas = []
    for contraste, valor, boot in [
        ("mejor modelo vs. azar (0.5)", auc_mejor - 0.5, boot_mejor - 0.5),
        ("mejor modelo vs. regresión logística", auc_mejor - auc_log, diferencia),
    ]:
        p = (1 + np.sum(boot <= 0)) / (N_BOOTSTRAP + 1)
        ic_inf, ic_sup = np.percentile(boot, [2.5, 97.5])
        filas.append({"modelo": nombre_modelo, "contraste": contraste, "auc_roc_mejor": auc_mejor, "auc_roc_logistica": auc_log,
                      "diferencia_auc": valor, "ic95_inferior": ic_inf, "ic95_superior": ic_sup, "p_valor": p,
                      "decision": "el mejor modelo es superior" if p < ALFA else "no se demuestra que sea superior"})
    ic_auc = np.percentile(boot_mejor, [2.5, 97.5])
    print(f"[{nombre_modelo}] AUC-ROC {auc_mejor:.4f} IC95 [{ic_auc[0]:.4f}, {ic_auc[1]:.4f}] | logistica {auc_log:.4f}")
    for f in filas:
        print(f"  {f['contraste']}: diferencia {f['diferencia_auc']:+.4f}, IC95 [{f['ic95_inferior']:+.4f}, {f['ic95_superior']:+.4f}], "
              f"p={f['p_valor']:.4f} -> {f['decision']}")
    return filas, boot_mejor, boot_log


def main():
    logistica = pd.read_parquet(CARPETA_FASE9 / "predicciones_prueba_logistica.parquet")

    final_fase8 = joblib.load(DATA_PROCESSED_DIR / "fase8_modelos_temporales.joblib")["modelo_final"]
    pred8 = pd.read_parquet(DATA_PROCESSED_DIR / "fase8_predicciones_prueba.parquet")
    a_evaluar = [(f"{NOMBRES[final_fase8]} (Fase 8, sin ajustar)", pred8, f"prob_{final_fase8}")]

    ruta_final9 = CARPETA_FASE9 / "predicciones_prueba_modelo_final.parquet"
    if ruta_final9.exists():
        pred9 = pd.read_parquet(ruta_final9)
        a_evaluar.append((f"{pred9['modelo'].iloc[0]} (Fase 9, ajustado)", pred9, "prob_sin_calibrar"))

    filas, distribuciones = [], {}
    for nombre, pred, columna in a_evaluar:
        unidas = pred[CLAVES + ["positivo", columna]].merge(logistica[CLAVES + ["prob_logistica"]], on=CLAVES)
        assert len(unidas) == len(logistica)
        f, boot_mejor, boot_log = contrastes(nombre, unidas["positivo"].to_numpy(), unidas[columna].to_numpy(),
                                             unidas["prob_logistica"].to_numpy())
        filas += f
        distribuciones[nombre] = boot_mejor
        distribuciones["Regresión logística"] = boot_log
    pd.DataFrame(filas).to_csv(CARPETA_FASE9 / "hipotesis_general.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(list(distribuciones.values()), tick_labels=[n.replace(" (", "\n(") for n in distribuciones], orientation="horizontal")
    ax.set_xlabel("AUC-ROC en 2022-2025 (1 000 remuestreos bootstrap)")
    ax.set_title("Hipótesis general: AUC-ROC del mejor modelo vs. regresión logística")
    fig.tight_layout()
    fig.savefig(CARPETA_FASE9 / "fig_hipotesis_general.png", dpi=150)


if __name__ == "__main__":
    main()
