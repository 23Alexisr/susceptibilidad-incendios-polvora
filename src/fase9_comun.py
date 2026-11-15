"""
Fase 9 - Piezas compartidas por los scripts fase9_*.py.

Reutiliza la tabla y las columnas de fase8_modelo.py (cargar_datos) para que
todos los pasos usen exactamente los mismos datos, la misma particion
temporal (2012-2021 / 2022-2025) y los mismos bloques de 5 x 5 km.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import GroupKFold

from config import ROOT_DIR
from fase8_modelo import ANIO_CORTE_TEST, N_FOLDS_ESPACIAL, cargar_datos

CARPETA_FASE9 = ROOT_DIR / "resultados" / "fase9"
CARPETA_FASE9.mkdir(parents=True, exist_ok=True)


def datos_separados():
    """Devuelve la tabla completa, el entrenamiento 2012-2021 y la prueba 2022-2025."""
    tabla, cols_features, cols_clase = cargar_datos()
    entrenamiento = tabla[tabla["anio"] < ANIO_CORTE_TEST].reset_index(drop=True)
    prueba = tabla[tabla["anio"] >= ANIO_CORTE_TEST].reset_index(drop=True)
    return tabla, entrenamiento, prueba, cols_features, cols_clase


def pliegues_espaciales(datos):
    """Los 5 pliegues por bloques de 5 x 5 km (GroupKFold, igual que fase8_modelo.py)."""
    return list(GroupKFold(n_splits=N_FOLDS_ESPACIAL).split(datos, datos["positivo"], groups=datos["bloque_espacial"]))
