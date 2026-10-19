"""
Configuracion compartida del proyecto.

Por que un modulo aparte: todos los scripts de las fases siguientes necesitan
la misma semilla aleatoria, las mismas rutas de carpetas y el mismo proyecto
de Earth Engine. Centralizarlo aqui evita que cada script repita estos valores
y que se desincronicen (por ejemplo, dos semillas distintas en dos scripts).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Semilla fija para que todo el pipeline (muestreo, particion train/test,
# entrenamiento de modelos) sea reproducible. Se usa en todas las fases
# que involucren aleatoriedad.
SEED = 42

# Raiz del proyecto = carpeta que contiene src/. Se calcula relativa a este
# archivo para que los scripts funcionen sin importar desde donde se ejecuten.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

# Carga las variables de entorno desde .env (no versionado). Aqui es donde
# vive el ID del proyecto de Google Cloud requerido por Earth Engine.
load_dotenv(ROOT_DIR / ".env")
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID")
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
