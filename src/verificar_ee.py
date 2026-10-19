"""
Fase 0 - Verificacion de autenticacion con Earth Engine.

Este script NO descarga datos: solo confirma que la sesion puede
autenticarse e inicializarse contra el proyecto de Google Cloud indicado
en .env (EE_PROJECT_ID), y hace una consulta minima (contar una imagen)
para probar que la API responde.

Como se usa:
    1. La primera vez, ejecuta en la terminal (fuera de este script):
       .venv\\Scripts\\python -m ee.cli.authenticate
       o simplemente deja que ee.Authenticate() abra el navegador la
       primera vez que corras este script.
    2. Completa EE_PROJECT_ID en tu archivo .env (ver .env.example).
    3. Corre: .venv\\Scripts\\python src\\verificar_ee.py
"""
import sys

import ee

from config import EE_PROJECT_ID


def main() -> None:
    if not EE_PROJECT_ID:
        print(
            "ERROR: no se encontro EE_PROJECT_ID en el archivo .env.\n"
            "Copia .env.example como .env y coloca el ID de tu proyecto de "
            "Google Cloud vinculado a Earth Engine."
        )
        sys.exit(1)

    # ee.Authenticate() guarda credenciales locales reutilizables (no las
    # copia el repo). Si ya existen, no abre el navegador de nuevo.
    try:
        ee.Authenticate()
    except Exception as exc:  # noqa: BLE001 - queremos reportar cualquier fallo de auth
        print(f"ERROR en ee.Authenticate(): {exc}")
        sys.exit(1)

    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR en ee.Initialize(project='{EE_PROJECT_ID}'): {exc}")
        sys.exit(1)

    # Consulta minima para confirmar que el backend responde de verdad,
    # no solo que la inicializacion local no lanzo error.
    try:
        imagen = ee.Image("USGS/SRTMGL1_003")
        info = imagen.getInfo()
        print("Conexion con Earth Engine OK.")
        print(f"Proyecto usado: {EE_PROJECT_ID}")
        print(f"Prueba de consulta (imagen SRTM): bandas = {[b['id'] for b in info['bands']]}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR al consultar Earth Engine: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
