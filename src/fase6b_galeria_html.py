"""
Fase 6b (paso 9, complemento) - Arma una pagina HTML local simple para
revisar en el navegador los pares antes/despues de Sentinel-2 generados
por fase6b_muestra_visual.py, con los metadatos de cada evento.
"""
import pandas as pd

from config import FIGURES_DIR

CARPETA = FIGURES_DIR / "fase6b_muestra_visual"


def main() -> None:
    df = pd.read_csv(CARPETA / "metadatos.csv")

    filas_html = []
    for _, f in df.iterrows():
        nombre_base = f"evento_{int(f['evento_id']):05d}"
        img_antes = f"{nombre_base}_antes.png" if f["imagen_antes_ok"] else None
        img_despues = f"{nombre_base}_despues.png" if f["imagen_despues_ok"] else None

        filas_html.append(f"""
        <div class="tarjeta">
          <div class="cabecera">
            <b>Evento {int(f['evento_id'])}</b> — {f['fecha_inicio']} a {f['fecha_fin']}
            · {int(f['n_detecciones'])} detección(es) · {f['categoria_dominante']}
          </div>
          <div class="imagenes">
            <div class="col">
              <div class="etiqueta">Antes</div>
              {f'<img src="fase6b_muestra_visual/{img_antes}">' if img_antes else '<div class="sin_imagen">Sin imagen (nubosidad)</div>'}
            </div>
            <div class="col">
              <div class="etiqueta">Después</div>
              {f'<img src="fase6b_muestra_visual/{img_despues}">' if img_despues else '<div class="sin_imagen">Sin imagen</div>'}
            </div>
          </div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Muestra visual — eventos de incendio forestal (Fase 6b)</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 24px; }}
  h1 {{ font-size: 1.4rem; }}
  p.nota {{ color: #aaa; max-width: 800px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; margin-top: 20px; }}
  .tarjeta {{ background: #262626; border-radius: 8px; padding: 12px; }}
  .cabecera {{ font-size: 0.85rem; margin-bottom: 8px; color: #ccc; }}
  .imagenes {{ display: flex; gap: 8px; }}
  .col {{ flex: 1; }}
  .etiqueta {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; text-transform: uppercase; }}
  img {{ width: 100%; border-radius: 4px; display: block; background: #000; }}
  .sin_imagen {{ width: 100%; aspect-ratio: 1; background: #333; border-radius: 4px; display: flex;
                 align-items: center; justify-content: center; color: #666; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
  <h1>Muestra visual — 50 eventos de incendio forestal (definición extendida, con VIIRS, 2018-2025)</h1>
  <p class="nota">
    Composite Sentinel-2 (bandas B12-SWIR2/B8-NIR/B4-Rojo, resalta cicatrices de quema en tonos marrón/rojizo
    sobre vegetación sana en verde). "Antes" = mediana de imágenes hasta 60-270 días previos al evento.
    "Después" = mediana de imágenes 60-365 días después del fin del evento (margen para que el humo se disipe).
    Revisa si el patrón visual (cicatriz de quema, apertura de dosel) es consistente con la clasificación de la Fase 3b.
  </p>
  <div class="grid">
    {"".join(filas_html)}
  </div>
</body>
</html>
"""

    ruta = FIGURES_DIR / "fase6b_galeria_muestra_visual.html"
    ruta.write_text(html, encoding="utf-8")
    print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
