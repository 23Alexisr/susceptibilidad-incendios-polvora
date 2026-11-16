const NOMBRE_DIMENSION = { climatica: "Climática", topografica: "Topográfica", ambiental: "Ambiental", antropogenica: "Antropogénica" };
const LETRAS = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
const UMBRAL_POBLADO_M = 8000;   // mas lejos que esto: "sin poblado cercano"
const NIVEL_ALTO = 3;            // niveles 3 (alta) y 4 (muy alta) entran a la lista
const AVISO_INDICE = "Índice relativo del modelo: sirve para comparar celdas y meses, no es la probabilidad real de que ocurra un incendio.";
const fmt = (v, d = 0) => v.toLocaleString("es-PE", { minimumFractionDigits: d, maximumFractionDigits: d });
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

// Escala de color del visor: la de datos.js, con el verde de "muy baja" menos
// saturado para que los niveles alto y muy alto resalten sobre el resto.
const COLORES = META.colores.slice();
COLORES[0] = "#7cab8b";

// Resumen por mes, solo con las celdas del dominio del modelo (>= 20 % de bosque)
const dominio = CELDAS.filter(c => c.dom);
const resumen = META.meses.map((_, m) => {
  const conteo = [0, 0, 0, 0, 0];
  dominio.forEach(c => conteo[c.n[m]]++);
  return { conteo, alto: conteo[3] + conteo[4], fuegos: CELDAS.reduce((s, c) => s + c.h[m], 0) };
});
const maxAlto = Math.max(...resumen.map(r => r.alto));

// --- estado ---
let mes = (new Date().getMonth() + 1) % 12;   // se abre en el mes siguiente al actual
let seleccion = null;    // celda fijada con clic
let resaltada = null;    // celda bajo el cursor en la lista
let temporizador = null;
let tecnico = false;     // modo tecnico: siempre apagado al abrir

// --- mapa ---
const mapa = L.map("mapa", { preferCanvas: true, zoomControl: false });
L.control.zoom({ position: "topleft" }).addTo(mapa);   // arriba a la izquierda: el panel de detalle ocupa la derecha
L.control.scale({ imperial: false, position: "bottomright" }).addTo(mapa);
// OpenStreetMap (403) y CARTO (piden clave) rechazan las paginas abiertas como archivo local; Esri si las acepta.
const fondos = {
  "Relieve": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Esri" }),
  "Calles": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Esri" }),
  "Satélite": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Esri, Maxar" }),
};
let fondoActual = "Relieve";
fondos[fondoActual].addTo(mapa);

const limite = L.geoJSON(META.distrito, { style: { color: "#1c2321", weight: 2.2, fill: false }, interactive: false }).addTo(mapa);
const mapaAncho = $("zonaMapa").clientWidth > 760;
mapa.fitBounds(limite.getBounds(), { paddingTopLeft: [mapaAncho ? 150 : 10, 20], paddingBottomRight: [mapaAncho ? 360 : 10, 20] });

// Centro poblado mas cercano a una celda, con la regla de los 8 km
function poblado(c) {
  const km = `${fmt(c.dres / 1000, 1)} km`;
  if (c.dres > UMBRAL_POBLADO_M) return { lejos: true, nombre: "Sin poblado cercano", detalle: "a más de 8 km", km };
  return { lejos: false, nombre: c.pob || "Centro poblado", detalle: `a ${km}`, km };
}

const poligonos = CELDAS.map((c, i) => {
  const capa = L.polygon(c.g.map(([x, y]) => [y, x]), { weight: 0.3, color: "#5b6663" });
  capa.on("click", () => seleccionar(i, false));
  capa.bindTooltip(() => {
    const p = poblado(c);
    return `<b>${fmt(c.p[mes] * 100, 1)} %</b> · ${META.niveles[c.n[mes]]}<br>${esc(p.nombre)}${p.lejos ? "" : ` · ${p.detalle}`}<br>` +
      `<span class="tenue">${c.id}${c.dom ? "" : " · fuera del dominio"}</span>`;
  }, { sticky: true, className: "tooltip-celda", direction: "top", offset: [0, -6] });
  return capa.addTo(mapa);
});
limite.bringToFront();

// --- contenido fijo ---
$("subtitulo").textContent = `Modelo ${META.modelo} · celdas de 1 km · climatología mensual 2012-2025`;
$("notaMetodo").innerHTML =
  `Cada celda se colorea según el índice que da el modelo ${META.modelo} con las condiciones típicas del mes: ` +
  `lluvia, temperatura y verdor (NDVI) promedio del mes anterior entre 2012 y 2025, cobertura MapBiomas de 2024, ` +
  `relieve y distancias a carreteras y centros poblados. Los cinco niveles se definieron con cortes naturales de Jenks, ` +
  `iguales para los doce meses, de modo que un color significa lo mismo en todo el año.<br><br>` +
  `El modelo se entrenó solo con celdas de al menos 20 % de bosque; en las demás (rayadas) el resultado es una extrapolación ` +
  `y no entran en la lista. Al elegir una celda se muestran los factores que más empujan su índice hacia arriba o hacia abajo.`;

if (META.metricas) {
  const m = META.metricas;
  const filas = [
    ["AUC-ROC, prueba 2022-2025", fmt(m.auc_roc_prueba, 3)],
    ["AUC-PR, prueba 2022-2025", fmt(m.auc_pr_prueba, 4)],
    ["AUC-ROC espacial", `${fmt(m.auc_roc_espacial, 3)} ± ${fmt(m.auc_roc_espacial_desv, 3)}`],
    ["AUC-PR espacial", `${fmt(m.auc_pr_espacial, 4)} ± ${fmt(m.auc_pr_espacial_desv, 4)}`],
    ["Prevalencia 2012-2025", `${fmt(m.prevalencia * 100, 2)} %`],
    ["Prevalencia 2022-2025", `${fmt(m.prevalencia_prueba * 100, 2)} %`],
    ["Celdas-mes", fmt(m.celdas_mes)],
    ["Con incendio", fmt(m.positivos)],
  ];
  $("metricas").innerHTML =
    `<h2>Métricas del modelo <span class="sin-versales">${esc(META.modelo)}</span></h2>` +
    `<dl class="metricas-rejilla">${filas.map(([k, v]) => `<div><dt>${k}</dt><dd class="num">${v}</dd></div>`).join("")}</dl>` +
    `<p>Validación del algoritmo entrenado con 2012-2021 (prueba temporal) y por bloques de 5 km (espacial, media ± desviación de 5 pliegues). ` +
    `El mapa usa el mismo algoritmo reentrenado con 2012-2025. Azar: AUC-ROC 0,5 y AUC-PR igual a la prevalencia. ` +
    `Valores SHAP en escala de ${esc(META.escala_shap)}.</p>`;
} else {
  $("metricas").innerHTML = `<h2>Métricas del modelo</h2><p>No disponibles: vuelve a generar datos.js con src/fase8_mapas.py.</p>`;
}

META.meses.forEach((nombre, m) => {
  const col = document.createElement("div");
  col.className = "barra-mes";
  col.title = `${nombre}: ${resumen[m].alto} celdas en nivel alto o muy alto`;
  col.innerHTML = `<div class="barra" style="height:${Math.max(4, resumen[m].alto / maxAlto * 42)}px"></div><span class="letra">${LETRAS[m]}</span>`;
  col.onclick = () => { detener(); cambiarMes(m); };
  $("lineaTiempo").appendChild(col);
});

Object.keys(fondos).forEach(nombre => {
  const b = document.createElement("button");
  b.textContent = nombre;
  b.onclick = () => {
    mapa.removeLayer(fondos[fondoActual]);
    fondoActual = nombre;
    fondos[nombre].addTo(mapa);
    if (nombre === "Satélite" && +$("opacidad").value > 0.55) $("opacidad").value = 0.5;
    actualizar();
  };
  $("fondos").appendChild(b);
});

// --- controles ---
function cambiarMes(m) { mes = m; resaltada = null; actualizar(); }
function detener() {
  clearInterval(temporizador); temporizador = null;
  $("reproducir").classList.remove("activo"); $("reproducir").innerHTML = "&#9654;";
}
$("reproducir").onclick = () => {
  if (temporizador) return detener();
  $("reproducir").classList.add("activo"); $("reproducir").innerHTML = "&#10074;&#10074;";
  temporizador = setInterval(() => cambiarMes((mes + 1) % 12), 1300);
};
$("anterior").onclick = () => { detener(); cambiarMes((mes + 11) % 12); };
$("siguiente").onclick = () => { detener(); cambiarMes((mes + 1) % 12); };
$("cerrar").onclick = () => { seleccion = null; resaltada = null; actualizar(); };
["verHistoricos", "verFueraDominio", "opacidad"].forEach(id => $(id).addEventListener("input", actualizar));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") $("anterior").click();
  if (e.key === "ArrowRight") $("siguiente").click();
  if (e.key === "Escape") $("cerrar").click();
  if (e.key === " ") { e.preventDefault(); $("reproducir").click(); }
});

$("modoTecnico").onclick = () => {
  tecnico = !tecnico;
  actualizar();
};
$("imprimir").onclick = () => { pintarImpresion(); window.print(); };
$("copiar").onclick = () => copiarLista();

// --- seleccion y resaltado ---
function seleccionar(i, desdeLista) {
  seleccion = i; resaltada = null;
  actualizar();
  if (desdeLista) {
    const centro = poligonos[i].getBounds().getCenter();
    if (!mapa.getBounds().pad(-0.15).contains(centro)) mapa.panTo(centro);
  } else {
    const fila = document.querySelector(`.fila[data-i="${i}"]`);
    if (fila) fila.scrollIntoView({ block: "nearest" });
  }
}
function resaltar(i) {
  if (i === resaltada) return;
  const previa = resaltada;
  resaltada = i;
  if (previa !== null) repintarCelda(previa);
  if (i !== null) { repintarCelda(i); poligonos[i].bringToFront(); }
  else if (seleccion !== null) poligonos[seleccion].bringToFront();
  pintarDetalle();
}
const filaDe = e => { const f = e.target.closest(".fila"); return f ? +f.dataset.i : null; };
$("lista").addEventListener("mouseover", e => { const i = filaDe(e); if (i !== null) resaltar(i); });
$("lista").addEventListener("mouseleave", () => resaltar(null));
$("lista").addEventListener("focusin", e => { const i = filaDe(e); if (i !== null) resaltar(i); });
$("lista").addEventListener("focusout", () => resaltar(null));
$("lista").addEventListener("click", e => { const i = filaDe(e); if (i !== null) seleccionar(i, true); });
$("lista").addEventListener("keydown", e => {
  const i = filaDe(e);
  if (i !== null && e.key === "Enter") { e.preventDefault(); seleccionar(i, true); }   // la barra espaciadora sigue recorriendo los meses
});

function rangoNivel(k) {
  const c = [0, ...META.cortes, 1].map(v => fmt(v * 100, 1));
  return k === 0 ? `< ${c[1]} %` : k === 4 ? `> ${c[4]} %` : `${c[k]}–${c[k + 1]} %`;
}

function valorIndicador(c, clave) {
  const mesPrevio = META.meses[(mes + 11) % 12].toLowerCase();
  switch (clave) {
    case "precip_mm": return `${fmt(c.pr[mes], 1)} mm en ${mesPrevio}`;
    case "temp_max_c": return `${fmt(c.tm[mes], 1)} °C en ${mesPrevio}`;
    case "ndvi": return `${fmt(c.nd[mes], 2)} en ${mesPrevio}`;
    case "elevacion": return `${fmt(c.elev)} m`;
    case "pendiente": return `${fmt(c.pend, 1)}°`;
    case "clase_mapbiomas": return META.clases[c.clase] || `clase ${c.clase}`;
    case "dist_carretera_m": return `${fmt(c.dcar / 1000, 1)} km`;
    case "dist_zona_residencial_m": return `${fmt(c.dres / 1000, 1)} km`;
  }
}

// Texto oscuro sobre los colores claros de la escala (muy baja, baja, media, alta)
const colorTexto = k => (k === 4) ? "#fff" : "#2b2f12";

// Celdas con menos de 20 % de bosque que el modelo pone en nivel alto o muy alto: no entran a la lista
const fueraDominioEnAlto = () => CELDAS.filter(c => !c.dom && c.n[mes] >= NIVEL_ALTO).length;
function textoFuera() {
  const n = fueraDominioEnAlto();
  return n === 0
    ? "Ninguna celda con menos de 20 % de bosque está en nivel alto o muy alto este mes."
    : `Además, ${fmt(n)} ${n === 1 ? "celda" : "celdas"} con menos de 20 % de bosque ${n === 1 ? "está" : "están"} en nivel alto o muy alto; ` +
      `no ${n === 1 ? "entra" : "entran"} en la lista porque ${n === 1 ? "está" : "están"} fuera del dominio del modelo.`;
}

// Celdas del dominio en nivel alto o muy alto, de mayor a menor nivel y, dentro del nivel, de mayor a menor indice
function celdasEnRiesgo() {
  return CELDAS.map((c, i) => ({ c, i }))
    .filter(({ c }) => c.dom && c.n[mes] >= NIVEL_ALTO)
    .sort((a, b) => b.c.n[mes] - a.c.n[mes] || b.c.p[mes] - a.c.p[mes]);
}

function estiloCelda(i, verHist, verFuera, op) {
  const c = CELDAS[i];
  const visible = c.dom || verFuera;
  const fuego = verHist && c.h[mes] > 0;
  const elegida = i === seleccion || i === resaltada;
  return {
    fillColor: COLORES[c.n[mes]],
    fillOpacity: visible ? (c.dom ? op : op * 0.4) : 0,
    opacity: visible || elegida ? 1 : 0,
    color: elegida ? "#111" : fuego ? "#7a1fa2" : "#5b6663",
    weight: elegida ? 3.5 : fuego ? 2 : 0.3,
    dashArray: c.dom || elegida ? null : "2 3",
  };
}
const controlesMapa = () => [$("verHistoricos").checked, $("verFueraDominio").checked, +$("opacidad").value];
function repintarCelda(i) { poligonos[i].setStyle(estiloCelda(i, ...controlesMapa())); }

function pintarMapa() {
  const controles = controlesMapa();
  CELDAS.forEach((_, i) => poligonos[i].setStyle(estiloCelda(i, ...controles)));
  if (seleccion !== null) poligonos[seleccion].bringToFront();
}

function pintarPanel() {
  const r = resumen[mes];
  document.body.classList.toggle("tecnico", tecnico);
  $("modoTecnico").setAttribute("aria-pressed", tecnico);
  $("mesNombre").textContent = META.meses[mes];
  document.querySelectorAll(".barra-mes").forEach((b, m) => b.classList.toggle("activo", m === mes));
  document.querySelectorAll("#fondos button").forEach(b => b.classList.toggle("activo", b.textContent === fondoActual));
  $("kpiAlto").textContent = fmt(r.alto);
  $("kpiPct").textContent = `${fmt(r.alto / dominio.length * 100, 1)} %`;
  $("kpiFuegos").textContent = fmt(r.fuegos);

  $("totalCeldas").textContent = `${fmt(dominio.length)} celdas con bosque`;
  $("barraProporcion").innerHTML = r.conteo.map((n, k) => `<span style="flex-grow:${n};background:${COLORES[k]}"></span>`).join("");
  $("leyenda").innerHTML = META.niveles.map((nombre, k) =>
    `<div class="fila-leyenda"><span class="muestra" style="background:${COLORES[k]}"></span>` +
    `<span>${nombre}<span class="rango">${rangoNivel(k)}</span></span><span class="conteo num">${fmt(r.conteo[k])}</span></div>`
  ).join("") +
    `<div class="fila-leyenda"><span class="muestra rayado"></span><span>Menos de 20 % de bosque</span><span class="conteo num">${fmt(CELDAS.length - dominio.length)}</span></div>`;
}

function pintarLista() {
  const filas = celdasEnRiesgo();
  const r = resumen[mes];
  $("listaResumen").textContent = filas.length
    ? `${META.meses[mes]} · ${fmt(filas.length)} celdas: ${fmt(r.conteo[4])} muy alta, ${fmt(r.conteo[3])} alta`
    : META.meses[mes];
  $("listaFuera").textContent = textoFuera();
  $("lista").innerHTML = filas.length ? filas.map(({ c, i }) => {
    const k = c.n[mes], p = poblado(c);
    return `<div class="fila n${k}${i === seleccion ? " activa" : ""}" data-i="${i}" tabindex="0" role="button">
      <span class="insignia" style="background:${COLORES[k]};color:${colorTexto(k)}">${META.niveles[k]}</span>
      <span class="fila-texto"><b class="${p.lejos ? "lejos" : ""}">${esc(p.nombre)}</b><small>${p.detalle}</small></span>
      <span class="fila-id num">${c.id}<small>${fmt(c.p[mes] * 100, 1)} %</small></span>
    </div>`;
  }).join("") :
    `<div class="lista-vacia">En ${META.meses[mes].toLowerCase()} ninguna celda con bosque llega al nivel alto o muy alto.</div>`;
}

// Lista del mes como texto con tabuladores (se pega bien en Excel o Word)
function textoLista() {
  const filas = celdasEnRiesgo();
  const cabecera = ["Nivel", "Centro poblado más cercano", "Distancia", "Celda", "Índice (%)"];
  const lineas = filas.map(({ c }) => {
    const p = poblado(c);
    return [META.niveles[c.n[mes]], p.nombre, p.lejos ? "más de 8 km" : p.km, c.id, fmt(c.p[mes] * 100, 1)].join("\t");
  });
  return [
    `Celdas en susceptibilidad alta y muy alta · ${META.meses[mes]} · distrito de Pólvora`,
    `${filas.length} celdas (solo celdas con al menos 20 % de bosque)`,
    textoFuera(),
    AVISO_INDICE,
    "",
    cabecera.join("\t"),
    ...lineas,
  ].join("\n");
}

function copiarLista() {
  const texto = textoLista();
  const listo = () => {
    const b = $("copiar"); b.textContent = "Copiada ✓"; b.classList.add("hecho");
    setTimeout(() => { b.textContent = "Copiar lista"; b.classList.remove("hecho"); }, 1600);
  };
  const respaldo = () => {   // para navegadores que no dan acceso al portapapeles en archivos locales
    const t = document.createElement("textarea");
    t.value = texto; t.style.position = "fixed"; t.style.opacity = "0";
    document.body.appendChild(t); t.select();
    try { document.execCommand("copy"); listo(); } finally { t.remove(); }
  };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(texto).then(listo, respaldo);
  else respaldo();
}

function pintarImpresion() {
  const filas = celdasEnRiesgo();
  const hoy = new Date().toLocaleDateString("es-PE", { day: "numeric", month: "long", year: "numeric" });
  $("impresion").innerHTML =
    `<h1>Celdas en susceptibilidad alta y muy alta · ${META.meses[mes]}</h1>` +
    `<div class="imp-sub">Distrito de Pólvora, Tocache, San Martín · ${filas.length} celdas de 1 km con al menos 20 % de bosque · generado el ${hoy}</div>` +
    `<div class="imp-aviso">${AVISO_INDICE}<br>${textoFuera()}</div>` +
    `<table><thead><tr><th>N.º</th><th>Nivel</th><th>Centro poblado más cercano</th><th class="der">Distancia</th><th>Celda</th>` +
    `<th class="der">Índice</th></tr></thead><tbody>` +
    filas.map(({ c }, n) => {
      const p = poblado(c);
      return `<tr><td>${n + 1}</td><td>${META.niveles[c.n[mes]]}</td><td>${esc(p.nombre)}</td>` +
        `<td class="der">${p.lejos ? "más de 8 km" : p.km}</td><td>${c.id}</td>` +
        `<td class="der">${fmt(c.p[mes] * 100, 1)} %</td></tr>`;
    }).join("") + `</tbody></table>`;
}

function pintarDetalle() {
  const idx = resaltada !== null ? resaltada : seleccion;
  const hay = idx !== null;
  $("detalleVacio").hidden = hay;
  $("detalleContenido").hidden = !hay;
  document.querySelectorAll(".fila.activa").forEach(f => f.classList.remove("activa"));
  if (seleccion !== null) {
    const fila = document.querySelector(`.fila[data-i="${seleccion}"]`);
    if (fila) fila.classList.add("activa");
  }
  if (!hay) return;
  const c = CELDAS[idx], k = c.n[mes], p = poblado(c);

  $("dCelda").textContent = `Celda ${c.id} · ${META.meses[mes]}`;
  $("dProb").textContent = `${fmt(c.p[mes] * 100, 1)} %`;
  $("dNivel").textContent = META.niveles[k];
  $("dNivel").style.background = COLORES[k];
  $("dNivel").style.color = colorTexto(k);
  $("dPoblado").innerHTML = p.lejos ? `Sin poblado cercano (el más próximo está a más de 8 km)` : `<b>${esc(p.nombre)}</b> ${p.detalle}`;
  $("dAviso").innerHTML = c.dom ? "" :
    `<div class="aviso">Esta celda tiene ${fmt(c.bosque, 1)} % de bosque, menos del 20 % con que se entrenó el modelo: tómalo como una extrapolación.</div>`;

  const factores = c.f[mes];
  const maxAbs = Math.max(...factores.map(([, v]) => Math.abs(v)));
  $("dFactores").innerHTML = factores.map(([i, v], r) => {
    const sube = v >= 0, largo = Math.abs(v) / maxAbs * 50;
    const relleno = sube ? `left:50%;width:${largo}%;background:var(--sube)` : `right:50%;width:${largo}%;background:var(--baja)`;
    return `<div class="factor">
      <div class="factor-cabecera"><span>${r + 1}. <b>${META.indicadores[i]}</b></span>
        <span class="efecto ${sube ? "sube" : "baja"}">${sube ? "▲ aumenta" : "▼ reduce"}</span></div>
      <div class="factor-meta">${NOMBRE_DIMENSION[META.dimension[i]]} · ${valorIndicador(c, META.claves[i])}<span class="solo-tecnico"> · SHAP ${sube ? "+" : ""}${fmt(v, 2)}</span></div>
      <div class="pista"><div class="relleno" style="${relleno}"></div></div>
    </div>`;
  }).join("");

  const maxP = Math.max(...c.p, 1e-9);
  $("dAnio").innerHTML = c.p.map((pm, m) =>
    `<div class="col${m === mes ? " activo" : ""}" data-mes="${m}"
       title="${META.meses[m]}: ${fmt(pm * 100, 1)} % · ${META.niveles[c.n[m]]}${c.h[m] ? ` · ${c.h[m]} incendio(s) registrado(s)` : ""}">
      <span class="fuego">${c.h[m] ? "●" : ""}</span>
      <div class="b" style="height:${Math.max(3, pm / maxP * 42)}px;background:${COLORES[c.n[m]]}"></div><span>${LETRAS[m]}</span></div>`
  ).join("");
  document.querySelectorAll("#dAnio .col").forEach(col => col.onclick = () => { detener(); seleccion = idx; cambiarMes(+col.dataset.mes); });

  $("dDatos").innerHTML = [
    ["Bosque en 2024", `${fmt(c.bosque, 1)} %`],
    ["Cobertura 2024", META.clases[c.clase] || `clase ${c.clase}`],
    ["Elevación", `${fmt(c.elev)} m`],
    ["Pendiente", `${fmt(c.pend, 1)}°`],
    ["Distancia a carretera", `${fmt(c.dcar / 1000, 1)} km`],
    ["Centro poblado más cercano", c.pob ? `${esc(c.pob)} (${p.km})` : p.km],
    [`Incendios en ${META.meses[mes].toLowerCase()} (2012-2025)`, fmt(c.h[mes])],
    ["Incendios en todo el año (2012-2025)", fmt(c.h.reduce((a, b) => a + b, 0))],
  ].map(([clave, v]) => `<dt>${clave}</dt><dd class="num">${v}</dd>`).join("");
}

function actualizar() { pintarMapa(); pintarPanel(); pintarLista(); pintarDetalle(); }
actualizar();
