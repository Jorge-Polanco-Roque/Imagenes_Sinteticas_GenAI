// Sala de Inspección — lógica del panel
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (p, o) => fetch(p, o).then(r => r.json());

let genMode = "inpaint";
let pollTimer = null, activeJob = null;

// ---------- Router ----------
function show(view) {
  $$(".view").forEach(v => v.classList.add("hidden"));
  $(`#view-${view}`).classList.remove("hidden");
  $$(".nav-item").forEach(n => n.classList.toggle("active", n.dataset.view === view));
  if (view === "galeria") loadGallery();
  if (view === "datos") loadCrops(currentCrop);
  if (view === "panel") loadPanel();
}
$$(".nav-item").forEach(n => n.onclick = () => { location.hash = n.dataset.view; });
window.addEventListener("hashchange", () => show(location.hash.slice(1) || "panel"));

// ---------- Telemetría ----------
async function loadStatus() {
  try {
    const s = await api("/api/status");
    const cd = $("#r-comfy");
    cd.innerHTML = `<i class="dot ${s.comfyui ? "on" : "off"}"></i>${s.comfyui ? "en línea" : "apagado"}`;
    $("#r-device").textContent = (s.device || "—").toUpperCase();
    $("#r-loras").textContent = s.loras.length;
    $("#r-crops").textContent = s.crops_total;
    $("#r-gen").textContent = s.generated;
    // llenar selector de LoRA
    const sel = $("#g-lora");
    if (sel && sel.dataset.filled !== "1" && s.loras.length) {
      sel.innerHTML = s.loras.map(l => `<option>${l}</option>`).join("");
      sel.dataset.filled = "1";
    }
    return s;
  } catch { return null; }
}

// ---------- Panel ----------
async function loadPanel() {
  const s = await loadStatus();
  if (!s) return;
  $("#panel-cards").innerHTML = `
    <div class="stat accent"><div class="k">Recortes semilla</div><div class="v">${s.crops_total}</div><div class="sub">defectos reales por clase</div></div>
    <div class="stat"><div class="k">Modelos LoRA</div><div class="v">${s.loras.length}</div><div class="sub">${s.loras[0] || "sin entrenar"}</div></div>
    <div class="stat flaw-accent"><div class="k">Imágenes generadas</div><div class="v">${s.generated}</div><div class="sub">con máscara + etiqueta</div></div>
    <div class="stat"><div class="k">Motor</div><div class="v" style="font-size:26px">${s.comfyui ? "LISTO" : "OFF"}</div><div class="sub">${(s.device || "").toUpperCase()} · ComfyUI</div></div>`;
  const steps = [
    ["01", "Datos semilla", s.crops_total > 0, `${s.crops_total} recortes`],
    ["02", "Entrenar LoRA", s.loras.length > 0, s.loras.length ? "listo" : "pendiente"],
    ["03", "Generar", s.generated > 0, `${s.generated} imágenes`],
    ["04", "Exportar dataset", false, "YOLO / máscaras"],
    ["05", "Entrenar detector", false, "holdout real"],
  ];
  $("#panel-flow").innerHTML = `<h3>Flujo de trabajo</h3>
    <div class="flow">${steps.map(([n, t, ok, d]) => `
      <div class="step"><div class="sn">${n}</div>
        <div class="st ${ok ? "done" : "pend"}">${ok ? "◉ " : "○ "}${t}</div>
        <div class="sd">${d}</div></div>`).join("")}</div>`;
}

// ---------- Galería ----------
async function loadGallery() {
  const { items } = await api("/api/gallery?dir=_test_salida");
  const grid = $("#gal-grid"), empty = $("#gal-empty");
  if (!items.length) {
    grid.innerHTML = ""; empty.classList.remove("hidden");
    empty.textContent = "Sin imágenes aún. Andá a Generar para producir las primeras.";
    return;
  }
  empty.classList.add("hidden");
  const overlay = $("#tg-overlay").checked;
  grid.innerHTML = items.map(it => {
    let boxHtml = "";
    if (it.box) {
      const [cx, cy, w, h] = it.box;
      const l = (cx - w / 2) * 100, t = (cy - h / 2) * 100;
      boxHtml = `<div class="flaw-box" style="left:${l}%;top:${t}%;width:${w * 100}%;height:${h * 100}%">
                   <span class="tag">${it.clase}</span></div>`;
    }
    return `<div class="inspect ${overlay ? "" : "hide-overlay"}" data-img="${it.image}">
      <div class="inspect-frame">
        <img src="${it.image}" loading="lazy">
        <span class="corner tr"></span><span class="corner bl"></span>
        <span class="scanline"></span>${boxHtml}
      </div>
      <div class="inspect-meta"><span class="nm">${it.name}</span><span class="cls">${it.clase}</span></div>
    </div>`;
  }).join("");
  $$(".inspect", grid).forEach(c => c.onclick = () => openViewer(c.dataset.img));
}
$("#tg-overlay").onchange = () => $$(".inspect").forEach(c => c.classList.toggle("hide-overlay", !$("#tg-overlay").checked));
$("#btn-refresh-gal").onclick = loadGallery;

// ---------- Visor ----------
function openViewer(src) { $("#viewer-img").src = src; $("#viewer").classList.remove("hidden"); }
$("#viewer-close").onclick = () => $("#viewer").classList.add("hidden");
$("#viewer").onclick = e => { if (e.target.id === "viewer") $("#viewer").classList.add("hidden"); };

// ---------- Crops ----------
let currentCrop = "scratch";
async function loadCrops(clase) {
  currentCrop = clase;
  const s = await api("/api/status");
  $("#crops-chips").innerHTML = Object.entries(s.crops).map(([c, n]) =>
    `<button class="chip ${c === clase ? "active" : ""}" data-c="${c}">${c}<b>${n}</b></button>`).join("");
  $$("#crops-chips .chip").forEach(ch => ch.onclick = () => loadCrops(ch.dataset.c));
  const { items } = await api(`/api/crops?clase=${clase}&limit=12`);
  $("#crops-strip").innerHTML = items.length
    ? items.map(u => `<img src="${u}" loading="lazy" onclick="__view('${u}')">`).join("")
    : `<p class="hint" style="grid-column:1/-1">Sin recortes de "${clase}". Descargá NEU-DET.</p>`;
}
window.__view = openViewer;

// ---------- Modo generar ----------
$$("#seg-mode .seg").forEach(b => b.onclick = () => {
  $$("#seg-mode .seg").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); genMode = b.dataset.v;
  $("#g-clean-wrap").style.display = genMode === "inpaint" ? "" : "none";
  $("#mode-hint").textContent = genMode === "inpaint"
    ? "Pinta defectos dentro de una máscara sobre fotos de rollos limpios. La máscara se exporta como etiqueta."
    : "Genera el rollo completo con defecto desde cero. Da variedad de fondos; sin etiqueta automática.";
});

// ---------- Lanzar procesos ----------
$("#form-generar").onsubmit = e => {
  e.preventDefault();
  run("/api/generate", {
    mode: genMode, n: +$("#g-n").value, denoise: +$("#g-denoise").value,
    lora: $("#g-lora").value, clean_dir: $("#g-clean").value, out: $("#g-out").value,
  });
};
$("#form-entrenar").onsubmit = e => {
  e.preventDefault();
  run("/api/train", { steps: +$("#t-steps").value, dim: +$("#t-dim").value, name: $("#t-name").value });
};
$("#btn-dataset").onclick = () => run("/api/dataset", { mode: "full", size: 512 });
$("#btn-prepare").onclick = () => run("/api/prepare", { limit: +$("#d-limit").value });
$("#btn-deploy").onclick = async () => {
  try { const r = await api("/api/deploy_lora", { method: "POST" });
    toastLog(`LoRA publicado: ${r.lora}. Reiniciá ComfyUI si no aparece en la lista.`); loadStatus(); }
  catch { toastLog("Aún no hay LoRA entrenado en runtime/lora_out."); }
};

async function run(url, body) {
  const r = await api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (r.job) openDrawer(r.job, url);
}

// ---------- Cajón de procesos ----------
function openDrawer(jid, url) {
  activeJob = jid;
  $("#drawer").classList.add("open");
  $("#drawer-name").textContent = jobName(url);
  $("#drawer-pulse").className = "pulse";
  if (pollTimer) clearInterval(pollTimer);
  poll();
  pollTimer = setInterval(poll, 1300);
}
function jobName(url) {
  return { "/api/generate": "Generación", "/api/train": "Entrenamiento LoRA",
           "/api/dataset": "Descarga NEU-DET", "/api/prepare": "Preparar dataset" }[url] || "Proceso";
}
async function poll() {
  if (!activeJob) return;
  const r = await api(`/api/jobs/${activeJob}/log?tail=200`);
  const log = $("#drawer-log");
  log.textContent = r.log || "iniciando…";
  log.scrollTop = log.scrollHeight;
  $("#drawer-status").textContent = r.status.toUpperCase();
  const p = $("#drawer-pulse");
  if (r.status === "running") p.className = "pulse";
  else {
    p.className = "pulse " + (r.status === "done" ? "ok" : "err");
    clearInterval(pollTimer); pollTimer = null;
    loadStatus();
    if (jobName && $("#view-galeria").classList.contains("hidden") === false) loadGallery();
  }
}
function toastLog(msg) {
  $("#drawer").classList.add("open");
  $("#drawer-name").textContent = "Aviso";
  $("#drawer-pulse").className = "pulse ok";
  $("#drawer-status").textContent = "OK";
  $("#drawer-log").textContent = msg;
}
$("#btn-close-drawer").onclick = () => { $("#drawer").classList.remove("open"); if (pollTimer) clearInterval(pollTimer); };
$("#btn-stop").onclick = async () => { if (activeJob) { await api(`/api/jobs/${activeJob}/stop`, { method: "POST" }); poll(); } };

// ---------- Init ----------
show(location.hash.slice(1) || "panel");
loadStatus();
setInterval(loadStatus, 5000);
