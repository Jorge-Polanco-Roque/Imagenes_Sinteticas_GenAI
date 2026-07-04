#!/usr/bin/env python3
"""
Sala de Inspección — backend FastAPI del studio de datos sintéticos.

Envuelve los scripts del proyecto (descarga_neu_det, prepara_dataset, genera_lote),
el entrenamiento kohya y la API de ComfyUI en una sola app local.

Lanzar:  python ui/app.py       (o)   uvicorn ui.app:app --port 8500
Abrir:   http://127.0.0.1:8500
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent            # raíz del proyecto
STATIC = Path(__file__).resolve().parent / "static"
RUNTIME = ROOT / "runtime"
PY_MAIN = RUNTIME / "venv" / "bin" / "python"            # pillow/numpy/requests
PY_KOHYA = RUNTIME / "venv_kohya" / "bin" / "python"     # entrenamiento
COMFY = "http://127.0.0.1:8188"
CLASES = ["scratch", "dent", "stain", "crease", "corrosion", "tear"]
LOGDIR = RUNTIME / "ui_jobs"
LOGDIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sala de Inspección")

# --------------------------------------------------------------------------- #
#  Gestor de trabajos (procesos largos: generar, entrenar, descargar)
# --------------------------------------------------------------------------- #
JOBS = {}          # id -> dict
_job_seq = 0
_lock = threading.Lock()


def launch(name, cmd, py=PY_MAIN, extra_env=None):
    global _job_seq
    with _lock:
        _job_seq += 1
        jid = f"{int(time.time())}-{_job_seq}"
    log_path = LOGDIR / f"{jid}.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    full = [str(py)] + cmd
    f = open(log_path, "w")
    proc = subprocess.Popen(full, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, env=env)
    JOBS[jid] = {"id": jid, "name": name, "cmd": " ".join(cmd[:2]),
                 "status": "running", "log": str(log_path), "proc": proc,
                 "started": time.time(), "file": f}
    return jid


def refresh(job):
    if job["status"] == "running":
        rc = job["proc"].poll()
        if rc is not None:
            job["status"] = "done" if rc == 0 else "failed"
            job["rc"] = rc
            try:
                job["file"].close()
            except Exception:
                pass
    return job


def safe_path(rel):
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(ROOT)):
        raise HTTPException(400, "ruta fuera del proyecto")
    return p


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def status():
    # ComfyUI
    comfy_up, device = False, "—"
    try:
        r = requests.get(f"{COMFY}/system_stats", timeout=2)
        if r.ok:
            comfy_up = True
            devs = r.json().get("devices", [])
            if devs:
                device = devs[0].get("type", "mps")
    except Exception:
        pass
    ck = RUNTIME / "ComfyUI" / "models" / "checkpoints"
    lo = RUNTIME / "ComfyUI" / "models" / "loras"
    checkpoints = sorted(f.name for f in ck.glob("*.safetensors")) if ck.exists() else []
    loras = sorted(f.name for f in lo.glob("*.safetensors")) if lo.exists() else []
    crops = {}
    for c in CLASES:
        d = ROOT / "defectos_reales" / c
        crops[c] = len(list(d.glob("*.png"))) if d.exists() else 0
    gen_dir = ROOT / "_test_salida" / "images"
    generated = len(list(gen_dir.glob("*.png"))) if gen_dir.exists() else 0
    running = sum(1 for j in JOBS.values() if refresh(j)["status"] == "running")
    return {"comfyui": comfy_up, "device": device, "checkpoints": checkpoints,
            "loras": loras, "crops": crops, "crops_total": sum(crops.values()),
            "generated": generated, "jobs_running": running}


@app.get("/api/gallery")
def gallery(dir: str = "_test_salida"):
    base = safe_path(dir)
    imgs_dir, masks_dir, labels_dir = base / "images", base / "masks", base / "labels"
    if not imgs_dir.exists():
        return {"items": []}
    classes = CLASES
    ctxt = base / "classes.txt"
    if ctxt.exists():
        classes = [l.strip() for l in ctxt.read_text().splitlines() if l.strip()]
    items = []
    for img in sorted(imgs_dir.glob("*.png"), reverse=True):
        stem = img.stem
        mask = masks_dir / f"{stem}.png"
        label = labels_dir / f"{stem}.txt"
        box, clase = None, stem.split("_")[0]
        if label.exists():
            parts = label.read_text().split()
            if len(parts) == 5:
                cid, cx, cy, w, h = parts
                box = [float(cx), float(cy), float(w), float(h)]
                idx = int(cid)
                clase = classes[idx] if idx < len(classes) else clase
        items.append({
            "name": stem, "clase": clase,
            "image": f"/file?path={dir}/images/{img.name}",
            "mask": f"/file?path={dir}/masks/{stem}.png" if mask.exists() else None,
            "box": box,
        })
    return {"items": items, "count": len(items)}


@app.get("/api/crops")
def crops(clase: str, limit: int = 24):
    d = ROOT / "defectos_reales" / clase
    if not d.exists():
        return {"items": [], "total": 0}
    files = sorted(d.glob("*.png"))
    items = [f"/file?path=defectos_reales/{clase}/{f.name}" for f in files[:limit]]
    return {"items": items, "total": len(files)}


@app.get("/file")
def serve_file(path: str):
    p = safe_path(path)
    if not p.is_file():
        raise HTTPException(404, "no encontrado")
    return FileResponse(str(p))


@app.post("/api/generate")
async def generate(req: Request):
    b = await req.json()
    mode = b.get("mode", "inpaint")
    n = int(b.get("n", 4))
    out = b.get("out", "_test_salida")
    lora = b.get("lora", "rollo_defectos_lora.safetensors")
    denoise = str(b.get("denoise", 0.85))
    clean = b.get("clean_dir", "_test_limpios")
    cmd = ["scripts/genera_lote.py", "--server", COMFY, "--mode", mode,
           "--n", str(n), "--out", out, "--lora", lora, "--denoise", denoise]
    if mode == "inpaint":
        cmd += ["--clean-dir", clean]
    jid = launch(f"Generar {n} ({mode})", cmd)
    return {"job": jid}


@app.post("/api/dataset")
async def dataset(req: Request):
    b = await req.json()
    mode = b.get("mode", "full")
    size = str(b.get("size", 512))
    cmd = ["scripts/descarga_neu_det.py", "--dst", "defectos_reales",
           "--mode", mode, "--size", size]
    return {"job": launch("Descargar NEU-DET", cmd)}


@app.post("/api/prepare")
async def prepare(req: Request):
    b = await req.json()
    limit = str(b.get("limit", 0))
    cmd = ["scripts/prepara_dataset.py", "--src", "defectos_reales",
           "--dst", "lora_train", "--repeats", "1", "--size", "512", "--limit", limit]
    return {"job": launch("Preparar dataset LoRA", cmd)}


@app.post("/api/train")
async def train(req: Request):
    b = await req.json()
    steps = str(b.get("steps", 220))
    name = b.get("name", "rollo_defectos_lora")
    dim = str(b.get("dim", 16))
    ckpt = "runtime/ComfyUI/models/checkpoints/RealVisXL_V5.0_fp16.safetensors"
    cmd = ["runtime/kohya/sdxl_train_network.py",
           "--pretrained_model_name_or_path", ckpt,
           "--train_data_dir", "lora_train", "--output_dir", "runtime/lora_out",
           "--output_name", name, "--resolution", "512,512",
           "--network_module", "networks.lora", "--network_dim", dim,
           "--network_alpha", str(int(dim) // 2 or 1),
           "--train_batch_size", "1", "--max_train_steps", steps,
           "--learning_rate", "1e-4", "--optimizer_type", "Adafactor",
           "--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False",
           "--lr_scheduler", "constant", "--gradient_checkpointing",
           "--network_train_unet_only", "--cache_latents", "--caption_extension", ".txt",
           "--mixed_precision", "no", "--save_precision", "fp16", "--sdpa",
           "--max_data_loader_n_workers", "0", "--save_model_as", "safetensors", "--seed", "42"]
    jid = launch(f"Entrenar LoRA ({steps} pasos)", cmd, py=PY_KOHYA,
                 extra_env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    return {"job": jid}


@app.post("/api/deploy_lora")
async def deploy_lora(req: Request):
    """Copia el LoRA recién entrenado a la carpeta de ComfyUI."""
    src = RUNTIME / "lora_out" / "rollo_defectos_lora.safetensors"
    if not src.exists():
        raise HTTPException(404, "aún no hay LoRA entrenado en runtime/lora_out")
    dst = RUNTIME / "ComfyUI" / "models" / "loras" / src.name
    dst.write_bytes(src.read_bytes())
    return {"ok": True, "lora": src.name}


@app.get("/api/jobs")
def jobs():
    out = []
    for j in sorted(JOBS.values(), key=lambda x: x["started"], reverse=True):
        refresh(j)
        out.append({"id": j["id"], "name": j["name"], "status": j["status"],
                    "started": j["started"], "elapsed": round(time.time() - j["started"])})
    return {"jobs": out}


@app.get("/api/jobs/{jid}/log")
def job_log(jid: str, tail: int = 120):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "job no encontrado")
    refresh(j)
    txt = Path(j["log"]).read_text(errors="ignore") if Path(j["log"]).exists() else ""
    lines = txt.replace("\r", "\n").splitlines()
    return {"status": j["status"], "log": "\n".join(lines[-tail:])}


@app.post("/api/jobs/{jid}/stop")
def job_stop(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "job no encontrado")
    if j["status"] == "running":
        j["proc"].terminate()
        j["status"] = "stopped"
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Sala de Inspección -> http://127.0.0.1:8500")
    uvicorn.run(app, host="127.0.0.1", port=8500, log_level="warning")
