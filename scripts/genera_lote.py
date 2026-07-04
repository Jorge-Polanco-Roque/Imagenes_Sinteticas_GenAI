#!/usr/bin/env python3
"""
genera_lote.py — Generación por lotes de datos sintéticos vía la API de ComfyUI.

Dos modos:
  * txt2img : genera rollos completos con defecto desde cero (variedad de fondos).
  * inpaint : sobre fotos de rollos LIMPIOS, pinta un defecto dentro de una máscara
              generada aleatoriamente, y exporta la etiqueta (bbox YOLO + máscara PNG).

El modo `inpaint` es el recomendado para entrenar detectores: la máscara es la
etiqueta de segmentación ground-truth, "gratis".

Requisitos: `pip install requests pillow numpy`  (ver requirements.txt)
ComfyUI debe estar corriendo y accesible (usa la URL trycloudflare del notebook,
o http://127.0.0.1:8188 si es local).

Ejemplos:
    # 20 imágenes full-scene
    python scripts/genera_lote.py --server http://127.0.0.1:8188 --mode txt2img --n 20

    # inpainting sobre una carpeta de rollos limpios (exporta imagen+bbox YOLO+máscara)
    python scripts/genera_lote.py --server https://xxxx.trycloudflare.com \
        --mode inpaint --clean-dir ./limpios --n 100 --out ./dataset_sintetico
"""
import argparse
import io
import json
import os
import random
import time
import uuid

import numpy as np
import requests
from PIL import Image, ImageDraw

# --- Configuración de defectos: (nombre_clase, prompt) --------------------------
CLASES = [
    ("scratch",   "(deep dark scratch:1.4), gouge scratches, metal scuff line, visible damage"),
    ("dent",      "(dent:1.4), dented deformation, impact mark, deep shadow, damaged metal"),
    ("stain",     "(dark stain:1.4), black oil discoloration, rust patch, dirty mark"),
    ("crease",    "(crease:1.4), deep wrinkle fold, buckled metal"),
    ("corrosion", "(corrosion:1.4), heavy oxidation pitting, rough granular rust texture"),
    ("tear",      "(torn ripped tear:1.4), jagged broken edge, hole in metal"),
]
CKPT_DEFAULT = "RealVisXL_V5.0_fp16.safetensors"
LORA_DEFAULT = "rollo_defectos_lora.safetensors"
TOKEN = "almdefect"
NEG = "blurry, cartoon, painting, illustration, lowres, text, watermark, deformed"


# --- Cliente API ComfyUI --------------------------------------------------------
class ComfyClient:
    def __init__(self, server):
        self.server = server.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def upload_image(self, pil_img, name):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        r = requests.post(
            f"{self.server}/upload/image",
            files={"image": (name, buf, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["name"]

    def queue(self, graph):
        r = requests.post(
            f"{self.server}/prompt",
            json={"prompt": graph, "client_id": self.client_id},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(self, prompt_id, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = requests.get(f"{self.server}/history/{prompt_id}", timeout=30)
            if r.ok and prompt_id in r.json():
                return r.json()[prompt_id]
            time.sleep(1.5)
        raise TimeoutError(f"prompt {prompt_id} no terminó en {timeout}s")

    def fetch_images(self, history):
        out = []
        for node in history.get("outputs", {}).values():
            for img in node.get("images", []):
                r = requests.get(
                    f"{self.server}/view",
                    params={"filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")},
                    timeout=60,
                )
                if r.ok:
                    out.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
        return out


# --- Construcción de grafos (formato API) ---------------------------------------
def graph_txt2img(ckpt, lora, pos, seed, lstr=0.85, w=1024, h=1024):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
            "strength_model": lstr, "strength_clip": lstr}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": pos}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": NEG}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": 30, "cfg": 6.5,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "rollo_defecto"}},
    }


def graph_inpaint(ckpt, lora, pos, seed, clean_name, mask_name, denoise=0.85, lstr=0.85, cfg=6.0):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["1", 1], "lora_name": lora,
            "strength_model": lstr, "strength_clip": lstr}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": pos}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": NEG}},
        "5": {"class_type": "LoadImage", "inputs": {"image": clean_name}},
        "10": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "6": {"class_type": "VAEEncodeForInpaint", "inputs": {
            "pixels": ["5", 0], "vae": ["1", 2], "mask": ["10", 0], "grow_mask_by": 8}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["6", 0], "seed": seed, "steps": 28, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "rollo_defecto"}},
    }


# --- Máscaras sintéticas + etiquetas --------------------------------------------
def make_mask(w, h, clase):
    """Genera una máscara blanca sobre negro con forma acorde al tipo de defecto."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    if clase in ("scratch", "tear"):  # lineal (más largo y grueso = más visible)
        x0 = random.randint(int(w * 0.1), int(w * 0.6))
        y0 = random.randint(int(h * 0.15), int(h * 0.7))
        length = random.randint(int(w * 0.28), int(w * 0.6))
        ang = random.uniform(0, 3.14159)
        x1 = int(x0 + length * np.cos(ang))
        y1 = int(y0 + length * np.sin(ang))
        d.line([(x0, y0), (x1, y1)], fill=255, width=random.randint(14, 34))
    else:  # ovalado / mancha (más grande)
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        rx = random.randint(int(w * 0.11), int(w * 0.26))
        ry = random.randint(int(h * 0.11), int(h * 0.26))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return m


def mask_to_yolo(mask, class_id):
    """bbox normalizada YOLO: 'class cx cy w h'. Devuelve None si la máscara está vacía."""
    a = np.array(mask)
    ys, xs = np.where(a > 127)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    W, H = mask.size
    cx = (x0 + x1) / 2 / W
    cy = (y0 + y1) / 2 / H
    bw = (x1 - x0) / W
    bh = (y1 - y0) / H
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# --- Main -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="URL de ComfyUI (trycloudflare o local)")
    ap.add_argument("--mode", choices=["txt2img", "inpaint"], default="inpaint")
    ap.add_argument("--n", type=int, default=20, help="número de imágenes a generar")
    ap.add_argument("--out", default="./dataset_sintetico", help="carpeta de salida")
    ap.add_argument("--clean-dir", default=None, help="[inpaint] carpeta de rollos limpios")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--lora", default=LORA_DEFAULT)
    ap.add_argument("--denoise", type=float, default=0.85)
    ap.add_argument("--lora-strength", type=float, default=0.85, dest="lora_strength",
                    help="fuerza del LoRA (0.8-1.4). Más alto = defecto más marcado.")
    ap.add_argument("--cfg", type=float, default=6.0,
                    help="CFG (guidance). Alto (10-13) fuerza el defecto contra el fondo limpio.")
    ap.add_argument("--seed", type=int, default=None, help="semilla base (por defecto aleatoria)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    client = ComfyClient(args.server)
    rng = random.Random(args.seed)

    cleans = []
    if args.mode == "inpaint":
        if not args.clean_dir or not os.path.isdir(args.clean_dir):
            raise SystemExit("--mode inpaint requiere --clean-dir con fotos de rollos limpios")
        cleans = [os.path.join(args.clean_dir, f) for f in sorted(os.listdir(args.clean_dir))
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
        if not cleans:
            raise SystemExit(f"No hay imágenes en {args.clean_dir}")

    # escribe classes.txt para YOLO
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(c[0] for c in CLASES))

    hechas = 0
    for i in range(args.n):
        class_id, (clase, cprompt) = rng.choice(list(enumerate(CLASES)))
        pos = (f"{TOKEN}, {cprompt}, defective damaged aluminum surface, "
               f"industrial defect close up, photorealistic, high detail")
        seed = rng.randint(0, 2**31 - 1)
        stem = f"{clase}_{i:05d}"

        try:
            if args.mode == "txt2img":
                graph = graph_txt2img(args.ckpt, args.lora, pos, seed, args.lora_strength)
                hist = client.wait(client.queue(graph))
                imgs = client.fetch_images(hist)
            else:
                clean = Image.open(rng.choice(cleans)).convert("RGB")
                clean.thumbnail((1024, 1024))
                W, H = clean.size
                mask = make_mask(W, H, clase)
                cname = client.upload_image(clean, f"{stem}_clean.png")
                mname = client.upload_image(mask, f"{stem}_mask.png")
                graph = graph_inpaint(args.ckpt, args.lora, pos, seed, cname, mname,
                                      args.denoise, args.lora_strength, args.cfg)
                hist = client.wait(client.queue(graph))
                imgs = client.fetch_images(hist)
                # guarda máscara y etiqueta YOLO
                mask.save(os.path.join(args.out, "masks", stem + ".png"))
                yolo = mask_to_yolo(mask, class_id)
                if yolo:
                    with open(os.path.join(args.out, "labels", stem + ".txt"), "w") as f:
                        f.write(yolo)

            if imgs:
                imgs[0].save(os.path.join(args.out, "images", stem + ".png"))
                hechas += 1
                print(f"[{hechas}/{args.n}] {stem} ({clase})")
            else:
                print(f"  aviso: sin imagen para {stem}")
        except Exception as e:
            print(f"  error en {stem}: {e}")

    print(f"\nListo: {hechas} imágenes en {args.out}")
    if args.mode == "inpaint":
        print("Etiquetas YOLO en labels/, máscaras en masks/ (val siempre con datos REALES).")


if __name__ == "__main__":
    main()
