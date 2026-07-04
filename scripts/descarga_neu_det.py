#!/usr/bin/env python3
"""
descarga_neu_det.py — Ejemplo desarrollado: descarga el dataset público NEU-DET
(defectos en superficie metálica, con anotaciones VOC) y lo convierte en recortes
listos para entrenar el LoRA, en tu estructura `defectos_reales/<clase>/`.

Fuente (git, sin login): siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection
Clases NEU: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.

Modos:
  full : usa cada imagen 200x200 completa como tile de defecto (rápido, limpio).
  bbox : recorta cada caja anotada (VOC XML) con padding (recortes precisos).

Uso:
    python scripts/descarga_neu_det.py --dst ./defectos_reales --mode full --size 512
    python scripts/descarga_neu_det.py --dst ./defectos_reales --mode bbox --pad 0.35

Ojo: NEU es ACERO (no aluminio). Úsalo como semilla para que el LoRA aprenda la
*apariencia* del defecto (rayadura, pitting, inclusión); los fondos limpios ponlos
con TUS fotos reales del rollo. Ver CLAUDE.md §10.
"""
import argparse
import os
import shutil
import subprocess
import sys
from glob import glob

try:  # parser XML endurecido contra XXE / billion-laughs
    from defusedxml import ElementTree as ET
except ImportError:  # fallback stdlib (el XML de NEU-DET es de fuente conocida)
    import xml.etree.ElementTree as ET

from PIL import Image

REPO = "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git"

# NEU (acero) -> tu taxonomía (aluminio). Editable a tu criterio.
MAP_CLASES = {
    "scratches":       "scratch",
    "pitted_surface":  "corrosion",
    "rolled-in_scale": "corrosion",
    "inclusion":       "stain",
    "patches":         "stain",
    "crazing":         "crease",
}


def clonar(cache_dir):
    """Sparse-clone: solo IMAGES/ y ANNOTATIONS/ (evita bajar modelos pesados del repo)."""
    if os.path.isdir(os.path.join(cache_dir, "IMAGES")):
        print("Dataset ya clonado en", cache_dir)
        return
    os.makedirs(cache_dir, exist_ok=True)
    run = lambda *a: subprocess.run(a, cwd=cache_dir, check=True)
    print("Clonando NEU-DET (sparse: IMAGES + ANNOTATIONS)...")
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", REPO)
    run("git", "config", "core.sparseCheckout", "true")
    run("git", "sparse-checkout", "init", "--no-cone")
    with open(os.path.join(cache_dir, ".git", "info", "sparse-checkout"), "w") as f:
        f.write("IMAGES/\nANNOTATIONS/\n")
    # blob:none + depth 1 = mínimo ancho de banda
    run("git", "fetch", "--depth", "1", "--filter=blob:none", "origin", "master")
    run("git", "checkout", "-q", "master")
    print("Clonado en", cache_dir)


def clase_de(nombre):
    """crazing_123.jpg -> crazing (el nombre NEU está en el prefijo)."""
    base = os.path.basename(nombre).rsplit(".", 1)[0]
    for c in MAP_CLASES:
        if base.startswith(c):
            return c
    return None


def exportar_full(cache_dir, dst, size):
    n = 0
    for img_path in sorted(glob(os.path.join(cache_dir, "IMAGES", "*.jpg"))):
        neu = clase_de(img_path)
        if not neu:
            continue
        destino = os.path.join(dst, MAP_CLASES[neu])
        os.makedirs(destino, exist_ok=True)
        im = Image.open(img_path).convert("RGB")
        if size:
            im = im.resize((size, size), Image.LANCZOS)
        im.save(os.path.join(destino, f"{os.path.basename(img_path).rsplit('.',1)[0]}.png"))
        n += 1
    return n


def exportar_bbox(cache_dir, dst, size, pad):
    n = 0
    for xml_path in sorted(glob(os.path.join(cache_dir, "ANNOTATIONS", "*.xml"))):
        neu = clase_de(xml_path)
        if not neu:
            continue
        img_path = os.path.join(cache_dir, "IMAGES",
                                os.path.basename(xml_path).replace(".xml", ".jpg"))
        if not os.path.exists(img_path):
            continue
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        destino = os.path.join(dst, MAP_CLASES[neu])
        os.makedirs(destino, exist_ok=True)
        for i, obj in enumerate(root.findall("object")):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x0, y0 = int(float(bb.find("xmin").text)), int(float(bb.find("ymin").text))
            x1, y1 = int(float(bb.find("xmax").text)), int(float(bb.find("ymax").text))
            # padding para dar contexto
            px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
            box = (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py))
            crop = im.crop(box)
            if size:
                crop = crop.resize((size, size), Image.LANCZOS)
            stem = os.path.basename(xml_path).replace(".xml", "")
            crop.save(os.path.join(destino, f"{stem}_{i}.png"))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", default="./defectos_reales", help="carpeta de salida (por clase)")
    ap.add_argument("--cache", default="./.cache_neu_det", help="dónde clonar el dataset")
    ap.add_argument("--mode", choices=["full", "bbox"], default="full")
    ap.add_argument("--size", type=int, default=512, help="lado de salida (0 = original)")
    ap.add_argument("--pad", type=float, default=0.35, help="[bbox] padding relativo de la caja")
    args = ap.parse_args()

    try:
        clonar(args.cache)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Fallo al clonar NEU-DET: {e}. ¿Tienes git y conexión?")

    n = (exportar_full(args.cache, args.dst, args.size) if args.mode == "full"
         else exportar_bbox(args.cache, args.dst, args.size, args.pad))

    print(f"\n{n} recortes exportados a {args.dst} (modo {args.mode})")
    for c in sorted(set(MAP_CLASES.values())):
        d = os.path.join(args.dst, c)
        cnt = len(glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else 0
        print(f"  {c:10s}: {cnt}")
    print("\nSiguiente: python scripts/prepara_dataset.py --src", args.dst,
          "--dst ./lora_train")


if __name__ == "__main__":
    main()
