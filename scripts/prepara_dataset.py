#!/usr/bin/env python3
"""
prepara_dataset.py — Convierte carpetas de recortes de defecto (una por clase) en el
formato de entrenamiento de kohya (train_data_dir con carpetas `{repeats}_{token}` y
un .txt de caption por imagen).

Uso:
    python scripts/prepara_dataset.py \
        --src  /ruta/defectos_reales \
        --dst  /ruta/lora_train \
        --repeats 10 --token almdefect --size 1024

Estructura esperada en --src (subcarpetas por clase):
    defectos_reales/scratch/*.jpg
    defectos_reales/dent/*.jpg
    ...
"""
import argparse
import glob
import os

from PIL import Image

# Prompt base por clase (inglés: SDXL responde mejor). Edita a tu proceso.
CLASE_PROMPT = {
    "scratch":   "deep scratch scratches on brushed aluminum surface, metal scuff",
    "dent":      "dent dented deformation on aluminum sheet, impact mark",
    "stain":     "dark oil stain discoloration on aluminum surface",
    "crease":    "crease wrinkle fold on aluminum foil",
    "corrosion": "corrosion oxidation pitting on aluminum",
    "tear":      "torn ripped edge on aluminum foil",
}

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="carpeta con subcarpetas por clase")
    ap.add_argument("--dst", required=True, help="salida (train_data_dir de kohya)")
    ap.add_argument("--repeats", type=int, default=10, help="repeticiones por época")
    ap.add_argument("--token", default="almdefect", help="token de activación del LoRA")
    ap.add_argument("--size", type=int, default=1024, help="lado máximo tras redimensionar")
    ap.add_argument("--limit", type=int, default=0,
                    help="máximo de imágenes por clase (0 = sin límite). Útil para MPS/pruebas.")
    args = ap.parse_args()

    folder = os.path.join(args.dst, f"{args.repeats}_{args.token}")
    os.makedirs(folder, exist_ok=True)

    n = 0
    for clase, prompt in CLASE_PROMPT.items():
        subdir = os.path.join(args.src, clase)
        if not os.path.isdir(subdir):
            continue
        vistos = 0
        for p in sorted(glob.glob(os.path.join(subdir, "*"))):
            if not p.lower().endswith(EXTS):
                continue
            if args.limit and vistos >= args.limit:
                break
            vistos += 1
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  aviso: no pude abrir {p}: {e}")
                continue
            im.thumbnail((args.size, args.size))
            base = f"{clase}_{n:04d}"
            im.save(os.path.join(folder, base + ".png"))
            with open(os.path.join(folder, base + ".txt"), "w") as f:
                f.write(f"{args.token}, {prompt}")
            n += 1

    print(f"{n} imágenes preparadas en {folder}")
    if n == 0:
        raise SystemExit(
            "No se encontraron imágenes. Verifica que --src tenga subcarpetas por clase "
            f"({', '.join(CLASE_PROMPT)}) con archivos de imagen."
        )


if __name__ == "__main__":
    main()
