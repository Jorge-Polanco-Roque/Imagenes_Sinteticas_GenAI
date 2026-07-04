<div align="center">

# Synthetic Defect Studio
### Generación de datos sintéticos de defectos en rollos de aluminio

**ComfyUI · SDXL · LoRA · Inpainting — para entrenar modelos de inspección industrial**

`SDXL (RealVisXL V5.0)` · `LoRA kohya` · `ComfyUI API` · `FastAPI + Web UI` · `Apple Silicon / Colab T4`

</div>

---

## Tabla de contenido

1. [Resumen](#1-resumen)
2. [Por qué este enfoque](#2-por-qué-este-enfoque)
3. [Arquitectura](#3-arquitectura)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Requisitos](#5-requisitos)
6. [Arranque rápido](#6-arranque-rápido)
7. [Flujo de trabajo detallado](#7-flujo-de-trabajo-detallado)
8. [Panel web — Sala de Inspección](#8-panel-web--sala-de-inspección)
9. [Modelos y datasets](#9-modelos-y-datasets)
10. [Parámetros de generación](#10-parámetros-de-generación)
11. [Entrenamiento del LoRA](#11-entrenamiento-del-lora)
12. [Referencia de scripts](#12-referencia-de-scripts)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Estado y hoja de ruta](#14-estado-y-hoja-de-ruta)
15. [Glosario](#15-glosario)

---

## 1. Resumen

**Synthetic Defect Studio** es un pipeline completo para producir **datos sintéticos
fotorrealistas** de defectos superficiales en rollos de aluminio —rayaduras, abolladuras,
manchas, pliegues, corrosión y rasgaduras— con el fin de **entrenar modelos de Machine
Learning para inspección de calidad** (detección y segmentación de defectos) partiendo de
muy pocos ejemplos reales.

El sistema cubre el ciclo completo, de punta a punta:

| Etapa | Qué hace |
|-------|----------|
| **Datos semilla** | Descarga y recorta datasets públicos de defectos, o usa tus fotos |
| **Entrenamiento** | Entrena un LoRA que aprende la *apariencia* de cada defecto |
| **Generación** | Inpaint de defectos sobre rollos limpios vía la API de ComfyUI |
| **Etiquetado** | Exporta cada imagen con su máscara y su bounding box en formato YOLO |
| **Operación** | Panel web local para ejecutar todo sin tocar la terminal |

> La documentación técnica exhaustiva —decisiones de arquitectura, fases con criterios de
> aceptación, benchmarks y notas de plataforma— vive en **[`CLAUDE.md`](./CLAUDE.md)**.
> Este README es la guía operativa.

---

## 2. Por qué este enfoque

**Estrategia central: inpainting de defectos sobre fotos de rollos reales limpios,
potenciado por un LoRA.** En lugar de generar el rollo completo desde cero, tomamos una foto
de un rollo *sin defecto* y el modelo "pinta" el desperfecto dentro de una máscara.

Ventajas frente a texto-a-imagen puro:

- **Realismo del dominio.** El aluminio tiene reflejos anisotrópicos, líneas de laminado y
  textura muy específicos. Al inpaintar sobre una foto real, todo eso ya está presente.
- **La etiqueta es gratis.** La máscara que define *dónde* se pinta el defecto **es** la
  segmentación ground-truth; su bounding box **es** la anotación YOLO.
- **Control de balance de clases.** Decides cuántos ejemplos de cada tipo de defecto generas.
- **Domain gap mínimo.** Si usas fotos de tu propia línea, el detector entrena con imágenes
  casi idénticas a las de producción.

---

## 3. Arquitectura

```
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  A · DATOS SEMILLA                                                             │
  │    Rollos LIMPIOS (fondos)  +  ejemplos de defecto (NEU-DET público o propios) │
  └───────────────┬──────────────────────────────────────────────────────────────┘
                  ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  B · ENTRENAMIENTO LoRA   (kohya · SDXL · Colab T4 recomendado)                │
  │    recortes de defecto + captions  ──►  rollo_defectos_lora.safetensors        │
  └───────────────┬──────────────────────────────────────────────────────────────┘
                  ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  C · GENERACIÓN   (ComfyUI + LoRA · local MPS o Colab)                         │
  │    foto limpia + máscara + prompt  ──►  inpaint  ──►  imagen con defecto        │
  │    la máscara se guarda emparejada  ──►  etiqueta de segmentación / YOLO        │
  └───────────────┬──────────────────────────────────────────────────────────────┘
                  ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  D · DATASET ETIQUETADO   ──►  entrenar detector (YOLO / RT-DETR / SAM)         │
  │    validar SIEMPRE contra un holdout 100% real                                 │
  └──────────────────────────────────────────────────────────────────────────────┘

  Operación transversal:  Panel web "Sala de Inspección"  (FastAPI + frontend)
```

| Componente | Elección | Motivo |
|---|---|---|
| Motor de inferencia | **ComfyUI** | Grafos reproducibles + API para lotes |
| Modelo base | **SDXL — RealVisXL V5.0** | Fotorrealismo de materiales; ecosistema maduro |
| Personalización | **LoRA (kohya)** | Aprende la apariencia del defecto con pocos ejemplos |
| Inpainting | Nodos nativos + opción Fooocus | Máxima compatibilidad, calidad opcional |
| Entrenamiento | kohya `sd-scripts` | fp16 + fused backward pass en T4 |
| Panel | **FastAPI + HTML/JS** (sin build) | Herramienta local robusta |

---

## 4. Estructura del repositorio

```
img_sintéticas/
├── CLAUDE.md                       # documentación técnica completa (fuente de verdad)
├── README.md                       # esta guía operativa
├── requirements.txt                # dependencias de los scripts locales
├── .gitignore
│
├── notebooks/
│   ├── hosting_ComfyUI.ipynb       # Colab: ComfyUI + túnel cloudflared + modelos
│   └── train_LoRA_kohya.ipynb      # Colab: entrena el LoRA (turnkey, auto-descarga dataset)
│
├── workflows/                      # arrastrables a ComfyUI
│   ├── inpaint_defectos.json       # inpainting con LoRA (principal)
│   ├── inpaint_defectos_sin_lora.json  # inpainting base, sin LoRA (para probar ya)
│   └── txt2img_defectos.json       # generación de rollo completo desde cero
│
├── scripts/
│   ├── descarga_neu_det.py         # descarga + recorta el dataset público NEU-DET
│   ├── prepara_dataset.py          # crops + captions en formato kohya
│   └── genera_lote.py              # generación por lotes vía API + export YOLO/máscaras
│
├── ui/                             # panel web "Sala de Inspección"
│   ├── app.py                      # backend FastAPI (envuelve scripts + API ComfyUI)
│   ├── run.sh                      # lanzador → http://127.0.0.1:8500
│   ├── requirements.txt            # fastapi · uvicorn · requests
│   └── static/                     # index.html · styles.css · app.js
│
└── runtime/                        # (gitignored) ComfyUI + venvs + modelos + salidas
```

---

## 5. Requisitos

**Para generar en local (Apple Silicon):**
- macOS con chip Apple (M-series). Probado en **M5, 24 GB**.
- **Python 3.12** (evita 3.13/3.14 — PyTorch aún no las soporta bien). Recomendado [`uv`](https://github.com/astral-sh/uv).
- ~15 GB de disco para ComfyUI + modelos.

**Para entrenar el LoRA:**
- **Google Colab con GPU T4** (recomendado) — fp16, ~20–40 min.
- El entrenamiento en Apple Silicon es posible pero lento (fp32, sin media precisión en MPS).

**Para los scripts locales:** `pip install -r requirements.txt` (Pillow, NumPy, Requests, defusedxml).

---

## 6. Arranque rápido

### Opción A — Todo en tu Mac (Apple Silicon)

```bash
# 1. ComfyUI + entorno (una sola vez)
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI runtime/ComfyUI
uv venv runtime/venv --python 3.12
source runtime/venv/bin/activate
uv pip install torch torchvision torchaudio            # incluye MPS en arm64
uv pip install -r runtime/ComfyUI/requirements.txt
uv pip install -r ui/requirements.txt                  # panel web

# 2. Modelo base SDXL → runtime/ComfyUI/models/checkpoints/
#    (RealVisXL V5.0 fp16 desde Hugging Face)

# 3. Datos semilla (dataset público, sin subir nada)
uv run --with pillow --with defusedxml \
  python scripts/descarga_neu_det.py --dst ./defectos_reales

# 4. Levantar ComfyUI  y  el panel web
(cd runtime/ComfyUI && python main.py --listen 127.0.0.1 --port 8188) &
bash ui/run.sh          # → http://127.0.0.1:8500
```

### Opción B — Entrenar en Colab, generar en local

1. Sube **`notebooks/train_LoRA_kohya.ipynb`** a [Google Colab](https://colab.research.google.com),
   pon la GPU en **T4** y ejecuta todo. Es *turnkey*: descarga el dataset y el modelo solos.
2. Descarga el `rollo_defectos_lora.safetensors` resultante.
3. Colócalo en `runtime/ComfyUI/models/loras/` y genera con el panel o con `genera_lote.py`.

---

## 7. Flujo de trabajo detallado

### Fase A — Datos semilla

Usa el dataset público (turnkey) o tus propias fotos:

```bash
# Dataset público NEU-DET → recortes por clase en ./defectos_reales/
python scripts/descarga_neu_det.py --dst ./defectos_reales --mode full --size 512
```

Estructura resultante (o la que armes con tus fotos):

```
defectos_reales/
├── scratch/  dent/  stain/  crease/  corrosion/  tear/
```

> **Nota de dominio:** NEU-DET es acero, no foil de aluminio. Sirve para que el LoRA aprenda
> la *apariencia* del defecto; los **fondos limpios** conviene tomarlos con **tus fotos reales**
> del rollo para minimizar el domain gap.

### Fase B — Entrenar el LoRA

```bash
# Preparar dataset en formato kohya (crops + captions con token 'almdefect')
python scripts/prepara_dataset.py --src ./defectos_reales --dst ./lora_train --limit 80
```

Luego entrena en **Colab** (`train_LoRA_kohya.ipynb`, ~20–40 min) — es la vía recomendada.
Ver [§11](#11-entrenamiento-del-lora) para la configuración y las notas de plataforma.

### Fase C — Generar

Con ComfyUI corriendo y el LoRA en su carpeta:

```bash
python scripts/genera_lote.py \
  --server http://127.0.0.1:8188 \
  --mode inpaint --clean-dir ./limpios \
  --lora rollo_defectos_lora.safetensors \
  --n 50 --out ./dataset_sintetico \
  --denoise 1.0 --lora-strength 1.3 --cfg 12
```

Produce, emparejados:

```
dataset_sintetico/
├── images/   # rollo con defecto
├── masks/    # máscara del defecto (segmentación)
├── labels/   # bbox YOLO  (class cx cy w h)
└── classes.txt
```

### Fase D — Entrenar el detector

Combina el sintético con tus pocos reales (p. ej. 80/20) y **valida siempre contra un
holdout 100% real**. La métrica que importa es el desempeño en producción, no en el sintético.

---

## 8. Panel web — Sala de Inspección

Consola local para operar todo el pipeline desde el navegador, con estética de instrumento
de inspección metrológica.

```bash
bash ui/run.sh        # → http://127.0.0.1:8500   (requiere ComfyUI corriendo)
```

| Vista | Función |
|-------|---------|
| **Panel** | Telemetría en vivo (ComfyUI, dispositivo, LoRAs, recortes, generadas) y flujo de trabajo |
| **Galería** | Tarjetas-inspección con overlay del bounding box del defecto (toggle *Marcar defecto*) |
| **Generar** | Lanza inpaint/txt2img con controles y consola de log en vivo |
| **Entrenar** | Dispara el entrenamiento del LoRA y lo publica en ComfyUI |
| **Datos** | Descarga NEU-DET, prepara el dataset y explora los recortes por clase |

Los procesos largos corren como *jobs* con log en vivo. El servido de archivos valida rutas
(anti-traversal). Backend FastAPI + frontend sin *build step*.

---

## 9. Modelos y datasets

### Modelos

| Modelo | Rol | Fuente | Tamaño |
|---|---|---|---|
| **RealVisXL V5.0** (fp16) | checkpoint base SDXL | HF `SG161222/RealVisXL_V5.0` | ~6.9 GB |
| sdxl-vae-fp16-fix | VAE estable | HF `madebyollin/sdxl-vae-fp16-fix` | ~330 MB |
| Fooocus inpaint (head+patch) | inpainting SDXL premium (opcional) | HF `lllyasviel/fooocus_inpaint` | ~1.3 GB |
| **LoRA de defectos** | apariencia de los defectos | lo entrenas tú (Fase B) | ~80 MB |

Alternativas de checkpoints: Juggernaut XL, CyberRealistic XL. Fuentes: Hugging Face
(preferida para automatizar), Civitai, Tensor.Art, ModelScope.

### Datasets públicos de defectos (semilla)

| Dataset | Contenido | Licencia | Integración |
|---|---|---|---|
| **NEU-DET** | 1,800 img acero, 6 clases + VOC | investigación | `descarga_neu_det.py` (automatizado) |
| **GC10-DET** | 3,570 img, 10 clases | **CC BY 4.0** | recomendado para uso comercial |
| **Tianchi Aluminum Profile** | ~2,090 img de **aluminio** | registro Tianchi | el más cercano al dominio |
| Severstal / MVTec AD | acero / anomalías | Kaggle / *non-commercial* | complementarios |

---

## 10. Parámetros de generación

El reto al inpaintar defectos sobre metal brillante es que el modelo tiende a "armonizar" el
parche hacia superficie limpia. La configuración que **produce defectos claramente visibles**:

| Parámetro | Valor | Efecto |
|---|---|---|
| `--cfg` | **12** | **Clave.** CFG alto fuerza el defecto contra el fondo limpio |
| `--denoise` | 1.0 | Regenera por completo la zona enmascarada |
| `--lora-strength` | 1.3 | Empuja la apariencia aprendida por el LoRA |
| máscara | grande | El defecto ocupa un área notoria (ajustado en `make_mask`) |
| prompt | `(corrosion:1.4)` | Peso explícito sobre el término del defecto |

**Visibilidad por tipo de defecto:** corrosión, mancha, rasgadura y pliegue se marcan bien.
Abolladura y rayadura quedan más sutiles por naturaleza (deformaciones en metal espejado casi
no contrastan) — algo realista en inspección industrial.

Controles disponibles en `genera_lote.py`: `--mode {inpaint,txt2img}`, `--n`, `--denoise`,
`--lora-strength`, `--cfg`, `--clean-dir`, `--out`, `--lora`, `--seed`.

---

## 11. Entrenamiento del LoRA

### En Colab T4 (recomendado)

`train_LoRA_kohya.ipynb` es **turnkey**: descarga el dataset NEU-DET y el modelo base solos,
y entrena con configuración probada para la T4 gratuita:

- `768px · network_dim 16 · AdamW8bit · 600 pasos · fp16`
- **`--lowram`** — imprescindible: carga el modelo a la VRAM de la T4 (16 GB) en vez de a la
  RAM del sistema (~13.6 GB en Colab gratis). Sin este flag, el proceso muere en silencio al
  cargar el checkpoint de 6.9 GB (OOM de RAM, justo tras *"Using DreamBooth method"*).

Para tus propias fotos: súbelas a Drive (`rollos_dataset/defectos_reales/<clase>/`) y pon
`USAR_NEU = False` en la celda 5.

### En local (Apple Silicon / MPS)

Funciona, pero **con limitaciones**: `accelerate` no habilita media precisión en MPS, así que
se entrena en **fp32** (~11 s/paso en régimen). Un LoRA de 220 pasos tarda ~40 min; dim 32 o
resoluciones altas saturan los 24 GB y provocan *swap*. **Para un LoRA de calidad, Colab es
notablemente más rápido y estable.** Detalle y benchmarks en [`CLAUDE.md` §14](./CLAUDE.md).

---

## 12. Referencia de scripts

| Script | Propósito | Ejemplo |
|---|---|---|
| `descarga_neu_det.py` | Baja y recorta NEU-DET a tu taxonomía | `--dst ./defectos_reales --mode full --size 512` |
| `prepara_dataset.py` | Crops + captions en formato kohya | `--src ./defectos_reales --dst ./lora_train --limit 80` |
| `genera_lote.py` | Generación por lotes + export YOLO/máscaras | `--mode inpaint --n 50 --cfg 12 --denoise 1.0` |

Todos aceptan `--help`. `genera_lote.py` usa la API de ComfyUI (`/prompt`), por lo que ComfyUI
debe estar corriendo (local o vía la URL del túnel de Colab).

---

## 13. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| Colab muere tras *"Using DreamBooth method"* | OOM de RAM del sistema al cargar el checkpoint | Añadir **`--lowram`** al comando de entrenamiento |
| El defecto casi no se ve en el inpaint | El modelo armoniza el parche a metal limpio | Subir **`--cfg` a 12**, `--denoise 1.0`, `--lora-strength 1.3`, máscara grande |
| El workflow pide "1 modelo requerido" | El LoRA aún no existe | Entrénalo, o usa `inpaint_defectos_sin_lora.json` |
| Entrenamiento local lentísimo / swap | fp32 en MPS supera los 24 GB | Reducir dim/resolución, o entrenar en Colab |
| OOM al entrenar (T4) | resolución/dim alto | `768,768`, `network_dim 16`, `--lowram` |
| Imágenes fp16 con manchas | VAE fp16 inestable | Usar `sdxl-vae-fp16-fix` |
| PyTorch no instala / rompe | Python 3.13/3.14 | Usar **Python 3.12** |
| ComfyUI no ve un LoRA nuevo | lista de modelos cacheada | Reiniciar ComfyUI |

---

## 14. Estado y hoja de ruta

| Fase | Entregable | Estado |
|------|-----------|--------|
| 0 · Scaffolding | `CLAUDE.md`, README, estructura | ✅ |
| 1 · Notebooks | hosting + entrenamiento (turnkey) | ✅ |
| 2 · Workflows | inpaint / txt2img / sin-LoRA | ✅ |
| 3 · Datos semilla | NEU-DET automatizado (1,770 recortes) | ✅ |
| 4 · LoRA | entrenado en Colab (`--lowram`) | ✅ |
| 5 · Generación | inpaint con defectos visibles (CFG 12) | ✅ |
| 6 · Panel web | Sala de Inspección | ✅ |
| 7 · Fotos reales de rollos limpios | captura del usuario | ⏳ |
| 8 · Dataset a escala + detector | generar volumen y entrenar YOLO | ⏳ |
| 9 · Validación en producción | mAP contra holdout real | ⏳ |

---

## 15. Glosario

- **LoRA** — adaptador ligero que enseña un concepto nuevo al modelo con pocas imágenes.
- **Inpainting** — regenerar solo la zona enmascarada de una imagen, manteniendo el resto.
- **Denoise** — cuánto se aleja el inpaint de la imagen original (0 = nada, 1 = todo nuevo).
- **CFG** — *classifier-free guidance*; cuánto obliga el modelo a seguir el prompt.
- **Domain gap** — diferencia entre los datos de entrenamiento y los reales de producción.
- **Ground truth** — etiqueta verdadera; aquí, la máscara del defecto.

---

<div align="center">
<sub>Documentación técnica completa en <a href="./CLAUDE.md">CLAUDE.md</a> · Uso interno / investigación</sub>
</div>
