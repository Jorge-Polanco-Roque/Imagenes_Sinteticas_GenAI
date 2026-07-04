# CLAUDE.md — Pipeline de datos sintéticos de defectos en rollos de aluminio

> Generación de datos sintéticos fotorrealistas (rayaduras, abolladuras, manchas,
> pliegues, corrosión, etc.) sobre rollos de aluminio para **entrenar modelos de
> Machine Learning de inspección industrial (detección/segmentación de defectos)**.
>
> Motor: **ComfyUI** en **Google Colab (GPU T4 16 GB)**. Base: **SDXL**.
> Estrategia principal: **inpainting de defectos sobre fotos de rollos reales limpios**,
> potenciado por un **LoRA de defectos** entrenado con muy pocos ejemplos.

---

## 0. TL;DR para retomar contexto rápido

- **Objetivo de negocio:** partimos de *muy pocas* fotos reales de defectos y queremos
  un dataset grande, variado y etiquetable para entrenar un detector (YOLO/RT-DETR)
  o segmentador (SAM-fine-tune / U-Net) de control de calidad.
- **Por qué inpainting y no texto→imagen puro:** el fondo (el rollo, la iluminación de
  planta, la textura del aluminio) lo aportan **fotos reales**; el modelo solo "pinta"
  el defecto dentro de una máscara. Resultado: mucho más realista, más controlable y
  —clave— **la máscara ES la etiqueta de segmentación gratis**.
- **Entregables:** este `CLAUDE.md`, `notebooks/hosting_ComfyUI.ipynb`,
  `notebooks/train_LoRA_kohya.ipynb`, workflows arrastrables en `workflows/`, y scripts
  de apoyo en `scripts/`.
- **Estado:** Fase 0-2 entregadas (scaffolding + notebooks + workflows). Gate pendiente:
  el usuario ejecuta en Colab y valida generación. Ver §9 Roadmap.

---

## 1. Principios de trabajo (cómo colaboramos)

Metodología por **fases con validation gates** (el usuario prueba en cada gate antes de avanzar):

1. Cada fase termina en un **entregable verificable** y un **criterio de aceptación** explícito.
2. No se avanza de fase sin que el usuario valide el gate anterior.
3. Preferimos **automatización** (aria2c, celdas idempotentes, detección de GPU) sobre pasos manuales.
4. Todo comentario/prompt/doc en **español**; código y nombres de archivo en snake_case.
5. Ante ambigüedad de diseño con impacto arquitectónico → se pregunta antes de construir.
6. Este archivo es la **fuente de verdad viva**: se actualiza al cerrar cada fase.

---

## 2. Arquitectura del pipeline

```
                    ┌──────────────────────────────────────────────────────────┐
                    │  FASE A · DATOS SEMILLA (usuario, local)                  │
                    │  - Fotos de rollos LIMPIOS (fondos)                        │
                    │  - Pocas fotos de rollos CON defecto (10-40 por tipo)     │
                    └───────────────┬──────────────────────────────────────────┘
                                    │ subir a Google Drive
                                    ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  FASE B · ENTRENAMIENTO LoRA  (notebooks/train_LoRA_kohya.ipynb, Colab T4)  │
   │  Recortes de defecto + captions  ──►  kohya_ss (SDXL, fused backward pass)  │
   │  salida: rollo_defectos_lora.safetensors                                    │
   └───────────────┬────────────────────────────────────────────────────────────┘
                   │ copiar a ComfyUI/models/loras
                   ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  FASE C · GENERACIÓN  (notebooks/hosting_ComfyUI.ipynb + workflows/*.json)  │
   │                                                                             │
   │   Foto rollo limpio ─┐                                                       │
   │                      ├─► [Máscara] ─► SDXL inpaint + LoRA defectos ─► IMG    │
   │   Prompt de defecto ─┘                    (workflow arrastrable)             │
   │                                                                             │
   │   La MÁSCARA se guarda junto a la imagen  ──►  etiqueta de segmentación     │
   └───────────────┬────────────────────────────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  FASE D · CURADO + EXPORT  (scripts/)                                       │
   │  filtro de calidad ─► pares (imagen, máscara/bbox) ─► COCO / YOLO / máscaras │
   └────────────────────────────────────────────────────────────────────────────┘
```

**Componentes:**

| Componente | Elección | Motivo |
|---|---|---|
| UI / motor de inferencia | ComfyUI | Grafos reproducibles, arrastrables, API para lotes |
| Modelo base | **SDXL 1.0** (RealVisXL V5.0) | Mejor balance realismo/ecosistema; corre en T4 con offload |
| Inpainting | **Fooocus Inpaint patch** (Acly/comfyui-inpaint-nodes) + fallback nativo `VAEEncodeForInpaint` | Mejor inpainting SDXL sin checkpoint dedicado |
| Personalización | **LoRA** (kohya_ss) | Aprende la *apariencia* del defecto con pocos ejemplos |
| Control de estructura (opcional) | ControlNet Canny/Depth | Fija bordes/geometría del rollo si el inpaint deforma |
| Entrenamiento | kohya_ss, Adafactor + fused backward pass | Cabe en 16 GB (T4) |
| Túnel de acceso | cloudflared (trycloudflare) | Cero config, URL pública temporal |
| Persistencia | Google Drive (opcional) | Sobrevive reinicios de Colab |

---

## 3. Estrategia de defectos (el corazón del proyecto)

### 3.1 Por qué inpainting sobre rollos reales
- **Realismo del dominio:** el aluminio tiene reflejos anisotrópicos, líneas de laminado
  y textura muy específica que un modelo genérico no reproduce bien "desde cero".
  Al inpaintar sobre una foto real, todo eso ya está presente.
- **Etiqueta gratis:** la máscara que define *dónde* se pinta el defecto es exactamente
  la segmentación ground-truth. De la bbox de la máscara sale la anotación YOLO.
- **Control de balance de clases:** decides cuántos "rayadura" vs "abolladura" generas.
- **Domain gap mínimo:** el detector entrenará con imágenes casi idénticas a la línea real.

### 3.2 Taxonomía de defectos (ajústala a tu proceso)
Definir de entrada las clases porque determinan captions, prompts y export:

| id | clase | prompt base (inglés, mejor para SDXL) | notas |
|----|-------|----------------------------------------|-------|
| 0 | rayadura / scratch | `deep scratch scratches on brushed aluminum surface, metal scuff` | lineal, fino |
| 1 | abolladura / dent | `dent dented deformation on aluminum sheet, impact mark` | volumétrico, sombra |
| 2 | mancha / stain | `dark oil stain discoloration on aluminum` | color/óxido |
| 3 | pliegue / crease | `crease wrinkle fold on aluminum foil roll` | estructural |
| 4 | corrosión / corrosion | `corrosion oxidation pitting on aluminum` | textura granular |
| 5 | rasgadura / tear | `torn ripped edge on aluminum foil` | discontinuidad |

> **Token de activación del LoRA:** usa un token raro y único por proyecto, p. ej.
> `almdefect`, para no colisionar con conceptos del modelo base. Los captions de
> entrenamiento empiezan por ese token: `almdefect, deep scratch on aluminum ...`.
> En el workflow, el prompt positivo incluye `almdefect` + la clase específica.

### 3.3 Dos LoRA vs uno multi-defecto
- **Recomendado para empezar:** **un solo LoRA** entrenado con todos los tipos, cada
  imagen con caption que nombra su defecto. En inferencia el prompt selecciona el tipo.
- Si un tipo domina o se "contamina", separar en LoRA por clase (más control, más costo).

---

## 4. Datos semilla: qué necesita capturar el usuario (FASE A — gate 0)

### 4.1 Rollos limpios (fondos para inpaintar) — **lo más importante**
- 30-100+ fotos de rollos **sin defecto**, variando: ángulo, distancia, iluminación,
  zona del rollo (canto, cara, núcleo). Estas son la "base" sobre la que se pinta.
- Resolución alta (≥1024 px lado corto). Formato PNG/JPG.
- Si la cámara final de inspección es fija, **fotografía con esa misma cámara/óptica/luz**
  → domain gap casi nulo.

### 4.2 Ejemplos reales de defecto (para el LoRA)
- 10-40 por tipo de defecto es suficiente para un LoRA (puede funcionar con menos).
- Recortes **centrados en el defecto** (crops), no la escena completa.
- Variedad > cantidad: distintos tamaños, orientaciones, severidades.
- **Anti-overfitting:** evita que todas las fotos compartan el mismo fondo/ángulo, o el
  LoRA aprenderá el fondo en vez del defecto.

### 4.3 Estructura en Drive
```
MyDrive/rollos_dataset/
├── limpios/                 # fondos para inpaint
├── defectos_reales/
│   ├── scratch/  dent/  stain/  crease/  corrosion/  tear/
├── lora_train/              # se genera en FASE B (crops + captions .txt)
└── salidas/                 # imágenes sintéticas generadas (FASE C)
```

**Criterio de aceptación gate 0:** existen ≥30 fotos limpias y ≥10 ejemplos de al menos
1 tipo de defecto en Drive con la estructura de arriba.

---

## 5. Entrenamiento del LoRA (FASE B — gate 1)

Notebook: `notebooks/train_LoRA_kohya.ipynb`.

### 5.1 Preparación del dataset
- `scripts/prepara_dataset.py` copia crops a `lora_train/<repeats>_almdefect/`, redimensiona,
  y genera un `.txt` de caption por imagen (plantilla editable).
- Formato kohya con **repeats en el nombre de carpeta**: `10_almdefect scratch`.
- Buckets de resolución activados (`--enable_bucket`) para aceptar aspect ratios varios.

### 5.2 Config recomendada para T4 16 GB (SDXL)
```
--pretrained_model_name_or_path  RealVisXL_V5.0_fp16.safetensors
--network_module    networks.lora
--network_dim       32          # 16-32 basta para "texturas de defecto"; menos VRAM
--network_alpha     16
--resolution        1024,1024   # si va justo de VRAM: 768,768
--train_batch_size  1
--gradient_checkpointing         # imprescindible en T4
--optimizer_type    Adafactor
--optimizer_args    "scale_parameter=False" "relative_step=False" "warmup_init=False"
--fused_backward_pass            # kohya >= 0.9, enero 2025: clave para caber en 16 GB
--learning_rate     1e-4
--cache_latents  --cache_latents_to_disk
--cache_text_encoder_outputs     # ahorra VRAM (implica no entrenar TE / sin shuffle caption)
--network_train_unet_only        # entrena solo UNet: más estable y ligero
--mixed_precision   fp16         # T4 no tiene bf16 real
--xformers                       # o --sdpa
--max_train_epochs  10           # ~1500-3000 steps según nº de imágenes
--save_every_n_epochs 2          # guarda checkpoints intermedios para elegir el mejor
--seed 42
```
> **Nota T4:** si sale OOM, baja `--resolution` a `768,768` y `--network_dim` a 16.
> Tiempos esperados: 20-60 min dentro de la ventana de Colab gratis.

### 5.3 Selección del mejor checkpoint
- Guardamos cada 2 épocas. En ComfyUI se prueban a distintos `strength_model`
  (0.6-1.0) sobre las mismas fotos limpias y se elige el que da defecto realista
  **sin** degradar el resto de la imagen.
- **Señal de overfit:** aparece el fondo de las fotos de entrenamiento, o el defecto
  sale idéntico siempre. → menos épocas / dim más bajo / más variedad de datos.

**Criterio de aceptación gate 1:** existe `rollo_defectos_lora.safetensors` y al aplicarlo
en el workflow de inpaint produce un defecto reconocible dentro de la máscara.

---

## 6. Generación en ComfyUI (FASE C — gate 2)

Notebook host: `notebooks/hosting_ComfyUI.ipynb`. Workflows en `workflows/`.

### 6.1 Modelos que descarga el notebook
| archivo | destino | fuente |
|---|---|---|
| `RealVisXL_V5.0_fp16.safetensors` | `models/checkpoints` | HF `SG161222/RealVisXL_V5.0` |
| `sdxl_vae.safetensors` (fp16-fix) | `models/vae` | HF `madebyollin/sdxl-vae-fp16-fix` |
| `fooocus_inpaint_head.pth` | `models/inpaint` | HF `lllyasviel/fooocus_inpaint` |
| `inpaint_v26.fooocus.patch` | `models/inpaint` | HF `lllyasviel/fooocus_inpaint` |
| *(tu)* `rollo_defectos_lora.safetensors` | `models/loras` | tu Drive (FASE B) |
| *(opc.)* ControlNet SDXL union | `models/controlnet` | HF `xinsir/controlnet-union-sdxl-1.0` |

> El notebook también acepta **token de Civitai** por si quieres traer checkpoints de ahí.

### 6.2 Workflows entregados
- **`workflows/inpaint_defectos.json`** — *principal, solo nodos nativos* (máxima
  compatibilidad, carga sin instalar nada). Flujo:
  `CheckpointLoader → LoraLoader → 2×CLIPTextEncode → LoadImage(+máscara) →
   VAEEncodeForInpaint → KSampler(denoise≈0.85) → VAEDecode → SaveImage`.
  Uso: carga foto limpia, clic derecho → *Open in MaskEditor*, pinta dónde va el defecto,
  ajusta el prompt (`almdefect, deep scratch ...`) y ejecuta.
- **`workflows/txt2img_defectos.json`** — genera el rollo completo con defecto desde
  cero (para variedad de fondos/ángulos que no tengas fotografiados). Solo nodos nativos.
- **Upgrade de calidad (opcional, documentado abajo):** versión Fooocus-inpaint usando
  `comfyui-inpaint-nodes` para un inpaint SDXL superior.

### 6.3 Parámetros de inpaint recomendados
- `denoise` **0.75-0.90**: más alto = defecto más marcado pero puede alterar textura base.
- `grow_mask_by` **6-16 px**: suaviza el borde de la máscara para fundir el defecto.
- `steps` 25-30, `cfg` 5-7, sampler `dpmpp_2m` + scheduler `karras`.
- LoRA `strength_model` 0.6-0.9 (calibrar en gate 1).
- **Variedad automática:** en la API/lote, aleatoriza seed, tipo de defecto, posición y
  tamaño de máscara → dataset diverso.

### 6.4 Upgrade a Fooocus Inpaint (mejor calidad, requiere custom node)
El notebook instala `comfyui-inpaint-nodes` (Acly) vía ComfyUI-Manager. Sustituye
`VAEEncodeForInpaint` por: `INPAINT_LoadFooocusInpaint` (head+patch) → `INPAINT_ApplyFooocusInpaint`
sobre el MODEL, + `InpaintModelConditioning`. Notas:
- Usa el checkpoint **regular** (no Turbo/Lightning/Hyper) para el patch.
- Head: `fooocus_inpaint_head.pth`; patch: `inpaint_v26.fooocus.patch` en `models/inpaint`.

**Criterio de aceptación gate 2:** el workflow arrastrado genera, sobre una foto limpia
real, un defecto plausible dentro de la máscara, y `SaveImage` guarda el resultado.

---

## 7. Curado y export a formato ML (FASE D — gate 3)

- `scripts/genera_lote.py`: usa la **API de ComfyUI** (`/prompt`) para generar N imágenes
  variando semilla/máscara/defecto sin tocar la UI. Guarda imagen **y** máscara emparejadas.
- Export:
  - **Segmentación:** máscara binaria PNG por clase → COCO RLE / máscaras.
  - **Detección:** bbox de la máscara → YOLO `class cx cy w h` normalizado.
- **Filtro de calidad (recomendado):** descartar imágenes donde el defecto no quedó dentro
  de la máscara o el fondo se degradó. Heurística simple (varianza dentro/fuera de máscara)
  o revisión visual en contact-sheet.
- **Mezcla real+sintético:** para entrenar el detector final, combinar los pocos reales con
  el sintético (p. ej. 80/20 sint/real) y **validar SIEMPRE contra un holdout 100% real**.

**Criterio de aceptación gate 3:** existe un dataset exportado (imágenes + anotaciones YOLO
o máscaras COCO) listo para `train/val`, con val = solo datos reales.

---

## 8. Estructura del repositorio

```
img_sintéticas/
├── CLAUDE.md                       # este archivo (fuente de verdad)
├── README.md                       # guía de arranque rápido
├── requirements.txt                # deps locales (scripts, export)
├── .gitignore                      # ignora modelos/salidas pesadas
├── notebooks/
│   ├── hosting_ComfyUI.ipynb       # Colab: ComfyUI + tunnel + descarga modelos
│   └── train_LoRA_kohya.ipynb      # Colab: entrena el LoRA (T4)
├── workflows/
│   ├── inpaint_defectos.json       # workflow principal (arrastrar a ComfyUI)
│   └── txt2img_defectos.json       # generación full-scene
├── scripts/
│   ├── prepara_dataset.py          # crops + captions para kohya
│   ├── descarga_neu_det.py         # descarga+recorta dataset público NEU-DET (semilla)
│   └── genera_lote.py              # generación por lotes vía API ComfyUI + export YOLO
├── ui/                             # panel web local "Sala de Inspección"
│   ├── app.py                      # backend FastAPI (envuelve scripts + API ComfyUI)
│   ├── run.sh                      # lanzador -> http://127.0.0.1:8500
│   └── static/                     # index.html + styles.css + app.js (frontend)
└── runtime/                        # (gitignored) ComfyUI local + venv + modelos + bench.py
```

---

## 9. Roadmap y estado

| Fase | Entregable | Estado | Gate (valida el usuario) |
|------|-----------|--------|--------------------------|
| 0 · Scaffolding | repo + CLAUDE.md + README | ✅ hecho | revisar plan |
| 1 · Host ComfyUI | `hosting_ComfyUI.ipynb` + workflows | ✅ hecho | abre ComfyUI en Colab y carga el workflow |
| 2 · Datos semilla | guía §4 | ⏳ usuario | subir fotos a Drive |
| 3 · LoRA | `train_LoRA_kohya.ipynb` + `prepara_dataset.py` | ✅ hecho | entrenar y obtener .safetensors |
| 4 · Generación | inpaint funcionando con LoRA | ⏳ gate | generar 1 defecto realista |
| 5 · Lote + export | `genera_lote.py` | ✅ hecho | dataset YOLO/COCO exportado |
| 6 · Validación ML | entrenar detector real | ⏳ futuro | mAP contra holdout real |

---

## 10. Recomendaciones de modelos y fuentes

### 10.1 Checkpoints base SDXL (fotorrealismo)
- **RealVisXL V5.0** — *default del pipeline*. Muy buen realismo de materiales/superficies.
  HF: `SG161222/RealVisXL_V5.0` (sin token).
- **Juggernaut XL (Ragnarok)** — el más versátil, corre en 8 GB sin refiner. Civitai (token).
- **CyberRealistic XL** — salida limpia y uniforme, útil si quieres poco "grano".
- Para el patch Fooocus usa siempre la variante **regular** (no Turbo/Lightning/Hyper).

### 10.1.b Datasets semilla públicos (para arrancar el LoRA sin muchas fotos propias)
| Dataset | Contenido | Licencia | Cómo usarlo |
|---|---|---|---|
| **NEU-DET** | 1,800 img acero, 6 clases (scratches, pitted, inclusion, patches, crazing, rolled-in scale) + VOC bbox | investigación | `scripts/descarga_neu_det.py` lo baja y recorta a `defectos_reales/` |
| **GC10-DET** | 3,570 img, 10 clases (incluye crease, oil spot) | **CC BY 4.0** | mejor licencia; ideal para uso comercial |
| **Tianchi Aluminum Profile** | ~2,090 img de **aluminio** real, 9 defectos | registro Tianchi | el más cercano a tu dominio |
| **Severstal / MVTec AD** | acero laminado / anomalías industriales | Kaggle / **non-commercial** | MVTec solo investigación |
> ⚠️ Casi todos son acero/perfil, no *foil roll*. Úsalos para que el LoRA aprenda la
> **apariencia** del defecto; los **fondos limpios** ponlos con TUS fotos reales. Colección
> maestra con enlaces: github.com/Charmve/Surface-Defect-Detection.
> **Ejemplo ya desarrollado y probado:** `scripts/descarga_neu_det.py` exportó 1,770 recortes
> (corrosion 589, stain 591, crease 295, scratch 295) desde NEU-DET a `defectos_reales/`.

### 10.2 Fuentes de modelos (además de Civitai y HuggingFace)
- **Hugging Face** — lo más fiable para automatizar (URLs `resolve/main`, sin login para
  muchos). Preferido en los notebooks.
- **Civitai** — mayor variedad de LoRA/checkpoints comunitarios; requiere API token para
  descarga programática (el notebook lo soporta).
- **Tensor.Art / SeaArt** — repos alternativos de modelos comunitarios.
- **ModelScope (Alibaba)** — mirror útil si HF va lento desde tu región.
- **GitHub releases** (p. ej. `lllyasviel/fooocus_inpaint`, ControlNets de `xinsir`) —
  fiables para modelos auxiliares.
- Para **defectos industriales** existen datasets públicos que pueden servir de semilla:
  MVTec AD, Severstal Steel Defect, NEU Surface Defect (revisar licencias antes de usar).

### 10.3 Modelos auxiliares recomendados
- VAE: `madebyollin/sdxl-vae-fp16-fix` (evita artefactos en fp16 en T4).
- ControlNet SDXL: `xinsir/controlnet-union-sdxl-1.0` (canny/depth en un solo modelo) si el
  inpaint deforma la geometría del rollo.
- Inpaint: Fooocus patch (Acly/comfyui-inpaint-nodes) para el mejor resultado SDXL.

---

## 11. Restricciones y notas de la T4 (Colab gratis)

- **VRAM 16 GB:** SDXL inference cabe con `--lowvram`/offload de ComfyUI (automático).
  Entrenar LoRA exige la config de §5.2 (fused backward pass + Adafactor + gradient ckpt).
- **Sesión efímera:** Colab corta por inactividad/tiempo. Monta Drive y guarda ahí LoRA y
  salidas. El notebook es **idempotente**: re-ejecutar reanuda sin duplicar descargas.
- **Sin bf16 real:** usar `fp16`. Ojo con NaNs → el VAE fp16-fix ayuda.
- **Descargas:** `aria2c` multihilo para modelos grandes (mucho más rápido que wget).
- **Túnel:** la URL `trycloudflare.com` cambia en cada arranque; se imprime en la celda.

---

## 12. Troubleshooting rápido

| Síntoma | Causa probable | Solución |
|---|---|---|
| OOM al entrenar | resolución/dim alto en T4 | `768,768`, `network_dim 16`, confirmar fused backward pass |
| Defecto no aparece | denoise bajo o LoRA débil | subir `denoise` a 0.85-0.9, `strength_model` a 0.9 |
| Se degrada todo el rollo | denoise muy alto / máscara enorme | bajar denoise, máscara más ajustada, `grow_mask_by` menor |
| Aparece el fondo de entrenamiento | LoRA sobreajustado | menos épocas, más variedad de crops, dim menor |
| Imágenes fp16 con manchas | VAE fp16 inestable | usar `sdxl-vae-fp16-fix` |
| El workflow no carga nodos | custom node faltante | instalar vía ComfyUI-Manager (Fooocus solo si usas ese upgrade) |
| Colab lento descargando HF | región/carga HF | usar mirror ModelScope o reintentar aria2c |

---

## 14. Ejecución LOCAL en Apple Silicon (probado en M5)

Alternativa a Colab: correr todo en el Mac usando **MPS (Metal)** — no es CPU puro.

**Setup (una vez):**
```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI runtime/ComfyUI
uv venv runtime/venv --python 3.12          # Python 3.14 rompe torch; usar 3.12
source runtime/venv/bin/activate
uv pip install torch torchvision torchaudio # trae MPS en arm64
uv pip install -r runtime/ComfyUI/requirements.txt
# descargar RealVisXL a runtime/ComfyUI/models/checkpoints/
```

**Lanzar:**
```bash
source runtime/venv/bin/activate
cd runtime/ComfyUI && python main.py --listen 127.0.0.1 --port 8188
# abrir http://127.0.0.1:8188  y arrastrar workflows/inpaint_defectos.json
```

**Benchmark real (M5, 24 GB, MPS, RealVisXL SDXL):**
| Resolución | Pasos | Tiempo | Notas |
|---|---|---|---|
| 768×768 | 20 | ~47 s | incluye carga inicial del modelo |
| 1024×1024 | 30 | **~93 s** | modelo ya en memoria (target real) |

→ **~1.5 min por imagen 1024² en local.** Viable para lotes pequeños/medianos sin Colab.
Para lotes grandes o entrenar el LoRA rápido, Colab/T4 sigue siendo más cómodo.
Notas: usar Python **3.12** (no 3.14); ComfyUI detecta MPS solo (`Device: mps`, VRAM SHARED);
si hay presión de memoria, añadir `--use-split-cross-attention`.

**Entrenamiento del LoRA en local (kohya + MPS) — probado, con caveats:**
- ✅ Funciona: `sdxl_train_network.py` carga SDXL, cachea latentes y entrena en `mps`.
- ❌ **Media precisión bloqueada en MPS**: `accelerate` solo permite fp16/bf16 en CUDA
  (`bf16 mixed precision requires ... a supported device`). Se entrena en **fp32**.
- ⏱️ fp32 + `--gradient_checkpointing` ≈ **11 s/paso en régimen** (M5, 512², dim 16, batch 1).
  OJO: los primeros ~3-5 pasos son lentos (~50-67 s, warmup/compilación MPS) y engañan; el
  ritmo real se estabiliza en ~11 s. Sin grad-checkpointing satura 24 GB y hace swap → peor.
- **Probado:** LoRA demo de 220 pasos (160 img, 512²) = **41 min**. Un LoRA sólido de ~800
  pasos ≈ **2.5-3 h** local vs **~20-40 min en Colab T4** (fp16 + fused backward pass).
- **Resultado validado:** el LoRA aprendió las texturas de defecto (txt2img con `almdefect`
  a strength 1.1 produce superficie metálica picada/rayada). Ciclo end-to-end completo OK:
  dataset→LoRA→ComfyUI inpaint→imagen+máscara+etiqueta YOLO (`scripts/genera_lote.py`).
- Entorno kohya local: `runtime/venv_kohya` (separado del de ComfyUI), `runtime/kohya`.
- **Importante:** nuestras captions son `.txt` → pasar `--caption_extension .txt` (kohya usa
  `.caption` por defecto), o el token `almdefect` no entra al entrenamiento.
- Flags MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `--sdpa` (no xformers),
  `--mixed_precision no`, `--max_data_loader_n_workers 0`.

## 15. Panel web local — "Sala de Inspección" (`ui/`)

Consola web para operar todo el pipeline sin tocar la terminal. Estética de instrumento
metrológico (grafito/acero, acentos cian=sistema y coral=defecto).

**Lanzar:**
```bash
bash ui/run.sh          # -> http://127.0.0.1:8500  (ComfyUI debe estar corriendo)
```

**Qué hace (FastAPI + frontend sin build):**
- **Panel**: telemetría en vivo (ComfyUI on/off, dispositivo MPS, LoRAs, recortes, generadas)
  y flujo de trabajo con pasos hecho/pendiente.
- **Galería**: tarjetas-inspección con las imágenes generadas; overlay del **bbox del defecto**
  (la etiqueta YOLO) con toggle "Marcar defecto". Clic → visor grande.
- **Generar**: lanza `genera_lote.py` (inpaint/txt2img) con controles (n, denoise, LoRA, carpeta
  limpia) y consola de log en vivo.
- **Entrenar**: lanza el entrenamiento kohya (MPS) y "Publicar LoRA en ComfyUI".
- **Datos**: descarga NEU-DET, prepara `lora_train/`, y muestra recortes semilla por clase.
- Procesos largos = jobs con log en vivo (cajón inferior). Servido de archivos con validación
  anti-traversal (rutas fuera del proyecto → 400).

Deps: `ui/requirements.txt` (fastapi, uvicorn, python-multipart, requests) en `runtime/venv`.

## 13. Glosario

- **LoRA:** adaptador ligero que enseña un concepto/estilo nuevo al modelo con pocas imágenes.
- **Inpainting:** regenerar solo la zona enmascarada de una imagen manteniendo el resto.
- **Denoise:** cuánto se aleja el inpaint de la imagen original (0 = nada, 1 = todo nuevo).
- **Domain gap:** diferencia entre datos de entrenamiento (sintéticos) y reales de producción.
- **Ground truth:** etiqueta verdadera; aquí, la máscara del defecto = segmentación real.
```
