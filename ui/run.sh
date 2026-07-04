#!/usr/bin/env bash
# Lanza la Sala de Inspección (panel web local del studio de datos sintéticos).
# Requiere que ComfyUI esté corriendo (notebooks/ o runtime local) para generar.
cd "$(dirname "$0")/.."
source runtime/venv/bin/activate
echo "Sala de Inspección -> http://127.0.0.1:8500"
exec python ui/app.py
