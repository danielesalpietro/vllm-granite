#!/bin/sh
python3 - << 'PYEOF'
import os, sys, time

# ── Monkey-patch: forza xet_file_data=None in tutti i download ─────────────
# (huggingface_hub decide di usare XetHub in base a xet_file_data; il
# vecchio parametro "use_xet" non esiste piu' nelle versioni recenti)
try:
    import huggingface_hub.file_download as _fd
    _orig = _fd._download_to_tmp_and_move
    def _no_xet(*a, **kw):
        kw["xet_file_data"] = None
        return _orig(*a, **kw)
    _fd._download_to_tmp_and_move = _no_xet
    print("[start] XetHub disabilitato via monkey-patch", flush=True)
except Exception as exc:
    print(f"[start] Avviso monkey-patch: {exc}", flush=True)

model = os.environ.get("MODEL_ID", "ibm-granite/granite-3.3-8b-instruct")
print(f"[start] Download {model} (HTTPS, no XetHub)...", flush=True)

from huggingface_hub import snapshot_download

for attempt in range(30):
    try:
        snapshot_download(model)
        print("[start] Modello pronto.", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"[start] Tentativo {attempt+1}/30 fallito: {exc}", flush=True)
        if attempt < 29:
            time.sleep(10)

print("[start] Tutti i tentativi esauriti, vLLM tenta autonomamente.", flush=True)
PYEOF

# ── Quantizzazione on-the-fly (opzionale) ───────────────────────────────────
# QUANTIZATION="bitsandbytes" carica i pesi bf16 originali quantizzandoli a
# 4-bit al volo (~4x meno VRAM per i pesi) — utile su GPU con poca VRAM
# (es. 16 GB) dove il modello non entrerebbe altrimenti. Lasciare vuoto
# (default) su GPU con VRAM sufficiente: nessun cambiamento di comportamento.
set -- vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE" \
  --trust-remote-code \
  --enable-prefix-caching \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --enable-auto-tool-choice \
  --tool-call-parser granite \
  --served-model-name granite

if [ -n "$QUANTIZATION" ]; then
  set -- "$@" --quantization "$QUANTIZATION" --load-format bitsandbytes
fi

exec "$@"
