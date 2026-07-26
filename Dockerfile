# ── vLLM + IBM Granite / MoE ──────────────────────────────────────────────────
FROM vllm/vllm-openai:v0.8.5

LABEL maintainer="daniele.salpietro@gmail.com" \
      description="vLLM server – Granite / MoE – 24 GB VRAM + 192 GB RAM"

# ── Variabili d'ambiente ───────────────────────────────────────────────────────
ENV MODEL_ID="ibm-granite/granite-3.3-8b-instruct" \
    MAX_MODEL_LEN=8192 \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.90 \
    DTYPE="bfloat16" \
    PORT=8000 \
    HOST="0.0.0.0" \
    HF_TOKEN=""

ENV HF_HOME=/root/.cache/huggingface

# Disabilita hf_transfer (Rust) e XetHub/CAS.
# HF_HUB_DISABLE_XET non è riconosciuto da tutte le versioni di huggingface_hub
# quindi usiamo anche un monkey-patch runtime (vedi start.sh).
ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1

# ── Script di avvio ────────────────────────────────────────────────────────────
# start.sh:
#   1. Monkey-patcha huggingface_hub per forzare HTTPS (no XetHub/CAS)
#   2. Scarica il modello con retry (fino a 30 tentativi)
#   3. Avvia vLLM con exec (sostituisce il processo shell)
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=15s --start-period=300s --retries=5 \
  CMD curl -sf http://localhost:${PORT}/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["/app/start.sh"]
CMD []
