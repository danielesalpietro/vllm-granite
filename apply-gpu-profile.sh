#!/bin/sh
# ── Applica un preset GPU/host da gpu-profiles.json a .env ─────────────────
#
# Non tocca start.sh/Dockerfile: legge gpu-profiles.json e scrive/aggiorna
# solo GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, QUANTIZATION, CPU_OFFLOAD_GB,
# VLLM_USE_V2_MODEL_RUNNER in .env. Tutte le altre variabili (MODEL_ID,
# HF_TOKEN, AUTH_TOKEN, JWT_SECRET, ...) restano invariate.
#
# Uso:
#   ./apply-gpu-profile.sh --list
#   ./apply-gpu-profile.sh <gpu-profile> <host-profile>
#
# Esempio:
#   ./apply-gpu-profile.sh rtx-5080-16gb wsl2

set -eu

usage() {
  cat <<EOF
Uso: $0 <gpu-profile> <host-profile>
     $0 --list

Esempio: $0 rtx-5080-16gb wsl2
EOF
}

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
  LIST=1
elif [ $# -ne 2 ]; then
  usage
  exit 1
else
  LIST=0
fi

PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "" >/dev/null 2>&1; then
    PYTHON="$cand"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Errore: serve un interprete Python 3 funzionante (python3 o python) per leggere gpu-profiles.json." >&2
  exit 1
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROFILES_FILE="$SCRIPT_DIR/gpu-profiles.json"

if [ ! -f "$PROFILES_FILE" ]; then
  echo "Errore: non trovo $PROFILES_FILE" >&2
  exit 1
fi

if [ "$LIST" = "1" ]; then
  "$PYTHON" - "$PROFILES_FILE" <<'PYEOF'
import json, sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

print("GPU profiles (VRAM/quantizzazione):")
for key, v in sorted(data.get("gpu_profiles", {}).items()):
    tag = "verificato" if v.get("verified") else "stimato, non testato"
    print(f"  {key:<20} vram={v.get('vram_gb')}GB quantization={v.get('quantization') or '(nessuna)'}  [{tag}]")

print()
print("Host profiles (ambiente Docker):")
for key, v in sorted(data.get("host_profiles", {}).items()):
    tag = "verificato" if v.get("verified") else "stimato, non testato"
    print(f"  {key:<20} [{tag}]")
PYEOF
  exit 0
fi

GPU_KEY="$1"
HOST_KEY="$2"

if [ ! -f ./.env ]; then
  if [ -f ./.env.docker.example ]; then
    cp ./.env.docker.example ./.env
    echo "[apply-gpu-profile] .env non trovato, creato da .env.docker.example."
  else
    echo "Errore: ne' .env ne' .env.docker.example trovati nella directory corrente." >&2
    echo "Esegui questo script dalla root del repo." >&2
    exit 1
  fi
fi

OUTPUT=$(GPU_KEY="$GPU_KEY" HOST_KEY="$HOST_KEY" "$PYTHON" - "$PROFILES_FILE" <<'PYEOF'
import json, os, sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

gpu_key = os.environ["GPU_KEY"]
host_key = os.environ["HOST_KEY"]
gpu_profiles = data.get("gpu_profiles", {})
host_profiles = data.get("host_profiles", {})

if gpu_key not in gpu_profiles:
    print(f"Profilo GPU sconosciuto: {gpu_key}", file=sys.stderr)
    print("Disponibili: " + ", ".join(sorted(gpu_profiles)), file=sys.stderr)
    sys.exit(1)
if host_key not in host_profiles:
    print(f"Profilo host sconosciuto: {host_key}", file=sys.stderr)
    print("Disponibili: " + ", ".join(sorted(host_profiles)), file=sys.stderr)
    sys.exit(1)

gpu = gpu_profiles[gpu_key]
host = host_profiles[host_key]

for key, value in (
    ("GPU_MEMORY_UTILIZATION", gpu["gpu_memory_utilization"]),
    ("MAX_MODEL_LEN", gpu["max_model_len"]),
    ("QUANTIZATION", gpu.get("quantization", "")),
    ("CPU_OFFLOAD_GB", host.get("cpu_offload_gb", 0)),
    ("VLLM_USE_V2_MODEL_RUNNER", host.get("vllm_use_v2_model_runner", "")),
):
    print(f"{key}={value}")

gpu_tag = "verificato" if gpu.get("verified") else "stimato, non testato"
host_tag = "verificato" if host.get("verified") else "stimato, non testato"
print(f"[apply-gpu-profile] GPU={gpu_key} [{gpu_tag}]: {gpu.get('notes', '')}", file=sys.stderr)
print(f"[apply-gpu-profile] Host={host_key} [{host_tag}]: {host.get('notes', '')}", file=sys.stderr)
PYEOF
) || exit 1

echo "$OUTPUT" | while IFS='=' read -r key value; do
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
  echo "[apply-gpu-profile] ${key}=${value}"
done

echo "[apply-gpu-profile] Fatto. Ricontrolla .env, poi: docker compose up -d --build"
