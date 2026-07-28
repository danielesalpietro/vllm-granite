#!/bin/sh
# ── Riapplica system prompt + temperatura al workspace AnythingLLM ──────────
#
# Le impostazioni di workspace (system prompt, temperatura) vivono solo nel
# DB SQLite dentro il volume anythingllm_storage, non in questo repo — se il
# volume viene ricreato vanno reimpostate. AnythingLLM non offre un file di
# provisioning per questi valori (a differenza di Grafana), ma espone una
# Developer API che questo script chiama per riapplicarli senza passare
# dalla UI.
#
# Prerequisito (una tantum, non automatizzabile — limite di AnythingLLM):
#   1. Completa l'onboarding iniziale in UI (crea l'account admin).
#   2. Genera una API key di sistema: UI → Impostazioni → API Keys.
#   3. Imposta in .env: ANYTHINGLLM_API_KEY e ANYTHINGLLM_WORKSPACE_SLUG
#      (lo slug è visibile nell'URL della workspace, es. /workspace/<slug>).
#
# Uso: ./sync-workspace-settings.sh

set -eu

# Carica .env se presente, senza richiedere che l'utente faccia export a mano.
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

: "${ANYTHINGLLM_API_KEY:?Imposta ANYTHINGLLM_API_KEY in .env (Admin UI -> Settings -> API Keys)}"
: "${ANYTHINGLLM_WORKSPACE_SLUG:?Imposta ANYTHINGLLM_WORKSPACE_SLUG in .env (slug della workspace)}"
ANYTHINGLLM_BASE_URL="${ANYTHINGLLM_BASE_URL:-http://localhost:3001}"
ANYTHINGLLM_TEMPERATURE="${ANYTHINGLLM_TEMPERATURE:-0.4}"
ANYTHINGLLM_SYSTEM_PROMPT="${ANYTHINGLLM_SYSTEM_PROMPT:-Always respond in the same language the user writes in, matching it exactly, unless the user explicitly asks you to switch or translate. Never claim you are only able to respond in English.}"

# Escape minimo per JSON (backslash e doppi apici). Il prompt è assunto su
# un'unica riga: se contiene newline reali, sostituiscili con \n prima di
# metterlo in .env.
esc_prompt=$(printf '%s' "$ANYTHINGLLM_SYSTEM_PROMPT" | sed 's/\\/\\\\/g; s/"/\\"/g')

response_file=$(mktemp)
trap 'rm -f "$response_file"' EXIT

http_status=$(curl -sS -o "$response_file" -w '%{http_code}' \
  -X POST "$ANYTHINGLLM_BASE_URL/api/v1/workspace/$ANYTHINGLLM_WORKSPACE_SLUG/update" \
  -H "Authorization: Bearer $ANYTHINGLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"openAiPrompt\":\"$esc_prompt\",\"openAiTemp\":$ANYTHINGLLM_TEMPERATURE}")

cat "$response_file"
echo

if [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
  echo "[sync] Workspace '$ANYTHINGLLM_WORKSPACE_SLUG' aggiornato (prompt + temperatura=$ANYTHINGLLM_TEMPERATURE)."
else
  echo "[sync] Richiesta fallita (HTTP $http_status)." >&2
  exit 1
fi
