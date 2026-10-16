#!/usr/bin/env bash
# run_cupath_llama.sh — capture CUPTI 12.9 du VRAI chemin de décodage (llama-server prod, verify MTP)
# Protocol d'alignement strict sur h2_capture.sh (gates FIT / DEVICE / model loaded / warmup /
# shortlist / prefill hors fenêtre / 2e demande chaude) + shim LD_PRELOAD libcupath_v2.so.
#
# Usage : ./run_cupath_llama.sh [nb_tokens]   (défaut 64)
# STOPPE le service ollama-v100 (~15-20 min) et se restaure seul (trap EXIT).
set -u

# --- Configuration (défauts = machine de référence; voir config/kit.env.example) ---
: "${LLAMA_ROOT:=/home/guile/llama-experiments}"
: "${OLLAMA_ROOT:=$LLAMA_ROOT/ollama-v100}"
: "${NSYS_BIN:=/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys}"
: "${MODEL_GGUF:=/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb}"
: "${FILLER_FILE:=$LLAMA_ROOT/v100-repro/work/filler-200k.txt}"
cd "${LLAMA_ROOT}"
CUPATH="${CUPATH_DIR:-${LLAMA_ROOT}/h2/cupti129-test}"
PORT=44901
NTOK=${1:-64}
OUT=/tmp/cupath8_$(date +%H%M%S)
API="http://127.0.0.1:$PORT"
MODEL_GGUF="${MODEL_GGUF}"
MODEL_TAG="qwen3.8-27b-q6-mtp:latest"
FILLER=$PWD/v100-repro/work/filler-200k.txt
CTX_CHARS=${H2_CTX_CHARS:-186000}
SOUT=/tmp/cupath8_server.out

restore(){
  echo "[restore] nettoyage server de capture"
  pkill -f "port $PORT" 2>/dev/null; sleep 3
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && break
    pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  done
  sudo systemctl start ollama-v100 || true
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
  sleep 8
  echo "[restore] service=$(systemctl is-active ollama-v100) V100=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB md5=$(md5sum ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so | cut -c1-8)"
  echo "[restore] traces : $OUT.bin / $OUT.bin.names"
}
trap restore EXIT

[ -f "$FILLER" ] || { echo "FILLER introuvable"; exit 1; }
[ -f "$MODEL_GGUF" ] || { echo "GGUF introuvable"; exit 1; }

# prompts (identiques a h2_capture.sh)
python3 - "$FILLER" "$CTX_CHARS" /tmp/cupath8_prompt.json "$NTOK" <<'PY'
import json, sys
filler = open(sys.argv[1], encoding="utf-8", errors="replace").read()
text = filler[: int(sys.argv[2])]
json.dump({"model": "qwen3.8-27b-q6-mtp:latest", "stream": False,
           "temperature": 0, "top_k": 1, "ignore_eos": True,
           "max_tokens": int(sys.argv[4]),
           "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "user", "content": text}]}, open(sys.argv[3], "w"), ensure_ascii=False)
PY
python3 - /tmp/cupath8_prompt.json /tmp/cupath8_prefill.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["max_tokens"] = 32
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY

sudo systemctl stop ollama-v100 2>/dev/null || true
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  pkill -9 -f "llama-server" 2>/dev/null; sleep 1
done
echo "V100 libre: ${u:-?} MiB"

export LD_LIBRARY_PATH="$PWD/ollama-v100/lib/ollama:$PWD/ollama-v100/lib/ollama/cuda_v12:$PWD/tools/cuda-12.6/usr/local/cuda-12.6/lib64:$PWD/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
export GGML_BACKEND_PATH="$PWD/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
export GGML_CUDA_VOLTA_Q6_W4R4=1
export GGML_CUDA_QWEN35_MTP_SHORTLIST=$PWD/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32
export LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ARG_FIT_TARGET=1024
export LD_PRELOAD="$CUPATH/libcupath_v2.so"
export CUPATH_TRACE=$OUT

timeout 3600 $PWD/ollama-v100/lib/ollama/llama-server --model "$MODEL_GGUF" --port $PORT --host 127.0.0.1 --no-webui -c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512 > "$SOUT" 2>&1 &
SPID=$!
echo "llama-server pid=$SPID"

ok=0
for i in $(seq 1 300); do
  curl -s --max-time 2 "$API/v1/models" >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ "$ok" = 1 ] || { tail -25 "$SOUT"; echo "server pas up"; exit 1; }
echo "server up (port)"

# GATE FIT
FITLINE=""
for i in $(seq 1 300); do
  if grep -q 'fit_params: failed\|unable to fit' "$SOUT" 2>/dev/null; then
    grep -B3 -A2 'fit_params: failed\|unable to fit' "$SOUT" | tail -12; echo "FIT ABORT"; exit 1
  fi
  FITLINE=$(grep -m1 'successfully fit params' "$SOUT" 2>/dev/null)
  [ -n "$FITLINE" ] && break
  kill -0 $SPID 2>/dev/null || { echo "server mort"; tail -20 "$SOUT"; exit 1; }
  sleep 1
done
[ -n "$FITLINE" ] || { tail -25 "$SOUT"; echo "fit non confirme"; exit 1; }
echo "$FITLINE" | head -c 200; echo " <- GATE FIT OK"

# GATE DEVICE
DEVLINE=$(grep -m1 'using device CUDA' "$SOUT" 2>/dev/null | tail -1)
echo "${DEVLINE:-AUCUNE LIGNE 'using device'}"
echo "$DEVLINE" | grep -q 'Tesla V100' || { echo "device != V100 — abort"; exit 1; }
echo "GATE DEVICE OK (V100)"

# GATE MODEL LOADED
MLOK=0
for i in $(seq 1 900); do
  if grep -q 'llama_server: model loaded' "$SOUT" 2>/dev/null; then MLOK=1; break; fi
  if grep -q 'out of memory' "$SOUT" 2>/dev/null; then echo "OOM"; tail -25 "$SOUT"; exit 1; fi
  kill -0 $SPID 2>/dev/null || { echo "server mort pendant load"; tail -25 "$SOUT"; exit 1; }
  sleep 1
done
[ "$MLOK" = 1 ] || { echo "model non charge"; tail -25 "$SOUT"; exit 1; }
echo "GATE MODEL LOADED OK (apres ${i}s)"

# WARMUP 16 tok (JIT/graphs/shortlist)
warm_ok=0
for w in 1 2 3 4 5; do
  curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_TAG\",\"stream\":false,\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}" \
    > /tmp/cupath8_warm.json 2>&1
  [ -s /tmp/cupath8_warm.json ] && ! grep -q '"error"' /tmp/cupath8_warm.json && { warm_ok=1; break; }
  echo "warmup essai $w: $(head -c 120 /tmp/cupath8_warm.json) — retry 10 s"; sleep 10
done
[ "$warm_ok" = 1 ] || { tail -5 /tmp/cupath8_warm.json; echo "warmup KO"; exit 1; }
echo "warmup done: $(python3 -c "import json;d=json.load(open('/tmp/cupath8_warm.json'));print(round(d.get('usage',{}).get('completion_tokens',0)),'tok')" 2>/dev/null)"

# GATE SHORTLIST (chargement lazy au 1er decode MTP)
grep -m1 'CUDA MTP shortlist' "$SOUT" 2>/dev/null || { tail -15 "$SOUT"; echo "shortlist non charge — abort"; exit 1; }
echo "GATE SHORTLIST OK"

# PREFILL ~100k HORS FENETRE (1er run 4-6 min) + 2e demande chaude
echo "prefill ~100k (hors fenêtre, 4-6 min)..."
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/cupath8_prefill.json > /tmp/cupath8_prefill.out 2>&1 \
  || { tail -5 /tmp/cupath8_prefill.out; echo "prefill rat"; exit 1; }
[ -s /tmp/cupath8_prefill.out ] && ! grep -q '"error"' /tmp/cupath8_prefill.out || { tail -5 /tmp/cupath8_prefill.out; echo "prefill erreur"; exit 1; }
echo "prefill done"
echo "2e demande identique (chaud, hors fenêtre)..."
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/cupath8_prefill.json > /tmp/cupath8_prefill2.out 2>&1 \
  || { tail -5 /tmp/cupath8_prefill2.out; echo "prefill2 rat"; exit 1; }
echo "chaud OK — demarrage fenêtre de decode"

# FENETRE DECODE : même prompt chaud, $NTOK tokens
TS_START=$(($(date +%s%N)/1000))
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/cupath8_prompt.json > /tmp/cupath8_gen.json 2>&1
sleep 2
TS_END=$(($(date +%s%N)/1000))
[ -s /tmp/cupath8_gen.json ] && ! grep -q '"error"' /tmp/cupath8_gen.json || { tail -5 /tmp/cupath8_gen.json; echo "generation KO"; }
grep -q '"finish_reason"' /tmp/cupath8_gen.json && echo "generation terminee" || echo "ATTENTION: pas de finish_reason"

echo "$TS_START $TS_END" > /tmp/cupath8_bracket
sleep 5
kill $SPID 2>/dev/null
wait $SPID 2>/dev/null

echo "=== analyse v2 (gates + union + occupation + overlap) ==="
WALL_MS=$(grep -oE "eval time = +[0-9.]+ ms" "$SOUT" | tail -1 | grep -oE "[0-9.]+")
echo "wall serveur (eval time, derniere req) : ${WALL_MS:-NON TROUVE} ms"
TRC=$(ls -S "$OUT"/*.names 2>/dev/null | head -1); TRC=${TRC%.names}
echo "trace du pid (plus gros fichier) : $TRC"
python3 - "$TRC" "$TS_START" "$TS_END" "$WALL_MS" <<'PY'
import sys
sys.argv = [sys.argv[0], sys.argv[1], "window", sys.argv[2], sys.argv[3], sys.argv[4]]
import os
exec(open(os.environ.get("CUPATH_ANALYZE", "/home/guile/llama-experiments/h2/cupti129-test/cupath_analyze_v2.py")).read())
PY
