#!/usr/bin/env bash
# =============================================================================
# h2_ab_mtp.sh — A/B MTP n_max 3→2→1 + perf (sur le serveur de CAPTURE)
# Protocole = h2_capture.sh sans nsys : stop prod → serveur direct → gates
# fit/V100/model-loaded → warmup → prefill 100k → 2e passe chaude → decode.
# Chaque config = serveur propre (n_max est un flag de lancement).
# Ordre alterné : A(3,2,1) puis B(1,2,3). 2 réplicats × 3 configs × 600 tok.
# Après : cycle perf (n_max=3, 1200 tok) : perf stat + perf record -g pendant
# le decode. Restauration prod garantie à chaque phase (trap).
# =============================================================================
set -uo pipefail

# --- Configuration (défauts = machine de référence; voir config/kit.env.example) ---
: "${LLAMA_ROOT:=/home/guile/llama-experiments}"
: "${OLLAMA_ROOT:=$LLAMA_ROOT/ollama-v100}"
: "${NSYS_BIN:=/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys}"
: "${MODEL_GGUF:=/usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb}"
: "${FILLER_FILE:=$LLAMA_ROOT/v100-repro/work/filler-200k.txt}"
BASE="${LLAMA_ROOT}"
TARGET="${OLLAMA_ROOT}"
SERVICE="${OLLAMA_SERVICE:-ollama-v100}"
MODEL_GGUF="${MODEL_GGUF}"
SRCRUN=$BASE/h2/runs/run-20261009-152723
AB=$BASE/h2/ab_results
PORT=44901
API="http://127.0.0.1:$PORT"
V100_IDX=0          # ordre CUDA (cuda:0 = V100 sur cette machine)
N_DEC=600
N_PERF=1200

mkdir -p "$AB"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
so_md5(){ md5sum "$TARGET/lib/ollama/cuda_v12/libggml-cuda.so" | cut -c1-8; }

# ---------- prompts (identiques pour toutes les configs) ----------
python3 - "$SRCRUN/prompt.json" "$AB/prompt600.json" "$AB/prefill.json" "$N_DEC" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["max_tokens"] = int(sys.argv[4])
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
p = dict(d); p["max_tokens"] = 32
json.dump(p, open(sys.argv[3], "w"), ensure_ascii=False)
PY

restore(){
  log "=== restauration service ==="
  pkill -f "port $PORT" 2>/dev/null; sleep 3
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && break
    pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  done
  sudo systemctl start "$SERVICE" || true
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
  sleep 8
  log "restore: service=$(systemctl is-active $SERVICE) V100=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB md5=$(so_md5) (attendu e4afbb4a)"
}
die(){ log "ERREUR: $*"; restore; exit 1; }

stop_and_free(){
  sudo systemctl stop "$SERVICE" 2>/dev/null || true
  local ok=0 u
  for i in $(seq 1 120); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && { ok=1; break; }
    pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  done
  [ "$ok" = 1 ] || { log "V100 pas libre (${u:-?} MiB)"; die "V100 pas libre"; }
  log "V100 libre: ${u} MiB"
}

# ---------- une config complète ----------
# args: <tag> <nmax> <run-dir> <decode-tokens> <perf:0|1>
run_cfg(){
  local tag=$1 nmax=$2 RUN=$3 ntok=$4 perfmode=${5:-0}
  mkdir -p "$RUN"
  log "=== [$tag] n_max=$nmax decode=$ntok perf=$perfmode ==="
  stop_and_free
  export LD_LIBRARY_PATH="$TARGET/lib/ollama:$TARGET/lib/ollama/cuda_v12:$BASE/tools/cuda-12.6/usr/local/cuda-12.6/lib64:$BASE/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
  export GGML_BACKEND_PATH="$TARGET/lib/ollama/cuda_v12/libggml-cuda.so"
  export GGML_CUDA_VOLTA_Q6_W4R4=1
  export GGML_CUDA_QWEN35_MTP_SHORTLIST=$BASE/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32
  export LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1
  export CUDA_VISIBLE_DEVICES=$V100_IDX
  export LLAMA_ARG_FIT_TARGET=1024

  "$TARGET/lib/ollama/llama-server" \
    --model "$MODEL_GGUF" --port $PORT --host 127.0.0.1 --no-webui \
    -c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps \
    --spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-backend-sampling \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512 \
    > "$RUN/server.out" 2>&1 &
  local SRV=$!
  echo $SRV > "$RUN/srv.pid"

  local ok=0
  for i in $(seq 1 300); do curl -s --max-time 2 "$API/v1/models" >/dev/null 2>&1 && { ok=1; break; }; sleep 1; done
  [ "$ok" = 1 ] || { tail -25 "$RUN/server.out"; die "[$tag] server pas up"; }

  local FITLINE=""
  for i in $(seq 1 150); do
    grep -q 'fit_params: failed\|unable to fit' "$RUN/server.out" 2>/dev/null && { tail -15 "$RUN/server.out"; die "[$tag] fit ABORT"; }
    FITLINE=$(grep -m1 'successfully fit params' "$RUN/server.out" 2>/dev/null); [ -n "$FITLINE" ] && break
    sleep 1
  done
  [ -n "$FITLINE" ] || { tail -25 "$RUN/server.out"; die "[$tag] fit non confirmé"; }
  local DEVLINE=$(grep -m1 'using device CUDA' "$RUN/server.out" 2>/dev/null)
  echo "$DEVLINE" | grep -q 'Tesla V100' || { echo "$DEVLINE"; die "[$tag] device != V100"; }
  log "[$tag] fit+device OK"

  local MLOK=0 MLERR=""
  for i in $(seq 1 900); do
    grep -q 'llama_server: model loaded' "$RUN/server.out" 2>/dev/null && { MLOK=1; break; }
    grep -q 'out of memory' "$RUN/server.out" 2>/dev/null && MLERR="OOM"
    kill -0 "$SRV" 2>/dev/null || MLERR="server mort"
    [ -n "$MLERR" ] && break
    sleep 1
  done
  [ "$MLOK" = 1 ] || { tail -25 "$RUN/server.out"; die "[$tag] model non chargé (${MLERR:-timeout})"; }
  log "[$tag] model loaded (${i}s)"

  local warm_ok=0
  for w in 1 2 3 4 5; do
    curl -s --max-time 900 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"qwen3.8-27b-q6-mtp:latest\",\"stream\":false,\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}" \
      > "$RUN/warm.json" 2>&1
    [ -s "$RUN/warm.json" ] && ! grep -q '"error"' "$RUN/warm.json" && { warm_ok=1; break; }
    sleep 10
  done
  [ "$warm_ok" = 1 ] || { tail -5 "$RUN/warm.json"; die "[$tag] warmup KO"; }
  grep -m1 'CUDA MTP shortlist' "$RUN/server.out" > "$RUN/shortlist.out" 2>/dev/null || { tail -15 "$RUN/server.out"; die "[$tag] shortlist non chargé"; }
  log "[$tag] warmup + shortlist OK"

  log "[$tag] prefill 100k..."
  curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$AB/prefill.json" > "$RUN/prefill.json.out" 2>&1 || die "[$tag] prefill rat"
  grep -q '"error"' "$RUN/prefill.json.out" && { tail -5 "$RUN/prefill.json.out"; die "[$tag] prefill erreur"; }
  log "[$tag] 2e passe chaude..."
  curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$AB/prefill.json" > "$RUN/prefill2.json.out" 2>&1 || die "[$tag] prefill2 rat"
  grep -q '"error"' "$RUN/prefill2.json.out" && die "[$tag] prefill2 erreur"
  log "[$tag] chaud — decode $ntok tok..."

  # prompt du decode (ntok)
  python3 - "$SRCRUN/prompt.json" "$RUN/prompt.json" "$ntok" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["max_tokens"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY

  local t0 t1
  if [ "$perfmode" = 1 ]; then
    local NT=$(awk '/^Threads:/{print $2}' /proc/$SRV/status)
    log "[$tag] threads du server: $NT — demarrage perf (stat + record -g cycles)"
    sudo perf stat -e cycles,instructions,task-clock,cpu-migrations,context-switches -p $SRV -o "$RUN/perfstat.txt" > "$RUN/perfstat.launch" 2>&1 &
    local PSTAT=$!
    sudo perf record -F 199 -g -e cycles -o "$RUN/perf.data" -p $SRV > "$RUN/perfrec.launch" 2>&1 &
    local PREC=$!
    sleep 2
  fi
  t0=$(date +%s)
  curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$RUN/prompt.json" > "$RUN/gen.json" 2>&1 || die "[$tag] decode rat"
  grep -q '"finish_reason"' "$RUN/gen.json" || die "[$tag] decode non terminé"
  t1=$(date +%s)
  log "[$tag] decode fini (wall $((t1-t0)) s)"

  if [ "$perfmode" = 1 ]; then
    sleep 2
    kill -INT $PREC 2>/dev/null; sleep 1; kill $PREC 2>/dev/null; wait $PREC 2>/dev/null
    kill -INT $PSTAT 2>/dev/null; sleep 1; kill $PSTAT 2>/dev/null; wait $PSTAT 2>/dev/null
    log "[$tag] perf stoppé: $(ls -la "$RUN/perf.data" 2>/dev/null | awk '{print $5}') octets"
  fi

  # kill propre du server + liberation V100 (SANS redemarrer prod)
  pkill -f "port $PORT" 2>/dev/null
  kill -9 $SRV 2>/dev/null
  for i in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && break
    sleep 1
  done
  log "[$tag] V100 liberee (${u:-?} MiB)"
}

# ---------- garde-fous globaux ----------
[ "$(so_md5)" = "e4afbb4a" ] || die "md5 .so prod inattendu au depart: $(so_md5)"
trap restore EXIT

# ---------- phase A/B : ordre alterné ----------
for rep in 1 2; do
  if [ "$rep" = 1 ]; then ORDRE="3 2 1"; else ORDRE="1 2 3"; fi
  for n in $ORDRE; do
    run_cfg "A${rep}-n$n" "$n" "$AB/ab-${rep}-n${n}" "$N_DEC" 0
  done
done
log "=== A/B termines — restauration prod (phase) ==="
restore

# ---------- phase perf : n_max=3, decode 1200 tok ----------
run_cfg "PERF-n3" 3 "$AB/perf-n3" "$N_PERF" 1
log "=== perf termines — restauration prod finale ==="
restore

log "=== VERIF FINALE ==="
echo "service=$(systemctl is-active $SERVICE) md5=$(so_md5)"
ls -la "$AB" | head -20
log "FIN h2_ab_mtp.sh"
