#!/usr/bin/env bash
# h2b_control_ab.sh — CONTROLE d'attribution : lib H2 (a4b6f440) vs lib H2b (0eb5afbd)
# dans la MEME session. Repond : le gain -13% de H2 (v2, autre session) est-il
# session-specifique, ou le partitionnement (BC=8 vs PC=16) pilote-t-il la perf
# applicative ? Meme protocole h2_capture, 1200 tok, ordre alterne.
set -uo pipefail
BASE=${LLAMA_ROOT:-/home/guile/llama-experiments}
TARGET=$BASE/ollama-v100
SERVICE=ollama-v100
H2LIB=$BASE/h2/cupti129-test/e2/h2lib
SRCRUN=$BASE/h2/runs/run-20261009-152723
AB=$BASE/h2/ab_results_h2b_control
PORT=44901
API="http://127.0.0.1:$PORT"
V100_IDX=0
N_DEC=1200
SO_A=$H2LIB/lib/libggml-cuda-h2.so      # H2
SO_B=$H2LIB/lib/libggml-cuda-h2b.so     # H2b
SO_A_MD5=$(md5sum "$SO_A" | cut -c1-8)
SO_B_MD5=$(md5sum "$SO_B" | cut -c1-8)
CUDA_LIBS=$BASE/tools/cuda-12.6/usr/local/cuda-12.6/lib64

mkdir -p "$AB"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
so_prod_md5(){ md5sum "$TARGET/lib/ollama/cuda_v12/libggml-cuda.so" | cut -c1-8; }

python3 - "$SRCRUN/prompt.json" "$AB/prompt1200.json" "$AB/prefill.json" "$N_DEC" <<'PY'
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
  log "restore: service=$(systemctl is-active $SERVICE) V100=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB md5=$(so_prod_md5) (attendu e4afbb4a)"
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

run_cfg(){
  local tag=$1 LIB=$2 RUN=$3
  local LIBMD5=$(md5sum "$LIB" | cut -c1-8)
  mkdir -p "$RUN"
  log "=== [$tag] lib=$LIBMD5 decode=$N_DEC ==="
  stop_and_free
  export LD_LIBRARY_PATH="$TARGET/lib/ollama:$TARGET/lib/ollama/cuda_v12:$CUDA_LIBS"
  export GGML_BACKEND_PATH="$LIB"
  export GGML_CUDA_VOLTA_Q6_W4R4=1
  export GGML_CUDA_QWEN35_MTP_SHORTLIST=$BASE/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32
  export LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1
  export CUDA_VISIBLE_DEVICES=$V100_IDX
  export LLAMA_ARG_FIT_TARGET=1024

  "$TARGET/lib/ollama/llama-server" \
    --model /usr/share/ollama/.ollama/models/blobs/sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb \
    --port $PORT --host 127.0.0.1 --no-webui \
    -c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling \
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
  log "[$tag] chaud — decode $N_DEC tok..."

  python3 - "$SRCRUN/prompt.json" "$RUN/prompt.json" "$N_DEC" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["max_tokens"] = int(sys.argv[3])
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY

  local t0 t1
  t0=$(date +%s.%N)
  curl -s --max-time 3600 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$RUN/prompt.json" > "$RUN/gen.json" 2>&1 || die "[$tag] decode rat"
  grep -q '"finish_reason"' "$RUN/gen.json" || die "[$tag] decode non terminé"
  t1=$(date +%s.%N)
  log "[$tag] decode fini (wall $(python3 -c "print(f'{$t1-$t0:.1f}')") s)"

  pkill -f "port $PORT" 2>/dev/null
  kill -9 $SRV 2>/dev/null
  for i in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && break
    sleep 1
  done
  log "[$tag] V100 liberee (${u:-?} MiB)"
}

[ "$SO_A_MD5" = "a4b6f440" ] || die "md5 lib A (H2) inattendu: $SO_A_MD5"
[ "$SO_B_MD5" = "0eb5afbd" ] || die "md5 lib B (H2b) inattendu: $SO_B_MD5"
trap restore EXIT

log "CONTROLE: A=$SO_A_MD5 (H2)  B=$SO_B_MD5 (H2b)  decode=$N_DEC tok x 2 réplicats (ordre alterné)"
run_cfg "r1-A-h2"   "$SO_A" "$AB/r1-A"
run_cfg "r1-B-h2b"  "$SO_B" "$AB/r1-B"
run_cfg "r2-B-h2b"  "$SO_B" "$AB/r2-B"
run_cfg "r2-A-h2"   "$SO_A" "$AB/r2-A"
log "=== CONTROLE termine — restauration prod ==="
restore
log "=== VERIF FINALE ==="
echo "service=$(systemctl is-active $SERVICE) md5=$(so_prod_md5)"
log "FIN h2b_control_ab.sh"
