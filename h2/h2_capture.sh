#!/usr/bin/env bash
# =============================================================================
# h2_capture.sh — FORAGE H2 : capture nsys d'un decode MTP long contexte
# (protocole : /home/guile/llama-experiments/h2/PROTOCOLE.md)
#
# ⚠️ STOPPE le service ollama-v100 (~15-25 min) et se restaure seul.
#    À lancer par l'utilisateur, uniquement après validation du protocole.
#    NE JAMAIS modifie le .so prod / l'unité / le GGUF.
#
# Séquence :
#   gate V100 → md5 .so → backup unité → stop service
#   → nsys launch (session h2) du llama-server DIRECT (payload+flags+env prod,
#     port 44901, CUDA_VISIBLE_DEVICES=0 = V100 dans l'ordre CUDA de cette machine)
#   → warmup 16 tok → prefill ~100k HORS capture → 2e demande identique (chaud)
#   → nsys start → decode 1800 tok (fenêtre ~60-90 s) → nsys stop
#   → export sqlite + stats csv → kill server → RESTORE service + vérifs
#
# Résultats : /home/guile/llama-experiments/h2/runs/run-<stamp>/
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
NSYS="${NSYS_BIN}"
SERVICE="${OLLAMA_SERVICE:-ollama-v100}"
MODEL_GGUF="${MODEL_GGUF}"
GGUF_SIZE=23088409504   # taille attendue (qwen3.8-27b-q6-mtp = meme blob que :vision)
FILLER="${FILLER_FILE}"
CTX_CHARS=${H2_CTX_CHARS:-550000}     # ~100k tokens (5,5 chars/tok sur le filler)
N_PRED_CAP=${H2_N_PRED:-1800}         # ~60-90 s a ~25-30 t/s long contexte
PORT=44901
API="http://127.0.0.1:$PORT"
# ORDRE CUDA != ORDRE nvidia-smi -L : ici cuda:0 = V100, cuda:1 = GTX 1070 (verifie par devmap, 9 oct).
# L'ollama scheduler prod utilise CUDA_VISIBLE_DEVICES=0 (lu dans /proc/<srv>/environ) -> parite stricte.
V100_IDX=0
STAMP=$(date -u +%Y%m%d-%H%M%S)
RES=$BASE/h2/runs/run-$STAMP
mkdir -p "$RES"
echo "$RES" > "$BASE/h2/last_run.txt"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RES/console.log"; }
so_md5(){ md5sum "$TARGET/lib/ollama/cuda_v12/libggml-cuda.so" | cut -c1-8; }
wait_api(){ for i in $(seq 1 180); do curl -s --max-time 2 "$API/v1/health" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
restore(){
  log "=== restauration service ==="
  pkill -f "port $PORT" 2>/dev/null; sleep 3
  # verifier que la V100 est libre AVANT de redemarrer (le fit prod est serre a ~58 MiB) ; sinon le service re-OOM
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
    [ -n "$u" ] && [ "$u" -lt 1000 ] && break
    pkill -9 -f "llama-server" 2>/dev/null; sleep 1
  done
  if [ -n "${u:-}" ] && [ "$u" -lt 1000 ] 2>/dev/null; then :; else
    log "ATTENTION: V100 pas libre au restore (${u:-?} MiB) - le service peut OOM (monitorer le journal)"
  fi
  sudo systemctl start "$SERVICE"
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
  log "service: $(systemctl is-active $SERVICE)"
  log "V100 used apres restore: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB (attendu ~32000 = modele recharge)"
  log "md5 .so apres restore : $(so_md5) (attendu e4afbb4a)"
  # le fit prod est serre : 5-10 s apres le start, verifier que le service n'a pas OOM (journal)
  sleep 8
  journalctl -u "$SERVICE" --since "$(date -u -d '5 minutes ago' +%Y-%m-%d\ %H:%M:%S 2>/dev/null || date +%Y-%m-%d\ %H:%M:%S)" --no-pager 2>/dev/null | grep -q 'out of memory' \
    && log "!! OOM detecte au restart — V100 etait occupee, voir journalctl -u $SERVICE" || true
}
die(){ log "ERREUR: $*"; restore; exit 1; }

# ---------------- preflight ----------------
log "=== H2 capture — dir $RES ==="
log "md5 .so de depart : $(so_md5) (attendu e4afbb4a)"
[ "$(so_md5)" = "e4afbb4a" ] || die "md5 .so prod inattendu"
"$BASE/run/v100probe" > "$RES/v100probe.out" 2>&1 || die "gate V100 : v100probe rc!=0 (driver pas pret?)"
log "gate V100 OK"
cp -a /etc/systemd/system/ollama-v100.service "$RES/ollama-v100.service.bak" 2>/dev/null || true
[ -f "$FILLER" ] || die "filler introuvable: $FILLER"
[ -f "$MODEL_GGUF" ] || die "GGUF introuvable: $MODEL_GGUF"
[ "$(stat -c%s "$MODEL_GGUF")" = "$GGUF_SIZE" ] || die "taille GGUF inattendue (attendu $GGUF_SIZE)"
"$NSYS" --version 2>&1 | head -1 | tee -a "$RES/console.log"

# prompt : premiers ~100k tokens du filler
python3 - "$FILLER" "$CTX_CHARS" "$RES/prompt.json" "$N_PRED_CAP" <<'PY'
import json, sys
filler = open(sys.argv[1], encoding="utf-8", errors="replace").read()
text = filler[: int(sys.argv[2])]
json.dump({"model": "qwen3.8-27b-q6-mtp:latest", "stream": False,
           "temperature": 0, "top_k": 1, "ignore_eos": True,
           "max_tokens": int(sys.argv[4]),
           "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "user", "content": text}]}, open(sys.argv[3], "w"), ensure_ascii=False)
PY
# variante prefill courte (32 tok)
python3 - "$RES/prompt.json" "$RES/prefill.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["max_tokens"] = 32
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False)
PY

# ---------------- stop service + lancement sous nsys ----------------
log "arret service..."
sudo systemctl stop "$SERVICE" || true

# attendre que la V100 soit reellement LIBRE (< 1 Go) avant de lancer —
# le fit prod est serre a ~58 MiB : tout residu VRAM (process orphelin) fait echouer ou degrader la capture.
log "attente liberation V100 (cuda idx $V100_IDX)..."
free_ok=0
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && { log "V100 libre: ${u} MiB (apres ${i}s)"; free_ok=1; break; }
  sleep 1
done
[ "$free_ok" = 1 ] || { log "V100 encore occupee (${u:-?} MiB)"; die "V100 pas libre apres 120 s"; }
# gate GPU visible ET libre cote CUDA (v100probe apres le stop, comme verra le serveur)
# (V100 occupee -> free=0/total=0 ; V100 libre -> free=32xxx MiB)
"$BASE/run/v100probe" 2>&1 | tee -a "$RES/v100probe.poststop.out" | grep -qE 'free=3[0-9]{4}' \
  || die "v100probe post-stop : V100 pas libre/visible en CUDA"

export LD_LIBRARY_PATH="$TARGET/lib/ollama:$TARGET/lib/ollama/cuda_v12:$BASE/tools/cuda-12.6/usr/local/cuda-12.6/lib64:$BASE/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"  # ordre = prod (lib/ollama AVANT cuda_v12)
export GGML_BACKEND_PATH="$TARGET/lib/ollama/cuda_v12/libggml-cuda.so"   # pose par l'ollama scheduler prod (pas dans l'unite) — parite stricte
export GGML_CUDA_VOLTA_Q6_W4R4=1
export GGML_CUDA_QWEN35_MTP_SHORTLIST=$BASE/src/llama-volta-merged/data/mtp-shortlists/qwen38-27b-exact-131072.i32
export LLAMA_ARG_SPEC_MTP_DEFER_PROMPT=1
export CUDA_VISIBLE_DEVICES=$V100_IDX
export LLAMA_ARG_FIT_TARGET=1024   # UNITE PROD : marge 1 Go -> le fit passe (parite prod stricte)
export GGML_CUDA_MOE_NVTX=1        # NVTX moe_metadata_wait / moe_... (gated, zero impact si absent du build prod)

log "lancement llama-server SOUS nsys launch (session h2, cuda+osrt, cuda-graph-trace=graph, -lv 4)"
"$NSYS" launch --session=h2 --trace=cuda,osrt --cuda-graph-trace=graph \
  "$TARGET/lib/ollama/llama-server" \
  --model "$MODEL_GGUF" --port $PORT --host 127.0.0.1 --no-webui \
  -c 200000 -np 1 -lv 4 --no-log-prefix --no-log-timestamps \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto -b 512 -ub 512 \
  > "$RES/server.out" 2>&1 &
SRV=$!
echo $SRV > "$RES/srv.pid"

ok=0
for i in $(seq 1 300); do
  curl -s --max-time 2 "$API/v1/models" >/dev/null 2>&1 && { ok=1; break; }
  sleep 1
done
[ "$ok" = 1 ] || { tail -25 "$RES/server.out" | tee -a "$RES/console.log"; die "server pas up sous nsys"; }
log "server up sous nsys (pid $SRV)"

# GATE FIT : le modele DOIT avoir ete fit sur le device (LOG_TRC, visible a -lv 4 comme en prod).
# ATTENTION : la gate doit ATTENDRE le fit (le port est up AVANT le chargement du modele ;
# fit prod typique : 0.5-2 s + chargement 23 Go). Le run de 13:55 l'a checkee trop tot (1 s).
FITLINE=""
for i in $(seq 1 150); do
  if grep -q 'fit_params: failed\|unable to fit' "$RES/server.out" 2>/dev/null; then
    grep -B3 -A2 'fit_params: failed\|unable to fit' "$RES/server.out" | tail -12 | tee -a "$RES/console.log"
    die "fit params ABORT (charge en cours) — voir ci-dessus"
  fi
  FITLINE=$(grep -m1 'successfully fit params' "$RES/server.out" 2>/dev/null)
  [ -n "$FITLINE" ] && break
  sleep 1
done
[ -n "$FITLINE" ] || { tail -25 "$RES/server.out" | tee -a "$RES/console.log"; die "fit params non confirme apres 150 s — server ne charge pas le modele comme en prod"; }
echo "$FITLINE" | tee -a "$RES/console.log"
log "GATE FIT OK"
# GATE GPU : le device choisi DOIT etre la V100 (protection contre un ordre CUDA inattendu).
DEVLINE=$(grep -m1 'using device CUDA' "$RES/server.out" 2>/dev/null | tail -1)
echo "${DEVLINE:-AUCUNE LIGNE 'using device'}" | tee -a "$RES/console.log"
echo "$DEVLINE" | grep -q 'Tesla V100' || die "device choisi != V100 — capture non representative"

# GATE MODEL LOADED : le port est up AVANT la fin du chargement (mmap 23 Go + offload
# CUDA : ~5 min comme en prod). Les requetes de ce stade renvoient 503 "Loading model"
# (transitoire, PAS un echec) : ne pas en tirer conclusion, attendre la marque.
log "attente 'model loaded' (mmap + offload, ~5 min en prod)"
MLOK=0; MLERR=""
for i in $(seq 1 900); do
  if grep -q 'llama_server: model loaded' "$RES/server.out" 2>/dev/null; then MLOK=1; break; fi
  if grep -q 'out of memory' "$RES/server.out" 2>/dev/null; then MLERR="OOM"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then MLERR="server mort"; break; fi
  sleep 1
done
[ "$MLOK" = 1 ] || { tail -25 "$RES/server.out" | tee -a "$RES/console.log"; die "model non charge apres 900 s (${MLERR:-timeout})"; }
grep -m1 'llama_server: model loaded' "$RES/server.out" | tee -a "$RES/console.log"
log "GATE MODEL LOADED OK (apres ${i}s)"

# warmup (charge CUDA graphs, autotune, et declenche le chargement LAZY du shortlist)
# retry court : si un 503 transitoire passe quand meme, on attend et on retente.
warm_ok=0
for w in 1 2 3 4 5; do
  curl -s --max-time 900 "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-q6-mtp:latest","stream":false,"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Say OK."}]}' \
    > "$RES/warm.json" 2>&1
  [ -s "$RES/warm.json" ] && ! grep -q '"error"' "$RES/warm.json" && { warm_ok=1; break; }
  log "warmup essai $w: reponse transitoire ($(head -c 120 "$RES/warm.json")) — retry dans 10 s"
  sleep 10
done
[ "$warm_ok" = 1 ] || { tail -5 "$RES/warm.json" | tee -a "$RES/console.log"; die "warmup: erreur server persistante"; }
log "warmup done"

# GATE PARITE : le shortlist (chargement lazy au 1er decode MTP) DOIT etre charge.
grep -m1 'CUDA MTP shortlist' "$RES/server.out" >> "$RES/console.log" 2>/dev/null \
  || { tail -15 "$RES/server.out" | tee -a "$RES/console.log"; die "shortlist non charge — capture non representative"; }
log "GATE PARITE OK : shortlist charge"

# prefill ~100k HORS capture (1er run : prefill complet ~4-6 min)
log "prefill ~100k HORS capture..."
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$RES/prefill.json" \
  > "$RES/prefill.json.out" 2>&1 || die "prefill rat"
[ -s "$RES/prefill.json.out" ] && ! grep -q '"error"' "$RES/prefill.json.out" || { tail -5 "$RES/prefill.json.out" | tee -a "$RES/console.log"; die "prefill: erreur server"; }
log "prefill done"

# 2e demande identique (checkpoint chaud, diff < ubatch) — HORS capture
log "2e demande identique (chaud) HORS capture..."
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$RES/prefill.json" \
  > "$RES/prefill2.json.out" 2>&1 || die "prefill2 rat"
[ -s "$RES/prefill2.json.out" ] && ! grep -q '"error"' "$RES/prefill2.json.out" || { tail -5 "$RES/prefill2.json.out" | tee -a "$RES/console.log"; die "prefill2: erreur server"; }
log "chaud OK — demarrage capture nsys"

"$NSYS" start --session=h2 2>>"$RES/nsys.log" || die "nsys start rat"
log "capture en cours (decode $N_PRED_CAP tok)..."
t0=$(date +%s)
curl -s --max-time 1800 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @"$RES/prompt.json" \
  > "$RES/gen.json" 2>&1 || log "gen rat"
[ -s "$RES/gen.json" ] && ! grep -q '"error"' "$RES/gen.json" || { tail -5 "$RES/gen.json" | tee -a "$RES/console.log"; die "generation: erreur server"; }
grep -q '"finish_reason"' "$RES/gen.json" || die "generation non terminee (pas de finish_reason)"
t1=$(date +%s)
log "decode fini (wall $((t1-t0)) s)"
sleep 4
# NB: 'nsys stop' ne prend PAS --output (seul 'start' l'accepte). Le nom du rapport
# est fixe a 'launch' ; par defaut il va dans ~ -> report1.nsys-rep (d'ou le rattrapage ci-dessous).
"$NSYS" stop --session=h2 2>>"$RES/nsys.log" || die "nsys stop rat"
# rattrapage du rapport si nsys l'a mis au defaut (~)
if [ ! -f "$RES/prof.nsys-rep" ]; then
  for cand in "$HOME"/report*.nsys-rep /tmp/report*.nsys-rep; do
    [ -f "$cand" ] && mv -f "$cand" "$RES/prof.nsys-rep" && log "rapport recupere depuis $cand" && break
  done
fi
[ -f "$RES/prof.nsys-rep" ] || { tail -10 "$RES/nsys.log" | tee -a "$RES/console.log"; die "prof.nsys-rep introuvable apres stop"; }
log "capture terminee: $(ls -la "$RES/prof.nsys-rep" | awk '{print $5}') octets"

pkill -f "port $PORT" 2>/dev/null
wait $SRV 2>/dev/null
# le wrapper nsys/llama-server peut survivre : kill precis par pid, V100 liberee de facon verifiee
for i in $(seq 1 20); do
  pkill -9 -f "llama-server.*port $PORT" 2>/dev/null; [ -n "$SRV" ] && kill -9 $SRV 2>/dev/null
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  sleep 1
done
sleep 2

# ---------------- export + stats ----------------
log "export sqlite..."
"$NSYS" export --type=sqlite -f --output="$RES/prof.sqlite" "$RES/prof.nsys-rep" >> "$RES/nsys.log" 2>&1 \
  && log "export sqlite OK" || log "ATTENTION: export sqlite rat (les stats csv peuvent suffire)"
log "stats csv..."
"$NSYS" stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum --format csv \
  --output "$RES/stats" "$RES/prof.nsys-rep" > "$RES/stats.out" 2>&1 \
  && log "stats OK" || { log "stats rat"; tail -10 "$RES/nsys.log" | tee -a "$RES/console.log"; }
ls -la "$RES" | grep -E 'nsys-rep|sqlite|stats' | tee -a "$RES/console.log"

restore
log "fin — analyser avec: python3 $BASE/h2/h2_analyze.py $RES"
