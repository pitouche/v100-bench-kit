#!/usr/bin/env bash
# =============================================================================
# p7_ab.sh — Expérience applicative A/B : kernel Q6_K P7 (X4) vs PROD (W4R4)
#
# Principe (decide l'utilite REELLE de l'optimisation, U5) :
#   A = .so PROD e4afbb4a (W4R4 Q6_K)      -> baseline
#   B = .so P7   a1b3b776 (X4  Q6_K)       -> candidate
# Seul le .so change (swap_revert.sh), meme service / GGUF / env / modeles.
# Charge comparable : 3 taches fixes, temperature 0, MTP actif.
# Metriques : eval t/s, latency/cycle (ms per token), acceptance, mean len,
#            taux par position, wall, VRAM, stabilite (aucun crash/Xid).
# Rollback garanti : revert vers known-good a la fin (meme en cas d'erreur).
# =============================================================================
set -uo pipefail

AB=/home/guile/llama-experiments/p7/bench
SWAP=/home/guile/llama-experiments/autopsy-20261005/q8-qpn/swap_revert.sh
P7SO=/home/guile/llama-experiments/p7/work/build-p7/llama-server-cuda_v12/bin/libggml-cuda.so
SERVICE=ollama-v100
PORT=11434
API="http://127.0.0.1:$PORT"
MODEL="qwen3.8-27b-q6-mtp:latest"
V100_IDX=1

STAMP=$(date -u +%Y%m%d-%H%M%S)
RES=$AB/res-$STAMP
mkdir -p "$RES"
echo "RES=$RES" > "$AB/last_res.txt"

log(){ echo "[$(date '+%H:%M:%S')] $*"; echo "[$(date '+%H:%M:%S')] $*" >> "$RES/console.log"; }
die(){ echo "ERREUR: $*" | tee -a "$RES/console.log"; sudo bash "$SWAP" revert >> "$RES/console.log" 2>&1; exit 1; }

# taches fixes (T=0, reproductible)
cat > "$RES/task1.json" <<'EOF'
{"model":"qwen3.8-27b-q6-mtp:latest","stream":false,"think":false,
 "options":{"num_predict":256,"temperature":0},
 "messages":[{"role":"user","content":"Write a concise history of lighthouses in 4 short paragraphs."}]}
EOF
cat > "$RES/task2.json" <<'EOF'
{"model":"qwen3.8-27b-q6-mtp:latest","stream":false,"think":false,
 "options":{"num_predict":512,"temperature":0},
 "messages":[{"role":"user","content":"Write a Python function that merges two sorted lists into one sorted list. Include a docstring and unit tests."}]}
EOF
cat > "$RES/task3.json" <<'EOF'
{"model":"qwen3.8-27b-q6-mtp:latest","stream":false,"think":false,
 "options":{"num_predict":256,"temperature":0},
 "messages":[{"role":"user","content":"Solve step by step: a train leaves Paris at 09:00 at 120 km/h toward Lyon. Another leaves Lyon at 09:40 at 150 km/h toward Paris. Paris-Lyon is 465 km. When and where do they meet?"}]}
EOF

so_md5(){ md5sum /home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so 2>/dev/null | cut -c1-8; }
wait_api(){ for i in $(seq 1 90); do curl -s --max-time 2 "$API/api/tags" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
vram_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $V100_IDX 2>/dev/null | tr -d ' '; }

warmup(){ curl -s --max-time 900 "$API/api/chat" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":16,\"temperature\":0},\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}]}" \
  > "$RES/warmup-$1.json" 2>&1 || die "warmup $1"; log "warmup $1 OK"; }

run_task(){ # phase label taskfile
  local phase=$1 label=$2 tf=$3 t0 t1 resp wall
  t0=$(date '+%F %T'); wall=$(date +%s.%N)
  resp=$(curl -s --max-time 900 "$API/api/chat" -H 'Content-Type: application/json' -d @"$tf")
  wall=$(python3 -c "print(f'{$wall*0+0:.2f}')" 2>/dev/null); local wall1; wall1=$(date +%s.%N)
  local wall_s; wall_s=$(python3 -c "print(f'{$wall1 - $(echo "$wall"|cut -d. -f1):.2f}')" 2>/dev/null)
  sleep 3
  t1=$(date '+%F %T')
  echo "$resp" > "$RES/${phase}-${label}.json"
  local parsed; parsed=$(python3 - "$RES/${phase}-${label}.json" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("NA NA NA NA"); raise SystemExit
et=d.get("eval_count",0); em=d.get("eval_duration",0)/1e6
pt=d.get("prompt_eval_count",0); pm=d.get("prompt_eval_duration",0)/1e6
print(f"{pt:.0f} {pm:.0f} {et} {em:.0f} {et/(em/1e3) if em else 0:.2f}")
PY
)
  set -- $parsed
  local p_tok=$1 p_ms=$2 e_tok=$3 e_ms=$4 tps=$5
  local acc mean rates
  acc=$(journalctl -u "$SERVICE" --no-pager --since "$t0" --until "$t1" 2>/dev/null | grep 'draft acceptance' | tail -1 | sed -nE 's/.*draft acceptance = ([0-9.]+).*/\1/p')
  mean=$(journalctl -u "$SERVICE" --no-pager --since "$t0" --until "$t1" 2>/dev/null | grep 'draft acceptance' | tail -1 | sed -nE 's/.*mean len = ([0-9.]+).*/\1/p')
  rates=$(journalctl -u "$SERVICE" --no-pager --since "$t0" --until "$t1" 2>/dev/null | grep 'acc per pos' | tail -1 | sed -nE 's/.*acc per pos = (\([^)]*\)).*/\1/p')
  acc=${acc:-NA}; mean=${mean:-NA}; rates=${rates:-NA}
  local wall_tot; wall_tot=$(python3 -c "print(f'{$wall1 - ${wall%%.*}+${wall#*.}00000000/1e9:.2f}')" 2>/dev/null || echo NA)
  echo "$label $p_tok $p_ms $e_tok $e_ms $tps $acc $mean $rates" >> "$RES/$phase.tsv"
  printf "  %-12s eval %s tok %s ms (%s t/s) | acc=%s mean=%s rates=%s\n" \
    "$label" "$e_tok" "$e_ms" "$tps" "$acc" "$mean" "$rates"
}

# resume d'une phase : tok total_ms tps acc mean (NA tolere dans acc/mean)
summary(){ # tsv
  python3 - "$1" <<'PY'
import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
te=sum(int(r[3]) for r in rows); tm=sum(float(r[4]) for r in rows)
tps=f"{te/(tm/1e3):.2f}" if tm else "0.00"
accs=[float(r[6]) for r in rows if r[6]!="NA"]
means=[float(r[7]) for r in rows if r[7]!="NA"]
acc=f"{sum(accs)/len(accs):.4f}" if accs else "NA"
mean=f"{sum(means)/len(means):.2f}" if means else "NA"
print(f"{te} {tm:.0f} {tps} {acc} {mean}")
PY
}

log "=== P7 A/B applicative — STAMP $STAMP ==="
log "so de depart : $(so_md5)"
systemctl is-active "$SERVICE" >/dev/null || die "service inactif"
wait_api || die "API pas up"

# ---------------- PHASE A : baseline (prod .so e4afbb4a) ----------------
log "=== PHASE A — baseline (prod $(so_md5)) ==="
warmup A
run_task A T1 "$RES/task1.json"; sleep 2
run_task A T2 "$RES/task2.json"; sleep 2
run_task A T3 "$RES/task3.json"
SUM_A=$(summary "$RES/A.tsv"); log "Phase A resume: $SUM_A  (VRAM $(vram_used) MiB)"
VRAM_A=$(vram_used)

# ---------------- swap vers P7 ----------------
log "=== SWAP vers .so P7 (a1b3b776) ==="
sudo bash "$SWAP" swap "$P7SO" >> "$RES/console.log" 2>&1 || die "swap P7"
log "so apres swap : $(so_md5)"
wait_api || die "API pas up apres swap"

# ---------------- PHASE B : candidate (P7 .so) ----------------
log "=== PHASE B — P7 X4 ($(so_md5)) ==="
warmup B
run_task B T1 "$RES/task1.json"; sleep 2
run_task B T2 "$RES/task2.json"; sleep 2
run_task B T3 "$RES/task3.json"
SUM_B=$(summary "$RES/B.tsv"); log "Phase B resume: $SUM_B  (VRAM $(vram_used) MiB)"
VRAM_B=$(vram_used)

# ---------------- REVERT (garanti) ----------------
log "=== REVERT vers known-good ==="
sudo bash "$SWAP" revert >> "$RES/console.log" 2>&1
wait_api || true
log "so final : $(so_md5)"

# ---------------- COMPARAISON ----------------
read -r TOKA MS_A TPS_A ACC_A MEAN_A <<< "$SUM_A"
read -r TOKB MS_B TPS_B ACC_B MEAN_B <<< "$SUM_B"
{
echo "================================================================"
echo " RESULTAT A/B — kernel Q6_K P7 (X4) vs PROD (W4R4) — applicatif"
echo "================================================================"
echo "  A = prod e4afbb4a (W4R4)   VRAM ${VRAM_A} MiB"
echo "  B = P7   a1b3b776 (X4)     VRAM ${VRAM_B} MiB"
echo "  taches : T1=256 / T2=512 / T3=256 tok, T=0, MTP actif"
echo "----------------------------------------------------------------"
printf "  %-22s %12s %12s   delta\n" "metrique" "A (prod)" "B (P7)"
printf "  %-22s %12s %12s\n" "eval tok total" "$TOKA" "$TOKB"
printf "  %-22s %12s %12s\n" "eval ms total" "$MS_A" "$MS_B"
python3 - "$TPS_A" "$TPS_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'eval t/s (global)':<22} {a:12.2f} {b:12.2f}   {((b-a)/a*100):+.1f}%")
PY
if [ "$ACC_A" != NA ] && [ "$ACC_B" != NA ]; then
  python3 - "$ACC_A" "$ACC_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'acceptance moyenne':<22} {a:12.4f} {b:12.4f}   {((b-a)/a*100):+.1f}%")
PY
else echo "  acceptance : NA (A=$ACC_A B=$ACC_B)"; fi
if [ "$MEAN_A" != NA ] && [ "$MEAN_B" != NA ]; then
  python3 - "$MEAN_A" "$MEAN_B" <<'PY'
import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f"  {'mean len moyenne':<22} {a:12.2f} {b:12.2f}   {((b-a)/a*100):+.1f}%")
PY
else echo "  mean len : NA"; fi
echo "================================================================"
echo " TSV detail : $RES/A.tsv  $RES/B.tsv"
echo "================================================================"
} | tee -a "$RES/console.log"
log "fin — service restaure (prod $(so_md5))"
exit 0
