#!/usr/bin/env bash
# e1_check_final.sh — verification definitive de correction H2 (base BC=16 vs H2 BC=8)
#   - n_kv=107344 (multiple de 16) : attendu BIT-IDENTIQUE (aucune tuile partiel)
#   - n_kv=107351 (cas reel decode, impair) : attendu divergence en queue
set -uo pipefail
BASE=${LLAMA_ROOT:-/home/guile/llama-experiments}
TARGET=$BASE/ollama-v100
SERVICE=ollama-v100
cd "$BASE/h2/cupti129-test/e2"
so_md5(){ md5sum "$TARGET/lib/ollama/cuda_v12/libggml-cuda.so" | cut -c1-8; }
restore(){
  echo "=== restauration service ==="
  pkill -9 -f "llama-server" 2>/dev/null; sleep 2
  sudo systemctl start "$SERVICE" || true
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
  sleep 6
  echo "restore: service=$(systemctl is-active $SERVICE) V100=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB md5=$(so_md5) (attendu e4afbb4a)"
}
trap restore EXIT
sudo systemctl stop "$SERVICE" 2>/dev/null || true
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  sleep 1
done
echo "V100 libre: ${u:-?} MiB"
./devcheck
run_pair(){
  local N=$1 TAG=$2
  rm -f /tmp/${TAG}_base.*.bin /tmp/${TAG}_h2.*.bin
  CUDA_VISIBLE_DEVICES=0 ./e1c_base /tmp/${TAG}_base 1 $N >/dev/null || { echo "ECHEC base n_kv=$N"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 ./e1c_h2b  /tmp/${TAG}_h2   1 $N >/dev/null || { echo "ECHEC h2   n_kv=$N"; exit 1; }
  echo "--- n_kv=$N ---"
  for t in pacc pm pl; do
    a=$(md5sum /tmp/${TAG}_base.$t.bin | cut -c1-12); b=$(md5sum /tmp/${TAG}_h2.$t.bin | cut -c1-12)
    [ "$a" = "$b" ] && echo "  $t : BIT-IDENTIQUE ($a)" || echo "  $t : DIFFERENT ($a vs $b)"
  done
}
run_pair 107344 align   # multiple de 16
run_pair 107351 odd     # cas reel decode
