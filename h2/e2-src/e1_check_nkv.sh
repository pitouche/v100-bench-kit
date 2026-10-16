#!/usr/bin/env bash
# e1_check_nkv.sh — correction base vs H2 sur un n_kv multiple de 16 (107344)
set -uo pipefail
BASE=${LLAMA_ROOT:-/home/guile/llama-experiments}
TARGET=$BASE/ollama-v100
SERVICE=ollama-v100
cd "$BASE/h2/cupti129-test/e2"
N=${1:-107344}
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
  echo "restore: service=$(systemctl is-active $SERVICE) V100=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) MiB md5=$(so_md5)"
}
trap restore EXIT
sudo systemctl stop "$SERVICE" 2>/dev/null || true
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  sleep 1
done
echo "V100 libre: ${u:-?} MiB ; n_kv=$N"
rm -f /tmp/nkv_base.*.bin /tmp/nkv_h2.*.bin
CUDA_VISIBLE_DEVICES=0 ./e1c_base /tmp/nkv_base 1 $N || { echo ECHEC base; exit 1; }
CUDA_VISIBLE_DEVICES=0 ./e1c_h2   /tmp/nkv_h2   1 $N || { echo ECHEC h2; exit 1; }
for t in pacc pm pl; do
  a=$(md5sum /tmp/nkv_base.$t.bin | cut -c1-12); b=$(md5sum /tmp/nkv_h2.$t.bin | cut -c1-12)
  [ "$a" = "$b" ] && echo "$t : IDENTIQUE ($a)" || echo "$t : DIFFERENT ($a vs $b)"
done
