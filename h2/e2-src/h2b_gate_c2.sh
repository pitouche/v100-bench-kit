#!/usr/bin/env bash
# h2b_gate_c2.sh — Gate C (confirm) : microbenchmark 5 passes x 30 iters, mediane.
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
echo "V100 libre: ${u:-?} MiB"; ./devcheck
run_variant(){
  local BIN=$1 TAG=$2; local vals=""
  for p in 1 2 3 4 5; do
    L=$(CUDA_VISIBLE_DEVICES=0 $BIN 30 2>/dev/null | grep -oE "durée/lance.*: +[0-9]+ us" | grep -oE "[0-9]+ us" | awk '{print $1}')
    vals="$vals $L"; sleep 1
  done
  echo -n "$TAG:$vals med="
  echo $vals | tr ' ' '\n' | sort -n | awk 'NR==3{m=$1} END{print m"us"}'
}
echo "=== Gate C (confirm) : 5 passes x 30 iters, mediane ==="
run_variant ./e1b_prod PROD
run_variant ./e1b_h2   H2
run_variant ./e1b_h2b  H2B
