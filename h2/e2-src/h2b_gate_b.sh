#!/usr/bin/env bash
# h2b_gate_b.sh — Gate B : gate numerique prod / H2 / H2b.
# 4 n_kv : 107344 (mult 16, <seuil), 107351 (cas reel, <seuil),
#          107521 (>seuil, H2 diverge), 108551 (>seuil).
# Comparaison pacc/pm/pl (md5) entre les 3 variantes.
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

# pre-dump: les 3 variantes sur les 4 n_kv
for N in 107344 107351 107521 108551; do
  CUDA_VISIBLE_DEVICES=0 ./e1c_prod /tmp/gb_prod_$N 1 $N 2>&1 | grep -E "check|dump OK|error" | sed "s/^/  [prod $N] /"
  CUDA_VISIBLE_DEVICES=0 ./e1c_h2   /tmp/gb_h2_$N   1 $N 2>&1 | grep -E "check|dump OK|error" | sed "s/^/  [h2   $N] /"
  CUDA_VISIBLE_DEVICES=0 ./e1c_h2b  /tmp/gb_h2b_$N  1 $N 2>&1 | grep -E "check|dump OK|error" | sed "s/^/  [h2b  $N] /"
done

echo
echo "=== Gate B : comparaison pacc/pm/pl (md5) ==="
echo "ATTENDU: H2b == PROD partout ; H2 == PROD sous le seuil, != PROD au-dessus."
for N in 107344 107351 107521 108551; do
  echo "--- n_kv=$N ---"
  for t in pacc pm pl; do
    a=$(md5sum /tmp/gb_prod_$N.$t.bin | cut -c1-12)
    b=$(md5sum /tmp/gb_h2_$N.$t.bin   | cut -c1-12)
    c=$(md5sum /tmp/gb_h2b_$N.$t.bin  | cut -c1-12)
    h2tag="DIFFERENT"; [ "$a" = "$b" ] && h2tag="identique"
    h2btag="DIFFERENT"; [ "$a" = "$c" ] && h2btag="identique"
    echo "  $t : prod=$a h2=$b h2b=$c  [H2=$h2tag | H2b=$h2btag]"
  done
done
