#!/usr/bin/env bash
# E2.1 — NCU baseline du partial_kernel (harnais e1, copie du kernel). Root requis.
set -u
cd ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2
NCU=/home/guile/ncu-2024.1.1/usr/lib/nsight-compute/ncu
RAW=/tmp/e2_base_ncu.txt
restore(){
  echo "[restore] redemarrage ollama-v100..."
  sudo systemctl start ollama-v100 2>/dev/null || true
  up=0
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && { up=1; break; }
    sleep 1
  done
  sleep 8
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  echo "[restore] service=$(systemctl is-active ollama-v100) api_ok=$up V100_mem=${m} MiB md5=$(md5sum ../../ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so | cut -c1-8)"
}
trap restore EXIT
echo "=== etat avant ==="
echo "service=$(systemctl is-active ollama-v100)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
sudo systemctl stop ollama-v100 2>/dev/null
pkill -9 -f "ollama-v100/lib/ollama/llama-server" 2>/dev/null
ok=0
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && { ok=1; break; }
  sleep 2
done
echo "V100_mem=${u:-?} MiB (ok=$ok)"
[ "$ok" = 1 ] || { echo "ABORT: V100 pas libre"; exit 1; }
echo "=== NCU baseline (root, V100=CUDA dev 0) ==="
timeout 600 sudo env CUDA_VISIBLE_DEVICES=0 $NCU \
  --launch-skip 1 --launch-count 1 \
  --kernel-name 'regex:partial_kernel' \
  --section SpeedOfLight --section WarpStateStats --section SchedulerStats \
  --section LaunchStats --section Occupancy --section ComputeWorkloadAnalysis \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,gpc__cycles_elapsed.max \
  ./e1b_base 2 > "$RAW" 2>&1
echo "ncu exit=$?"
echo "=== FIN (le trap redémarre le service) ==="
