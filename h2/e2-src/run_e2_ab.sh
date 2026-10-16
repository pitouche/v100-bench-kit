#!/usr/bin/env bash
# E2.2 — A/B microbench (V100 seule, service stoppe, trap restore).
# 3 passes x (base, a3, a4), 30 iters chacun. Alternance pour lisser le drift.
set -u
cd ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2
restore(){
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
echo "service=$(systemctl is-active ollama-v100)"
sudo systemctl stop ollama-v100 2>/dev/null
pkill -9 -f "ollama-v100/lib/ollama/llama-server" 2>/dev/null
ok=0
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && { ok=1; break; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "ABORT V100 pas libre"; exit 1; }
./devcheck | grep -q V100 || { echo "ABORT dev!=V100"; exit 1; }
echo "GATE DEVICE OK : $(./devcheck)"
PASS=3; ITERS=30
echo "variant,pass,med_us,probe0,probe1" > ab_results.csv
for p in $(seq 1 $PASS); do
  for V in base a3 a4; do
    OUT=$(CUDA_VISIBLE_DEVICES=0 ./e1b_$V $ITERS 2>&1)
    US=$(echo "$OUT" | grep -oP 'iters\) : \K[0-9.]+')
    P0=$(echo "$OUT" | grep -oP 'pacc probe\[0..1\] = \K[0-9.e-]+')
    P1=$(echo "$OUT" | grep -oP 'pacc probe\[0..1\] = [0-9.e-]+ \K[0-9.e-]+')
    echo "$V,$p,$US,$P0,$P1"
    echo "$V,$p,$US,$P0,$P1" >> ab_results.csv
  done
done
python3 - <<'PY'
import statistics as S, csv
d = {}
for r in csv.DictReader(open("ab_results.csv")):
    d.setdefault(r["variant"], []).append(float(r["med_us"]))
base = d["base"]
print("\n=== RECAP A/B microbench (us/lance, median des passes) ===")
for v in ("base","a3","a4"):
    m = S.median(d[v]); mn = min(d[v]); mx = max(d[v])
    rel = f"{(m/base[1]-1)*100:+.1f} %" if v != "base" else "  ref"
    print(f"  {v:5s} : med={m:7.1f} us  (pass: {['%.0f'%x for x in d[v]]})  {rel}")
PY
echo "=== FIN (le trap redémarre le service) ==="
