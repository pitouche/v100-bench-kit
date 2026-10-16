#!/usr/bin/env bash
# e1_check_ab.sh — verification de correction numerique base vs H2.
# Arrete ollama-v100 (trap de restauration garanti), execute les 2 checkers
# sur la V100 seule (CUDA dev 0), puis compare les sorties.
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

echo "=== stop prod ==="
sudo systemctl stop "$SERVICE" 2>/dev/null || true
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | head -1)
  [ -n "$u" ] && [ "$u" -lt 1000 ] && break
  sleep 1
done
echo "V100 libre: ${u:-?} MiB"
# identite GPU (CUDA dev 0 = V100 sur cette machine)
./devcheck

rm -f /tmp/chk_base.*.bin /tmp/chk_h2.*.bin
echo "=== dump base ==="
CUDA_VISIBLE_DEVICES=0 ./e1c_base /tmp/chk_base 1 || { echo "ECHEC base"; exit 1; }
echo "=== dump H2 ==="
CUDA_VISIBLE_DEVICES=0 ./e1c_h2   /tmp/chk_h2   1 || { echo "ECHEC H2"; exit 1; }

echo "=== comparaison (donnees identiques seed 12345, meme geometrie) ==="
ok=1
for t in pacc pm pl; do
  md5a=$(md5sum /tmp/chk_base.$t.bin | cut -c1-12)
  md5b=$(md5sum /tmp/chk_h2.$t.bin   | cut -c1-12)
  if [ "$md5a" = "$md5b" ]; then echo "  $t : IDENTIQUE ($md5a)"; else
    echo "  $t : DIFFERENT ($md5a vs $md5b)"; ok=0
  fi
done
# tolerance bf16 (si un ULP d'ecart de fma change un bit bf16)
python3 - <<'PY'
import struct, math
for t, fmt, n in [("pacc", 'e', 256*24*160*4*8), ("pm", 'f', 24*4*160*4), ("pl", 'f', 24*4*160*4)]:
    a = open(f"/tmp/chk_base.{t}.bin","rb").read()
    b = open(f"/tmp/chk_h2.{t}.bin","rb").read()
    if len(a) != len(b):
        print(f"  {t}: TAILLE DIFF ({len(a)} vs {len(b)})"); continue
    n = len(a)//2
    xa = struct.unpack(f"<{n}e", a) if fmt=='e' else struct.unpack(f"<{n}f", a)
    xb = struct.unpack(f"<{n}e", b) if fmt=='e' else struct.unpack(f"<{n}f", b)
    diff = max_diff = 0
    for i in range(n):
        if xa[i] != xb[i]:
            diff += 1
            try:
                rel = abs(xa[i]-xb[i])/max(1e-12, abs(xa[i]))
                max_diff = max(max_diff, rel)
            except: max_diff = 1.0
    print(f"  {t}: {diff}/{n} elements differents (max_rel={max_diff:.2e})" + ("" if diff==0 else "   <-- verifier si > tolerance"))
PY
[ "$ok" = 1 ] && echo "VERDICT CHECK : SORTIES BIT-IDENTIQUES" || echo "VERDICT CHECK : ECARTS (voir tolerance ci-dessus)"
