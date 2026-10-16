#!/usr/bin/env bash
# h2b_build_harness.sh — Gate A (compilation isolee) des harnais H2b.
# Compile 3 variantes (prod / H2 / H2b) du checker (e1_check) et du bench (e1_bench)
# sur V100 (sm_70), commande nvcc identique, empreintes consignées.
set -uo pipefail
E2=${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2
CU=${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/cuda-12.6/usr/local/cuda-12.6
GCC=${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/gcc-13-bin
NVCC="$CU/bin/nvcc"
cd "$E2"

# copier les .cu sources dans les bons repertoires (include quote -> meme dir)
cp -f e1_check.cu  h2b/h2b_variant/e1_check.cu
cp -f e1_bench.cu  h2b/h2b_variant/e1_bench.cu

build(){ # $1=src_cu $2=out
  $NVCC -O2 -arch=sm_70 -ccbin "$GCC/g++" -o "$2" "$1" 2>"${2}.log"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "ECHEC compile $2"; tail -20 "${2}.log"; return 1; fi
  echo "OK $2  md5=$(md5sum "$2" | cut -c1-12)"
}

echo "=== Gate A : compilation isolee (sm_70, nvcc 12.6, -O2) ==="
# PROD (BC=16, pas de PC)
build ./e1_check.cu            e1c_prod
build ./e1_bench.cu            e1b_prod
# H2 (BC=8, pas de PC, partitionnement sur BC)
build h2b/e1_check.cu          e1c_h2
build h2b/e1_bench.cu          e1b_h2
# H2b (BC=8 + PC=16, partitionnement decouple sur PC)
build h2b/h2b_variant/e1_check.cu  e1c_h2b
build h2b/h2b_variant/e1_bench.cu  e1b_h2b

echo
echo "=== verification : chaque binaire rapporte sa geo (BC/PC/splits) ==="
for b in e1c_prod e1c_h2 e1c_h2b; do
  echo -n "$b: "
done
echo "(voir Gate B : le checker imprime splits par n_kv)"
echo "FIN build harness"
