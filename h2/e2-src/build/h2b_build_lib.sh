#!/usr/bin/env bash
# h2b_build_lib.sh — Lib H2b ISOLEE : 144 .o PROD + 1 fattn.cu.o recompile (H2b).
# Reutilise le build PROPRE A (build-clean, fork intact) pour les .o de prod,
# recompile fattn.cu depuis fork-h2b (patch H2b), re-link.
set -uo pipefail
BASE=${LLAMA_ROOT:-/home/guile/llama-experiments}
E2=$BASE/h2/cupti129-test/e2
H2LIB=$E2/h2lib
TOOLS=$BASE/tools
NVCC=$TOOLS/cuda-12.6/usr/local/cuda-12.6/bin/nvcc
GCC=$TOOLS/gcc-13-bin/g++
PROD_SO=$BASE/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
cd "$H2LIB"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

PROD_MD5=$(md5sum "$PROD_SO" | cut -c1-8)
log "prod md5=$PROD_MD5 (attendu e4afbb4a)"

# 1) recompiler fattn.cu depuis fork-h2b avec la commande nvcc EXACTE du build prod
#    (memes flags, memes includes du build-clean)
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
log "compile fattn.h2b.o depuis fork-h2b (commande nvcc EXACTE du build prod)"
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
( cd "$BD" && \
  ${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/cuda-12.6/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler -ccbin=${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/gcc-13-bin/g++ -DGGML_BACKEND_BUILD -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED -DGGML_CUDA_FA_BF16_BF16=1 -DGGML_CUDA_FA_BF16_F16=0 -DGGML_CUDA_FA_BF16_Q4_0=0 -DGGML_CUDA_FA_BF16_Q4_1=0 -DGGML_CUDA_FA_BF16_Q5_0=0 -DGGML_CUDA_FA_BF16_Q5_1=0 -DGGML_CUDA_FA_BF16_Q8_0=0 -DGGML_CUDA_FA_F16_BF16=0 -DGGML_CUDA_FA_F16_F16=1 -DGGML_CUDA_FA_F16_Q4_0=0 -DGGML_CUDA_FA_F16_Q4_1=0 -DGGML_CUDA_FA_F16_Q5_0=0 -DGGML_CUDA_FA_F16_Q5_1=0 -DGGML_CUDA_FA_F16_Q8_0=0 -DGGML_CUDA_FA_Q4_0_BF16=0 -DGGML_CUDA_FA_Q4_0_F16=0 -DGGML_CUDA_FA_Q4_0_Q4_0=1 -DGGML_CUDA_FA_Q4_0_Q4_1=0 -DGGML_CUDA_FA_Q4_0_Q5_0=0 -DGGML_CUDA_FA_Q4_0_Q5_1=0 -DGGML_CUDA_FA_Q4_0_Q8_0=0 -DGGML_CUDA_FA_Q4_1_BF16=0 -DGGML_CUDA_FA_Q4_1_F16=0 -DGGML_CUDA_FA_Q4_1_Q4_0=0 -DGGML_CUDA_FA_Q4_1_Q4_1=0 -DGGML_CUDA_FA_Q4_1_Q5_0=0 -DGGML_CUDA_FA_Q4_1_Q5_1=0 -DGGML_CUDA_FA_Q4_1_Q8_0=0 -DGGML_CUDA_FA_Q5_0_BF16=0 -DGGML_CUDA_FA_Q5_0_F16=0 -DGGML_CUDA_FA_Q5_0_Q4_0=0 -DGGML_CUDA_FA_Q5_0_Q4_1=0 -DGGML_CUDA_FA_Q5_0_Q5_0=0 -DGGML_CUDA_FA_Q5_0_Q5_1=0 -DGGML_CUDA_FA_Q5_0_Q8_0=0 -DGGML_CUDA_FA_Q5_1_BF16=0 -DGGML_CUDA_FA_Q5_1_F16=0 -DGGML_CUDA_FA_Q5_1_Q4_0=0 -DGGML_CUDA_FA_Q5_1_Q4_1=0 -DGGML_CUDA_FA_Q5_1_Q5_0=0 -DGGML_CUDA_FA_Q5_1_Q5_1=0 -DGGML_CUDA_FA_Q5_1_Q8_0=0 -DGGML_CUDA_FA_Q8_0_BF16=0 -DGGML_CUDA_FA_Q8_0_F16=0 -DGGML_CUDA_FA_Q8_0_Q4_0=0 -DGGML_CUDA_FA_Q8_0_Q4_1=0 -DGGML_CUDA_FA_Q8_0_Q5_0=0 -DGGML_CUDA_FA_Q8_0_Q5_1=0 -DGGML_CUDA_FA_Q8_0_Q8_0=1 -DGGML_CUDA_FA_QUANTS=\"q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16\" -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cuda_EXPORTS --options-file CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp -Wno-deprecated-gpu-targets -t 2 -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_70,code=[sm_70]" -Xcompiler=-fPIC -use_fast_math -extended-lambda -Xcompiler "-Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-pedantic" -x cu -c ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/fork-h2b/ggml/src/ggml-cuda/fattn.cu -o ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/fattn.h2b.o \
  > "$H2LIB/compile_h2b.log" 2>&1 ) || { log "ECHEC compile fattn.h2b.o"; tail -30 "$H2LIB/compile_h2b.log"; exit 1; }
log "fattn.h2b.o md5=$(md5sum "$H2LIB/fattn.h2b.o" | cut -c1-12)"

# 2) link : memes .o de prod (objs_prod_h2.rsp remplace fattn.h2.o -> fattn.h2b.o)
log "link lib H2b (144 .o prod + fattn.h2b.o)"
cd "$H2LIB/prod-obj"
# construire un rsp H2b : remplacer le chemin fattn.h2.o par fattn.h2b.o
sed "s|$H2LIB/fattn.h2.o|$H2LIB/fattn.h2b.o|" "$H2LIB/objs_prod_h2.rsp" > "$H2LIB/objs_h2b.rsp"
$GCC -fPIC -shared -o "$H2LIB/lib/libggml-cuda-h2b.so" @"$H2LIB/objs_h2b.rsp" @CMakeFiles/ggml-cuda.dir/linkLibs.rsp \
  -L"$TOOLS/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs" \
  -L"$TOOLS/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib" \
  > "$H2LIB/link_h2b.log" 2>&1 || { log "ECHEC link H2b"; tail -30 "$H2LIB/link_h2b.log"; exit 1; }
B_MD5=$(md5sum "$H2LIB/lib/libggml-cuda-h2b.so" | cut -c1-8)
log "lib H2b md5=$B_MD5"

# 3) gates
[ "$B_MD5" != "$PROD_MD5" ] && log "OK: H2b != prod" || { log "H2b == prod (erreur)"; exit 1; }
H2MD5=$(md5sum "$H2LIB/lib/libggml-cuda-h2.so" | cut -c1-8)
[ "$B_MD5" != "$H2MD5" ] && log "OK: H2b != H2 ($H2MD5)" || { log "H2b == H2 (erreur)"; exit 1; }
M=$(strings "$H2LIB/lib/libggml-cuda-h2b.so" | grep -c "GGML_CUDA_VOLTA_Q6_W4R4")
[ "$M" -ge 1 ] && log "OK: marqueur GGML_CUDA_VOLTA_Q6_W4R4 present ($M)" || { log "marqueurs absents"; exit 1; }
log "prod intact: $(md5sum "$PROD_SO" | cut -c1-8) (attendu $PROD_MD5)"
{
  echo "lib prod (fork intact)   : md5=$PROD_MD5"
  echo "lib H2  (BC 16->8)       : md5=$H2MD5"
  echo "lib H2b (BC=8 + PC=16)   : md5=$B_MD5"
  echo "fattn.h2b.o              : md5=$(md5sum "$H2LIB/fattn.h2b.o" | cut -c1-12)"
  echo "marqueur W4R4 present    : $M"
  echo "prod                     : intact"
} | tee "$H2LIB/rapport-build-h2b.txt"
log "FIN build H2b"
