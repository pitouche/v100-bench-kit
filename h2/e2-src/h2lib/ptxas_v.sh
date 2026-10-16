#!/usr/bin/env bash
# ptxas_v.sh — extraire smem/regs du partial_kernel via ptxas -v sur les 3 variantes
set -uo pipefail
CU=${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/cuda-12.6/usr/local/cuda-12.6
GCC=${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/gcc-13-bin/g++
H2LIB=${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
compile(){ # $1=fork_dir $2=tag
  local FK=$1 TAG=$2
  ( cd "$BD" && $CU/bin/nvcc -forward-unknown-to-host-compiler -ccbin=$GCC \
    -DGGML_BACKEND_BUILD -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED \
    -DGGML_CUDA_FA_BF16_BF16=1 -DGGML_CUDA_FA_F16_F16=1 \
    -DGGML_CUDA_FA_Q4_0_Q4_0=1 -DGGML_CUDA_FA_Q8_0_Q8_0=1 \
    -DGGML_CUDA_FA_QUANTS=\"q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16\" \
    -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cuda_EXPORTS \
    --options-file CMakeFiles/ggml-cuda.dir/includes_CUDA.rsp \
    -Wno-deprecated-gpu-targets -t 2 -O3 -DNDEBUG -std=c++17 \
    "--generate-code=arch=compute_70,code=[sm_70]" -Xcompiler=-fPIC -use_fast_math -extended-lambda \
    -Xcompiler "-Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-pedantic" \
    -Xptxas -v -x cu -c "$FK/ggml/src/ggml-cuda/fattn.cu" -o /tmp/ptxas_$TAG.o 2>/tmp/ptxas_$TAG.log )
  echo "######## $TAG ########"
  # le partial_kernel ggml_q8v est le kernel le plus gros en smem. On liste tous les entry + smem
  grep -E "Compiling entry function|bytes smem|registers in use" /tmp/ptxas_$TAG.log | awk '
    /Compiling entry/{sub(/.*entry function .*/,"");name=$0}
    /smem/{ if(name!=""){printf "  %s : %s\n", name, $0; name=""}}
  ' | sort -t= -k2 -rn | head -5
}
compile ${LLAMA_ROOT:-/home/guile/llama-experiments}/v100-repro/work/llama-volta-merged control
compile "$H2LIB/fork"   h2
compile "$H2LIB/fork-h2b" h2b
