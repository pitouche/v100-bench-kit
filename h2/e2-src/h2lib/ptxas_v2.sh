#!/usr/bin/env bash
# ptxas_v2.sh — smem/regs du partial_kernel via ptxas -v, commande EXACTE du build prod.
set -uo pipefail
H2LIB=${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib
BD="$H2LIB/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
# la ligne nvcc exacte du build H2 (tous les -D), on ne change que la source et on ajoute -Xptxas -v
NVCCBASE=$(grep -E "bin/nvcc" ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/compile_h2.sh | head -1)
compile(){ # $1=fork_dir $2=tag
  local FK=$1 TAG=$2
  local cmd
  cmd=$(echo "$NVCCBASE" | sed "s|$H2LIB/fork/ggml/src/ggml-cuda/fattn.cu|$FK/ggml/src/ggml-cuda/fattn.cu|; s|-o [^ ]*|-o /tmp/ptxas_$TAG.o|; s|-t 2|-t 2 -Xptxas -v|")
  ( cd "$BD" && eval "$cmd" 2>/tmp/ptxas_$TAG.log )
  echo "######## $TAG ########"
  awk '
    /Compiling entry function/{name=$0; sub(/.*entry function /,"",name); sub(/ for .*/,"",name); n=++c}
    /bytes smem/{sm[c]=$0; nm[c]=name}
    END{ for(i=1;i<=n;i++) if(sm[i]!="") print "  " sm[i] "  <=  " nm[i] }
  ' /tmp/ptxas_$TAG.log | sort -t'=' -k2 -rn | head -4
}
compile ${LLAMA_ROOT:-/home/guile/llama-experiments}/v100-repro/work/llama-volta-merged control
compile "$H2LIB/fork"   h2
compile "$H2LIB/fork-h2b" h2b
echo "FIN ptxas_v2"
