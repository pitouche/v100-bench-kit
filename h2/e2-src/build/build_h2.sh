#!/usr/bin/env bash
# =============================================================================
# build_h2.sh — Lib H2 ISOLEE (BC 16->8) — v2 (deux builds propres, aucune
# chirurgie .o).
#
# A) build PROPRE A (gate recette) : configure + gmake du sub-build CUDA avec
#    la commande EXACTE du prod (cfgcmd du build v100-repro) sur le fork
#    v100-repro NON modifie -> la lib DOIT etre bit-identique au prod
#    (md5 e4afbb4a). Si gate A echoue : diagnostic, on n'avance pas.
# B) build PROPRE B (variante H2) : MEME recette, fork = COPIE patchee
#    (2 lignes : BC=8, sub<BC/8) -> lib H2.
# C) gates : md5(H2) != md5(prod) et != md5(A) ; marqueurs Volta presents ;
#    prod et fork v100-repro intacts (md5 controles avant/apres).
#
# AUCUN fichier en dehors de h2lib/ n'est ecrit.
# =============================================================================
set -uo pipefail
BASE=${LLAMA_ROOT:-/home/guile/llama-experiments}
E2=$BASE/h2/cupti129-test/e2
H2LIB=$E2/h2lib
SRC=$BASE/v100-repro/work/ollama-v100-src/llama/server
FORK=$BASE/v100-repro/work/llama-volta-merged
TOOLS=$BASE/tools
CMAKE=$TOOLS/cmake/bin/cmake
PROD_SO=$BASE/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so
PROD_MD5=$(md5sum "$PROD_SO" | cut -c1-8)

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
die(){ log "ERREUR BLOQUANTE : $*"; exit 1; }

# env du build de reference (herite par le configure nvcc)
export CUDACXX="$TOOLS/cuda-12.6/usr/local/cuda-12.6/bin/nvcc"
export CUDAHOSTCXX="$TOOLS/gcc-13-bin/g++"

# configure + build d'un sub-build propre
# $1 = binary dir, $2 = prefix install, $3 = fork, $4 = nom log
do_build(){
  local BD=$1 IP=$2 FK=$3 TAG=$4
  mkdir -p "$BD" "$H2LIB/lib"
  ( cd "$BD" && "$CMAKE" -S "$SRC" --preset llama_cuda_v12_user_arch -B . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$IP" \
      -DOLLAMA_LIB_DIR:STRING=lib/ollama \
      -DOLLAMA_RUNNER_DIR=cuda_v12 \
      -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP="$FK" \
      -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON \
      -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=70-real > "$H2LIB/configure-$TAG.log" 2>&1 ) \
    || { tail -30 "$H2LIB/configure-$TAG.log"; die "configure $TAG"; }
  ( cd "$BD" && gmake -j8 ggml-cuda > "$H2LIB/build-$TAG.log" 2>&1 ) \
    || { tail -40 "$H2LIB/build-$TAG.log"; die "gmake $TAG"; }
}

log "prod md5=$PROD_MD5 (attendu e4afbb4a)"
FORK_MD5_BEFORE=$(md5sum "$FORK/ggml/src/ggml-cuda/fattn-q8-volta.cuh" | cut -c1-8)

# ---------------- [A] build propre A (gate recette) --------------------------
log "=== A: build propre (fork v100-repro intact) — gate recette ==="
rm -rf "$H2LIB/build-clean"
do_build "$H2LIB/build-clean" "$H2LIB/build-clean" "$FORK" clean
cp -a "$H2LIB/build-clean/bin/libggml-cuda.so" "$H2LIB/lib/libggml-cuda-base.so"
A_MD5=$(md5sum "$H2LIB/lib/libggml-cuda-base.so" | cut -c1-8)
log "build propre A : md5=$A_MD5 (prod=$PROD_MD5)"
[ "$A_MD5" = "$PROD_MD5" ] || { log "GATE A ECHOUEE (md5 propre != prod)"; die "gate A"; }
log "GATE A PASSE : recette reproduite bit-identique au prod"

# ---------------- [B] build propre B (variante H2) ---------------------------
log "=== B: build propre (fork COPIE patchee H2) ==="
rm -rf "$H2LIB/fork" "$H2LIB/build-h2"
cp -a "$FORK" "$H2LIB/fork"
F="$H2LIB/fork/ggml/src/ggml-cuda/fattn-q8-volta.cuh"
cp "$F" "$F.orig"
sed -i 's/constexpr int BC = 16;/constexpr int BC = 8; \/\/ H2: tuile key 16->8/' "$F"
sed -i 's/for (int sub=0;sub<2;sub++) {/for (int sub=0;sub<BC\/8;sub++) { \/\/ H2: BC\/8 subtiles/' "$F"
NDIFF=$(diff "$F.orig" "$F" | grep -c '^[<>]')
[ "$NDIFF" = 4 ] || die "patch H2 : attendu 4 lignes, vu $NDIFF"
log "patch H2 sur la COPIE : $(diff "$F.orig" "$F" | grep '^[<>]' | tr '\n' ' | ')"
do_build "$H2LIB/build-h2" "$H2LIB/build-h2" "$H2LIB/fork" h2
cp -a "$H2LIB/build-h2/bin/libggml-cuda.so" "$H2LIB/lib/libggml-cuda-h2.so"
B_MD5=$(md5sum "$H2LIB/lib/libggml-cuda-h2.so" | cut -c1-8)
log "lib H2 : md5=$B_MD5"

# ---------------- [C] gates --------------------------------------------------
log "=== C: gates ==="
[ "$B_MD5" != "$PROD_MD5" ] && log "OK: H2 != prod"   || die "H2 == prod"
[ "$B_MD5" != "$A_MD5" ]    && log "OK: H2 != build A" || die "H2 == A (patch non efficace?)"
M=$(strings "$H2LIB/lib/libggml-cuda-h2.so" | grep -c "GGML_CUDA_VOLTA_Q6_W4R4")
[ "$M" -ge 1 ] && log "OK: marqueur GGML_CUDA_VOLTA_Q6_W4R4 present" || die "marqueurs Volta absents"
FORK_MD5_AFTER=$(md5sum "$FORK/ggml/src/ggml-cuda/fattn-q8-volta.cuh" | cut -c1-8)
[ "$FORK_MD5_BEFORE" = "$FORK_MD5_AFTER" ] && log "OK: fork v100-repro intact" || die "fork v100-repro modifie"
[ "$(md5sum "$PROD_SO" | cut -c1-8)" = "$PROD_MD5" ] && log "OK: prod intact" || die "prod modifie"

{
  echo "build propre A (fork intact)  : md5=$A_MD5  (bit-identique prod=$PROD_MD5)"
  echo "lib H2 (BC 16->8, sub<BC/8)   : md5=$B_MD5"
  echo "patch source (2 lignes, copie): $(diff "$F.orig" "$F" | grep -c '^[<>]') lignes"
  echo "marqueur W4R4 present : $M"
  echo "prod + fork v100-repro : intacts"
} | tee "$H2LIB/rapport-build.txt"
log "FIN build H2"
