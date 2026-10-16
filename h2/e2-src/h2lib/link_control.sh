#!/bin/bash
set -e
cd "${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/build-clean/_deps/llama_cpp-build/ggml/src/ggml-cuda"
${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/gcc-13-bin/g++ -fPIC -shared  -o ${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/lib/libggml-cuda-control.so @${LLAMA_ROOT:-/home/guile/llama-experiments}/h2/cupti129-test/e2/h2lib/objects_control.rsp @CMakeFiles/ggml-cuda.dir/linkLibs.rsp -L"${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs" -L"${LLAMA_ROOT:-/home/guile/llama-experiments}/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"

