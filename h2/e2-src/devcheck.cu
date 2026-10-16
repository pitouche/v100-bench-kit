// (c) 2026 guile — harnais du kit, MIT (voir LICENSE + NOTICE.md à la racine du kit).
#include <cuda_runtime.h>
#include <cstdio>
int main(){cudaFree(0);cudaDeviceProp p;cudaGetDeviceProperties(&p,0);printf("DEV=%s sm_%d%d\n",p.name,p.major,p.minor);return 0;}
