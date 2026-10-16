// e1_check.cu — E1 : verification de CORRECTION numerique de la variante.
// Identique a e1_bench.cu (memes donnees aléatoires, meme seed, meme géométrie)
// mais DUMPE la sortie COMPLETE (pacc, pm, pl) au lieu d'un probe [0..1].
// Comparaison base vs variante faite en hote (diff binaire / tolerance bf16).
//
// Sortie : <out_prefix>.pacc.bin  <out_prefix>.pm.bin  <out_prefix>.pl.bin
// Usage : ./e1_check <out_prefix>  [iters=1]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cinttypes>
#include "e1_bench.cuh"

#define CK(x) do{ cudaError_t e_=(x); if(e_){fprintf(stderr,"CUDA err %s @%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

static void dump(const void* d, size_t n, const char* p){
    void* h = malloc(n);
    CK(cudaMemcpy(h,d,n,cudaMemcpyDeviceToHost));
    FILE* f=fopen(p,"wb"); fwrite(h,1,n,f); fclose(f); free(h);
}

int main(int argc, char** argv) {
    if (argc<2){ fprintf(stderr,"usage: %s <out_prefix> [iters]\n",argv[0]); return 2; }
    const char* prefix = argv[1];
    int iters = (argc>2)? atoi(argv[2]) : 1;

    const int    n_kv     = (argc>3)? atoi(argv[3]) : 107351;
    const int    kvh      = 4;
    const int    G        = e1q8v::G;
    const int    T        = e1q8v::T;
    const int    D        = e1q8v::D;
    const int    qh_count = kvh*G;
    const int    row_count= T*G;
    const int    splits   = e1q8v::split_count(n_kv);
    printf("[check] BC=%d n_kv=%d splits=%d\n", e1q8v::BC, n_kv, splits);

    const int64_t q_s1 = D;
    const int64_t q_s2 = (int64_t)D*qh_count;
    const int    blocks_per_head = D/32;
    const int64_t k_s1 = (int64_t)blocks_per_head;
    const int64_t k_s2 = (int64_t)blocks_per_head;
    size_t kv_blocks = (size_t)n_kv * kvh * blocks_per_head;
    const int64_t mask_s1 = T;
    size_t mask_elems = (size_t)n_kv * T;
    const int64_t pacc_rows = (int64_t)qh_count * T * splits;
    size_t pacc_elems = (size_t)D * pacc_rows;
    size_t pstat_elems= (size_t)pacc_rows;

    std::vector<float>   hq((size_t)T * (size_t)qh_count * (size_t)D);
    std::vector<uint8_t> hkv(kv_blocks*sizeof(block_q8_0));
    std::vector<half>    hmask(mask_elems);
    srand(12345);
    for (size_t i=0;i<hq.size();++i) hq[i] = (rand()/(float)RAND_MAX - 0.5f) * 1.0f;
    for (size_t b=0;b<kv_blocks;++b) {
        block_q8_0 * blk = (block_q8_0*)hkv.data() + b;
        blk->d = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.5f);
        for (int j=0;j<32;++j) blk->qs[j] = (int8_t)((rand()%251)-125);
    }
    for (size_t i=0;i<hmask.size();++i) hmask[i] = __float2half(0.0f);

    float *dq; block_q8_0 *dK,*dV; half *dmask;
    __nv_bfloat16 *dpacc; float *dpm,*dpl;
    CK(cudaMalloc(&dq,   (size_t)T*qh_count*D*sizeof(float)));
    CK(cudaMalloc(&dK,   kv_blocks*sizeof(block_q8_0)));
    CK(cudaMalloc(&dV,   kv_blocks*sizeof(block_q8_0)));
    CK(cudaMalloc(&dmask,mask_elems*sizeof(half)));
    CK(cudaMalloc(&dpacc,pacc_elems*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dpm,  pstat_elems*sizeof(float)));
    CK(cudaMalloc(&dpl,  pstat_elems*sizeof(float)));
    CK(cudaMemcpy(dq,   hq.data(),   (size_t)T*qh_count*D*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,   hkv.data(),   kv_blocks*sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,   hkv.data(),   kv_blocks*sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmask,hmask.data(), mask_elems*sizeof(half),         cudaMemcpyHostToDevice));
    // initialiser les sorties a 0 pour un dump stable
    CK(cudaMemset(dpacc,0,pacc_elems*sizeof(__nv_bfloat16)));
    CK(cudaMemset(dpm,0,pstat_elems*sizeof(float)));
    CK(cudaMemset(dpl,0,pstat_elems*sizeof(float)));

    dim3 grid(kvh, splits), block(e1q8v::THREADS);
    const float scale = 1.0f / sqrtf((float)D);
    for (int it=0; it<iters; ++it) {
        e1q8v::partial_kernel<<<grid,block,0,0>>>(
            dq, dK, dV, dmask,
            n_kv, splits, qh_count, scale,
            q_s1, q_s2, k_s1, k_s2, k_s1, k_s2, mask_s1,
            dpacc, dpm, dpl);
        CK(cudaDeviceSynchronize());
        cudaError_t le = cudaGetLastError();
        if (le){ fprintf(stderr,"kernel error: %s\n", cudaGetErrorString(le)); return 1; }
    }

    char p[512];
    snprintf(p,sizeof p,"%s.pacc.bin",prefix); dump(dpacc, pacc_elems*sizeof(__nv_bfloat16), p);
    snprintf(p,sizeof p,"%s.pm.bin",prefix);    dump(dpm,   pstat_elems*sizeof(float), p);
    snprintf(p,sizeof p,"%s.pl.bin",prefix);    dump(dpl,   pstat_elems*sizeof(float), p);
    printf("dump OK : %s (pacc=%zu B, pm/pl=%zu B)\n", prefix, pacc_elems*sizeof(__nv_bfloat16), pstat_elems*sizeof(float));
    return 0;
}
