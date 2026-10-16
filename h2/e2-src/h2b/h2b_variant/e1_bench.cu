// e1_bench.cu — E1 : harnais hôte du partial_kernel (sm_70).
//
// Reproduit la VRAIE géométrie decode de run6 pour UN lance partial :
//   n_kv = 107351   (long contexte, ~100k + tokens générés)
//   kvh  = 4        (head_count_kv = 4, du GGUF)
//   G    = 6        (24 qh / 4 kvh)
//   T    = 4        (batch tokens vérifiés ; row_count = T*G = 24)
//   D    = 256, BC = 16, THREADS = 128, splits = 160
//   grid = (kvh=4, splits=160), block = 128
//   K/V en Q8_0 (34 octets/block de 32 clés)
//
// Lancement : ./e1_bench [iters]
//   iters=1 suffit pour un profil NCU ; iters>1 permet un warm-up + repli médiane.
//
// Le harnais n'ajoute AUCUNE heuristique : il alloue les pointeurs, remplit avec des
// données aléatoires bornées (q8_0 d borné, qs aléatoires), et lance partial_kernel.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include "e1_bench.cuh"

#define CK(x) do{ cudaError_t e_=(x); if(e_){fprintf(stderr,"CUDA err %s @%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

int main(int argc, char** argv) {
    int iters = (argc>1)? atoi(argv[1]) : 1;
    int splits_arg = (argc>2)? atoi(argv[2]) : 0;   // 0 => split_count(n_kv)

    // --- géométrie exacte du decode run6 ---
    const int    n_kv     = 107351;          // clés KV (long contexte)
    const int    kvh      = 4;               // head_count_kv
    const int    G        = e1q8v::G;        // 6
    const int    T        = e1q8v::T;        // 4
    const int    D        = e1q8v::D;        // 256
    const int    qh_count = kvh*G;           // 24 query heads
    const int    row_count= T*G;             // 24 rows per warp-tile
    const int    splits   = splits_arg>0 ? splits_arg : e1q8v::split_count(n_kv);   // 160 par defaut

    // --- tailles hôte ---
    // q : qh_count * n_kv * D floats, laid out [tok][qh][d] with q_s1 = D, q_s2 = D*qh_count
    //     (strides : q[lane+32p + q_s1*tok + q_s2*qh])
    const int64_t q_s1 = D;                        // stride between tokens  = D
    const int64_t q_s2 = (int64_t)D*qh_count;      // stride between heads    = D*qh_count
    // q[tok][qh][d] with tok in [0,T), qh in [0,qh_count), d in [0,D)
    size_t q_elems = (size_t)T * (size_t)qh_count * (size_t)D;

    // K / V : block_q8_0 layout [key][kvh][D/32 blocks of 32]
    const int        blocks_per_head = D/32;         // 8 blocks de 32 clés => D=256
    const int64_t    k_s1 = (int64_t)blocks_per_head; // stride between keys   (blocks)
    const int64_t    k_s2 = (int64_t)blocks_per_head; // stride between kvh     (blocks)
    size_t kv_blocks = (size_t)n_kv * kvh * blocks_per_head;

    // mask : half [key][tok] with mask_s1 = T (tok stride)
    const int64_t mask_s1 = T;
    size_t mask_elems = (size_t)n_kv * T;

    // pacc : bf16 [d][qh][tok in T*splits] ; pstat : float [qh][tok in T*splits] x2
    const int64_t pacc_rows = (int64_t)qh_count * T * splits;
    size_t pacc_elems = (size_t)D * pacc_rows;
    size_t pstat_elems= (size_t)pacc_rows;

    std::vector<float>   hq(q_elems);
    std::vector<uint8_t> hkv(kv_blocks*sizeof(block_q8_0));
    std::vector<half>    hmask(mask_elems);
    // deterministe + borné pour éviter les NaN/Inf qui faussent le timing
    srand(12345);
    for (size_t i=0;i<hq.size();++i) hq[i] = (rand()/(float)RAND_MAX - 0.5f) * 1.0f;
    for (size_t b=0;b<kv_blocks;++b) {
        block_q8_0 * blk = (block_q8_0*)hkv.data() + b;
        blk->d = __float2half((rand()/(float)RAND_MAX - 0.5f) * 0.5f);   // scale borné
        for (int j=0;j<32;++j) blk->qs[j] = (int8_t)((rand()%251)-125);
    }
    for (size_t i=0;i<hmask.size();++i) hmask[i] = __float2half(0.0f);    // pas de masque additif

    // --- alloc GPU ---
    float *dq; block_q8_0 *dK,*dV; half *dmask;
    __nv_bfloat16 *dpacc; float *dpm,*dpl;
    CK(cudaMalloc(&dq,        q_elems*sizeof(float)));
    CK(cudaMalloc(&dK,        kv_blocks*sizeof(block_q8_0)));
    CK(cudaMalloc(&dV,        kv_blocks*sizeof(block_q8_0)));
    CK(cudaMalloc(&dmask,     mask_elems*sizeof(half)));
    CK(cudaMalloc(&dpacc,     pacc_elems*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dpm,       pstat_elems*sizeof(float)));
    CK(cudaMalloc(&dpl,       pstat_elems*sizeof(float)));
    CK(cudaMemcpy(dq,   hq.data(),   q_elems*sizeof(float),      cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,   hkv.data(),  kv_blocks*sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,   hkv.data(),  kv_blocks*sizeof(block_q8_0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmask,hmask.data(),mask_elems*sizeof(half),     cudaMemcpyHostToDevice));

    dim3 grid(kvh, splits), block(e1q8v::THREADS);
    const float scale = 1.0f / sqrtf((float)D);   // typique (1/sqrt(head_dim)); n'influe pas sur le BW
    auto launch_once = [&](){
        e1q8v::partial_kernel<<<grid,block,0,0>>>(
            dq, dK, dV, dmask,
            n_kv, splits, qh_count, scale,
            q_s1, q_s2, k_s1, k_s2, k_s1, k_s2, mask_s1,
            dpacc, dpm, dpl);
    };

    // warm-up (JIT/const cache) hors mesure
    launch_once();
    CK(cudaDeviceSynchronize());
    cudaError_t le = cudaGetLastError();
    if (le) { fprintf(stderr,"kernel error: %s\n", cudaGetErrorString(le)); return 1; }

    // Timing CUDA-events par itération (pas NCU) : robuste, aucun tooling GPU requis.
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    float total_ms = 0.0f;
    for (int it=0; it<iters; ++it) {
        CK(cudaEventRecord(ev0));
        launch_once();
        CK(cudaEventRecord(ev1));
        CK(cudaEventSynchronize(ev1));
        float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
        total_ms += ms;
        le = cudaGetLastError();
        if (le) { fprintf(stderr,"kernel error (iter %d): %s\n", it, cudaGetErrorString(le)); return 1; }
    }
    CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
    const double us_med = total_ms*1000.0 / iters;

    // --- estimation BW DRAM (METHODE 1 : modèle exacte, PAS compteur) ---
    // Trafic exact du kernel (chaque élément lu/écrit compté une fois par bloc qui le lit) :
    //   K : n_kv × kvh × 272 B (8 blocks q8_0 × 34 B / head), lu 1× par un bloc (split unique)
    //   V : idem
    //   Q : lu par CHAQUE bloc (grid = kvh × splits) : 24 rows × 256 × 4 B
    //   mask : lu par chaque kvh sur sa plage de tiles => tout le mask × kvh
    //   pacc : écrit 1× ; pm/pl : écrits 1×
    const int grid_blocks = kvh * splits;
    const double kv_bytes   = (double)n_kv * kvh * 272.0;             // K seul (272 B/head)
    const double q_bytes    = (double)grid_blocks * (double)row_count * D * sizeof(float);
    const double mask_bytes = (double)kvh * (double)n_kv * T * sizeof(half);
    const double pacc_bytes = (double)D * qh_count * T * splits * sizeof(__nv_bfloat16);
    const double pstat_bytes= (double)qh_count * T * splits * 2 * sizeof(float);
    const double rw_bytes = kv_bytes*2.0 /* K+V */ + q_bytes + mask_bytes + pacc_bytes + pstat_bytes;
    const double gb_s = rw_bytes / (us_med*1e-6) / 1e9;
    printf("n_kv=%d kvh=%d G=%d T=%d D=%d splits=%d qh_count=%d row_count=%d\n", n_kv,kvh,G,T,D,splits,qh_count,row_count);
    printf("[E1-modele] durée/lance (CUDA events, %d iters) : %.0f us   (run6 CUPTI: 1457-1840 us, med 1615)\n", iters, us_med);
    printf("[E1-modele] volume/lance (modele exact)           : %.1f Mo  (KV %.0f + Q %.1f + mask %.1f + out %.1f)\n",
           rw_bytes/1e6, kv_bytes*2/1e6, q_bytes/1e6, mask_bytes/1e6, (pacc_bytes+pstat_bytes)/1e6);
    printf("[E1-modele] BW DRAM effectif (estimation)      : %.0f GB/s   (plafond HBM V100 ~900)\n", gb_s);
    printf("[E1-modele] CRITERE E1 : <300 => kernel-bound ; >600 => mem-bound ; sinon zone grise\n");

    // sanity : un bout de pacc
    __nv_bfloat16 probe[8];
    CK(cudaMemcpy(probe, dpacc, 8*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    printf("pacc probe[0..1] = %f %f  (fini, %d iters)\n",
           __bfloat162float(probe[0]), __bfloat162float(probe[1]), iters);
    (void)us_med;

    CK(cudaFree(dq)); CK(cudaFree(dK)); CK(cudaFree(dV)); CK(cudaFree(dmask));
    CK(cudaFree(dpacc)); CK(cudaFree(dpm)); CK(cudaFree(dpl));
    printf("OK\n");
    return 0;
}
