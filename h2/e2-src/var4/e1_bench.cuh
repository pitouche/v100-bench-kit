// e1_bench.cuh — E1 : harnais standalone du ggml_q8v::partial_kernel (sm_70).
//
// Objectif : mesurer le DRAM réel (dram__bytes_read/write.sum) et la durée d'un lance
//   `partial_kernel` sur la vraie géométrie decode (n_kv=107351, 4 kvh, 160 splits,
//   128 threads, grid=(4,160)), en dehors du serveur, avec NCU 13.1 (replay application).
//
// Le code du kernel ci-dessous est une COPIE EXACTE de
//   src/llama-volta-merged/ggml/src/ggml-cuda/fattn-q8-volta.cuh
//   (namespace ggml_q8v, partial_kernel, sm_70 seulement) — aucun changement de logique,
//   de paramètres ni d'heuristique. On remplace uniquement le type `ggml_half` par `__half`
//   (identique en ABI) pour que le harnais compile sans l'arborescence ggml complète.
//
// Métriques visées :
//   dram__bytes_read.sum / dur  => BW DRAM effectif.
//   CRITÈRE E1 : < 300 GB/s  => kernel-bound confirmé (H2/H3)
//                 > 600 GB/s  => mem-bound, H2 morte (réduire les octets)
//   300..600 => zone grise : croiser avec sm__cycles_active / occ / tensor pipe.

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ggml q8_0 block — layout EXACT de ggml/src/ggml-common.h : half d + 32 int8 = 34 octets.
struct block_q8_0 {
    __half   d;
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
// --- Fragment addressing (copie exacte) ---
__device__ __forceinline__ int volta_qp_get_i() { return threadIdx.x & 31; }
__device__ __forceinline__ int volta_qp_get_j(int l) { return l; }
__device__ __forceinline__ int volta_k_get_i() {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_k_get_j(int l) { return l; }
__device__ __forceinline__ int volta_d_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return (l & 2) + (lane & ~2);
}
__device__ __forceinline__ int volta_d_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return (lane & 2) + (l & (4 + 1));
}
__device__ __forceinline__ int volta_v_get_i(int l) {
    const int lane = threadIdx.x & 31;
    return ((l / 2) * 4) + (lane % 4);
}
__device__ __forceinline__ int volta_v_get_j(int l) {
    const int lane = threadIdx.x & 31;
    return ((lane / 16) * 2) + (l % 2);
}
__device__ __forceinline__ void volta_load_qp(half2 (&dst)[4], const half2* __restrict__ base, int stride) {
    const int row = volta_qp_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}
__device__ __forceinline__ void volta_load_k(half2 (&dst)[4], const half2* __restrict__ base, int stride) {
    const int row = volta_k_get_i();
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[row * stride + l]; }
}
__device__ __forceinline__ void volta_load_v(half2 (&dst)[4], const half2* __restrict__ base, int stride) {
#pragma unroll
    for (int l = 0; l < 4; ++l) { dst[l] = base[volta_v_get_i(l) * stride + volta_v_get_j(l)]; }
}
__device__ __forceinline__ void volta_mma_qk(float (&d)[8], const half2 (&q)[4], const half2 (&k)[4]) {
    const int* Axi = reinterpret_cast<const int*>(q);
    const int* Bxi = reinterpret_cast<const int*>(k);
    int* Dxi        = reinterpret_cast<int*>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]), "r"(Bxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3]), "+r"(Dxi[4]),
                   "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
                 : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]), "r"(Bxi[3]));
}
__device__ __forceinline__ void volta_mma_pv(half2 (&pv)[4], const half2 (&p)[4], const half2 (&v)[4]) {
    const int* Pxi = reinterpret_cast<const int*>(p);
    const int* Vxi = reinterpret_cast<const int*>(v);
    int* PVxi = reinterpret_cast<int*>(pv);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[0]), "r"(Pxi[1]), "r"(Vxi[0]), "r"(Vxi[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
                 : "+r"(PVxi[0]), "+r"(PVxi[1]), "+r"(PVxi[2]), "+r"(PVxi[3])
                 : "r"(Pxi[2]), "r"(Pxi[3]), "r"(Vxi[2]), "r"(Vxi[3]));
}
__device__ __forceinline__ void volta_softmax_to_half2(half2 (&p)[4], const float (&d)[8]) {
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 4) {
        p[l0 / 2 + 0] = __floats2half2_rn(d[l0 + 0], d[l0 + 1]);
        p[l0 / 2 + 1] = __floats2half2_rn(d[l0 + 2], d[l0 + 3]);
        const int lane      = threadIdx.x & 31;
        const int swap_idx = l0 / 2 + (((lane % 4) / 2) ^ 1);
        p[swap_idx]          = __shfl_xor_sync(0xFFFFFFFFu, p[swap_idx], 2, 32);
    }
}
#endif

namespace e1q8v {

constexpr int D = 256;
constexpr int G = 6;
constexpr int T = 4;
constexpr int BC = 16;
constexpr int BR = 32;
constexpr int THREADS = 128;
constexpr int D_SLICE = 64;
constexpr int SMEM_STRIDE = 264;
constexpr float LOG2E = 1.4426950408889634074f;
constexpr int LONG_SPLITS = 160;

static inline int split_count(int n_kv) {
    const int tiles = (n_kv + BC - 1) / BC;
    const int requested = std::min(LONG_SPLITS, tiles);
    const int tiles_per_split = (tiles + requested - 1) / requested;
    return (tiles + tiles_per_split - 1) / tiles_per_split;
}

__device__ __forceinline__ int row_qh(int row, int kvh) { return kvh*G + row % G; }
__device__ __forceinline__ int row_tok(int row) { return row/G; }
__device__ __forceinline__ int64_t pacc_idx(int qh,int d,int tok,int split,int qh_count) {
    return d + (int64_t)D*(qh + (int64_t)qh_count*(tok + (int64_t)T*split));
}
__device__ __forceinline__ int64_t pstat_idx(int qh,int tok,int split,int qh_count) {
    return qh + (int64_t)qh_count*(tok + (int64_t)T*split);
}
__device__ __forceinline__ int4 deq8(const block_q8_0 & b, int off) {
    const float s = __half2float(b.d);
    half2 h[4];
#pragma unroll
    for (int i=0;i<4;i++) h[i]=__floats2half2_rn((float)b.qs[off+2*i]*s,(float)b.qs[off+2*i+1]*s);
    return *reinterpret_cast<int4 *>(h);
}

__launch_bounds__(128,4)
__global__ void partial_kernel(
        const float * q, const block_q8_0 * K, const block_q8_0 * V, const half * mask,
        int n_kv, int splits, int qh_count, float scale,
        int64_t q_s1, int64_t q_s2, int64_t k_s1, int64_t k_s2, int64_t v_s1, int64_t v_s2, int64_t mask_s1,
        __nv_bfloat16 * pacc, float * pm, float * pl) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    __shared__ __align__(16) half q_s[BR*SMEM_STRIDE];
    __shared__ __align__(16) half k_s[BC*SMEM_STRIDE];
    __shared__ __align__(16) half v_s[BC*SMEM_STRIDE];
    __shared__ half m_s[T*BC];
    const int kvh=blockIdx.x, split=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31, dim_warp=warp;
    constexpr int row_count=T*G;
    const int logical_tiles=(n_kv+BC-1)/BC;
    const int tiles_per_split=(logical_tiles+splits-1)/splits;
    const int first_tile=split*tiles_per_split*BC;
    const int split_end=min(n_kv, first_tile+tiles_per_split*BC);
    if (first_tile>=split_end) return;
    const int key_blocks=(split_end-first_tile+BC-1)/BC;

    for (int row=warp; row<BR; row+=4) {
        float vals[8]={};
        if (row<row_count) {
            const int qh=row_qh(row,kvh), tok=row_tok(row);
#pragma unroll
            for (int p=0;p<8;p++) vals[p]=q[lane+32*p + q_s1*tok + q_s2*qh];
        }
#pragma unroll
        for (int p=0;p<8;p++) q_s[row*SMEM_STRIDE+lane+32*p]=__float2half(vals[p]);
    }
    __syncthreads();

    float acc[D_SLICE/8][8];
#pragma unroll
    for(int c=0;c<D_SLICE/8;c++) for(int i=0;i<8;i++) acc[c][i]=0.0f;
    float mlo=-CUDART_INF_F,mhi=-CUDART_INF_F,llo=0.0f,lhi=0.0f;

    for (int kb=0; kb<key_blocks; kb++) {
        const int k0=first_tile+kb*BC;
        for (int chunk=tid; chunk<BC*(D/8); chunk+=THREADS) {
            const int key_l=chunk/(D/8), d=(chunk-key_l*(D/8))*8, key=k0+key_l;
            half * kd=&k_s[key_l*SMEM_STRIDE+d], *vd=&v_s[key_l*SMEM_STRIDE+d];
            if (key<split_end) {
                const int bo=d/32, off=d&31;
                *reinterpret_cast<int4*>(kd)=deq8(K[key*k_s1 + kvh*k_s2 + bo],off);
                *reinterpret_cast<int4*>(vd)=deq8(V[key*v_s1 + kvh*v_s2 + bo],off);
            } else { const int4 z=make_int4(0,0,0,0); *reinterpret_cast<int4*>(kd)=z; *reinterpret_cast<int4*>(vd)=z; }
        }
        for (int i=tid; i<T*BC; i+=THREADS) {
            const int tok=i/BC, col=i%BC, key=k0+col;
            m_s[i]=key<n_kv ? mask[key + mask_s1*tok] : __float2half(-CUDART_INF_F);
        }
        __syncthreads();

#pragma unroll
        for (int sub=0;sub<2;sub++) {
            const int sub_k0=k0+sub*8;
            float ds[8]={};
#pragma unroll
            for (int c=0;c<D/8;c++) {
                half2 qf[4],kf[4];
                volta_load_qp(qf,reinterpret_cast<const half2*>(&q_s[c*8]),SMEM_STRIDE/2);
                volta_load_k(kf,reinterpret_cast<const half2*>(&k_s[sub*8*SMEM_STRIDE+c*8]),SMEM_STRIDE/2);
                volta_mma_qk(ds,qf,kf);
            }
#pragma unroll
            for(int l=0;l<8;l++) {
                const int row=volta_d_get_i(l), col=volta_d_get_j(l), key=sub_k0+col;
                if (row<row_count && key<split_end) {
                    const int tok=row_tok(row); ds[l]=ds[l]*scale+__half2float(m_s[tok*BC+sub*8+col]);
                } else ds[l]=-CUDART_INF_F;
            }
            float bmlo=fmaxf(fmaxf(ds[0],ds[1]),fmaxf(ds[4],ds[5]));
            float bmhi=fmaxf(fmaxf(ds[2],ds[3]),fmaxf(ds[6],ds[7]));
            bmlo=fmaxf(bmlo,__shfl_xor_sync(0xffffffffu,bmlo,2)); bmhi=fmaxf(bmhi,__shfl_xor_sync(0xffffffffu,bmhi,2));
            const float nmlo=fmaxf(mlo,bmlo), nmhi=fmaxf(mhi,bmhi);
            const float alo=mlo==-CUDART_INF_F?0.0f:exp2f((mlo-nmlo)*LOG2E);
            const float ahi=mhi==-CUDART_INF_F?0.0f:exp2f((mhi-nmhi)*LOG2E);
#pragma unroll
            for(int l=0;l<8;l++){const float nm=(l&2)?nmhi:nmlo;ds[l]=(nm>-CUDART_INF_F&&ds[l]>-CUDART_INF_F)?exp2f((ds[l]-nm)*LOG2E):0.0f;}
            float bllo=ds[0]+ds[1]+ds[4]+ds[5],blhi=ds[2]+ds[3]+ds[6]+ds[7];
            bllo+=__shfl_xor_sync(0xffffffffu,bllo,2);blhi+=__shfl_xor_sync(0xffffffffu,blhi,2);
            llo=llo*alo+bllo;lhi=lhi*ahi+blhi;mlo=nmlo;mhi=nmhi;
            const float alpha=(lane&2)?ahi:alo; half2 pp[4]; volta_softmax_to_half2(pp,ds);
#pragma unroll
            for(int c=0;c<D_SLICE/8;c++){half2 vf[4];volta_load_v(vf,reinterpret_cast<const half2*>(&v_s[sub*8*SMEM_STRIDE+dim_warp*D_SLICE+c*8]),SMEM_STRIDE/2);half2 pv[4]={{0,0},{0,0},{0,0},{0,0}};volta_mma_pv(pv,pp,vf);
#pragma unroll
                for(int n=0;n<4;n++){const float2 f=__half22float2(pv[n]);acc[c][2*n]=acc[c][2*n]*alpha+f.x;acc[c][2*n+1]=acc[c][2*n+1]*alpha+f.y;}}
        }
        __syncthreads();
    }
    const int row=lane; const float ownm=(lane&2)?mhi:mlo, ownl=(lane&2)?lhi:llo;
    if(dim_warp==0&&row<row_count){const int qh=row_qh(row,kvh),tok=row_tok(row);pm[pstat_idx(qh,tok,split,qh_count)]=ownm;pl[pstat_idx(qh,tok,split,qh_count)]=ownl;}
    if(row<row_count){const int qh=row_qh(row,kvh),tok=row_tok(row);
#pragma unroll
        for(int c=0;c<D_SLICE/8;c++){const int d=dim_warp*D_SLICE+c*8;__nv_bfloat16 o[8];for(int i=0;i<8;i++)o[i]=__float2bfloat16(acc[c][i]);*reinterpret_cast<int4*>(&pacc[pacc_idx(qh,d,tok,split,qh_count)])=*reinterpret_cast<int4*>(o);}}
#endif
}

} // namespace e1q8v
