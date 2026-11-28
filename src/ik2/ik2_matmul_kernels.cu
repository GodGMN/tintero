#include "ik2_kernel_common.cuh"
#include "ik2_cuda_mma.cuh"
#include "ik2_iq_tables.h"

namespace ik2::k {
namespace {

#ifdef STRATA_USE_HIP
// The high word sorts by FP32 score, the low bits break ties in ascending expert order.
__device__ __forceinline__ uint64_t route_key(float score, int expert) {
    uint32_t bits = score == 0.f ? 0 : __float_as_uint(score);
    bits ^= (0u - (bits >> 31)) | 0x80000000u;
    return ((uint64_t)bits << 8) | (127 - expert);
}

__device__ __forceinline__ uint64_t route_sort_wave(uint64_t key, int lane) {
#pragma unroll
    for (int group = 2; group <= 32; group *= 2) {
#pragma unroll
        for (int stride = group / 2; stride; stride /= 2) {
            const uint64_t peer = __shfl_xor(key, stride, 32);
            const bool larger = ((lane & group) == 0) == ((lane & stride) == 0);
            key = larger ? max(key, peer) : min(key, peer);
        }
    }
    return key;
}

struct RouteGather {
    const uint8_t* const* table;
    const float *gate_scale, *up_scale, *down_scale;
    const uint8_t** blob;
    int *tok, *inv;
    float *sg, *su, *wt;
};

// Four waves select their best eight experts. A final wave finds the eighth of those 32.
template <bool Gather>
__device__ __forceinline__ void rdna3_select_eight(float logit, const float* bias, float scale,
                                                    int* top, float* weights, RouteGather tables) {
    constexpr int K = 8;
    const int e = threadIdx.x, lane = e & 31, wave = e >> 5;
    const int token = blockIdx.x;
    const float sig = 1.f / (1.f + expf(-logit));
    const uint64_t key = route_key(sig + bias[e], e);
    const uint64_t local = route_sort_wave(key, lane);
    __shared__ uint64_t finalists[32], cutoff;
    __shared__ uint32_t masks[4];
    __shared__ float chosen[K];
    if (lane < K) finalists[wave * K + lane] = local;
    __syncthreads();
    if (!wave) {
        const uint64_t ranked = route_sort_wave(finalists[lane], lane);
        if (lane == K - 1) cutoff = ranked;
    }
    __syncthreads();
    const bool pick = key >= cutoff;
    const uint32_t mine = __ballot(pick);
    if (!lane) masks[wave] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
#pragma unroll
    for (int w = 0; w < 3; ++w) if (w < wave) pos += __popc(masks[w]);
    if (pick) chosen[pos] = sig;
    __syncthreads();
    if (!pick) return;
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < K; ++i) sum += chosen[i];
    const float weight = sig / (sum + 1e-20f) * scale;
    top[(size_t)token * K + pos] = e;
    weights[(size_t)token * K + pos] = weight;
    if constexpr (Gather) {
        tables.blob[pos] = tables.table[e];
        tables.tok[pos] = 0;
        tables.inv[pos] = pos;
        tables.sg[pos] = tables.gate_scale ? tables.gate_scale[e] : 1.f;
        tables.su[pos] = tables.up_scale ? tables.up_scale[e] : 1.f;
        tables.wt[pos] = weight * (tables.down_scale ? tables.down_scale[e] : 1.f);
    }
}
template <bool Gather>
__global__ void rdna3_route_eight(const float* logits, const float* bias, float scale, int* top, float* weights,
                                  RouteGather tables) {
    rdna3_select_eight<Gather>(logits[blockIdx.x * 128 + threadIdx.x], bias, scale, top, weights, tables);
}

// Wave32 router projection adapted from CIRU hip_glue_decode.hip (Copyright 2026 Ciru, Apache-2.0).
// Keep one token's logits in LDS so the workgroup can also select and normalize the eight experts.
__global__ void rdna3_project_route_kernel(const uint16_t* weights, const float* x, int width, const float* bias,
                                           float scale, int* top, float* picks) {
    __shared__ float logits[128];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    x += size_t(blockIdx.x) * width;
    for (int expert = wave; expert < 128; expert += 4) {
        const uint16_t* w = weights + size_t(expert) * width;
        float acc = 0.f;
        for (int i = lane; i < width; i += 32) acc = fmaf(h2f(w[i]), x[i], acc);
        const float sum = wsum(acc);
        if (!lane) logits[expert] = sum;
    }
    __syncthreads();
    rdna3_select_eight<false>(logits[threadIdx.x], bias, scale, top, picks, {});
}
bool rdna3_route_rows(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
                      cudaStream_t stream) {
    if (E != 128 || K != 8) return false;
    rdna3_route_eight<false><<<T, 128, 0, stream>>>(logits, bias, scale, top, w, {});
    check("rdna3_route_rows");
    return true;
}
#endif

constexpr int kRouteMax = 16;
// Single-warp zero-barrier router for E <= 512: each lane holds <= 16 experts in registers.
__global__ void route_warp_kernel(const float* logits, const float* bias, int T, int E, int K, float scale, int* top,
                                  float* w) {
    const int t = blockIdx.x, lane = threadIdx.x;   // 32 threads
    const float* l = logits + (size_t) t * E;
    float val[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int e = lane + m * 32;
        val[m] = (e < E) ? (1.0f / (1.0f + expf(-__ldg(l + e))) + __ldg(bias + e)) : -INFINITY;
    }
    int ch[kRouteMax];
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int e = lane + m * 32;
            if (val[m] > best || (val[m] == best && e < idx)) {
                best = val[m];
                idx = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) {
                best = ob;
                idx = oi;
            }
        }
        ch[k] = idx;
        if (idx < E && (idx & 31) == lane) val[idx >> 5] = -INFINITY;
    }
    if (lane == 0) {
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = ch[a];
            int b = a - 1;
            for (; b >= 0 && ch[b] > x; --b) ch[b + 1] = ch[b];
            ch[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += (sc[k] = 1.0f / (1.0f + expf(-__ldg(l + ch[k]))));
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = ch[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

// a block a token: the scores in shared memory, then K rounds of the block's argmax (ties: the lower expert)
__global__ void route_kernel(const float* logits, const float* bias, int T, int E, int K, float scale, int* top,
                             float* w) {
    extern __shared__ float v[];   // [E]
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int ch[kRouteMax];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const float* l = logits + (size_t) t * E;
    for (int e = tid; e < E; e += blockDim.x) v[e] = 1.0f / (1.0f + expf(-l[e])) + bias[e];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x)
            if (v[e] > best || (v[e] == best && e < idx)) best = v[e], idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) best = ob, idx = oi;
        }
        if (lane == 0) bv[warp] = best, bi[warp] = idx;
        __syncthreads();
        if (tid == 0) {
            float b = bv[0];
            int i = bi[0];
            for (int j = 1; j < nw; ++j)
                if (bv[j] > b || (bv[j] == b && bi[j] < i)) b = bv[j], i = bi[j];
            ch[k] = i;
            v[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int c[kRouteMax];
        for (int k = 0; k < K; ++k) c[k] = ch[k];
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = c[a];
            int b = a - 1;
            for (; b >= 0 && c[b] > x; --b) c[b + 1] = c[b];
            c[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += (sc[k] = 1.0f / (1.0f + expf(-l[c[k]])));
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = c[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

__global__ void gather_f16_kernel(const float* x, const int* idx, int n, int C, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * C) return;
    y[i] = f2h(x[(size_t) idx[i / C] * C + i % C]);
}

__global__ void swiglu_cap_kernel(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap,
                                  float ucap, uint16_t* h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = h2f(gu[r * 2 * F + c]) * sg[r];
    const float u = h2f(gu[r * 2 * F + F + c]) * su[r];
    h[i] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
}

__global__ void moe_combine_kernel(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out,
                                   bool add) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[t * K + k] * h2f(y[(size_t) inv[t * K + k] * C + c]);
    out[i] = add ? out[i] + a : a;
}

__global__ void swiglu_cap_f32_kernel(const float* gu, int n, int F, const float* sg, const float* su, float gcap,
                                      float ucap, uint16_t* h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = gu[r * 2 * F + c] * sg[r];
    const float u = gu[r * 2 * F + F + c] * su[r];
    h[i] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
}

// one token's moe_combine_f32 and the conv4_res after it in one launch: out = conv(sum_k wt[k] y[inv[k]]) + res
__global__ void moe_combine_conv_kernel(const float* __restrict__ y, const int* __restrict__ inv,
                                        const float* __restrict__ wt, int K, int C, float* hist,
                                        const float* __restrict__ w, const float* res, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[k] * y[(size_t) inv[k] * C + c];
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float b = w0 * h0 + w1 * h1 + w2 * h2 + w3 * a;
    if (res) b += res[c];
    out[c] = b;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = a;
}
__global__ void moe_combine_f32_kernel(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out,
                                       bool add) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[t * K + k] * y[(size_t) inv[t * K + k] * C + c];
    out[i] = add ? out[i] + a : a;
}

// ---- the decodes: value v of a run of blocks, ggml's dequantize_row_* arithmetic

__host__ __device__ __forceinline__ float ue4m3(uint8_t x) {   // ggml_ue4m3_to_fp32 (branchless bit-ops)
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    const float norm_f = bits_to_float(((e + 119u) << 23) | (m << 20));
    const float sub_f = (float) m * 0x1.0p-10f;
    const float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}
// IQ3_XXS / IQ4_XS tables: a device copy (read with __ldg: lanes index them divergently) and a host copy.
__device__ const uint32_t d_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
[[maybe_unused]] static const uint32_t h_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
__device__ const int8_t d_iq4nl[16] = {IK2_IQ4NL_VALUES};
[[maybe_unused]] static const int8_t h_iq4nl[16] = {IK2_IQ4NL_VALUES};
__host__ __device__ __forceinline__ uint32_t iq3xxs_grid_at(int i) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ldg(d_iq3xxs_grid + i);
#else
    return h_iq3xxs_grid[i];
#endif
}
__host__ __device__ __forceinline__ int iq4nl_at(int i) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return d_iq4nl[i];
#else
    return h_iq4nl[i];
#endif
}
// ggml's ksigns_iq2xs[s]: the 7 sign bits plus an 8th that makes the count of set bits even
__host__ __device__ __forceinline__ uint32_t iq_signs8(uint32_t s7) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return s7 | ((uint32_t) (__popc(s7) & 1) << 7);
#else
    uint32_t p = s7 ^ (s7 >> 4);   // (the parity by folding: no GNU builtin, no C++20 <bit> - MSVC builds this too)
    p ^= p >> 2;
    p ^= p >> 1;
    return s7 | (p & 1) << 7;
#endif
}

__host__ __device__ inline float decode_value(int type, const uint8_t* blocks, size_t v) {
    if (type == kNVFP4) {
        const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        const uint8_t* b = blocks + (v / 64) * 36;
        const int i = (int) (v % 64), sb = i / 16, j = i % 16;
        const uint8_t byte = b[4 + sb * 8 + (j & 7)];
        return (float) kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * ue4m3(b[sb]);
    }
    if (type == kQ8_0) {
        const uint8_t* b = blocks + (v / 32) * 34;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        return d * (float) (int8_t) b[2 + v % 32];
    }
    if (type == kQ6_K) {   // ql [128], qh [64], scales int8 [16], d
        const uint8_t* b = blocks + (v / 256) * 210;
        const int i = (int) (v % 256), n = i / 128, lq = i % 128, qd = lq / 32, l = lq % 32;
        const uint8_t* ql = b + n * 64;
        const uint8_t* qh = b + 128 + n * 32;
        const int8_t* sc = (const int8_t*) b + 192 + n * 8;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const uint8_t lo = ql[l + ((qd & 1) ? 32 : 0)];
        const int nib = qd >= 2 ? (lo >> 4) : (lo & 0xF);
        const int q = (int8_t) (nib | (((qh[l] >> (2 * qd)) & 3) << 4)) - 32;
        return d * (float) sc[l / 16 + 2 * qd] * (float) q;
    }
    if (type == kIQ3_XXS) {   // d, qs [64] grid indices, [8] u32 scale (top 4 bits) + 4 x 7 sign bits
        const uint8_t* b = blocks + (v / 256) * 98;
        const int i = (int) (v % 256), ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint8_t* q = b + 2 + 64 + 4 * ib32;
        const uint32_t aux = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint32_t signs = iq_signs8((aux >> (7 * l)) & 127u);
        const uint32_t g = iq3xxs_grid_at(b[2 + 8 * ib32 + 2 * l + (j >> 2)]);
        const float w = (float) ((g >> (8 * (j & 3))) & 0xFFu);
        return db * w * ((signs >> j) & 1u ? -1.0f : 1.0f);
    }
    if (type == kIQ4_NL) {
        const uint8_t* b = blocks + (v / 32) * 18;
        const uint8_t q = b[2 + (v & 15)];
        return half_bits_to_float((uint16_t) (b[0] | (b[1] << 8))) * iq4nl_at((v & 16) ? q >> 4 : q & 15);
    }
    if (type == kIQ4_XS) {   // d, scales_h, scales_l [4], qs [128]
        const uint8_t* b = blocks + (v / 256) * 136;
        const int i = (int) (v % 256), ib = i / 32, j = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint32_t sh = (uint32_t) b[2] | ((uint32_t) b[3] << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (int) (((sh >> (2 * ib)) & 3u) << 4);
        const uint8_t q = b[8 + 16 * ib + (j & 15)];
        return d * (float) (ls - 32) * (float) iq4nl_at(j < 16 ? (q & 0xF) : (q >> 4));
    }
    if (type == kQ5_K) {   // d, dmin, scales [12], qh [32], qs [128]
        const uint8_t* b = blocks + (v / 256) * 176;
        const int i = (int) (v % 256), j64 = i / 64, half = (i % 64) / 32, l = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const int is = 2 * j64 + half;
        int sc, m;
        if (is < 4) {
            sc = q[is] & 63, m = q[is + 4] & 63;
        } else {
            sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
        }
        const uint8_t lo = b[48 + 32 * j64 + l];
        const int hb = (b[16 + l] >> (2 * j64 + half)) & 1;
        const int val = (half ? (lo >> 4) : (lo & 0xF)) + (hb ? 16 : 0);
        return d * (float) sc * (float) val - mn * (float) m;
    }
    return 0.0f;
}

// The input vector in shared memory, padded: value i at xp(i) = (i >> 5) * 36 + (i & 31), so every 16-float
// NVFP4 sub-block and 32-float Q8_0 block is 16-byte (float4 / LDS.128) aligned with 0 bank conflicts.
__device__ __forceinline__ int xp(int i) { return (i >> 5) * 36 + (i & 31); }
__host__ __device__ constexpr size_t xp_floats(int K) {
    return K <= 0 ? 0 : (size_t) ((K - 1) >> 5) * 36 + ((K - 1) & 31) + 1;
}

// Fast register E2M1 pair decode via FP16 high-byte LUT in __byte_perm -> __half22float2:
// FP16 representations of {0, 1, 2, 3, 4, 6, 8, 12} all have low byte 0x00 and high bytes {0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a}.
__device__ __forceinline__ float2 e2m1_pair_lo(uint32_t s) {
    const uint32_t sel = (s << 4) & 0x7070u;
    const uint32_t sgn = ((s & 8u) << 12) | ((s & 0x800u) << 20);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}
__device__ __forceinline__ float2 e2m1_pair_hi(uint32_t s) {
    const uint32_t sel = s & 0x7070u;
    const uint32_t sgn = ((s & 0x80u) << 8) | ((s & 0x8000u) << 16);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}

// Four signed FP16 high bytes in the order of the serialized E2M1 nibbles.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
__device__ __forceinline__ uint32_t decode_signed_nibbles(uint32_t nib) {
    const uint32_t p = __byte_perm(0x42403c00u, 0x4a484644u, nib);
    const uint32_t n = __byte_perm(0xc2c0bc80u, 0xcac8c6c4u, nib);
    return __byte_perm(p, n, 0x3210u | ((nib & 0x8888u) >> 1));
}
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, float4 x0, float4 x1, float4 x2, float4 x3) {
    const uint32_t a = decode_signed_nibbles(q0), b = decode_signed_nibbles(q0 >> 16);
    const uint32_t c = decode_signed_nibbles(q1), d = decode_signed_nibbles(q1 >> 16);

    const auto v01 = u32_to_f2(__byte_perm(0, a, 0x6040)), v23 = u32_to_f2(__byte_perm(0, b, 0x6040));
    const auto v45 = u32_to_f2(__byte_perm(0, c, 0x6040)), v67 = u32_to_f2(__byte_perm(0, d, 0x6040));
    const auto v89 = u32_to_f2(__byte_perm(0, a, 0x7050)), vAB = u32_to_f2(__byte_perm(0, b, 0x7050));
    const auto vCD = u32_to_f2(__byte_perm(0, c, 0x7050)), vEF = u32_to_f2(__byte_perm(0, d, 0x7050));
    return (v01.x * x0.x + v01.y * x0.y + v23.x * x0.z + v23.y * x0.w) +
           (v45.x * x1.x + v45.y * x1.y + v67.x * x1.z + v67.y * x1.w) +
           (v89.x * x2.x + v89.y * x2.y + vAB.x * x2.z + vAB.y * x2.w) +
           (vCD.x * x3.x + vCD.y * x3.y + vEF.x * x3.z + vEF.y * x3.w);
}
#else
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, float4 x0, float4 x1, float4 x2, float4 x3) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    return (w01.x * x0.x + w01.y * x0.y + w23.x * x0.z + w23.y * x0.w) +
           (w45.x * x1.x + w45.y * x1.y + w67.x * x1.z + w67.y * x1.w) +
           (w89.x * x2.x + w89.y * x2.y + wAB.x * x2.z + wAB.y * x2.w) +
           (wCD.x * x3.x + wCD.y * x3.y + wEF.x * x3.z + wEF.y * x3.w);
}
#endif

__device__ __forceinline__ float dot_u32_f4(uint32_t q4, float4 f) {
    return (float) (int8_t) (q4 & 0xFFu) * f.x +
           (float) (int8_t) ((q4 >> 8) & 0xFFu) * f.y +
           (float) (int8_t) ((q4 >> 16) & 0xFFu) * f.z +
           (float) (int8_t) (q4 >> 24) * f.w;
}

__device__ __forceinline__ float q8_blk0_dot(const uint32_t* p, float4 f0, float4 f1, float4 f2, float4 f3,
                                             float4 f4, float4 f5, float4 f6, float4 f7, float& d1_out) {
    const uint32_t u0 = __ldg(p + 0), u1 = __ldg(p + 1), u2 = __ldg(p + 2), u3 = __ldg(p + 3), u4 = __ldg(p + 4);
    const uint32_t u5 = __ldg(p + 5), u6 = __ldg(p + 6), u7 = __ldg(p + 7), u8 = __ldg(p + 8);
    const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
    float a0 = (float) (int8_t) ((u0 >> 16) & 0xFFu) * f0.x + (float) (int8_t) (u0 >> 24) * f0.y;
    a0 += dot_u32_f4(u1, make_float4(f0.z, f0.w, f1.x, f1.y));
    a0 += dot_u32_f4(u2, make_float4(f1.z, f1.w, f2.x, f2.y));
    a0 += dot_u32_f4(u3, make_float4(f2.z, f2.w, f3.x, f3.y));
    a0 += dot_u32_f4(u4, make_float4(f3.z, f3.w, f4.x, f4.y));
    a0 += dot_u32_f4(u5, make_float4(f4.z, f4.w, f5.x, f5.y));
    a0 += dot_u32_f4(u6, make_float4(f5.z, f5.w, f6.x, f6.y));
    a0 += dot_u32_f4(u7, make_float4(f6.z, f6.w, f7.x, f7.y));
    a0 += (float) (int8_t) (u8 & 0xFFu) * f7.z + (float) (int8_t) ((u8 >> 8) & 0xFFu) * f7.w;
    d1_out = half_bits_to_float((uint16_t) (u8 >> 16));
    return d0 * a0;
}

__device__ __forceinline__ float q8_blk1_dot(const uint32_t* p, float d1, float4 g0, float4 g1, float4 g2, float4 g3,
                                             float4 g4, float4 g5, float4 g6, float4 g7) {
    return d1 * (dot_u32_f4(__ldg(p + 9), g0) + dot_u32_f4(__ldg(p + 10), g1) + dot_u32_f4(__ldg(p + 11), g2) +
                 dot_u32_f4(__ldg(p + 12), g3) + dot_u32_f4(__ldg(p + 13), g4) + dot_u32_f4(__ldg(p + 14), g5) +
                 dot_u32_f4(__ldg(p + 15), g6) + dot_u32_f4(__ldg(p + 16), g7));
}


// ---- IQ3_XXS / IQ4_XS: one 32-value sub-block against 32 inputs (xv: 8 float4, the padded xs of that sub-block)

__device__ __forceinline__ float sdot4_u8(uint32_t g, uint32_t s, float4 x) {   // bytes of g, sign bit k of s
    return __int_as_float(__float_as_int((float) (g & 0xFFu) * x.x) ^ ((s & 1u) << 31)) +
           __int_as_float(__float_as_int((float) ((g >> 8) & 0xFFu) * x.y) ^ (((s >> 1) & 1u) << 31)) +
           __int_as_float(__float_as_int((float) ((g >> 16) & 0xFFu) * x.z) ^ (((s >> 2) & 1u) << 31)) +
           __int_as_float(__float_as_int((float) (g >> 24) * x.w) ^ (((s >> 3) & 1u) << 31));
}
// block b (98 bytes, 2-aligned), sub-block ib32
__device__ __forceinline__ float iq3xxs_sub_dot(const uint8_t* b, int ib32, const float4 (&xv)[8]) {
    const float d = half_bits_to_float(__ldg((const uint16_t*) b));
    const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib32);
    const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib32);
    const uint32_t aux = (uint32_t) __ldg(sp) | ((uint32_t) __ldg(sp + 1) << 16);
    float a = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = __ldg(qs + l);
        const uint32_t g1 = __ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = __ldg(d_iq3xxs_grid + (qq >> 8));
        const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
        a += sdot4_u8(g1, sg, xv[2 * l]) + sdot4_u8(g2, sg >> 4, xv[2 * l + 1]);
    }
    return d * (0.5f + (float) (aux >> 28)) * 0.5f * a;
}
// IQ4_NL values of the 4 low (hi = false) or high nibbles of the bytes of w, as 4 int8 in a u32
__device__ __forceinline__ uint32_t iq4nl_lut4(uint32_t w, bool hi) {
    const uint32_t t = (hi ? (w >> 4) : w) & 0x0F0F0F0Fu;
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t p = t & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(0xF6EADDCFu, 0xBFAD9881u, p);
    const uint32_t hi4 = __builtin_amdgcn_perm(0x71594535u, 0x26190D01u, p);
    return __builtin_amdgcn_perm(hi4, lo, ((t >> 1) & 0x04040404u) | 0x03020100u);
#else
    const uint32_t x = t | (t >> 4);
    const uint32_t sel = (x & 0xFFu) | ((x >> 8) & 0xFF00u);
    const uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
    const uint32_t lo = __byte_perm(T0, T1, sel & 0x7777u), hv = __byte_perm(T2, T3, sel & 0x7777u);
    const uint32_t m = ((t >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hv & m);
#endif
}
#ifdef STRATA_USE_HIP
#include "ik2_expert_mv_rdna3.cuh"
#endif
// block b (136 bytes, 4-aligned), sub-block ib
__device__ __forceinline__ float iq4xs_sub_dot(const uint8_t* b, int ib, const float4 (&xv)[8]) {
    const uint32_t h = __ldg((const uint32_t*) b), sl = __ldg((const uint32_t*) (b + 4));
    const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
    const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
    const uint32_t* q = (const uint32_t*) (b + 8 + 16 * ib);
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t w = __ldg(q + k);
        a += dot_u32_f4(iq4nl_lut4(w, false), xv[k]) + dot_u32_f4(iq4nl_lut4(w, true), xv[4 + k]);
    }
    return d * (float) (ls - 32) * a;
}
__device__ __forceinline__ void load_xv8(const float* x, float4 (&xv)[8]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) xv[k] = *(const float4*) (x + 4 * k);
}

// Four lanes decode an IQ4_NL block. A lane reads four packed bytes and the two corresponding float4s.
template <bool Pair>
__device__ __forceinline__ void iq4nl_rows(const uint8_t* row0, const uint8_t* row1, const float* xs, int K,
                                          float& acc0, float& acc1) {
    const int lane = threadIdx.x & 31;
    const int word = lane & 3;
    for (int block = lane >> 2; block < K / 32; block += 8) {
        const float* x = xs + block * 36 + 4 * word;
        const float4 lo = *(const float4*) x, hi = *(const float4*) (x + 16);
        const size_t off = (size_t) block * 18;
#pragma unroll
        for (int r = 0; r < (Pair ? 2 : 1); ++r) {
            const uint8_t* b = (r ? row1 : row0) + off;
            uint32_t packed;
            memcpy(&packed, b + 2 + 4 * word, sizeof(packed));
            const float dot = dot_u32_f4(iq4nl_lut4(packed, false), lo) + dot_u32_f4(iq4nl_lut4(packed, true), hi);
            (r ? acc1 : acc0) += h2f(*(const uint16_t*) b) * dot;
        }
    }
}

// a warp's dot of one row of K values (blocks of `type` from w) with xs (shared, padded: xp), every lane gets it
__device__ float row_dot(int type, const uint8_t* w, const float* xs, int K) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (__ldg((const uint32_t*) b) >> (8 * sub)));
            const uint32_t q0 = __ldg((const uint32_t*) (b + 4 + sub * 8));
            const uint32_t q1 = __ldg((const uint32_t*) (b + 8 + sub * 8));
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const float4 x0 = *(const float4*) (x + 0), x1 = *(const float4*) (x + 4);
            const float4 x2 = *(const float4*) (x + 8), x3 = *(const float4*) (x + 12);
            acc += d * nvfp4_sub_dot(q0, q1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const float* x0 = xs + bi * 36;
                const float4 f0 = *(const float4*) (x0 + 0), f1 = *(const float4*) (x0 + 4);
                const float4 f2 = *(const float4*) (x0 + 8), f3 = *(const float4*) (x0 + 12);
                const float4 f4 = *(const float4*) (x0 + 16), f5 = *(const float4*) (x0 + 20);
                const float4 f6 = *(const float4*) (x0 + 24), f7 = *(const float4*) (x0 + 28);
                float d1;
                acc += q8_blk0_dot(p, f0, f1, f2, f3, f4, f5, f6, f7, d1);
                const float4 g0 = *(const float4*) (x0 + 36), g1 = *(const float4*) (x0 + 40);
                const float4 g2 = *(const float4*) (x0 + 44), g3 = *(const float4*) (x0 + 48);
                const float4 g4 = *(const float4*) (x0 + 52), g5 = *(const float4*) (x0 + 56);
                const float4 g6 = *(const float4*) (x0 + 60), g7 = *(const float4*) (x0 + 64);
                acc += q8_blk1_dot(p, d1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                acc += d * (float) (int8_t) __ldg(b + 2 + lane) * xs[xp(bi * 32 + lane)];
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                const float* x0 = xs + bi * 36;
                float a = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) a += (float) (int8_t) __ldg(b + 2 + j) * x0[j];
                acc += d * a;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (__ldg(b + 208) | (__ldg(b + 209) << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const uint8_t* ql = b + n * 64;
                const uint8_t qh = __ldg(b + 128 + n * 32 + lane);
                const uint8_t l0 = __ldg(ql + lane), l1 = __ldg(ql + lane + 32);
                const int is = n * 8 + (lane >> 4);
                const int xn = x0 + n * 128 + lane;
                a += (float) __ldg(sc + is + 0) * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32) * xs[xp(xn)];
                a += (float) __ldg(sc + is + 2) * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32) * xs[xp(xn + 32)];
                a += (float) __ldg(sc + is + 4) * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32) * xs[xp(xn + 64)];
                a += (float) __ldg(sc + is + 6) * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32) * xs[xp(xn + 96)];
            }
            acc += d * a;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) w & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            float d, mn;
            if (hdr4) {
                const uint32_t dm = __ldg((const uint32_t*) b);
                d = half_bits_to_float((uint16_t) (dm & 0xFFFFu));
                mn = half_bits_to_float((uint16_t) (dm >> 16));
            } else {
                d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                mn = half_bits_to_float((uint16_t) (__ldg(b + 2) | (__ldg(b + 3) << 8)));
            }
            const uint8_t* q = b + 4;
            const uint8_t qh = __ldg(b + 16 + lane);
            const int x0 = bi * 256 + lane;
            float a = 0.0f;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = __ldg(b + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    int sc, m;
                    if (is < 4) {
                        sc = __ldg(q + is) & 63, m = __ldg(q + is + 4) & 63;
                    } else {
                        const uint8_t q_hi = __ldg(q + is + 4);
                        sc = (q_hi & 0xF) | ((__ldg(q + is - 4) >> 6) << 4);
                        m = (q_hi >> 4) | ((__ldg(q + is) >> 6) << 4);
                    }
                    const int val = (half ? (lo >> 4) : (lo & 0xF)) + (((qh >> is) & 1) ? 16 : 0);
                    a += (d * (float) sc * (float) val - mn * (float) m) * xs[xp(x0 + 64 * j64 + 32 * half)];
                }
            }
            acc += a;
        }
    } else if (type == kIQ4_NL) {
        float unused = 0.0f;
        iq4nl_rows<false>(w, nullptr, xs, K, acc, unused);
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        for (int sb = lane; sb < K / 32; sb += 32) {
            float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            acc += iq3 ? iq3xxs_sub_dot(w + (size_t) (sb >> 3) * 98, sb & 7, xv)
                       : iq4xs_sub_dot(w + (size_t) (sb >> 3) * 136, sb & 7, xv);
        }
    } else {
        for (int i = lane; i < K; i += 32) acc += decode_value(type, w, (size_t) i) * xs[xp(i)];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// Single-pass dot of two weight rows (w0, w1) of the same quant type against xs (reading xs once for both rows).
__device__ void row_dot_2(int type, const uint8_t* wg, const uint8_t* wu, const float* xs, int K, float& og, float& ou) {
    const int lane = threadIdx.x & 31;
    float ag = 0.0f, au = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const float4 x0 = *(const float4*) (x + 0), x1 = *(const float4*) (x + 4);
            const float4 x2 = *(const float4*) (x + 8), x3 = *(const float4*) (x + 12);
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const float dg = ue4m3((uint8_t) (__ldg((const uint32_t*) bg) >> (8 * sub)));
            const uint32_t gq0 = __ldg((const uint32_t*) (bg + 4 + sub * 8));
            const uint32_t gq1 = __ldg((const uint32_t*) (bg + 8 + sub * 8));
            ag += dg * nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3);

            const uint8_t* bu = wu + blk_off;
            const float du = ue4m3((uint8_t) (__ldg((const uint32_t*) bu) >> (8 * sub)));
            const uint32_t uq0 = __ldg((const uint32_t*) (bu + 4 + sub * 8));
            const uint32_t uq1 = __ldg((const uint32_t*) (bu + 8 + sub * 8));
            au += du * nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) wg & 3) == 0 && ((uintptr_t) wu & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const float* x0 = xs + bi * 36;
                const float4 f0 = *(const float4*) (x0 + 0), f1 = *(const float4*) (x0 + 4);
                const float4 f2 = *(const float4*) (x0 + 8), f3 = *(const float4*) (x0 + 12);
                const float4 f4 = *(const float4*) (x0 + 16), f5 = *(const float4*) (x0 + 20);
                const float4 f6 = *(const float4*) (x0 + 24), f7 = *(const float4*) (x0 + 28);
                const uint32_t* pg = (const uint32_t*) (wg + (size_t) bi * 34);
                const uint32_t* pu = (const uint32_t*) (wu + (size_t) bi * 34);
                float dg1, du1;
                ag += q8_blk0_dot(pg, f0, f1, f2, f3, f4, f5, f6, f7, dg1);
                au += q8_blk0_dot(pu, f0, f1, f2, f3, f4, f5, f6, f7, du1);
                const float4 g0 = *(const float4*) (x0 + 36), g1 = *(const float4*) (x0 + 40);
                const float4 g2 = *(const float4*) (x0 + 44), g3 = *(const float4*) (x0 + 48);
                const float4 g4 = *(const float4*) (x0 + 52), g5 = *(const float4*) (x0 + 56);
                const float4 g6 = *(const float4*) (x0 + 60), g7 = *(const float4*) (x0 + 64);
                ag += q8_blk1_dot(pg, dg1, g0, g1, g2, g3, g4, g5, g6, g7);
                au += q8_blk1_dot(pu, du1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const float xv = xs[xp(bi * 32 + lane)];
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                ag += half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8))) *
                      (float) (int8_t) __ldg(bg + 2 + lane) * xv;
                au += half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8))) *
                      (float) (int8_t) __ldg(bu + 2 + lane) * xv;
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const float* x0 = xs + bi * 36;
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                const float dg = half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8)));
                const float du = half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8)));
                float sg = 0.0f, su = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) {
                    const float xv = x0[j];
                    sg += (float) (int8_t) __ldg(bg + 2 + j) * xv;
                    su += (float) (int8_t) __ldg(bu + 2 + j) * xv;
                }
                ag += dg * sg;
                au += du * su;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (__ldg(bg + 208) | (__ldg(bg + 209) << 8)));
            const float du = half_bits_to_float((uint16_t) (__ldg(bu + 208) | (__ldg(bu + 209) << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg = 0.0f, su = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int xn = x0 + n * 128 + lane;
                const float xv0 = xs[xp(xn)], xv1 = xs[xp(xn + 32)], xv2 = xs[xp(xn + 64)], xv3 = xs[xp(xn + 96)];
                const int is = n * 8 + (lane >> 4);

                const uint8_t* qlg = bg + n * 64;
                const uint8_t qhg = __ldg(bg + 128 + n * 32 + lane);
                const uint8_t lg0 = __ldg(qlg + lane), lg1 = __ldg(qlg + lane + 32);
                sg += (float) __ldg(scg + is + 0) * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32) * xv0;
                sg += (float) __ldg(scg + is + 2) * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32) * xv1;
                sg += (float) __ldg(scg + is + 4) * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32) * xv2;
                sg += (float) __ldg(scg + is + 6) * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32) * xv3;

                const uint8_t* qlu = bu + n * 64;
                const uint8_t qhu = __ldg(bu + 128 + n * 32 + lane);
                const uint8_t lu0 = __ldg(qlu + lane), lu1 = __ldg(qlu + lane + 32);
                su += (float) __ldg(scu + is + 0) * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32) * xv0;
                su += (float) __ldg(scu + is + 2) * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32) * xv1;
                su += (float) __ldg(scu + is + 4) * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32) * xv2;
                su += (float) __ldg(scu + is + 6) * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32) * xv3;
            }
            ag += dg * sg;
            au += du * su;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) wg & 3) == 0) && (((uintptr_t) wu & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            float dg, mng, du, mnu;
            if (hdr4) {
                const uint32_t dmg = __ldg((const uint32_t*) bg);
                const uint32_t dmu = __ldg((const uint32_t*) bu);
                dg = half_bits_to_float((uint16_t) (dmg & 0xFFFFu));
                mng = half_bits_to_float((uint16_t) (dmg >> 16));
                du = half_bits_to_float((uint16_t) (dmu & 0xFFFFu));
                mnu = half_bits_to_float((uint16_t) (dmu >> 16));
            } else {
                dg = half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8)));
                mng = half_bits_to_float((uint16_t) (__ldg(bg + 2) | (__ldg(bg + 3) << 8)));
                du = half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8)));
                mnu = half_bits_to_float((uint16_t) (__ldg(bu + 2) | (__ldg(bu + 3) << 8)));
            }
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = __ldg(bg + 16 + lane), qhu = __ldg(bu + 16 + lane);
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = __ldg(bg + 48 + 32 * j64 + lane);
                const uint8_t lou = __ldg(bu + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    const float xv = xs[xp(x0 + 64 * j64 + 32 * half)];
                    int scg, mg, scu, mu;
                    if (is < 4) {
                        scg = __ldg(qg + is) & 63, mg = __ldg(qg + is + 4) & 63;
                        scu = __ldg(qu + is) & 63, mu = __ldg(qu + is + 4) & 63;
                    } else {
                        const uint8_t qgh = __ldg(qg + is + 4), quh = __ldg(qu + is + 4);
                        scg = (qgh & 0xF) | ((__ldg(qg + is - 4) >> 6) << 4), mg = (qgh >> 4) | ((__ldg(qg + is) >> 6) << 4);
                        scu = (quh & 0xF) | ((__ldg(qu + is - 4) >> 6) << 4), mu = (quh >> 4) | ((__ldg(qu + is) >> 6) << 4);
                    }
                    const int vg = (half ? (log_ >> 4) : (log_ & 0xF)) + (((qhg >> is) & 1) ? 16 : 0);
                    const int vu = (half ? (lou >> 4) : (lou & 0xF)) + (((qhu >> is) & 1) ? 16 : 0);
                    ag += (dg * (float) scg * (float) vg - mng * (float) mg) * xv;
                    au += (du * (float) scu * (float) vu - mnu * (float) mu) * xv;
                }
            }
        }
    } else if (type == kIQ4_NL) {
        iq4nl_rows<true>(wg, wu, xs, K, ag, au);
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        const size_t bb = iq3 ? 98 : 136;
        for (int sb = lane; sb < K / 32; sb += 32) {
            float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            const size_t off = (size_t) (sb >> 3) * bb;
            ag += iq3 ? iq3xxs_sub_dot(wg + off, sb & 7, xv) : iq4xs_sub_dot(wg + off, sb & 7, xv);
            au += iq3 ? iq3xxs_sub_dot(wu + off, sb & 7, xv) : iq4xs_sub_dot(wu + off, sb & 7, xv);
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float xv = xs[xp(i)];
            ag += decode_value(type, wg, (size_t) i) * xv;
            au += decode_value(type, wu, (size_t) i) * xv;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ag += __shfl_xor_sync(0xffffffffu, ag, o);
        au += __shfl_xor_sync(0xffffffffu, au, o);
    }
    og = ag;
    ou = au;
}
__device__ __forceinline__ void row_dot_pair(int type, const uint8_t* wg, const uint8_t* wu, const float* xs, int K,
                                             float& og, float& ou) {
#ifdef STRATA_USE_HIP
    if (type == kQ8_0 && K == 32) {
        // A single 34-byte block can start on a halfword. Give each lane one
        // product and retain cancellation bits until the entire dot is reduced.
        const int lane = threadIdx.x & 31;
        double g = (double)decode_value(kQ8_0, wg, lane) * xs[lane];
        double u = (double)decode_value(kQ8_0, wu, lane) * xs[lane];
#pragma unroll
        for (int d = 16; d; d /= 2) {
            g += __shfl_xor(g, d, 32);
            u += __shfl_xor(u, d, 32);
        }
        og = (float)g;
        ou = (float)u;
        return;
    }
    if (type == kIQ4_NL) {
        float ag = 0.f, au = 0.f;
        iq4nl_rows<true>(wg, wu, xs, K, ag, au);
        og = rdna3_iq4nl_gate_sum<true>(ag, wg, xs, K);
        ou = rdna3_iq4nl_gate_sum<true>(au, wu, xs, K);
        return;
    }
#endif
    row_dot_2(type, wg, wu, xs, K, og, ou);
}

__device__ __forceinline__ void load_xs_f32(const float* x, int K, float* xs) {
    if (((uintptr_t) x & 15) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const float4* x4 = (const float4*) x;
        for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x)
            *(float4*) (xs + xp(i4 << 2)) = __ldg(x4 + i4);
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) xs[xp(i)] = x[i];
    }
}

__device__ __forceinline__ void load_xs_f16(const uint16_t* h, int K, float* xs) {
    if (((uintptr_t) h & 7) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const uint2* h4 = (const uint2*) h;
        for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
            const uint2 u = __ldg(h4 + i4);
            const float2 f0 = u32_to_f2(u.x), f1 = u32_to_f2(u.y);
            *(float4*) (xs + xp(i4 << 2)) = make_float4(f0.x, f0.y, f1.x, f1.y);
        }
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) xs[xp(i)] = h2f(h[i]);
    }
}

size_t mv_row_bytes(int type, int K) {
    return (size_t) (K / block_values(type)) * block_bytes(type);
}

// Two signed-byte limbs per activation, one FP32 scale per sixteen values.
// The byte order follows the GGUF nibbles, so each four-weight lookup feeds
// two dp4a accumulators without reshuffling either set of bytes.
struct NvInput {
    int high[4], low[4];
    float scale;
};
template <bool Gate>
__device__ __forceinline__ void pack_limbs(const float* x, const uint16_t* h, int K, NvInput* packed, int i, int slot,
                                           bool q8) {
    if (i >= K / 16) return;
    const int base = slot * K + 16 * i;
    float vals[16], mx = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        vals[j] = Gate ? x[base + j] : h2f(h[base + j]);
        mx = fmaxf(mx, fabsf(vals[j]));
    }
    const float inv = mx > 0 ? 16256.f / mx : 0;
    NvInput dst;
    dst.scale = mx / 16256.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned hi = 0, lo = 0;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int ix = q8 ? 4 * j + t : 2 * j + (t >> 1) + 8 * (t & 1);
            const int q = __float2int_rn(vals[ix] * inv);
            const int qh = (q + 64) >> 7;
            hi |= (unsigned)(uint8_t)qh << (8 * t);
            lo |= (unsigned)(uint8_t)(q - 128 * qh) << (8 * t);
        }
        dst.high[j] = hi;
        dst.low[j] = lo;
    }
    packed[base / 16] = dst;
}

template <bool Gate> __global__ void nv_input(const float* x, const uint16_t* h, int K, NvInput* packed, bool q8) {
    pack_limbs<Gate>(x, h, K, packed, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y, q8);
}

__device__ __forceinline__ float nv_limb_dot(const uint8_t* row, int sub, const NvInput& x) {
    const uint8_t* b = row + 36 * (sub / 4);
    const uint32_t* words = (const uint32_t*)(b + 4 + 8 * (sub % 4));
    int hi = 0, lo = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t nib = words[j / 2] >> (16 * (j % 2));
        const uint32_t pos = __byte_perm(0x03020100, 0x0c080604, nib);
        const uint32_t neg = __byte_perm(0xfdfeff00, 0xf4f8fafc, nib);
        const int val = __byte_perm(pos, neg, 0x3210 | ((nib & 0x8888) >> 1));
        hi = __dp4a(val, x.high[j], hi);
        lo = __dp4a(val, x.low[j], lo);
    }
    return float(128 * hi + lo) * x.scale * ue4m3(b[sub % 4]);
}
template <int Type>
__device__ __forceinline__ float linear_limb_dot(const uint8_t* row, int sub, const NvInput& x) {
    constexpr int bytes = Type == kIQ4_NL ? 18 : 34;
    const uint8_t* block = row + (sub / 2) * bytes;
    const uint16_t* q = (const uint16_t*)(block + 2 + (Type == kIQ4_NL ? 0 : (sub % 2) * 16));
    int hi = 0, lo = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t four = (uint32_t)q[2 * j] | ((uint32_t)q[2 * j + 1] << 16);
        if constexpr (Type == kIQ4_NL) four = iq4nl_lut4(four, sub & 1);
        hi = __dp4a(static_cast<int>(four), x.high[j], hi);
        lo = __dp4a(static_cast<int>(four), x.low[j], lo);
    }
    return float(128 * hi + lo) * x.scale * h2f(*(const uint16_t*)block);
}

template <bool Gate, int Type>
__global__ void nv_limb_mv(const uint8_t* const* blobs, const NvInput* packed, size_t offset, int C, int F,
                           const float* sg, const float* su, float gcap, float ucap, uint16_t* hidden, float* y) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, slot = blockIdx.y;
    const int K = Gate ? C : F, N = Gate ? F : C;
    const int r = (blockIdx.x * (blockDim.x / 32) + warp) * (Gate ? 1 : 2);
    if (r >= N) return;
    const int stride = Type == kQ8_0 ? K / 32 * 34 : Type == kIQ4_NL ? K / 32 * 18 : K / 64 * 36;
    const uint8_t* row = blobs[slot] + (Gate ? 0 : offset) + (size_t)r * stride;
    const uint8_t* second = row + (Gate ? offset : stride);
    const NvInput* xp = packed + (Gate ? 0 : slot * K / 16);
    float a = 0, b = 0;
    for (int s = lane; s < K / 16; s += 32) {
        const NvInput z = xp[s];
        a += Type == kNVFP4 ? nv_limb_dot(row, s, z) : linear_limb_dot<Type>(row, s, z);
        b += Type == kNVFP4 ? nv_limb_dot(second, s, z) : linear_limb_dot<Type>(second, s, z);
    }
    a = wsum(a);
    b = wsum(b);
    if (lane) return;
    if (Gate) {
        const float g = a * (sg ? sg[slot] : 1.f), u = b * (su ? su[slot] : 1.f);
        hidden[(size_t)slot * F + r] = f2h(gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap));
    } else {
        y[(size_t)slot * C + r] = a;
        y[(size_t)slot * C + r + 1] = b;
    }
}

// Each warp multiplies 16 expert rows by the distinct token inputs in its group.
#ifndef STRATA_USE_HIP
__device__ __forceinline__ uint32_t ik2_fp4pair(uint32_t v, uint32_t scale) {
    const uint32_t mag = __byte_perm(0x42403c00u, 0x4a484644u, (v & 0x0707u) << 4);
    const uint32_t sign = ((v & 8u) << 12) | ((v & 0x800u) << 20);
    uint32_t ret;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(ret) : "r"(mag | sign), "r"(scale));
    return ret;
}
__device__ __forceinline__ uint32_t q8_pair(uint32_t w) {
    const uint32_t u = __byte_perm(w ^ 0x80808080u, 0x64646464u, 0x4140);
    uint32_t f;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(u), "r"(0x64806480u));
    return f;
}
__device__ __forceinline__ uint32_t q8_h2(uint32_t w, uint32_t scale) {
    uint32_t f = q8_pair(w);
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(f) : "r"(f), "r"(scale));
    return f;
}

#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define IK2_HAS_NATIVE_NVFP4
#include "ik2_blackwell.cuh"
#endif
#endif

// counts[e] becomes a cursor in the padded member table after the prefix sum.
__global__ void bw_prefill_groups(const int* top, const float* weights, int slots, int K, int experts,
                                  const uint8_t* const* table, const float* gs, const float* us, const float* ds,
                                  int capacity, const uint8_t** blobs, int* sizes, int* tokens, int* rows,
                                  float* gate, float* up, float* weight, int* inv) {
    extern __shared__ int counts[];
    int* starts = counts + experts;
    const int tid = threadIdx.x;
    for (int e = tid; e < experts; e += blockDim.x) counts[e] = 0;
    for (int g = tid; g < capacity; g += blockDim.x) sizes[g] = 0;
    __syncthreads();
    for (int i = tid; i < slots; i += blockDim.x) atomicAdd(counts + top[i], 1);
    __syncthreads();
    if (!tid) {
        int next = 0;
        for (int e = 0; e < experts; ++e) {
            starts[e] = next;
            next += (counts[e] + kBlackwellPrefillRows - 1) / kBlackwellPrefillRows;
        }
    }
    __syncthreads();
    for (int e = tid; e < experts; e += blockDim.x) {
        for (int m = 0; m < counts[e]; m += kBlackwellPrefillRows) {
            const int g = starts[e] + m / kBlackwellPrefillRows;
            blobs[g] = table[e];
            sizes[g] = min(kBlackwellPrefillRows, counts[e] - m);
        }
        counts[e] = starts[e] * kBlackwellPrefillRows;
    }
    __syncthreads();
    for (int i = tid; i < slots; i += blockDim.x) {
        const int e = top[i], t = i / K;
        const int member = atomicAdd(counts + e, 1);
        tokens[member] = t;
        rows[member] = i;
        gate[i] = gs ? gs[e] : 1.f;
        up[i] = us ? us[e] : 1.f;
        weight[i] = weights[i] * (ds ? ds[e] : 1.f);
        int rank = 0;
        for (int j = 0; j < K; ++j) rank += top[t*K+j] < e;
        inv[t*K + rank] = i;
    }
}

__global__ void bw_pack_rows_kernel(const uint16_t* x, size_t blocks, BlackwellActivation* packed) {
#ifdef IK2_HAS_NATIVE_NVFP4
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 4;
    // All lanes execute the shuffles, including a final partial warp.
    const auto frag = bw_quantize(x + (i < blocks ? i : 0) * 64);
    if (i >= blocks) return;
    bw_store(packed[i], frag);
#endif
}

__device__ __forceinline__ void bw_fp16_block(float (&v)[4], const uint8_t* ap, const uint8_t* bp,
                                               const uint16_t* input) {
    const int sub = threadIdx.x & 3;
    uint32_t sa = f2h(ue4m3(ap[sub])), sb = f2h(ue4m3(bp[sub]));
    sa |= sa << 16;
    sb |= sb << 16;
    uint32_t xx[8];

    const uint32_t* xp = (const uint32_t*)(input + sub * 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) xx[i] = xp[i];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const uint32_t wa = ((const uint32_t*)(ap + 4))[sub * 2 + j];
        const uint32_t wb = ((const uint32_t*)(bp + 4))[sub * 2 + j];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const uint32_t ra = wa >> (16 * m), rb = wb >> (16 * m);
            const uint32_t af[4] = {ik2_fp4pair(ra, sa), ik2_fp4pair(rb, sb), ik2_fp4pair(ra >> 4, sa),
                                    ik2_fp4pair(rb >> 4, sb)};
            mma16816(v, af, xx[2 * j + m], xx[4 + 2 * j + m]);
        }
    }
}

template <bool Gate, int Splits, bool Q8 = false>
__global__ void grouped_expert_mma(const uint8_t* const* blobs, const int* tok, size_t offset, int C, int F,
                                   const uint16_t* h, uint16_t* hidden, float* out, const float* sg, const float* su,
                                   float gcap, float ucap, const int* counts, const int* rows_out,
                                   const uint16_t* input16 = nullptr, bool native_fp4 = false) {
    static constexpr int rows = Gate ? 8 : 16;
    const int lane = threadIdx.x & 31, split = threadIdx.x / 32, grp = lane / 4, sub = lane % 4;
    const int s = blockIdx.y, r = blockIdx.x * rows + grp, K = Gate ? C : F;
    const int count = counts[s];
    if (count <= 0) return;
    const int gi = s * kGroupMax + (grp < count ? grp : 0);
    const int input_row = Gate ? tok[gi] : rows_out[gi];
    const int stride = Q8 ? K / 32 * 34 : K / 64 * 36;
    const uint8_t* a = blobs[s] + (Gate ? 0 : offset) + (size_t)r * stride;
    const uint8_t* b = a + (Gate ? offset : 8 * stride);
    float v[4] = {};
    if constexpr (Q8) {
        for (int k = split; k < K / 32; k += Splits) {
            const uint16_t* ap = (const uint16_t*)(a + k * 34);
            const uint16_t* bp = (const uint16_t*)(b + k * 34);
            const float sa = h2f(ap[0]), sb = h2f(bp[0]);
            const int off = k * 32 + sub * 8;
            uint32_t xx[4];
                const uint32_t* xp = (const uint32_t*)((Gate ? input16 : h) + (size_t)input_row * K + off);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    xx[j] = xp[j];
            ap += 1 + 4 * sub;
            bp += 1 + 4 * sub;
            // Q8's signed integers are exactly representable in FP16. Apply its original
            // per-row block scale after the FP32 dot, avoiding FP16 weight rounding.
            float block_sum[4]{};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t af[4] = {q8_pair(ap[j]), q8_pair(bp[j]), q8_pair(ap[j + 2]), q8_pair(bp[j + 2])};
                mma16816(block_sum, af, xx[j], xx[j + 2]);
            }
            v[0] += sa * block_sum[0];
            v[1] += sa * block_sum[1];
            v[2] += sb * block_sum[2];
            v[3] += sb * block_sum[3];
        }
    } else
        for (int k = split; k < K / 64; k += Splits) {
            const uint8_t* ap = a + k * 36;
            const uint8_t* bp = b + k * 36;
#ifdef IK2_HAS_NATIVE_NVFP4
            // 0xff is outside the serialized UE4M3 range, but the reference
            // decoder accepts it. Preserve its legacy value on that block.
            if (native_fp4 && __all_sync(0xffffffff, ap[sub] != 255 && bp[sub] != 255)) {
                bw_nvfp4_dot(v, bw_weights(ap, bp), bw_quantize((Gate ? input16 : h) + (size_t)input_row * K + k * 64));
                continue;
            }
#endif
            bw_fp16_block(v, ap, bp, (Gate ? input16 : h) + (size_t)input_row * K + k * 64);
        }
    __shared__ float4 partial[Splits][32];
    partial[split][lane] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    if (split) return;
    float4 total = partial[0][lane];
#pragma unroll
    for (int i = 1; i < Splits; ++i) {
        const float4 z = partial[i][lane];
        total.x += z.x;
        total.y += z.y;
        total.z += z.z;
        total.w += z.w;
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int token = sub * 2 + j;
        if (token >= count) continue;
        const int dst = rows_out[s * kGroupMax + token];
        const float a0 = j ? total.y : total.x, a1 = j ? total.w : total.z;
        if (Gate) {
            const float g = a0 * (sg ? sg[dst] : 1.f), u = a1 * (su ? su[dst] : 1.f);
            hidden[(size_t)dst * F + r] =
                f2h(gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap));
        } else {
            out[(size_t)dst * C + r] = a0;
            out[(size_t)dst * C + r + 8] = a1;
        }
    }
}

// Each warp owns separate output channels. Its weight registers feed all M/8 MMA
// fragments before advancing K. Only the final scatter depends on the destination order.
template <bool Gate, int M, int Warps>
__global__ void __launch_bounds__(Warps * 32, M == kGroupMax ? (Gate ? 12 : 10) : 3) bw_tiled_product(
    BlackwellExpertBatch batch, size_t offset, int C, int F, const uint16_t* input,
    const BlackwellActivation* packed, const float* sg, const float* su, float gcap, float ucap,
    uint16_t* hidden, float* y, BlackwellActivation* packed_hidden) {
#ifdef IK2_HAS_NATIVE_NVFP4
    constexpr bool pack_tile = Gate && M == kBlackwellPrefillRows;
    __shared__ uint16_t tile[pack_tile ? M : 1][pack_tile ? 64 : 1];
    constexpr int channels = Gate ? 8 : 16;
    const int lane = threadIdx.x % 32, quad = lane / 4;
    const int r0 = (blockIdx.x * Warps + threadIdx.x / 32) * channels;
    const int group = blockIdx.y, first = group * M;
    const bool contiguous = M != kGroupMax && batch.expert;
    const int count = contiguous ? min(M, batch.size - first) : batch.counts[group];
    if (!count || r0 >= (Gate ? F : C)) return;
    const int blocks = (Gate ? C : F) / 64, r = r0 + quad;
    const auto* expert = contiguous ? batch.expert : batch.groups[group];
    const auto* a = expert + (Gate ? 0 : offset) + size_t(r) * blocks * 36;
    const auto* b = a + (Gate ? offset : size_t(8) * blocks * 36);
    size_t xblock[M / 8];
#pragma unroll
    for (int m = 0; m < M / 8; ++m) {
        const int i = first + min(8 * m + quad, count - 1);
        const int row = Gate ? batch.tokens[i] : contiguous ? i : batch.rows[i];
        xblock[m] = size_t(row) * blocks;
    }
    float v[M / 8][4]{};
    for (int k = 0; k < blocks; ++k) {
        const auto* ap = a + k * 36;
        const auto* bp = b + k * 36;
        auto w = bw_weights(ap, bp);
        if (__any_sync(0xffffffff, w.legacy)) {
#pragma unroll
            for (int m = 0; m < M / 8; ++m)
                bw_fp16_block(v[m], ap, bp, input + 64 * (xblock[m] + k));
        } else {
#pragma unroll
            for (int m = 0; m < M / 8; ++m)
                bw_nvfp4_dot(v[m], w, bw_load(packed[xblock[m] + k]));
        }
    }
#pragma unroll
    for (int m = 0; m < M / 8; ++m) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int member = 8 * m + 2 * (lane % 4) + j;
            if (member >= count) continue;
            const int dst = contiguous ? first + member : batch.rows[first + member];
            if constexpr (Gate) {
                const float g = v[m][j] * (sg ? sg[dst] : 1.f);
                const float u = v[m][j+2] * (su ? su[dst] : 1.f);
                const uint16_t h =
                    f2h(gcap * tanhf(g / gcap) * (1.f / (1.f + expf(-g))) * ucap * tanhf(u / ucap));
                hidden[size_t(dst) * F + r] = h;
                if constexpr (pack_tile)
                    if (packed_hidden) tile[member][r % 64] = h;
            } else {
                y[size_t(dst) * C + r] = v[m][j];
                y[size_t(dst) * C + r + 8] = v[m][j+2];
            }
        }
    }
    if constexpr (pack_tile) {
        // Packed output is requested only for full 64-channel CTAs. Four lanes
        // quantize one row, including its scales, directly from the SwiGLU tile.
        if (packed_hidden) {
            __syncthreads();
            const int member = threadIdx.x / 4;
            if (member < M) {
                const auto frag = bw_quantize(tile[min(member, count - 1)]);
                if (member < count) {
                    const int dst = contiguous ? first + member : batch.rows[first + member];
                    bw_store(packed_hidden[size_t(dst) * (F / 64) + blockIdx.x], frag);
                }
            }
        }
    }
#endif
}

// ---- one token's NVFP4 experts through the tensor cores, straight from their blocks into the registers.
// A warp is 16 rows (gate / up: 8 gate rows and the same 8 of up, so that SwiGLU is a lane's; down: 16 rows) over
// the whole K, no shared memory.  Lane (g, t): sub-block t of rows g and g + 8 of each 64-value block - four MMA a
// block, MMA m the sub-block's bytes 2 m and 2 m + 1 (low nibbles values 2 m, 2 m + 1 in k slots 2 t, 2 t + 1; high
// nibbles values 2 m + 8, 2 m + 9 in slots 2 t + 8, 2 t + 9).  B: the input in FP16 where it is, the same in all
// eight columns.  A code times its sub-block's scale is exact in FP16 (a code is at most 12, ue4m3 at most 240 and
// at least 2^-10); the sum is the tensor cores' FP32.
__device__ __forceinline__ uint32_t ue4m3_h2(uint32_t x) {   // FP16 bits of ue4m3(x) in both halves
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    const uint32_t sub = __byte_perm(0x1a181400u, 0x1f1e1d1cu, m) << 8;   // m 2^-10
    uint32_t h = e ? (((e + 7u) << 10) | (m << 7)) : sub;
    h = (x == 0x7Fu) ? 0u : h;
    return h | (h << 16);
}
__device__ __forceinline__ uint32_t nv_pair_h2(uint32_t s, uint32_t scale) {   // bytes 0 and 1's low nibbles
    const uint32_t mag = __byte_perm(0x42403c00u, 0x4a484644u, (s & 0x0707u) << 4);
    const uint32_t sign = ((s & 0x0808u) * 0x101000u) & 0x80008000u;
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(mag | sign), "r"(scale));
    return r;
}
struct NvStep {   // a lane's part of a 64-value block of its two rows: the sub-block's scales and codes
    uint32_t sa, sb, wa0, wa1, wb0, wb1;
};
__device__ __forceinline__ NvStep nv_step_load(const uint8_t* a, const uint8_t* b, int k, int t) {
    const uint32_t* ap = (const uint32_t*) (a + (size_t) k * 36);
    const uint32_t* bp = (const uint32_t*) (b + (size_t) k * 36);
    NvStep r;
    r.sa = __ldg(ap), r.sb = __ldg(bp);
    r.wa0 = __ldg(ap + 1 + 2 * t), r.wa1 = __ldg(ap + 2 + 2 * t);
    r.wb0 = __ldg(bp + 1 + 2 * t), r.wb1 = __ldg(bp + 2 + 2 * t);
    return r;
}
// A block is a group of 16 rows, its SPLIT warps each a stride of the K blocks (more rows of weights in flight than
// a warp alone keeps: a warp waits for a block's bytes before it asks for more than the next one's).
template <bool Gate, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_kernel(const uint8_t* const* __restrict__ blob, size_t off, int C, int F, const uint16_t* __restrict__ x16,
                   int x_ld, uint16_t* hidden, float* y, const float* sg, const float* su, float gcap, float ucap) {
    __shared__ float2 part[SPLIT][8];
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, g = lane >> 2, t = lane & 3, slot = blockIdx.y;
    const int K = Gate ? C : F;
    const int r = blockIdx.x * (Gate ? 8 : 16) + g;
    const size_t rb = (size_t) (K / 64) * 36;
    const uint8_t* a = blob[slot] + (Gate ? 0 : off) + (size_t) r * rb;
    const uint8_t* b = Gate ? a + off : a + 8 * rb;
    // (gate / up: the input as two FP16 limbs, x = hi + lo, [2][C] - column 0 takes hi, column 1 lo, and a row's
    //  two columns add up to the product with the FP32 input; down: h is FP16 itself, every column the same)
    const uint4* xp = (const uint4*) (x16 + (Gate ? (g == 1 ? (size_t) C : 0) : (size_t) slot * x_ld)) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = K / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        const uint4 x0 = __ldg(xp + k * 8), x1 = __ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {nv_pair_h2(qa, sa), nv_pair_h2(qb, sb), nv_pair_h2(qa >> 4, sa),
                                    nv_pair_h2(qb >> 4, sb)};
            mma16816(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    if (t == 0) part[split][g] = Gate ? make_float2(v[0] + v[1], v[2] + v[3]) : make_float2(v[0], v[2]);
    __syncthreads();
    if (split || t) return;
    float2 tot = part[0][g];
#pragma unroll
    for (int i = 1; i < SPLIT; ++i) tot.x += part[i][g].x, tot.y += part[i][g].y;
    if (Gate) {
        const float gg = tot.x * (sg ? sg[slot] : 1.0f), u = tot.y * (su ? su[slot] : 1.0f);
        hidden[(size_t) slot * F + r] =
            f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(u / ucap));
    } else {
        y[(size_t) slot * C + r] = tot.x;
        y[(size_t) slot * C + r + 8] = tot.y;
    }
}
#endif

template <int WARPS, int ROWS>
__global__ void experts_gate_up_kernel(const uint8_t* const* blob, const int* tok, int ty_gate, int ty_up, size_t eb0,
                                       const float* x, int C, int F, size_t rb_gate, size_t rb_up, float* gu) {
    extern __shared__ float xs[];
    const int s = blockIdx.y;
    load_xs_f32(x + (size_t) tok[s] * C, C, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* bl = blob[s];
    for (int j = 0; j < ROWS; ++j) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;   // of 2 F: gate's then up's
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        const float v = row_dot(up ? ty_up : ty_gate, bl + (up ? eb0 + r * rb_up : r * rb_gate), xs, C);
        if ((threadIdx.x & 31) == 0) gu[(size_t) s * 2 * F + row] = v;
    }
}

template <int WARPS, int ROWS>
__global__ void experts_gate_up_swiglu_kernel(const uint8_t* const* blob, const int* tok, int ty_gate, int ty_up,
                                              size_t eb0, const float* x, int C, int F, size_t rb_gate, size_t rb_up,
                                              const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                                              const int* orow, int h_ld, const int* counts = nullptr) {
    extern __shared__ float xs[];
    const int group = blockIdx.y;
    if (counts && blockIdx.z >= counts[group]) return;
    const int member = counts ? group * kGroupMax + blockIdx.z : group;
    const int s = counts ? orow[member] : group;
    load_xs_f32(x + (size_t) tok[member] * C, C, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* bl = blob[group];
    const float sgs = sg ? sg[s] : 1.0f;
    const float sus = su ? su[s] : 1.0f;
    for (int j = 0; j < ROWS; ++j) {
        const int r = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (r >= F) break;
        float vg, vu;
        if (ty_gate == ty_up) {
            row_dot_pair(ty_gate, bl + (size_t) r * rb_gate, bl + eb0 + (size_t) r * rb_up, xs, C, vg, vu);
        } else {
            vg = row_dot(ty_gate, bl + (size_t) r * rb_gate, xs, C);
            vu = row_dot(ty_up, bl + eb0 + (size_t) r * rb_up, xs, C);
        }
        if ((threadIdx.x & 31) == 0) {
            const float g = vg * sgs, u = vu * sus;
            h[(size_t) (!counts && orow ? orow[s] : s) * h_ld + r] =
                f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
        }
    }
}

template <int WARPS, int ROWS>
__global__ void experts_down_kernel(const uint8_t* const* blob, int ty_down, size_t off_down, const uint16_t* h,
                                    int C, int F, size_t rb_down, float* y, const int* counts = nullptr, const int* rows = nullptr) {
    extern __shared__ float xs[];
    const int group = blockIdx.y;
    if (counts && blockIdx.z >= counts[group]) return;
    const int s = counts ? rows[group * kGroupMax + blockIdx.z] : group;
    load_xs_f16(h + (size_t) s * F, F, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* w = blob[group] + off_down;
    for (int j = 0; j < ROWS;) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (row >= C) break;
        if (j + 1 < ROWS && row + 1 < C) {
            float v0, v1;
            row_dot_2(ty_down, w + (size_t) row * rb_down, w + (size_t) (row + 1) * rb_down, xs, F, v0, v1);
            if ((threadIdx.x & 31) == 0) {
                y[(size_t) s * C + row] = v0;
                y[(size_t) s * C + row + 1] = v1;
            }
            j += 2;
        } else {
            const float v = row_dot(ty_down, w + (size_t) row * rb_down, xs, F);
            if ((threadIdx.x & 31) == 0) y[(size_t) s * C + row] = v;
            j += 1;
        }
    }
}

// ---- grouped: an expert's rows read once for every token that picked it (a verify window's few tokens)

// a warp's dots of one row (blocks of `type`) with n <= N_CAP inputs (each K values, F32 or FP16 rows in global memory)
template <class X>
__device__ __forceinline__ float xval(const X* p, int i);
template <>
__device__ __forceinline__ float xval<float>(const float* p, int i) { return __ldg(p + i); }
template <>
__device__ __forceinline__ float xval<uint16_t>(const uint16_t* p, int i) { return h2f(__ldg(p + i)); }

template <class X>
__device__ __forceinline__ float dot16_vec(const X* p, const float* wv);

template <>
__device__ __forceinline__ float dot16_vec<float>(const float* p, const float* wv) {
    if (((uintptr_t) p & 15) == 0) {
        const float4* p4 = (const float4*) p;
        const float4 v0 = __ldg(p4 + 0), v1 = __ldg(p4 + 1), v2 = __ldg(p4 + 2), v3 = __ldg(p4 + 3);
        return wv[0] * v0.x + wv[1] * v0.y + wv[2] * v0.z + wv[3] * v0.w +
               wv[4] * v1.x + wv[5] * v1.y + wv[6] * v1.z + wv[7] * v1.w +
               wv[8] * v2.x + wv[9] * v2.y + wv[10] * v2.z + wv[11] * v2.w +
               wv[12] * v3.x + wv[13] * v3.y + wv[14] * v3.z + wv[15] * v3.w;
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * __ldg(p + i);
    return a;
}

template <>
__device__ __forceinline__ float dot16_vec<uint16_t>(const uint16_t* p, const float* wv) {
    if (((uintptr_t) p & 15) == 0) {
        const uint4* p4 = (const uint4*) p;
        const uint4 u0 = __ldg(p4 + 0), u1 = __ldg(p4 + 1);
        const float2 f0 = u32_to_f2(u0.x), f1 = u32_to_f2(u0.y), f2 = u32_to_f2(u0.z), f3 = u32_to_f2(u0.w);
        const float2 f4 = u32_to_f2(u1.x), f5 = u32_to_f2(u1.y), f6 = u32_to_f2(u1.z), f7 = u32_to_f2(u1.w);
        return wv[0] * f0.x + wv[1] * f0.y + wv[2] * f1.x + wv[3] * f1.y +
               wv[4] * f2.x + wv[5] * f2.y + wv[6] * f3.x + wv[7] * f3.y +
               wv[8] * f4.x + wv[9] * f4.y + wv[10] * f5.x + wv[11] * f5.y +
               wv[12] * f6.x + wv[13] * f6.y + wv[14] * f7.x + wv[15] * f7.y;
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * h2f(__ldg(p + i));
    return a;
}

__device__ __forceinline__ void decode_nvfp4_wv(uint32_t q0, uint32_t q1, float* wv) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    wv[0] = w01.x, wv[1] = w01.y, wv[2] = w23.x, wv[3] = w23.y;
    wv[4] = w45.x, wv[5] = w45.y, wv[6] = w67.x, wv[7] = w67.y;
    wv[8] = w89.x, wv[9] = w89.y, wv[10] = wAB.x, wv[11] = wAB.y;
    wv[12] = wCD.x, wv[13] = wCD.y, wv[14] = wEF.x, wv[15] = wEF.y;
}

__device__ __forceinline__ void unpack_u32_i8(uint32_t q4, float* wv) {
    wv[0] = (float) (int8_t) (q4 & 0xFFu);
    wv[1] = (float) (int8_t) ((q4 >> 8) & 0xFFu);
    wv[2] = (float) (int8_t) ((q4 >> 16) & 0xFFu);
    wv[3] = (float) (int8_t) (q4 >> 24);
}

template <class X, int N_CAP>
__device__ void row_dot_multi(int type, const uint8_t* w, const X* const* xs, int n, int K, float* out) {
    const int lane = threadIdx.x & 31;
    float acc[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) acc[t] = 0.0f;
#ifndef STRATA_USE_HIP
    // Match iq4nl_rows' lane assignment, block scaling and reduction order.
    if (type == kIQ4_NL) {
        const int word = lane & 3;
        for (int block = lane >> 2; block < K / 32; block += 8) {
            const uint8_t* b = w + (size_t)block * 18;
            uint32_t packed;
            memcpy(&packed, b + 2 + 4 * word, sizeof(packed));
            const uint32_t lo = iq4nl_lut4(packed, false), hi = iq4nl_lut4(packed, true);
            const float scale = h2f(*(const uint16_t*)b);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                const X* x = xs[t] + block * 32 + 4 * word;
                const float4 xl = make_float4(xval(x, 0), xval(x, 1), xval(x, 2), xval(x, 3));
                const float4 xh = make_float4(xval(x, 16), xval(x, 17), xval(x, 18), xval(x, 19));
                acc[t] += scale * (dot_u32_f4(lo, xl) + dot_u32_f4(hi, xh));
            }
        }
    } else
#endif
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
            const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
            float wv[16];
            decode_nvfp4_wv(q0, q1, wv);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * dot16_vec(xs[t] + sb * 16, wv);
            }
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const uint32_t u0 = p[0], u1 = p[1], u2 = p[2], u3 = p[3], u4 = p[4];
                const uint32_t u5 = p[5], u6 = p[6], u7 = p[7], u8 = p[8];
                const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
                float a0[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a0[t] = 0.0f;
                {
                    float wv[16];
                    wv[0] = (float) (int8_t) ((u0 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u0 >> 24);
                    unpack_u32_i8(u1, wv + 2);
                    unpack_u32_i8(u2, wv + 6);
                    unpack_u32_i8(u3, wv + 10);
                    wv[14] = (float) (int8_t) (u4 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u4 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32, wv);
                    }
                    wv[0] = (float) (int8_t) ((u4 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u4 >> 24);
                    unpack_u32_i8(u5, wv + 2);
                    unpack_u32_i8(u6, wv + 6);
                    unpack_u32_i8(u7, wv + 10);
                    wv[14] = (float) (int8_t) (u8 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u8 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32 + 16, wv);
                        acc[t] += d0 * a0[t];
                    }
                }
                const float d1 = half_bits_to_float((uint16_t) (u8 >> 16));
                const uint32_t v0 = p[9], v1 = p[10], v2 = p[11], v3 = p[12];
                const uint32_t v4 = p[13], v5 = p[14], v6 = p[15], v7 = p[16];
                float a1[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a1[t] = 0.0f;
                {
                    float wv[16];
                    unpack_u32_i8(v0, wv);
                    unpack_u32_i8(v1, wv + 4);
                    unpack_u32_i8(v2, wv + 8);
                    unpack_u32_i8(v3, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 32, wv);
                    }
                    unpack_u32_i8(v4, wv);
                    unpack_u32_i8(v5, wv + 4);
                    unpack_u32_i8(v6, wv + 8);
                    unpack_u32_i8(v7, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 48, wv);
                        acc[t] += d1 * a1[t];
                    }
                }
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                const float wv = d * (float) (int8_t) b[2 + lane];
                const int off = bi * 32 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += wv * xval(xs[t], off);
                }
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                for (int h = 0; h < 2; ++h) {
                    float wv[16];
#pragma unroll
                    for (int j = 0; j < 16; ++j) wv[j] = (float) (int8_t) b[2 + h * 16 + j];
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        acc[t] += d * dot16_vec(xs[t] + bi * 32 + h * 16, wv);
                    }
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) a[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const uint8_t* ql = b + nn * 64;
                const uint8_t qh = b[128 + nn * 32 + lane];
                const uint8_t l0 = ql[lane], l1 = ql[lane + 32];
                const int is = nn * 8 + (lane >> 4);
                const float w0 = (float) sc[is + 0] * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                const float w1 = (float) sc[is + 2] * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                const float w2 = (float) sc[is + 4] * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                const float w3 = (float) sc[is + 6] * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    a[t] += w0 * xval(xs[t], xn) + w1 * xval(xs[t], xn + 32) +
                            w2 * xval(xs[t], xn + 64) + w3 * xval(xs[t], xn + 96);
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * a[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
            const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
            const uint8_t* q = b + 4;
            const uint8_t qh = b[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = b[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int sc0, m0, sc1, m1;
                if (is0 < 4) {
                    sc0 = q[is0] & 63, m0 = q[is0 + 4] & 63;
                    sc1 = q[is1] & 63, m1 = q[is1 + 4] & 63;
                } else {
                    sc0 = (q[is0 + 4] & 0xF) | ((q[is0 - 4] >> 6) << 4), m0 = (q[is0 + 4] >> 4) | ((q[is0] >> 6) << 4);
                    sc1 = (q[is1 + 4] & 0xF) | ((q[is1 - 4] >> 6) << 4), m1 = (q[is1 + 4] >> 4) | ((q[is1] >> 6) << 4);
                }
                const float w0 = d * (float) sc0 * (float) ((lo & 0xF) + (((qh >> is0) & 1) ? 16 : 0)) - mn * (float) m0;
                const float w1 = d * (float) sc1 * (float) ((lo >> 4) + (((qh >> is1) & 1) ? 16 : 0)) - mn * (float) m1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += w0 * xval(xs[t], xp0) + w1 * xval(xs[t], xp0 + 32);
                }
            }
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float wv = decode_value(type, w, (size_t) i);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += wv * xval(xs[t], i);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        out[t] = a;
    }
}

// Single-pass grouped (gate, up) dot of the same quant type against xs[0..n-1] (reading each xs[t] once)
template <int N_CAP>
__device__ void row_dot_pair_multi(int type, const uint8_t* wg, const uint8_t* wu, const float* const* xs, int n,
                                   int K, float* og, float* ou) {
    const int lane = threadIdx.x & 31;
    float ag[N_CAP], au[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) ag[t] = 0.0f, au[t] = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const uint8_t* bu = wu + blk_off;
            const float dg = ue4m3((uint8_t) (*(const uint32_t*) bg >> (8 * sub)));
            const float du = ue4m3((uint8_t) (*(const uint32_t*) bu >> (8 * sub)));
            float wvg[16], wvu[16];
            decode_nvfp4_wv(*(const uint32_t*) (bg + 4 + sub * 8), *(const uint32_t*) (bg + 8 + sub * 8), wvg);
            decode_nvfp4_wv(*(const uint32_t*) (bu + 4 + sub * 8), *(const uint32_t*) (bu + 8 + sub * 8), wvu);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                const float* xp_t = xs[t] + sb * 16;
                if (((uintptr_t) xp_t & 15) == 0) {
                    const float4* p4 = (const float4*) xp_t;
                    const float4 v0 = __ldg(p4 + 0), v1 = __ldg(p4 + 1), v2 = __ldg(p4 + 2), v3 = __ldg(p4 + 3);
                    ag[t] += dg * (wvg[0] * v0.x + wvg[1] * v0.y + wvg[2] * v0.z + wvg[3] * v0.w +
                                   wvg[4] * v1.x + wvg[5] * v1.y + wvg[6] * v1.z + wvg[7] * v1.w +
                                   wvg[8] * v2.x + wvg[9] * v2.y + wvg[10] * v2.z + wvg[11] * v2.w +
                                   wvg[12] * v3.x + wvg[13] * v3.y + wvg[14] * v3.z + wvg[15] * v3.w);
                    au[t] += du * (wvu[0] * v0.x + wvu[1] * v0.y + wvu[2] * v0.z + wvu[3] * v0.w +
                                   wvu[4] * v1.x + wvu[5] * v1.y + wvu[6] * v1.z + wvu[7] * v1.w +
                                   wvu[8] * v2.x + wvu[9] * v2.y + wvu[10] * v2.z + wvu[11] * v2.w +
                                   wvu[12] * v3.x + wvu[13] * v3.y + wvu[14] * v3.z + wvu[15] * v3.w);
                } else {
                    float sg = 0.0f, su = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float xv = __ldg(xp_t + i);
                        sg += wvg[i] * xv;
                        su += wvu[i] * xv;
                    }
                    ag[t] += dg * sg;
                    au[t] += du * su;
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (bg[208] | (bg[209] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[208] | (bu[209] << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg[N_CAP], su[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) sg[t] = 0.0f, su[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const int is = nn * 8 + (lane >> 4);
                const uint8_t* qlg = bg + nn * 64;
                const uint8_t qhg = bg[128 + nn * 32 + lane];
                const uint8_t lg0 = qlg[lane], lg1 = qlg[lane + 32];
                const float wg0 = (float) scg[is + 0] * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32);
                const float wg1 = (float) scg[is + 2] * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32);
                const float wg2 = (float) scg[is + 4] * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32);
                const float wg3 = (float) scg[is + 6] * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32);

                const uint8_t* qlu = bu + nn * 64;
                const uint8_t qhu = bu[128 + nn * 32 + lane];
                const uint8_t lu0 = qlu[lane], lu1 = qlu[lane + 32];
                const float wu0 = (float) scu[is + 0] * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32);
                const float wu1 = (float) scu[is + 2] * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32);
                const float wu2 = (float) scu[is + 4] * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32);
                const float wu3 = (float) scu[is + 6] * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32);

                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = __ldg(xs[t] + xn), xv1 = __ldg(xs[t] + xn + 32);
                    const float xv2 = __ldg(xs[t] + xn + 64), xv3 = __ldg(xs[t] + xn + 96);
                    sg[t] += wg0 * xv0 + wg1 * xv1 + wg2 * xv2 + wg3 * xv3;
                    su[t] += wu0 * xv0 + wu1 * xv1 + wu2 * xv2 + wu3 * xv3;
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                ag[t] += dg * sg[t];
                au[t] += du * su[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            const float dg = half_bits_to_float((uint16_t) (bg[0] | (bg[1] << 8)));
            const float mng = half_bits_to_float((uint16_t) (bg[2] | (bg[3] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[0] | (bu[1] << 8)));
            const float mnu = half_bits_to_float((uint16_t) (bu[2] | (bu[3] << 8)));
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = bg[16 + lane], qhu = bu[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = bg[48 + 32 * j64 + lane], lou = bu[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int scg0, mg0, scg1, mg1, scu0, mu0, scu1, mu1;
                if (is0 < 4) {
                    scg0 = qg[is0] & 63, mg0 = qg[is0 + 4] & 63, scg1 = qg[is1] & 63, mg1 = qg[is1 + 4] & 63;
                    scu0 = qu[is0] & 63, mu0 = qu[is0 + 4] & 63, scu1 = qu[is1] & 63, mu1 = qu[is1 + 4] & 63;
                } else {
                    scg0 = (qg[is0 + 4] & 0xF) | ((qg[is0 - 4] >> 6) << 4), mg0 = (qg[is0 + 4] >> 4) | ((qg[is0] >> 6) << 4);
                    scg1 = (qg[is1 + 4] & 0xF) | ((qg[is1 - 4] >> 6) << 4), mg1 = (qg[is1 + 4] >> 4) | ((qg[is1] >> 6) << 4);
                    scu0 = (qu[is0 + 4] & 0xF) | ((qu[is0 - 4] >> 6) << 4), mu0 = (qu[is0 + 4] >> 4) | ((qu[is0] >> 6) << 4);
                    scu1 = (qu[is1 + 4] & 0xF) | ((qu[is1 - 4] >> 6) << 4), mu1 = (qu[is1 + 4] >> 4) | ((qu[is1] >> 6) << 4);
                }
                const float wg0 = dg * (float) scg0 * (float) ((log_ & 0xF) + (((qhg >> is0) & 1) ? 16 : 0)) - mng * (float) mg0;
                const float wg1 = dg * (float) scg1 * (float) ((log_ >> 4) + (((qhg >> is1) & 1) ? 16 : 0)) - mng * (float) mg1;
                const float wu0 = du * (float) scu0 * (float) ((lou & 0xF) + (((qhu >> is0) & 1) ? 16 : 0)) - mnu * (float) mu0;
                const float wu1 = du * (float) scu1 * (float) ((lou >> 4) + (((qhu >> is1) & 1) ? 16 : 0)) - mnu * (float) mu1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = __ldg(xs[t] + xp0), xv1 = __ldg(xs[t] + xp0 + 32);
                    ag[t] += wg0 * xv0 + wg1 * xv1;
                    au[t] += wu0 * xv0 + wu1 * xv1;
                }
            }
        }
    } else {
        row_dot_multi<float, N_CAP>(type, wg, xs, n, K, og);
        row_dot_multi<float, N_CAP>(type, wu, xs, n, K, ou);
        return;
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float g_a = ag[t], u_a = au[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            g_a += __shfl_xor_sync(0xffffffffu, g_a, o);
            u_a += __shfl_xor_sync(0xffffffffu, u_a, o);
        }
        og[t] = g_a;
        ou[t] = u_a;
    }
}

constexpr int kGroupWarps = 4, kGroupRows = 4, kGroupSwigluRows = 2;

template <int N_CAP>
__device__ __forceinline__ void experts_gate_up_g_run(const uint8_t* bl, const int* tok_g, const int* orow_g,
                                                      int ng, int ty_gate, int ty_up, size_t eb0, const float* x,
                                                      int C, int F, size_t rb_gate, size_t rb_up, float* gu) {
    const float* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = threadIdx.x >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row = (blockIdx.x * kGroupWarps + warp) * kGroupRows + j;
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        row_dot_multi<float, N_CAP>(up ? ty_up : ty_gate, bl + (up ? eb0 + r * rb_up : r * rb_gate), xs, ng, C, out);
        if ((threadIdx.x & 31) == 0)
            for (int t = 0; t < ng; ++t) gu[(size_t) orow_g[t] * 2 * F + row] = out[t];
    }
}

template <int N_CAP>
__device__ __forceinline__ void experts_gate_up_swiglu_g_run(const uint8_t* bl, const int* tok_g, const int* orow_g,
                                                             int ng, int ty_gate, int ty_up, size_t eb0,
                                                             const float* x, int C, int F, size_t rb_gate,
                                                             size_t rb_up, const float* sg, const float* su,
                                                             float gcap, float ucap, uint16_t* h) {
    const float* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = threadIdx.x >> 5;
    float og[N_CAP], ou[N_CAP];
    for (int j = 0; j < kGroupSwigluRows; ++j) {
        const int r = (blockIdx.x * kGroupWarps + warp) * kGroupSwigluRows + j;
        if (r >= F) break;
        if (ty_gate == ty_up) {
            row_dot_pair_multi<N_CAP>(ty_gate, bl + (size_t) r * rb_gate, bl + eb0 + (size_t) r * rb_up, xs, ng, C,
                                      og, ou);
        } else {
            row_dot_multi<float, N_CAP>(ty_gate, bl + (size_t) r * rb_gate, xs, ng, C, og);
            row_dot_multi<float, N_CAP>(ty_up, bl + eb0 + (size_t) r * rb_up, xs, ng, C, ou);
        }
        if ((threadIdx.x & 31) == 0) {
            for (int t = 0; t < ng; ++t) {
                const int ro = orow_g[t];
                const float g = og[t] * (sg ? sg[ro] : 1.0f);
                const float u = ou[t] * (su ? su[ro] : 1.0f);
                h[(size_t) ro * F + r] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            }
        }
    }
}

template <int N_CAP>
__device__ __forceinline__ void experts_down_g_run(const uint8_t* w, const int* orow_g, int ng, int ty_down,
                                                   const uint16_t* h, int C, int F, size_t rb_down, float* y) {
    const uint16_t* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = h + (size_t) orow_g[t < ng ? t : 0] * F;
    const int warp = threadIdx.x >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row = (blockIdx.x * kGroupWarps + warp) * kGroupRows + j;
        if (row >= C) break;
        row_dot_multi<uint16_t, N_CAP>(ty_down, w + row * rb_down, xs, ng, F, out);
        if ((threadIdx.x & 31) == 0)
            for (int t = 0; t < ng; ++t) y[(size_t) orow_g[t] * C + row] = out[t];
    }
}

// group g: blob[g], n[g] tokens, their rows of x tok[g][j] and their output rows row[g][j]
__global__ void experts_gate_up_g_kernel(const uint8_t* const* blob, const int* n, const int* tok, const int* orow,
                                         int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F,
                                         size_t rb_gate, size_t rb_up, float* gu) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else if (ng == 2)
        experts_gate_up_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else if (ng <= 4)
        experts_gate_up_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else
        experts_gate_up_g_run<kGroupMax>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
}

__global__ void experts_gate_up_swiglu_g_kernel(const uint8_t* const* blob, const int* n, const int* tok,
                                                const int* orow, int ty_gate, int ty_up, size_t eb0, const float* x,
                                                int C, int F, size_t rb_gate, size_t rb_up, const float* sg,
                                                const float* su, float gcap, float ucap, uint16_t* h) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_swiglu_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else if (ng == 2)
        experts_gate_up_swiglu_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else if (ng <= 4)
        experts_gate_up_swiglu_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else
        experts_gate_up_swiglu_g_run<kGroupMax>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up,
                                                sg, su, gcap, ucap, h);
}

__global__ void experts_down_g_kernel(const uint8_t* const* blob, const int* n, const int* orow, int ty_down,
                                      size_t off_down, const uint16_t* h, int C, int F, size_t rb_down, float* y) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* w = blob[g] + off_down;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_down_g_run<1>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else if (ng == 2)
        experts_down_g_run<2>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else if (ng <= 4)
        experts_down_g_run<4>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else
        experts_down_g_run<kGroupMax>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
}

template <int WARPS, int ROWS>
__global__ void mv_blocks_kernel(int type, const uint8_t* W, int N, int K, size_t rb, const float* x, float* y) {
    extern __shared__ float xs[];
    load_xs_f32(x, K, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    for (int j = 0; j < ROWS; ++j) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (row >= N) break;
        const float v = row_dot(type, W + (size_t) row * rb, xs, K);
        if ((threadIdx.x & 31) == 0) y[row] = v;
    }
}

// one block a head, 4 warps: register Flash-Decode online softmax per warp, 1 SMEM combine

// ---- the resident token path: every expert in VRAM and nothing of a token on the host

// y [N] = W [N][K] FP16 . x [K]: a warp a row (the small dense matrices kept in FP16: sinks, dynamic taps, the router)
template <int WARPS>
__global__ void mv_f16_kernel(const uint16_t* __restrict__ W, int N, int K, const float* __restrict__ x,
                              float* __restrict__ y) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float a = 0.0f;
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];
    a = wsum(a);
    if (lane == 0) y[row] = a;
}

#ifdef STRATA_USE_HIP
// Keep the original 32-lane accumulation order while giving each row its own workgroup.
__device__ __forceinline__ float rdna3_f16_dot_2304(const uint16_t* weights, const float* x, int lane) {
    constexpr int K = 2304, batch = 8;
    const auto* w = weights + lane;
    float acc = 0;
    for (int base = 0; base < K; base += 32 * batch) {
        float a[batch], b[batch];
#pragma unroll
        for (int i = 0; i < batch; ++i) {
            a[i] = h2f(w[base + 32 * i]);
            b[i] = x[base + 32 * i + lane];
        }
#pragma unroll
        for (int i = 0; i < batch; ++i) acc = fmaf(a[i], b[i], acc);
    }
    return wsum(acc);
}
__global__ void rdna3_mv_f16_exact(const uint16_t* weights, const float* x, float* y) {
    const float sum = rdna3_f16_dot_2304(weights + (size_t)blockIdx.x * 2304, x, threadIdx.x);
    if (!threadIdx.x) y[blockIdx.x] = sum;
}

// A small routing or dynamic-convolution matrix has too few rows to fill the card with four rows per block.
__global__ void rdna3_mv_f16_row(const uint16_t* weights, int K, const float* input, float* output) {
    constexpr int threads = 128;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const auto* w = reinterpret_cast<const uint2*>(weights + (size_t)row * K);
    const auto* x = reinterpret_cast<const float4*>(input);
    float sum = 0;
    for (int i = tid; i < K / 4; i += threads) {
        const uint2 halves = w[i];
        const float2 lo = u32_to_f2(halves.x), hi = u32_to_f2(halves.y);
        const float4 v = x[i];
        sum += lo.x * v.x + lo.y * v.y + hi.x * v.z + hi.y * v.w;
    }
    sum = wsum(sum);
    __shared__ float part[threads / 32];
    if (lane == 0) part[tid / 32] = sum;
    __syncthreads();
    if (tid == 0) output[row] = (part[0] + part[1]) + (part[2] + part[3]);
}
#endif

// a product's row through its short conv (conv4_fused_kernel's arithmetic for one token): false if the row has none
__device__ __forceinline__ bool mv_conv_row(const MvConv& cv, int row, float x_new) {
    const int c = row - cv.r0;
    if (c < 0 || c >= cv.n) return false;
    const float h0 = cv.hist[c], h1 = cv.hist[(size_t) cv.n + c], h2 = cv.hist[(size_t) 2 * cv.n + c];
    const float* wc = cv.w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
    if (cv.dyn) {
        const float* dc = cv.dyn + (size_t) (c / (cv.n / cv.heads)) * 4;
        a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
    }
    if (cv.res) a += cv.res[c];
    cv.y[c] = a;
    cv.hist[c] = h1;
    cv.hist[(size_t) cv.n + c] = h2;
    cv.hist[(size_t) 2 * cv.n + c] = x_new;
    return true;
}

// y [N] = W . x for a Q8_0 matrix kept planar - its values q [N][K] int8 and its blocks' scales d [N][K / 32] FP16 -
// so that a warp reads 128 values an instruction; x where it is.  A warp a row; a row of c0's or c1's goes through
// its short conv instead of to y (a launch less a conv: a token's launches each cost 2 us and more).
// Quantized planar Q8 GEMM for the small verify/prefill tiles. One scale covers
// both 16-wide MMA instructions. Packing the signed bytes via 1024+u is exact
// for all 256 inputs; the product rounds to half as in decode_q8p_f16_kernel.
#ifndef STRATA_USE_HIP
template <int Splits, int Panels = 1>
__global__ void q8_mma_tile(const int8_t* q, const uint16_t* scales, int N, int K, const uint16_t* x, int T, float* out,
                            int ldy) {
    const int lane = threadIdx.x % 32, split = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row = 16 * blockIdx.x + g, token = Panels * 8 * blockIdx.y + g;
    const int row0 = min(row, N - 1), row1 = min(row + 8, N - 1);
    const int blocks = K / 32;
    float acc[Panels][4] = {};
    for (int i = split; i < blocks; i += Splits) {
        const int col = i * 32 + t * 8;
        const uint2 a0 = *(const uint2*)(q + (size_t)row0 * K + col);
        const uint2 a1 = *(const uint2*)(q + (size_t)row1 * K + col);
        uint32_t s0 = scales[(size_t)row0 * blocks + i], s1 = scales[(size_t)row1 * blocks + i];
        s0 |= s0 << 16;
        s1 |= s1 << 16;
        const uint32_t lo[4] = {q8_h2(a0.x, s0), q8_h2(a1.x, s1), q8_h2(a0.y, s0), q8_h2(a1.y, s1)};
        const uint32_t hi[4] = {q8_h2(a0.x >> 16, s0), q8_h2(a1.x >> 16, s1),
                                q8_h2(a0.y >> 16, s0), q8_h2(a1.y >> 16, s1)};
#pragma unroll
        for (int p = 0; p < Panels; ++p) {
            const uint4 v = *(const uint4*)(x + size_t(min(token + 8*p, T - 1)) * K + col);
            mma16816(acc[p], lo, v.x, v.z);
            mma16816(acc[p], hi, v.y, v.w);
        }
    }
    __shared__ float4 part[Splits][32];
#pragma unroll
    for (int p = 0; p < Panels; ++p) {
        part[split][lane] = make_float4(acc[p][0], acc[p][1], acc[p][2], acc[p][3]);
        __syncthreads();
        if (!split) {
            float4 v = part[0][lane];
#pragma unroll
            for (int i = 1; i < Splits; ++i) {
                const float4 z = part[i][lane];
                v.x += z.x;
                v.y += z.y;
                v.z += z.z;
                v.w += z.w;
            }
            const int t0 = (blockIdx.y * Panels + p) * 8 + 2 * t;
            if (t0 < T && row < N) out[(size_t)t0 * ldy + row] = v.x;
            if (t0 < T && row + 8 < N) out[(size_t)t0 * ldy + row + 8] = v.z;
            if (t0 + 1 < T && row < N) out[(size_t)(t0 + 1) * ldy + row] = v.y;
            if (t0 + 1 < T && row + 8 < N) out[(size_t)(t0 + 1) * ldy + row + 8] = v.w;
        }
        if constexpr (Panels > 1) __syncthreads();
    }
}
#endif

template <int StaticK, int Unroll = 4>
__device__ __forceinline__ void mv_q8p_row(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, int runtime_k,
                              const float* __restrict__ x, float* y, MvConv c0, MvConv c1, MvAlt a0, MvAlt a1, int row, int lane) {
    const int K = StaticK ? StaticK : runtime_k;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    // (a0's and a1's rows have an input of their own: several products of a layer's fused matrix in one launch)
    const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                        : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                     : x);
    float acc = 0.0f;
#pragma unroll Unroll
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = __ldg(qr + it * 32 + lane);
        const float4 xv = __ldg(x4 + it * 32 + lane);
        const float dv = h2f(__ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float) (int8_t) (w & 0xFFu) * xv.x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv.y +
                     (float) (int8_t) ((w >> 16) & 0xFFu) * xv.z + (float) (int8_t) (w >> 24) * xv.w);
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
// Four sequences share each planar weight load. Each wave reduces one output channel
// in the same lane and iteration order as mv_q8p_row.
template <int Rows>
__global__ void mv_q8p_sequences_kernel(const int8_t* q, const uint16_t* d, int N, int K,
                                        const float* x, int T, float* y, int ldy) {
    const int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= N) return;
    const int first = blockIdx.y * Rows;
    const uint32_t* qr = (const uint32_t*)(q + (size_t)r * K);
    const uint16_t* dr = d + (size_t)r * (K >> 5);
    float acc[Rows] = {};
#pragma unroll 4
    for (int i = 0; i < K / 128; ++i) {
        const uint32_t w = qr[i * 32 + lane];
        const float scale = h2f(dr[i * 4 + (lane >> 3)]);
#pragma unroll
        for (int t = 0; t < Rows; ++t) {
            if (first + t >= T) break;
            const float4 v = ((const float4*)(x + (size_t)(first + t) * K))[i * 32 + lane];
            acc[t] += scale * ((int8_t)w * v.x + (int8_t)(w >> 8) * v.y +
                               (int8_t)(w >> 16) * v.z + (int8_t)(w >> 24) * v.w);
        }
    }
#pragma unroll
    for (int t = 0; t < Rows && first + t < T; ++t) {
        const float sum = wsum(acc[t]);
        if (!lane) y[(size_t)(first + t) * ldy + r] = sum;
    }
}

__global__ void conv_sequences_kernel(const int* slots, const float* x, int ldx, MvConv cv) {
    const int row = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cv.n) return;
    cv.hist += (size_t)slots[row] * 3 * cv.n;
    if (cv.dyn) cv.dyn += row * cv.heads * 4;
    if (cv.res) cv.res += row * cv.n;
    cv.y += row * cv.n;
    mv_conv_row(cv, c, x[(size_t)row * ldx + c]);
}

template <int WARPS, int StaticK = 0, int Unroll = 4>
__global__ void mv_q8p_kernel(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y,
                              MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    mv_q8p_row<StaticK, Unroll>(q, d, N, K, x, y, c0, c1, a0, a1,
                        blockIdx.x * WARPS + (threadIdx.x >> 5), threadIdx.x & 31);
}

#ifdef STRATA_USE_HIP
__global__ void rdna3_attention_projections(const int8_t* q, const uint16_t* d, int N, const float* x, float* y,
                                           MvConv c0, MvConv c1, MvAlt a0, MvAlt a1, MvF16Jobs side) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const MvF16Job& job = side.job[i];
        if (row < job.rows) {
            const float sum = rdna3_f16_dot_2304(job.weights + (size_t)row * 2304, job.input, lane);
            if (!lane) job.output[row] = sum;
            return;
        }
        row -= job.rows;
    }
    mv_q8p_row<2304>(q, d, N, 2304, x, y, c0, c1, a0, a1, row, lane);
}
#endif
__global__ void decode_q8p_f16_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, size_t n,
                                      uint16_t* __restrict__ out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(h2f(d[i >> 5]) * (float) q[i]);
}

// the routed slots' tables from the picks, on the device: slot i's expert e = top[i] - its blocks, its token, its
// scales; wt = the route weight times down's scale
template <bool GroupByExpert>
__global__ void route_gather_kernel(const int* __restrict__ top, const float* __restrict__ w,
                                    const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                                    const float* __restrict__ su_t, const float* __restrict__ sd_t, int S, int K,
                                    const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    const int i = threadIdx.x;
    int ordered = i;
    if constexpr (GroupByExpert) {
        // Run neighboring picks of the same expert together so they can share cached weights.
        // The inverse map restores the original top-K summation order in moe_combine_f32.
        extern __shared__ int counts[];
        if (i < 128) counts[i] = 0;
        __syncthreads();
        const int e = i < S ? top[i] : 0;
        const int within = i < S ? atomicAdd(counts + e, 1) : 0;
        __syncthreads();
        if (i >= S) return;
        ordered = within;
        for (int j = 0; j < e; ++j) ordered += counts[j];
    } else if (i >= S) return;
    const int e = top[i];
    blob[ordered] = tab[e];
    tok[ordered] = i / K;
    inv[i] = ordered;
    sg[ordered] = sg_t ? sg_t[e] : 1.0f;
    su[ordered] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
}
// One token's routing in one launch: route's picks and weights and route_gather's tables, the same values.  A
// thread an expert: its rank is the count of the experts ahead of it (larger sigmoid + bias; a tie: the lower
// expert), the K of rank under K are the picks, and a pick's place among them - ascending experts - is the count of
// picks under it.  (route_warp_kernel finds them by K rounds of one warp's argmax: 10 us of a lone warp's latency.)
__global__ void route_token_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int E, int K,
                                   float scale, const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                                   const float* __restrict__ su_t, const float* __restrict__ sd_t, int* top, float* w,
                                   const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                                   const float* x, int width, NvInput* packed, bool q8, uint16_t* x_limbs, unsigned long long* usage) {
    if (blockIdx.x) {
        const int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
        if (x_limbs) {   // x = hi + lo in FP16, [2][width] (experts_gate_up_swiglu_mma's input)
            if (i < width) {
                const uint16_t hi = f2h(x[i]);
                x_limbs[i] = hi, x_limbs[width + i] = f2h(x[i] - h2f(hi));
            }
        } else
            pack_limbs<true>(x, nullptr, width, packed, i, 0, q8);
        return;
    }
    extern __shared__ float rt_sm[];
    float* val = rt_sm;                                       // [E] sigmoid + bias
    uint32_t* picked = (uint32_t*) (rt_sm + E);               // [warps] a warp's picks, a bit a lane
    float* sig = (float*) (picked + (blockDim.x >> 5));       // [K] the picks' sigmoids in order, [K]: their sum
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float sl = e < E ? 1.0f / (1.0f + expf(-__ldg(logits + e))) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {   // (the sum in route's order: the same weights to the bit)
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        if (usage) ++usage[e];  // Each selected expert has one writer on the routing stream.
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
    }
}

// fast FP16 decodes: NVFP4 a thread a 16-value sub-block (__byte_perm + uint4 stores); Q8_0 8 threads a 32-value block
// (uint2 stores); Q5_K / Q6_K a warp a 256-value block (4 values/lane with coalesced uint2 stores)
__global__ void decode_f16_nvfp4_kernel(const uint8_t* blocks, size_t nsub, uint16_t* out) {
    const size_t sb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsub) return;
    const uint8_t* b = blocks + (sb >> 2) * 36;
    const int sub = (int) (sb & 3);
    const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
    const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    const uint4 o0 = make_uint4(f2_to_u32(w01.x * d, w01.y * d), f2_to_u32(w23.x * d, w23.y * d),
                                f2_to_u32(w45.x * d, w45.y * d), f2_to_u32(w67.x * d, w67.y * d));
    const uint4 o1 = make_uint4(f2_to_u32(w89.x * d, w89.y * d), f2_to_u32(wAB.x * d, wAB.y * d),
                                f2_to_u32(wCD.x * d, wCD.y * d), f2_to_u32(wEF.x * d, wEF.y * d));
    uint4* o = (uint4*) (out + sb * 16);
    o[0] = o0;
    o[1] = o1;
}

__global__ void decode_f16_q8_kernel(const uint8_t* blocks, size_t nblk, uint16_t* out) {
    const size_t tid_g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t bi = tid_g >> 3;
    if (bi >= nblk) return;
    const int sub = (int) (tid_g & 7);
    const uint8_t* b = blocks + bi * 34;
    const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
    const uint8_t* qp = b + 2 + sub * 4;
    int8_t q0, q1, q2, q3;
    if (((uintptr_t) qp & 3) == 0) {
        const uint32_t q4 = *(const uint32_t*) qp;
        q0 = (int8_t) (q4 & 0xFFu);
        q1 = (int8_t) ((q4 >> 8) & 0xFFu);
        q2 = (int8_t) ((q4 >> 16) & 0xFFu);
        q3 = (int8_t) (q4 >> 24);
    } else {
        q0 = (int8_t) qp[0], q1 = (int8_t) qp[1], q2 = (int8_t) qp[2], q3 = (int8_t) qp[3];
    }
    uint16_t* dst = out + bi * 32 + sub * 4;
    if (((uintptr_t) dst & 7) == 0) {
        *(uint2*) dst = make_uint2(f2_to_u32(d * (float) q0, d * (float) q1),
                                   f2_to_u32(d * (float) q2, d * (float) q3));
    } else {
        dst[0] = f2h(d * (float) q0), dst[1] = f2h(d * (float) q1);
        dst[2] = f2h(d * (float) q2), dst[3] = f2h(d * (float) q3);
    }
}

__device__ __forceinline__ void store_f16x4(uint16_t* dst, float v0, float v1, float v2, float v3) {
    if (((uintptr_t) dst & 7) == 0) {
        *(uint2*) dst = make_uint2(f2_to_u32(v0, v1), f2_to_u32(v2, v3));
    } else {
        dst[0] = f2h(v0), dst[1] = f2h(v1), dst[2] = f2h(v2), dst[3] = f2h(v3);
    }
}

// one thread per 32-value sub-block of IQ3_XXS / IQ4_XS -> 32 FP16 (ggml's dequantize_row_iq3_xxs / iq4_xs values)
__global__ void decode_f16_iq_kernel(int type, const uint8_t* blocks, size_t nsub, uint16_t* out) {
    const size_t sb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsub) return;
    const int ib = (int) (sb & 7);
    float y[32];
    if (type == kIQ3_XXS) {
        const uint8_t* b = blocks + (sb >> 3) * 98;
        const float d = half_bits_to_float(__ldg((const uint16_t*) b));
        const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib);
        const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib);
        const uint32_t aux = (uint32_t) __ldg(sp) | ((uint32_t) __ldg(sp + 1) << 16);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qq = __ldg(qs + l);
            const uint32_t g1 = __ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = __ldg(d_iq3xxs_grid + (qq >> 8));
            const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                y[8 * l + j] = db * (float) ((g1 >> (8 * j)) & 0xFFu) * ((sg >> j) & 1u ? -1.0f : 1.0f);
                y[8 * l + 4 + j] = db * (float) ((g2 >> (8 * j)) & 0xFFu) * ((sg >> (4 + j)) & 1u ? -1.0f : 1.0f);
            }
        }
    } else {
        const uint8_t* b = blocks + (sb >> 3) * 136;
        const uint32_t h = __ldg((const uint32_t*) b), sl = __ldg((const uint32_t*) (b + 4));
        const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
        const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* q = b + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = __ldg(q + j);
            y[j] = dl * (float) d_iq4nl[v & 0xF];
            y[16 + j] = dl * (float) d_iq4nl[v >> 4];
        }
    }
    uint32_t* o = (uint32_t*) (out + sb * 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = (uint32_t) f2h(y[2 * j]) | ((uint32_t) f2h(y[2 * j + 1]) << 16);
}

__global__ void decode_f16_k_kernel(int type, const uint8_t* blocks, size_t nblk, uint16_t* out) {
    const size_t bi = (size_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (bi >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int gr = lane >> 3, l0 = (lane & 7) * 4;
    uint16_t* y = out + bi * 256;
    if (type == kQ6_K) {
        const uint8_t* b = blocks + bi * 210;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const int8_t* sc = (const int8_t*) (b + 192);
        const int qd = gr;
        const int sh = 2 * qd;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float d_sc = d * (float) sc[n * 8 + (l0 >> 4) + 2 * qd];
            const uint8_t* qlp = b + n * 64 + ((qd & 1) ? 32 : 0) + l0;
            const uint8_t* qhp = b + 128 + n * 32 + l0;
            uint8_t lo[4], qh[4];
            if (((uintptr_t) qlp & 1) == 0) {
                const uint16_t lo0 = *(const uint16_t*) qlp, lo1 = *(const uint16_t*) (qlp + 2);
                const uint16_t qh0 = *(const uint16_t*) qhp, qh1 = *(const uint16_t*) (qhp + 2);
                lo[0] = (uint8_t) lo0, lo[1] = (uint8_t) (lo0 >> 8);
                lo[2] = (uint8_t) lo1, lo[3] = (uint8_t) (lo1 >> 8);
                qh[0] = (uint8_t) qh0, qh[1] = (uint8_t) (qh0 >> 8);
                qh[2] = (uint8_t) qh1, qh[3] = (uint8_t) (qh1 >> 8);
            } else {
                for (int k = 0; k < 4; ++k) lo[k] = qlp[k], qh[k] = qhp[k];
            }
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int nib = qd >= 2 ? (lo[k] >> 4) : (lo[k] & 0xF);
                const int q = (int) (nib | (((qh[k] >> sh) & 3) << 4)) - 32;
                f[k] = d_sc * (float) q;
            }
            store_f16x4(y + n * 128 + qd * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    } else {   // Q5_K
        const uint8_t* b = blocks + bi * 176;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const uint8_t* qhp = b + 16 + l0;
        const uint32_t qh4 = (((uintptr_t) qhp & 3) == 0)
                                 ? *(const uint32_t*) qhp
                                 : ((uint32_t) qhp[0] | ((uint32_t) qhp[1] << 8) | ((uint32_t) qhp[2] << 16) |
                                    ((uint32_t) qhp[3] << 24));
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int is = pass * 4 + gr, j64 = is >> 1, half = is & 1;
            int sc, m;
            if (is < 4) {
                sc = q[is] & 63, m = q[is + 4] & 63;
            } else {
                sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
            }
            const float dsc = d * (float) sc, mnm = mn * (float) m;
            const uint8_t* lop = b + 48 + 32 * j64 + l0;
            const uint32_t lo4 = (((uintptr_t) lop & 3) == 0)
                                     ? *(const uint32_t*) lop
                                     : ((uint32_t) lop[0] | ((uint32_t) lop[1] << 8) | ((uint32_t) lop[2] << 16) |
                                        ((uint32_t) lop[3] << 24));
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t lo_k = (lo4 >> (8 * k)) & 0xFFu;
                const uint32_t qh_k = (qh4 >> (8 * k)) & 0xFFu;
                const int val = (half ? (lo_k >> 4) : (lo_k & 0xF)) + (((qh_k >> is) & 1) ? 16 : 0);
                f[k] = dsc * (float) val - mnm;
            }
            store_f16x4(y + is * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    }
}

__global__ void decode_f16_kernel(int type, const uint8_t* blocks, size_t n, uint16_t* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(decode_value(type, blocks, i));
}
__global__ void decode_f32_kernel(int type, const uint8_t* blocks, size_t n, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = decode_value(type, blocks, i);
}

// y [T][N] = W [N][K] (FP16) . x [T][K]: mv_f16_kernel a token a grid row
template <int WARPS>
__global__ void mv_f16_rows_kernel(const uint16_t* __restrict__ W, int N, int K, const float* __restrict__ x, int x_ld,
                                   float* __restrict__ y, int y_ld) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    const float* xt = x + (size_t) t * x_ld;
    float a = 0.0f;
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * xt[i];
    a = wsum(a);
    if (lane == 0) y[(size_t) t * y_ld + row] = a;
}

template <int TFixed>
__global__ void mv_f16_rows_vec_kernel(const uint16_t* __restrict__ W, int N, int K, const float* __restrict__ x,
                                       int x_ld, float* __restrict__ y, int y_ld) {
    constexpr int WARPS = 4;
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float acc[TFixed] = {};
    for (int i = lane * 4; i < K; i += 128) {
        const uint2 wh = *(const uint2*)(w + i);
        const float2 w01 = u32_to_f2(wh.x), w23 = u32_to_f2(wh.y);
#pragma unroll
        for (int t = 0; t < TFixed; ++t) {
            const float4 xv = *(const float4*)(x + (size_t) t * x_ld + i);
            acc[t] = fmaf(w01.x, xv.x, fmaf(w01.y, xv.y, fmaf(w23.x, xv.z, fmaf(w23.y, xv.w, acc[t]))));
        }
    }
#pragma unroll
    for (int t = 0; t < TFixed; ++t) {
        const float a = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_ld + row] = a;
    }
}

// x [T][C] as two FP16 limbs a token, out [T][2][C] (hi, lo; x = hi + lo)
__global__ void f16_limbs_rows_kernel(const float* __restrict__ x, int T, int C, uint16_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int t = i / C, c = i % C;
    const uint16_t hi = f2h(x[i]);
    out[(size_t) t * 2 * C + c] = hi, out[(size_t) t * 2 * C + C + c] = f2h(x[i] - h2f(hi));
}

// The first pick reserves consecutive chunks for its expert. Slot order within each chunk
// and the original router order used by moe_combine_f32 are both preserved.
template<int Capacity>
__global__ void group_route_kernel(const int* __restrict__ top, const float* __restrict__ w, int S, int K,
                                   const uint8_t* const* __restrict__ tab, const float* sg_t, const float* su_t,
                                   const float* sd_t, const uint8_t** blob, int* n, int* tok, int* row, float* sg,
                                   float* su, float* wt, int* inv) {
    __shared__ int ex[Capacity], chunks[Capacity], gid[Capacity];
    const int i = threadIdx.x;
    const int e = i < S ? top[i] : -1;
    ex[i] = e;
    if (i < S) n[i] = 0;
    __syncthreads();
    int first = i, m = 0, cnt = 0;
    if (i < S) {
        for (int j = 0; j < S; ++j)
            if (ex[j] == e) {
                if (j < first) first = j;
                if (j < i) ++m;
                ++cnt;
            }
    }
    chunks[i] = first == i ? (cnt + kGroupMax - 1) / kGroupMax : 0;
    __syncthreads();
    if (chunks[i]) {
        int gx = 0;
        for (int j = 0; j < i; ++j) gx += chunks[j];
        gid[i] = gx;
    }
    __syncthreads();
    if (i >= S) return;
    const int g = gid[first] + m / kGroupMax;
    const int member = m % kGroupMax;
    tok[g * kGroupMax + member] = i / K, row[g * kGroupMax + member] = i;
    if (!member) blob[g] = tab[e], n[g] = min(cnt - m, kGroupMax);
    sg[i] = sg_t ? sg_t[e] : 1.0f, su[i] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
    inv[i] = i;
}

#ifndef STRATA_USE_HIP
// A group's NVFP4 experts on the tensor cores (experts_mma_kernel for the tokens of a group): gate | up, a block 8
// gate rows and the same 8 of up, SPLIT warps each a stride of the 64-value blocks; the B columns are (member,
// limb) pairs - column c the member c / 2's limb c % 2 of its input (limbs [T][2][C]) - NT tiles of 8 (members <=
// 4 NT), so that the members' products read the expert's rows once.  h [slot][F] = SwiGLU of the member's rows.
template <int SPLIT, int NT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_gu_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n_g, const int* __restrict__ tok_g,
                        const int* __restrict__ row_g, size_t off_up, int C, int F, const uint16_t* __restrict__ limbs,
                        const float* __restrict__ sg, const float* __restrict__ su, float gcap, float ucap, uint16_t* h) {
    __shared__ float4 part[SPLIT][NT][32];
    const int grp = blockIdx.y, n = n_g[grp];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t) (C / 64) * 36;
    const uint8_t* a = blob[grp] + (size_t) r * rb;
    const uint8_t* b = a + off_up;
    const uint4* xp[NT];
    bool x_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] = (const uint4*) (limbs + ((size_t) tk * 2 + limb) * C) + 2 * t;
    }
    float v[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint32_t af[4][4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            af[m][0] = nv_pair_h2(qa, sa), af[m][1] = nv_pair_h2(qb, sb);
            af[m][2] = nv_pair_h2(qa >> 4, sa), af[m][3] = nv_pair_h2(qb >> 4, sb);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
            if (x_ok[j]) x0 = __ldg(xp[j] + k * 8), x1 = __ldg(xp[j] + k * 8 + 1);
            const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int m = 0; m < 4; ++m) mma16816(v[j], af[m], xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) part[split][j][lane] = make_float4(v[j][0] + v[j][1], 0, v[j][2] + v[j][3], 0);
    __syncthreads();
    if (split) return;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        float4 tot = part[0][j][lane];
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            const float4 z = part[s][j][lane];
            tot.x += z.x, tot.z += z.z;
        }
        const int mem = j * 4 + t;   // (the lane's columns 2 t, 2 t + 1 of tile j: member 4 j + t's two limbs)
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = tot.x * (sg ? sg[slot] : 1.0f), u = tot.z * (su ? su[slot] : 1.0f);
            h[(size_t) slot * F + r] = f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(u / ucap));
        }
    }
}
// ... down: a block 16 rows, the B columns a group's members (<= 8, their FP16 h rows); y [slot][C]
template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_down_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n_g,
                          const int* __restrict__ row_g, size_t off_down, int C, int F, const uint16_t* __restrict__ h,
                          float* y) {
    __shared__ float4 part[SPLIT][32];
    const int grp = blockIdx.y, n = n_g[grp];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t) (F / 64) * 36;
    const uint8_t* a = blob[grp] + off_down + (size_t) r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const uint4* xp = (const uint4*) (h + (size_t) (x_ok ? row_g[grp * kGroupMax + gr] : 0) * F) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
        if (x_ok) x0 = __ldg(xp + k * 8), x1 = __ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {nv_pair_h2(qa, sa), nv_pair_h2(qb, sb), nv_pair_h2(qa >> 4, sa),
                                    nv_pair_h2(qb >> 4, sb)};
            mma16816(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    part[split][lane] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    if (split) return;
    float4 tot = part[0][lane];
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const float4 z = part[s][lane];
        tot.x += z.x, tot.y += z.y, tot.z += z.z, tot.w += z.w;
    }
    // the lane's columns 2 t, 2 t + 1: members 2 t, 2 t + 1; rows r and r + 8
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        y[(size_t) slot * C + r] = tot.x, y[(size_t) slot * C + r + 8] = tot.z;
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        y[(size_t) slot * C + r] = tot.y, y[(size_t) slot * C + r + 8] = tot.w;
    }
}
#endif


// route() for a window, a block of E threads a token (route_token_kernel's picks: the same weights to the bit)
__global__ void route_rows_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int E, int K,
                                  float scale, int* top, float* w) {
    extern __shared__ float rr_sm[];
    float* val = rr_sm;
    uint32_t* picked = (uint32_t*) (rr_sm + E);
    float* sig = (float*) (picked + (blockDim.x >> 5));
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float* lg = logits + (size_t) t * E;
    const float sl = e < E ? 1.0f / (1.0f + expf(-__ldg(lg + e))) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
    }
    __syncthreads();
    if (pick) top[(size_t) t * K + pos] = e, w[(size_t) t * K + pos] = sl / (sig[K] + 1e-20f) * scale;
}

// a lane's part of a Q8_0 block of its two rows (34 bytes: the FP16 scale, 32 int8): the scales, and the values
// 8 t .. 8 t + 7 as four pairs (2-byte loads: a block is 2-byte aligned)
#ifndef STRATA_USE_HIP
struct LinearBlockStep {
    uint32_t sa, sb;
    uint32_t a[4], b[4];
};
template <bool IQ4>
__device__ __forceinline__ LinearBlockStep block_step_load(const uint8_t* a, const uint8_t* b, int k, int t) {
    constexpr int bytes = IQ4 ? 18 : 34;
    const uint16_t* ap = (const uint16_t*) (a + (size_t) k * bytes);
    const uint16_t* bp = (const uint16_t*) (b + (size_t) k * bytes);
    LinearBlockStep r;
    r.sa = __ldg(ap), r.sb = __ldg(bp);
    if constexpr (IQ4) {
        // A lane consumes eight consecutive values. Low and high nibbles span 16 values each.
        ap += 1 + 4 * (t & 1);
        bp += 1 + 4 * (t & 1);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t av = iq4nl_lut4(__ldg(ap + 2 * i) | (uint32_t(__ldg(ap + 2 * i + 1)) << 16), t >= 2);
            const uint32_t bv = iq4nl_lut4(__ldg(bp + 2 * i) | (uint32_t(__ldg(bp + 2 * i + 1)) << 16), t >= 2);
            r.a[2 * i] = av, r.a[2 * i + 1] = av >> 16;
            r.b[2 * i] = bv, r.b[2 * i + 1] = bv >> 16;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) r.a[i] = __ldg(ap + 1 + 4 * t + i), r.b[i] = __ldg(bp + 1 + 4 * t + i);
    }
    return r;
}
// Each 32-value block contributes unscaled integer products, then its FP16 row scale in FP32.
template <int SPLIT, int NT, bool IQ4>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_gu_linear_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n_g, const int* __restrict__ tok_g,
                           const int* __restrict__ row_g, size_t off_up, int C, int F, const uint16_t* __restrict__ limbs,
                           const float* __restrict__ sg, const float* __restrict__ su, float gcap, float ucap,
                           uint16_t* h) {
    __shared__ float4 part[SPLIT][NT][32];
    const int grp = blockIdx.y, n = n_g[grp];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t) (C / 32) * (IQ4 ? 18 : 34);
    const uint8_t* a = blob[grp] + (size_t) r * rb;
    const uint8_t* b = a + off_up;
    const uint4* xp[NT];
    bool x_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] = (const uint4*) (limbs + ((size_t) tk * 2 + limb) * C) + t;
    }
    float v[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 32;
    LinearBlockStep cur{};
    if (split < nk) cur = block_step_load<IQ4>(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        LinearBlockStep nxt = cur;
        if (k + SPLIT < nk) nxt = block_step_load<IQ4>(a, b, k + SPLIT, t);
        const float sa = h2f((uint16_t) cur.sa), sb = h2f((uint16_t) cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint4 x = make_uint4(0, 0, 0, 0);
            if (x_ok[j]) x = __ldg(xp[j] + k * 4);
            float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816(bs, af0, x.x, x.y);
            mma16816(bs, af1, x.z, x.w);
            v[j][0] += sa * bs[0], v[j][1] += sa * bs[1], v[j][2] += sb * bs[2], v[j][3] += sb * bs[3];
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) part[split][j][lane] = make_float4(v[j][0], v[j][1], v[j][2], v[j][3]);
    __syncthreads();
    if (split) return;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        float4 tot = part[0][j][lane];
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            const float4 z = part[s][j][lane];
            tot.x += z.x, tot.y += z.y, tot.z += z.z, tot.w += z.w;
        }
        const int mem = j * 4 + t;
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = (tot.x + tot.y) * (sg ? sg[slot] : 1.0f), u = (tot.z + tot.w) * (su ? su[slot] : 1.0f);
            h[(size_t) slot * F + r] = f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(u / ucap));
        }
    }
}
template <int SPLIT, bool IQ4>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_down_linear_kernel(const uint8_t* const* __restrict__ blob, const int* __restrict__ n_g,
                             const int* __restrict__ row_g, size_t off_down, int C, int F,
                             const uint16_t* __restrict__ h, float* y) {
    __shared__ float4 part[SPLIT][32];
    const int grp = blockIdx.y, n = n_g[grp];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t) (F / 32) * (IQ4 ? 18 : 34);
    const uint8_t* a = blob[grp] + off_down + (size_t) r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const uint4* xp = (const uint4*) (h + (size_t) (x_ok ? row_g[grp * kGroupMax + gr] : 0) * F) + t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 32;
    LinearBlockStep cur{};
    if (split < nk) cur = block_step_load<IQ4>(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        LinearBlockStep nxt = cur;
        if (k + SPLIT < nk) nxt = block_step_load<IQ4>(a, b, k + SPLIT, t);
        const float sa = h2f((uint16_t) cur.sa), sb = h2f((uint16_t) cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        uint4 x = make_uint4(0, 0, 0, 0);
        if (x_ok) x = __ldg(xp + k * 4);
        float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma16816(bs, af0, x.x, x.y);
        mma16816(bs, af1, x.z, x.w);
        v[0] += sa * bs[0], v[1] += sa * bs[1], v[2] += sb * bs[2], v[3] += sb * bs[3];
        cur = nxt;
    }
    part[split][lane] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    if (split) return;
    float4 tot = part[0][lane];
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const float4 z = part[s][lane];
        tot.x += z.x, tot.y += z.y, tot.z += z.z, tot.w += z.w;
    }
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        y[(size_t) slot * C + r] = tot.x, y[(size_t) slot * C + r + 8] = tot.z;
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        y[(size_t) slot * C + r] = tot.y, y[(size_t) slot * C + r + 8] = tot.w;
    }
}
#endif

}  // namespace

void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s) {
    if (T <= 0) return;
#ifdef STRATA_USE_HIP
    if (rdna3_route_rows(logits, bias, T, E, K, scale, top, w, s)) return;
#endif
    if (K < 1 || K > kRouteMax) {
        std::fprintf(stderr, "ik2::k::route: K %d\n", K);
        std::abort();
    }
    if (E <= 512) {
        route_warp_kernel<<<T, 32, 0, s>>>(logits, bias, T, E, K, scale, top, w);
        check("route_warp");
        return;
    }
    route_kernel<<<T, 128, (size_t) E * 4, s>>>(logits, bias, T, E, K, scale, top, w);
    check("route");
}

void gather_f16(const float* x, const int* idx, int n, int C, uint16_t* y, cudaStream_t s) {
    if (n <= 0) return;
    gather_f16_kernel<<<grid((size_t) n * C), kBlk, 0, s>>>(x, idx, n, C, y);
    check("gather_f16");
}

void swiglu_cap(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                uint16_t* h, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_cap_kernel<<<grid((size_t) n * F), kBlk, 0, s>>>(gu, n, F, sg, su, gcap, ucap, h);
    check("swiglu_cap");
}

void moe_combine(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                 cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_kernel<<<grid((size_t) T * C), kBlk, 0, s>>>(y, inv, wt, T, K, C, out, add);
    check("moe_combine");
}

void swiglu_cap_f32(const float* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                    uint16_t* h, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_cap_f32_kernel<<<grid((size_t) n * F), kBlk, 0, s>>>(gu, n, F, sg, su, gcap, ucap, h);
    check("swiglu_cap_f32");
}
void moe_combine_f32(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                     cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_f32_kernel<<<grid((size_t) T * C), kBlk, 0, s>>>(y, inv, wt, T, K, C, out, add);
    check("moe_combine_f32");
}
void moe_combine_conv(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                      const float* res, float* out, cudaStream_t s) {
    moe_combine_conv_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(y, inv, wt, K, C, hist, w, res, out);
    check("moe_combine_conv");
}

int block_values(int type) {
    return type == kNVFP4 ? 64 : (type == kQ8_0 || type == kIQ4_NL) ? 32
         : type == kQ5_K || type == kQ6_K || type == kIQ3_XXS || type == kIQ4_XS ? 256 : 0;
}
size_t block_bytes(int type) {
    return type == kNVFP4 ? 36 : type == kQ8_0 ? 34 : type == kQ5_K ? 176 : type == kQ6_K ? 210
         : type == kIQ4_NL ? 18 : type == kIQ3_XXS ? 98 : type == kIQ4_XS ? 136 : 0;
}

template <int WARPS, int ROWS>
void launch_experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, float* gu, cudaStream_t s) {
    const dim3 g((unsigned) ((2 * F + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_gate_up_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(C) * 4, s>>>(
        blob, tok, ty_gate, ty_up, eb0, x, C, F, mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), gu);
}
template <int WARPS, int ROWS>
void launch_experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C,
                         int F, float* y, cudaStream_t s) {
    const dim3 g((unsigned) ((C + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_down_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(F) * 4, s>>>(
        blob, ty_down, off_down, h, C, F, mv_row_bytes(ty_down, F), y);
}

void expert_limb_mv(bool gate, const uint8_t* const* blob, size_t offset, int C, int F, int S, const float* x,
                    uint16_t* h, float* y, const float* sg, const float* su, float gcap, float ucap, void* scratch,
                    cudaStream_t stream, int type, bool packed) {
    const bool ordered = type != kNVFP4;
    auto* p = (NvInput*)scratch;
    if (gate) {
        if (!packed) nv_input<true><<<(C / 16 + 31) / 32, 32, 0, stream>>>(x, nullptr, C, p, ordered);
        if (type == kQ8_0)
            nv_limb_mv<true, kQ8_0>
                <<<dim3((F + 3) / 4, S), 128, 0, stream>>>(blob, p, offset, C, F, sg, su, gcap, ucap, h, nullptr);
        else if (type == kIQ4_NL)
            nv_limb_mv<true, kIQ4_NL>
                <<<dim3((F + 3) / 4, S), 128, 0, stream>>>(blob, p, offset, C, F, sg, su, gcap, ucap, h, nullptr);
        else
            nv_limb_mv<true, kNVFP4>
                <<<dim3((F + 3) / 4, S), 128, 0, stream>>>(blob, p, offset, C, F, sg, su, gcap, ucap, h, nullptr);
    } else {
        nv_input<false><<<dim3((F / 16 + 31) / 32, S), 32, 0, stream>>>(nullptr, h, F, p, ordered);
        if (type == kQ8_0)
            nv_limb_mv<false, kQ8_0>
                <<<dim3((C + 7) / 8, S), 128, 0, stream>>>(blob, p, offset, C, F, nullptr, nullptr, 0, 0, nullptr, y);
        else if (type == kIQ4_NL)
            nv_limb_mv<false, kIQ4_NL>
                <<<dim3((C + 7) / 8, S), 128, 0, stream>>>(blob, p, offset, C, F, nullptr, nullptr, 0, 0, nullptr, y);
        else
            nv_limb_mv<false, kNVFP4>
                <<<dim3((C + 7) / 8, S), 128, 0, stream>>>(blob, p, offset, C, F, nullptr, nullptr, 0, 0, nullptr, y);
    }
    check("nv_expert_limb");
}
bool experts_mma_fits(int ty_gate, int ty_up, int ty_down, int C, int F) {
    return ty_gate == kNVFP4 && ty_up == kNVFP4 && ty_down == kNVFP4 && C % 64 == 0 && F % 64 == 0;
}
#ifndef STRATA_USE_HIP
static int experts_mma_split(bool gate) {
    static const int wg = getenv("IK2_EXPERT_MMA_SPLIT") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT")) : 8;
    static const int wd = getenv("IK2_EXPERT_MMA_SPLIT_DOWN") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT_DOWN")) : 4;
    return gate ? wg : wd;
}
template <bool Gate>
static void launch_experts_mma(const uint8_t* const* blob, int S, size_t off, const uint16_t* x16, int x_ld, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                               cudaStream_t s) {
    const int w = experts_mma_split(Gate);
    const dim3 g((unsigned) (Gate ? F / 8 : C / 16), (unsigned) S);
#define IK2_MMA_LAUNCH(W) experts_mma_kernel<Gate, W><<<g, W * 32, 0, s>>>(blob, off, C, F, x16, x_ld, h, y, sg, su, gcap, ucap)
    if (w == 1) IK2_MMA_LAUNCH(1);
    else if (w == 2) IK2_MMA_LAUNCH(2);
    else if (w == 3) IK2_MMA_LAUNCH(3);
    else if (w == 6) IK2_MMA_LAUNCH(6);
    else if (w == 8) IK2_MMA_LAUNCH(8);
    else IK2_MMA_LAUNCH(4);
#undef IK2_MMA_LAUNCH
}
void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int S, size_t eb0, const uint16_t* x16, int C, int F,
                                const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                                cudaStream_t s) {
    if (S <= 0) return;
    launch_experts_mma<true>(blob, S, eb0, x16, 0, C, F, sg, su, gcap, ucap, h, nullptr, s);
    check("experts_gate_up_swiglu_mma");
}
void experts_down_mma(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                      cudaStream_t s) {
    if (S <= 0) return;
    launch_experts_mma<false>(blob, S, off_down, h, F, C, F, nullptr, nullptr, 0.0f, 0.0f, nullptr, y, s);
    check("experts_down_mma");
}
#endif
void experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                     const float* x, int C, int F, float* gu, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 2) launch_experts_gate_up<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    else launch_experts_gate_up<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    check("experts_gate_up");
}
void experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F,
                  float* y, cudaStream_t s) {

    if (S <= 0) return;
#ifdef STRATA_USE_HIP
    static const bool nvfp4_mv = !getenv("IK2_NVFP4_MV") || atoi(getenv("IK2_NVFP4_MV"));
    if (nvfp4_mv && ty_down == kNVFP4 && F % 64 == 0 && ((uintptr_t)h & 15) == 0) {
        rdna3_nvfp4_token_down(blob, S, off_down, h, C, F, y, s);
        return;
    }
    if (try_rdna3_expert<false>(ty_down, blob, S, off_down, nullptr, nullptr, h, F, C, nullptr, nullptr,
                                0, 0, nullptr, y, s)) return;
#endif
    if (S <= 4) launch_experts_down<4, 2>(blob, S, ty_down, off_down, h, C, F, y, s);
    else launch_experts_down<4, 4>(blob, S, ty_down, off_down, h, C, F, y, s);
    check("experts_down");
}
void experts_gate_up_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int ty_gate,
                       int ty_up, size_t eb0, const float* x, int C, int F, float* gu, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned) ((2 * F + kGroupWarps * kGroupRows - 1) / (kGroupWarps * kGroupRows)), (unsigned) G);
    experts_gate_up_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                                                            mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), gu);
    check("experts_gate_up_g");
}
// The serving engine can retain the original short-prompt arithmetic and enable
// the grouped tensor-core path only in a window whose tokens are being verified.
thread_local bool grouped_tensor_products = true;
#ifndef STRATA_USE_HIP
bool use_blackwell_nvfp4() {
    static const bool enabled = getenv("IK2_BLACKWELL_NVFP4") && atoi(getenv("IK2_BLACKWELL_NVFP4"));
    if (!enabled) return false;
    int dev, major, minor;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static thread_local int last_device = -1;
    static thread_local bool available = false;
    if (dev == last_device) return available;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) return false;
    last_device = dev;
    // A binary built with older virtual PTX can run on Blackwell via JIT,
    // but its device pass omitted the architecture-specific MMA instructions.
    cudaFuncAttributes attr{};
    available = major == 12 && minor <= 1 &&
                cudaFuncGetAttributes(&attr, bw_pack_rows_kernel) == cudaSuccess &&
                attr.ptxVersion == 10 * major + minor;
    return available;
}
int blackwell_group_capacity(int slots, int experts) {
    const int distinct = std::min(slots, experts);
    return distinct + (slots - distinct) / kBlackwellPrefillRows;
}

size_t blackwell_group_bytes(int groups) {
    return size_t(groups) * (sizeof(const uint8_t*) + (1 + 2 * kBlackwellPrefillRows) * sizeof(int));
}

BlackwellExpertBatch blackwell_prefill_groups(const int* top, const float* weights, int slots, int K, int experts,
                                              const uint8_t* const* table, const float* gs, const float* us,
                                              const float* ds, void* scratch, float* gate, float* up, float* weight,
                                              int* inv, cudaStream_t s) {
    const int groups = blackwell_group_capacity(slots, experts);
    auto** blobs = static_cast<const uint8_t**>(scratch);
    int* counts = reinterpret_cast<int*>(blobs + groups);
    int* tokens = counts + groups;
    int* rows = tokens + groups * kBlackwellPrefillRows;
    bw_prefill_groups<<<1, 256, 2 * experts * sizeof(int), s>>>(top, weights, slots, K, experts,
        table, gs, us, ds, groups, blobs, counts, tokens, rows, gate, up, weight, inv);
    check("bw_prefill_groups");
    return {nullptr, blobs, counts, tokens, rows, groups, true};
}

void blackwell_pack_rows(const uint16_t* x, int rows, int K, BlackwellActivation* packed, cudaStream_t s) {
    const size_t blocks = size_t(rows) * (K / 64);
    bw_pack_rows_kernel<<<(blocks + 31) / 32, 128, 0, s>>>(x, blocks, packed);
    check("bw_pack_rows");
}

template <bool Gate, int M>
static void launch_bw_product(BlackwellExpertBatch batch, size_t offset, int C, int F,
                              const uint16_t* input, const BlackwellActivation* packed,
                              const float* sg, const float* su, float gcap, float ucap,
                              uint16_t* hidden, float* y, cudaStream_t s, BlackwellActivation* packed_hidden) {
    constexpr int warps = M == kGroupMax ? 4 : 8;
    constexpr int N = warps * (Gate ? 8 : 16);
    const int width = Gate ? F : C;
    const int groups = batch.expert ? (batch.size + M - 1) / M : batch.size;
    bw_tiled_product<Gate, M, warps><<<dim3((width + N - 1) / N, groups), warps * 32, 0, s>>>(
        batch, offset, C, F, input, packed, sg, su, gcap, ucap, hidden, y, packed_hidden);
    check("bw_tiled_product");
}

void blackwell_expert_product(bool gate, BlackwellExpertBatch batch, size_t offset, int C, int F,
                              const uint16_t* input, const BlackwellActivation* packed,
                              const float* sg, const float* su, float gcap, float ucap,
                              uint16_t* hidden, float* y, cudaStream_t s, BlackwellActivation* packed_hidden) {
    const auto launch = batch.prefill
        ? (gate ? launch_bw_product<true, kBlackwellPrefillRows> : launch_bw_product<false, kBlackwellPrefillRows>)
        : (gate ? launch_bw_product<true, kGroupMax> : launch_bw_product<false, kGroupMax>);
    launch(batch, offset, C, F, input, packed, sg, su, gcap, ucap, hidden, y, s, packed_hidden);
}
int block_nvfp4_mma_mask() {
    static const int mask = getenv("IK2_GROUP_MMA") ? atoi(getenv("IK2_GROUP_MMA")) : 0;
    if (!grouped_tensor_products) return 0;
    // Small speculative-verify blocks (T <= 8) stay memory-bandwidth bound on GB10 and
    // run faster with higher acceptance on the FP16 block path unless explicitly opted in.
    return mask & 3;
}
#endif
void enable_grouped_tensor_products(bool enabled) { grouped_tensor_products = enabled; }
#ifndef STRATA_USE_HIP
// The last four Mini layers use Q8. Run the token kernels on group members to keep
// their padding, dot products and rounding identical to independent decode.
void experts_q8_verify(const uint8_t* const* blob, const int* counts, const int* tokens, const int* rows,
                       int groups, int T, size_t up, size_t down, const float* x, int C, int F,
                       const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                       cudaStream_t stream) {
    const int members = std::min(kGroupMax, T);
    experts_gate_up_swiglu_kernel<4, 2><<<dim3((F+7)/8, groups, members), 128, xp_floats(C)*4, stream>>>(
        blob, tokens, kQ8_0, kQ8_0, up, x, C, F, mv_row_bytes(kQ8_0, C), mv_row_bytes(kQ8_0, C),
        sg, su, gcap, ucap, h, rows, F, counts);
    experts_down_kernel<4, 4><<<dim3((C+15)/16, groups, members), 128, xp_floats(F)*4, stream>>>(
        blob, kQ8_0, down, h, C, F, mv_row_bytes(kQ8_0, F), y, counts, rows);
    check("grouped Q8 verify");
}
#endif

void experts_down_g(const uint8_t* const* blob, const int* n, const int* row, int G, int ty_down, size_t off_down,
                    const uint16_t* h, int C, int F, float* y, cudaStream_t s) {
    if (G <= 0) return;
#ifndef STRATA_USE_HIP
    static const bool use_group_mma = getenv("IK2_GROUP_MMA") && (atoi(getenv("IK2_GROUP_MMA")) & 2);
    const bool native_fp4 = use_blackwell_nvfp4();
    if (grouped_tensor_products && (use_group_mma || native_fp4) && ty_down == kNVFP4 && C % 16 == 0 && F % 64 == 0) {
        grouped_expert_mma<false, 4><<<dim3(C / 16, G), 128, 0, s>>>(blob, nullptr, off_down, C, F, h, nullptr, y,
                                                                     nullptr, nullptr, 0, 0, n, row, nullptr, native_fp4);
        check("grouped_expert_mma (grouped)");
        return;
    }

    if (grouped_tensor_products && use_group_mma && ty_down == kQ8_0 && C % 16 == 0 && F % 32 == 0) {
        grouped_expert_mma<false, 4, true><<<dim3(C / 16, G), 128, 0, s>>>(blob, nullptr, off_down, C, F, h, nullptr, y,
                                                                           nullptr, nullptr, 0, 0, n, row);
        check("q8 grouped MMA");
        return;
    }
#endif
    const dim3 g((unsigned) ((C + kGroupWarps * kGroupRows - 1) / (kGroupWarps * kGroupRows)), (unsigned) G);
    experts_down_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, row, ty_down, off_down, h, C, F,
                                                         mv_row_bytes(ty_down, F), y);
    check("experts_down_g");
}

template <int WARPS, int ROWS>
void launch_experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                   size_t eb0, const float* x, int C, int F, const float* sg, const float* su,
                                   float gcap, float ucap, uint16_t* h, cudaStream_t s, const int* orow = nullptr,
                                   int h_ld = 0) {
    const dim3 g((unsigned) ((F + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_gate_up_swiglu_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(C) * 4, s>>>(
        blob, tok, ty_gate, ty_up, eb0, x, C, F, mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), sg, su, gcap,
        ucap, h, orow, h_ld > 0 ? h_ld : F);
}

void experts_gate_up_swiglu_rows(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                 size_t eb0, const float* x, int C, int F, const float* sg, const float* su, float gcap,
                                 float ucap, uint16_t* h, const int* orow, int h_ld, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 8) launch_experts_gate_up_swiglu<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    else launch_experts_gate_up_swiglu<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    check("experts_gate_up_swiglu_rows");
}
void experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                            uint16_t* h, cudaStream_t s) {

    if (S <= 0) return;
#ifdef STRATA_USE_HIP
    static const bool nvfp4_mv = !getenv("IK2_NVFP4_MV") || atoi(getenv("IK2_NVFP4_MV"));
    if (nvfp4_mv && ty_gate == kNVFP4 && ty_up == kNVFP4 && C % 64 == 0 && ((uintptr_t)x & 15) == 0) {
        rdna3_nvfp4_token_gu(blob, tok, S, eb0, x, C, F, sg, su, gcap, ucap, h, s);
        return;
    }
    if (ty_gate == ty_up && try_rdna3_expert<true>(ty_gate, blob, S, eb0, tok, x, nullptr, C, F, sg, su,
                                                  gcap, ucap, h, nullptr, s)) return;
#endif
    if (S <= 8) launch_experts_gate_up_swiglu<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    else launch_experts_gate_up_swiglu<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    check("experts_gate_up_swiglu");
}

void experts_gate_up_swiglu_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                              int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                              const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s,
                              const uint16_t* input16) {
    if (G <= 0) return;
#ifndef STRATA_USE_HIP
    static const bool use_group_mma = getenv("IK2_GROUP_MMA") && (atoi(getenv("IK2_GROUP_MMA")) & 1);
    const bool native_fp4 = use_blackwell_nvfp4();
    if (grouped_tensor_products && (use_group_mma || native_fp4) && input16 && ty_gate == kNVFP4 && ty_up == kNVFP4 && F % 8 == 0 && C % 64 == 0) {
        grouped_expert_mma<true, 4><<<dim3(F / 8, G), 128, 0, s>>>(blob, tok, eb0, C, F, nullptr, h, nullptr, sg, su,
                                                                   gcap, ucap, n, row, input16, native_fp4);
        check("grouped_expert_mma (grouped)");
        return;
    }

    if (grouped_tensor_products && use_group_mma && input16 && ty_gate == kQ8_0 && ty_up == kQ8_0 && F % 8 == 0 && C % 32 == 0) {
        grouped_expert_mma<true, 4, true><<<dim3(F / 8, G), 128, 0, s>>>(blob, tok, eb0, C, F, nullptr, h, nullptr, sg,
                                                                         su, gcap, ucap, n, row, input16);
        check("q8 grouped MMA");
        return;
    }
#endif
    const dim3 g((unsigned) ((F + kGroupWarps * kGroupSwigluRows - 1) / (kGroupWarps * kGroupSwigluRows)), (unsigned) G);
    experts_gate_up_swiglu_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                                                                   mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C),
                                                                   sg, su, gcap, ucap, h);
    check("experts_gate_up_swiglu_g");
}

template <int WARPS, int ROWS>
void launch_mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    mv_blocks_kernel<WARPS, ROWS><<<(unsigned) ((N + WARPS * ROWS - 1) / (WARPS * ROWS)), WARPS * 32,
                                    xp_floats(K) * 4, s>>>(type, W, N, K, mv_row_bytes(type, K), x, y);
}

void mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    if (N <= 0) return;
    if (K > 8192 || !block_values(type) || K % block_values(type)) {
        std::fprintf(stderr, "ik2::k::mv_blocks: type %d, K %d\n", type, K);
        std::abort();
    }
    if (N <= 512) launch_mv_blocks<2, 1>(type, W, N, K, x, y, s);
    else if (N <= 2048) launch_mv_blocks<4, 1>(type, W, N, K, x, y, s);
    else launch_mv_blocks<4, 2>(type, W, N, K, x, y, s);
    check("mv_blocks");
}

#ifndef STRATA_USE_HIP
void q8_tile(const int8_t* q, const uint16_t* d, int N, int K, const uint16_t* x, int T, float* y, int ldy,
             cudaStream_t s) {

    const unsigned blocks = (N + 15) / 16;
    if (T > 32 && T <= 64) q8_mma_tile<4, 8><<<blocks, 128, 0, s>>>(q, d, N, K, x, T, y, ldy);
    else if (T > 16 && T <= 32) q8_mma_tile<4, 4><<<blocks, 128, 0, s>>>(q, d, N, K, x, T, y, ldy);
    else if (T > 8 && T <= 16) q8_mma_tile<4, 2><<<blocks, 128, 0, s>>>(q, d, N, K, x, T, y, ldy);
    else q8_mma_tile<4><<<dim3(blocks, (T + 7) / 8), 128, 0, s>>>(q, d, N, K, x, T, y, ldy);
    check("q8_mma_tile");
}

#endif
void mv_q8p_sequences(const int8_t* q, const uint16_t* d, int N, int K, const float* x, int rows,
                      float* y, int ldy, cudaStream_t s) {
    mv_q8p_sequences_kernel<4><<<dim3((N + 3) / 4, (rows + 3) / 4), 128, 0, s>>>(q, d, N, K, x, rows, y, ldy);
    check("mv_q8p_sequences");
}

void conv_sequences(const int* slots, int rows, const float* x, int ldx, MvConv cv, cudaStream_t s) {
    conv_sequences_kernel<<<dim3((cv.n + 127) / 128, rows), 128, 0, s>>>(slots, x, ldx, cv);
    check("conv_sequences");
}

void mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1, const MvF16Jobs& side) {
    if (N <= 0) return;
    if (K % 128 || ((uintptr_t) q & 3) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    const int extras = side.job[0].rows + side.job[1].rows + side.job[2].rows;
    if (extras) {
#ifdef STRATA_USE_HIP
        if (K == 2304) {
            rdna3_attention_projections<<<(extras + N + 3) / 4, 128, 0, s>>>(q, d, N, x, y, c0, c1, a0, a1, side);
            check("attention projections");
            return;
        }
#endif
        for (const MvF16Job& job : side.job)
            if (job.rows) mv_f16(job.weights, job.rows, K, job.input, job.output, s);
    }
    static const bool specialize = getenv("IK2_Q8_UNROLL") && atoi(getenv("IK2_Q8_UNROLL"));
    if (specialize && K == 2304)
        mv_q8p_kernel<4, 2304><<<(unsigned)((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1);
    else if (specialize && K == 3072) {
#ifdef STRATA_USE_HIP
        mv_q8p_kernel<8, 3072, 8><<<(unsigned)((N + 7) / 8), 256, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1);
#else
        mv_q8p_kernel<4, 3072><<<(unsigned)((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1);
#endif
    } else
        mv_q8p_kernel<4><<<(unsigned)((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1);
    check("mv_q8p");
}
void mv_q8p(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, cudaStream_t s) {
    mv_q8p_conv(q, d, N, K, x, y, MvConv{}, MvConv{}, s);
}
void decode_q8p_f16(const int8_t* q, const uint16_t* d, size_t n, uint16_t* out, cudaStream_t s) {
    if (!n) return;
    decode_q8p_f16_kernel<<<grid(n), kBlk, 0, s>>>(q, d, n, out);
    check("decode_q8p_f16");
}

void mv_f16(const uint16_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    if (N <= 0) return;
#ifdef STRATA_USE_HIP
    static const int mode = getenv("IK2_F16_MV") ? atoi(getenv("IK2_F16_MV")) : 0;
    if (mode == 2 && K == 2304 && (N == 32 || N == 128)) {
        rdna3_mv_f16_exact<<<N, 32, 0, s>>>(W, x, y);
        check("rdna3_mv_f16_exact");
        return;
    }
    if (mode == 1 && (N == 32 || N == 128) && K % 4 == 0 && ((uintptr_t)W & 7) == 0 && ((uintptr_t)x & 15) == 0) {
        rdna3_mv_f16_row<<<N, 128, 0, s>>>(W, K, x, y);
        check("rdna3_mv_f16_row");
        return;
    }
#else
    if ((K & 127) == 0 && ((uintptr_t)W & 7) == 0 && ((uintptr_t)x & 15) == 0) {
        mv_f16_rows_vec_kernel<1><<<(unsigned)((N + 3) / 4), 128, 0, s>>>(W, N, K, x, K, y, N);
        check("mv_f16_vec");
        return;
    }
#endif
    mv_f16_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(W, N, K, x, y);
    check("mv_f16");
}

void route_gather(const int* top, const float* w, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                  const float* sd_t, int S, int K, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
                  float* wt, cudaStream_t s, bool group) {
    if (S <= 0) return;
    if (S > 1024) {
        std::fprintf(stderr, "ik2::k::route_gather: %d slots\n", S);
        std::abort();
    }
    const unsigned threads = (S + 31) / 32 * 32;
    if (group)
        route_gather_kernel<true><<<1, std::max(128u, threads), 128 * sizeof(int), s>>>(
            top, w, tab, sg_t, su_t, sd_t, S, K, blob, tok, inv, sg, su, wt);
    else
        route_gather_kernel<false><<<1, threads, 0, s>>>(top, w, tab, sg_t, su_t, sd_t, S, K, blob, tok, inv,
                                                        sg, su, wt);
    check("route_gather");
}
__global__ void count_routes_kernel(const int* top, int picks, unsigned long long* usage) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < picks) atomicAdd(usage + top[i], 1ULL);
}

void count_routes(const int* top, int picks, unsigned long long* usage, cudaStream_t stream) {
    count_routes_kernel<<<(picks + 255) / 256, 256, 0, stream>>>(top, picks, usage);
    check("count_routes");
}

void route_token(const float* logits, const float* bias, int E, int K, float scale, const uint8_t* const* tab,
                 const float* sg_t, const float* su_t, const float* sd_t, int* top, float* w, const uint8_t** blob,
                 int* tok, int* inv, float* sg, float* su, float* wt, cudaStream_t s, const float* x, int width,
                 void* packed, bool q8, uint16_t* x_limbs, unsigned long long* usage) {
#ifdef STRATA_USE_HIP
    if (E == 128 && K == 8 && !packed && !x_limbs) {
        const RouteGather gather{tab, sg_t, su_t, sd_t, blob, tok, inv, sg, su, wt};
        rdna3_route_eight<true><<<1, 128, 0, s>>>(logits, bias, scale, top, w, gather);
        check("rdna3_route_token");
        if (usage) count_routes(top, K, usage, s);
        return;
    }
#endif
    if (E > 1024 || K > E) {   // (more experts than a block's threads: the two kernels)
        if (packed) nv_input<true><<<(width / 16 + 31) / 32, 32, 0, s>>>(x, nullptr, width, (NvInput*)packed, q8);
        if (x_limbs) to_f16_limbs(x, (size_t) width, x_limbs, s);
        route(logits, bias, 1, E, K, scale, top, w, s);
        route_gather(top, w, tab, sg_t, su_t, sd_t, K, K, blob, tok, inv, sg, su, wt, s);
        if (usage) count_routes(top, K, usage, s);
        return;
    }
    const unsigned nt = (unsigned) ((E + 31) / 32 * 32);
    const unsigned blocks = 1 + (x_limbs ? (width + nt - 1) / nt : packed ? (width / 16 + nt - 1) / nt : 0);
    route_token_kernel<<<blocks, nt, (size_t)(E + nt / 32 + K + 1) * sizeof(float), s>>>(
        logits, bias, E, K, scale, tab, sg_t, su_t, sd_t, top, w, blob, tok, inv, sg, su, wt, x, width,
        (NvInput*)packed, q8, x_limbs, usage);
    check("route_token");
}

void decode_f16(int type, const uint8_t* blocks, size_t n, uint16_t* out, cudaStream_t s) {
    if (!n) return;
    if (type == kNVFP4 && n % 16 == 0 && (((uintptr_t) blocks & 3) == 0) && (((uintptr_t) out & 15) == 0)) {
        decode_f16_nvfp4_kernel<<<grid(n / 16), kBlk, 0, s>>>(blocks, n / 16, out);
        check("decode_f16_nvfp4");
        return;
    }
    if (type == kQ8_0 && n % 32 == 0) {
        decode_f16_q8_kernel<<<grid(n / 32, kBlk / 8), kBlk, 0, s>>>(blocks, n / 32, out);
        check("decode_f16_q8");
        return;
    }
    if ((type == kIQ3_XXS || type == kIQ4_XS) && n % 256 == 0 && (((uintptr_t) out & 3) == 0) &&
        (((uintptr_t) blocks & (type == kIQ4_XS ? 3 : 1)) == 0)) {
        decode_f16_iq_kernel<<<grid(n / 32), kBlk, 0, s>>>(type, blocks, n / 32, out);
        check("decode_f16_iq");
        return;
    }
    if ((type == kQ5_K || type == kQ6_K) && n % 256 == 0) {
        decode_f16_k_kernel<<<grid(n / 256, kBlk / 32), kBlk, 0, s>>>(type, blocks, n / 256, out);
        check("decode_f16_k");
        return;
    }
    decode_f16_kernel<<<grid(n), kBlk, 0, s>>>(type, blocks, n, out);
    check("decode_f16");
}
void decode_f32(int type, const uint8_t* blocks, size_t n, float* out, cudaStream_t s) {
    if (!n) return;
    decode_f32_kernel<<<grid(n), kBlk, 0, s>>>(type, blocks, n, out);
    check("decode_f32");
}
void decode_host(int type, const uint8_t* blocks, size_t n, float* out) {
    for (size_t i = 0; i < n; ++i) out[i] = decode_value(type, blocks, i);
}

void mv_f16_rows(const uint16_t* W, int N, int K, const float* x, int x_ld, int T, float* y, int y_ld,
                 cudaStream_t s) {
    if (N <= 0 || T <= 0) return;
    if ((K & 127) == 0 && (x_ld & 3) == 0 && ((uintptr_t)W & 7) == 0 && ((uintptr_t)x & 15) == 0) {
        const unsigned blocks = (unsigned) ((N + 3) / 4);
        for (int off = 0; off < T; off += 8) {
            const int chunk = std::min(8, T - off);
            const float* xp = x + (size_t) off * x_ld;
            float* yp = y + (size_t) off * y_ld;
            switch (chunk) {
                case 1: mv_f16_rows_vec_kernel<1><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 2: mv_f16_rows_vec_kernel<2><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 3: mv_f16_rows_vec_kernel<3><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 4: mv_f16_rows_vec_kernel<4><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 5: mv_f16_rows_vec_kernel<5><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 6: mv_f16_rows_vec_kernel<6><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 7: mv_f16_rows_vec_kernel<7><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
                case 8: mv_f16_rows_vec_kernel<8><<<blocks, 128, 0, s>>>(W, N, K, xp, x_ld, yp, y_ld); break;
            }
        }
        check("mv_f16_rows_vec");
        return;
    }
    mv_f16_rows_kernel<4><<<dim3((unsigned) ((N + 3) / 4), (unsigned) T), 128, 0, s>>>(W, N, K, x, x_ld, y, y_ld);
    check("mv_f16_rows");
}
void f16_limbs_rows(const float* x, int T, int C, uint16_t* out, cudaStream_t s) {
    if (T <= 0) return;
    f16_limbs_rows_kernel<<<(unsigned) ((T * C + 255) / 256), 256, 0, s>>>(x, T, C, out);
    check("f16_limbs_rows");
}
void group_route(const int* top, const float* w, int S, int K, const uint8_t* const* tab, const float* sg_t,
                 const float* su_t, const float* sd_t, const uint8_t** blob, int* n, int* tok, int* row, float* sg,
                 float* su, float* wt, int* inv, cudaStream_t s) {
    if (S <= 0 || S > kGroupRouteSlots) {
        std::fprintf(stderr, "ik2::k::group_route: %d slots (1 .. %d)\n", S, kGroupRouteSlots);
        std::abort();
    }
    const int threads = ((S + 31) / 32) * 32;
    if (S <= 256)
        group_route_kernel<256><<<1, threads, 0, s>>>(top, w, S, K, tab, sg_t, su_t, sd_t, blob, n, tok, row, sg, su, wt, inv);
    else
        group_route_kernel<kGroupRouteSlots><<<1, threads, 0, s>>>(top, w, S, K, tab, sg_t, su_t, sd_t, blob, n, tok, row, sg, su, wt, inv);
    check("group_route");
}
void experts_block_gu(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                      size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su, float gcap,
                      float ucap, uint16_t* h, cudaStream_t s) {
#ifdef STRATA_USE_HIP
    unavailable_mma(__func__);
#else
    if (G <= 0) return;
    const dim3 g((unsigned) (F / 8), (unsigned) G);
    if (T <= 4) experts_block_gu_kernel<8, 1><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
    else experts_block_gu_kernel<8, 2><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
    check("experts_block_gu");
#endif
}
void experts_block_down(const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                        int F, const uint16_t* h, float* y, cudaStream_t s) {
#ifdef STRATA_USE_HIP
    unavailable_mma(__func__);
#else
    if (G <= 0) return;
    experts_block_down_kernel<4><<<dim3((unsigned) (C / 16), (unsigned) G), 128, 0, s>>>(blob, n, row, off_down, C, F,
                                                                                         h, y);
    check("experts_block_down");
#endif
}

#ifdef STRATA_USE_HIP
void rdna3_project_route(const uint16_t* weights, const float* x, int width, int rows, const float* bias,
                         float scale, int* top, float* picks, cudaStream_t stream) {
    rdna3_project_route_kernel<<<rows, 128, 0, stream>>>(weights, x, width, bias, scale, top, picks);
    check("rdna3_project_route");
}
#endif
void route_rows(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
                cudaStream_t s) {
    if (T <= 0) return;
#ifdef STRATA_USE_HIP
    if (rdna3_route_rows(logits, bias, T, E, K, scale, top, w, s)) return;
#endif
    if (E > 1024 || K > E || K > kRouteMax) {
        route(logits, bias, T, E, K, scale, top, w, s);
        return;
    }
    const unsigned nt = (unsigned) ((E + 31) / 32 * 32);
    route_rows_kernel<<<(unsigned) T, nt, (size_t) (E + nt / 32 + K + 1) * sizeof(float), s>>>(logits, bias, E, K,
                                                                                               scale, top, w);
    check("route_rows");
}
void experts_block_gu_linear(int type, const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int T,
                         size_t off_up, int C, int F, const uint16_t* limbs, const float* sg, const float* su,
                         float gcap, float ucap, uint16_t* h, cudaStream_t s) {
#ifdef STRATA_USE_HIP
    unavailable_mma(__func__);
#else
    if (G <= 0) return;
    const dim3 g((unsigned) (F / 8), (unsigned) G);
    if (type == kIQ4_NL) {
        if (T <= 4) experts_block_gu_linear_kernel<8, 1, true><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
        else experts_block_gu_linear_kernel<8, 2, true><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
    } else {
        if (T <= 4) experts_block_gu_linear_kernel<8, 1, false><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
        else experts_block_gu_linear_kernel<8, 2, false><<<g, 256, 0, s>>>(blob, n, tok, row, off_up, C, F, limbs, sg, su, gcap, ucap, h);
    }
    check("experts_block_gu_linear");
#endif
}
void experts_block_down_linear(int type, const uint8_t* const* blob, const int* n, const int* row, int G, size_t off_down, int C,
                           int F, const uint16_t* h, float* y, cudaStream_t s) {
#ifdef STRATA_USE_HIP
    unavailable_mma(__func__);
#else
    if (G <= 0) return;
    if (type == kIQ4_NL) {
        experts_block_down_linear_kernel<4, true><<<dim3((unsigned) (C / 16), (unsigned) G), 128, 0, s>>>(blob, n, row, off_down, C,
                                                                                                F, h, y);
    } else {
        experts_block_down_linear_kernel<4, false><<<dim3((unsigned) (C / 16), (unsigned) G), 128, 0, s>>>(blob, n, row, off_down, C,
                                                                                                F, h, y);
    }
    check("experts_block_down_linear");
#endif
}

}  // namespace ik2::k
