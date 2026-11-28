#include "ik2_kernel_common.cuh"
#include "ik2_cuda_mma.cuh"

namespace ik2::k {
namespace {

__device__ __forceinline__ void l2clip_warp(const float* src, const float* sc, float* dst32, uint16_t* dst16,
                                            int D, float th, float tau) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (!sc || ((uintptr_t) sc & 15) == 0) &&
        (!dst32 || ((uintptr_t) dst32 & 15) == 0) && (!dst16 || ((uintptr_t) dst16 & 7) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float4 sv = sc ? __ldg((const float4*) sc + lane) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
        const float nrm = sqrtf(wsum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3));
        const float f = fminf(1.0f, th / (nrm + 1e-6f)) * tau;
        const float o0 = v0 * f, o1 = v1 * f, o2 = v2 * f, o3 = v3 * f;
        if (dst32) ((float4*) dst32)[lane] = make_float4(o0, o1, o2, o3);
        if (dst16) ((uint2*) dst16)[lane] = make_uint2(f2_to_u32(o0, o1), f2_to_u32(o2, o3));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, th / (nrm + 1e-6f)) * tau;
    for (int d = lane; d < D; d += 32) {
        const float o = src[d] * (sc ? sc[d] : 1.0f) * f;
        if (dst32) dst32[d] = o;
        if (dst16) dst16[d] = f2h(o);
    }
}

__device__ __forceinline__ void rms_f16_warp(const float* src, uint16_t* dst16, int D, float eps) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst16 & 7) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float rs = rsqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w) / (float) D + eps);
        ((uint2*) dst16)[lane] = make_uint2(f2_to_u32(xv.x * rs, xv.y * rs), f2_to_u32(xv.z * rs, xv.w * rs));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = rsqrtf(wsum(a) / (float) D + eps);
    for (int d = lane; d < D; d += 32) dst16[d] = f2h(src[d] * rs);
}

__device__ __forceinline__ void l2clip_fp8_warp(const float* src, uint8_t* dst8, int D, float th, float k_scale) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst8 & 3) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float nrm = sqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w));
        const float f = fminf(1.0f, th / (nrm + 1e-6f)) * k_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x * f, xv.y * f);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z * f, xv.w * f);
        ((uint32_t*) dst8)[lane] = p0 | (p1 << 16);
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, th / (nrm + 1e-6f)) * k_scale;
    for (int d = lane; d < D; d += 32) dst8[d] = f32_to_e4m3fn(src[d] * f);
}

__device__ __forceinline__ size_t kv_page_base(int token, int head, int heads) {
    return kv_page_head(token & ~15, head, heads) * 128;
}

__device__ __forceinline__ void rms_fp8_warp(const float* src, uint8_t* dst8, int D, float eps, float v_scale, bool paged = false) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (paged || ((uintptr_t) dst8 & 3) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float rs = rsqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w) / (float) D + eps) *
                         v_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x * rs, xv.y * rs);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z * rs, xv.w * rs);
        const uint32_t bytes = p0 | (p1 << 16);
        if (paged) {
#pragma unroll
            for (int d = 0; d < 4; ++d) dst8[(4 * lane + d) * 16] = bytes >> (8 * d);
        } else ((uint32_t*)dst8)[lane] = bytes;
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = rsqrtf(wsum(a) / (float) D + eps) * v_scale;
    for (int d = lane; d < D; d += 32) dst8[paged ? 16 * d : d] = f32_to_e4m3fn(src[d] * rs);
}

__device__ __forceinline__ void store_k_warp(const float* kp, uint16_t* kdst, int D, float qk_clip, float) {
    l2clip_warp(kp, nullptr, nullptr, kdst, D, qk_clip, 1.0f);
}
__device__ __forceinline__ void store_k_warp(const float* kp, uint8_t* kdst, int D, float qk_clip, float k_scale) {
    l2clip_fp8_warp(kp, kdst, D, qk_clip, k_scale);
}
__device__ __forceinline__ void store_v_warp(const float* vp, uint16_t* vdst, int D, float eps, float, bool) {
    rms_f16_warp(vp, vdst, D, eps);
}
__device__ __forceinline__ void store_v_warp(const float* vp, uint8_t* vdst, int D, float eps, float v_scale, bool paged) {
    rms_fp8_warp(vp, vdst, D, eps, v_scale, paged);
}

__device__ __forceinline__ unsigned fp4_encode(float x) {
    const float a = fabsf(x);
    // Adjacent E2M1 bin midpoints, ties to the even code.
    return ((x < 0.f) << 3) | ((a > .25f) + (a >= .75f) + (a > 1.25f) +
        (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f));
}

__device__ __forceinline__ float fp4_value(unsigned code) {
    const unsigned m = code & 7;
    const unsigned bits = m > 1 ? 0x3800 + (m << 9) : m * 0x3800;
    return h2f((uint16_t)(bits | ((code & 8) << 12)));
}

__device__ __forceinline__ float fp4_read(const uint8_t* cache, size_t element) {
    const auto& block = reinterpret_cast<const Fp4KvBlock*>(cache)[element / 32];
    unsigned pair = block.pairs[(element % 32) / 2];
    return fp4_value((pair >> (4 * (element % 2))) & 15) * h2f(block.scale);
}

__device__ __forceinline__ void store_fp4_head(const float* src, uint8_t* cache, size_t head,
                                               bool key, float clip, float eps, float scale) {
    const int lane = threadIdx.x & 31;
    const float4 x = reinterpret_cast<const float4*>(src)[lane];
    const float ss = wsum(x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w);
    const float multiplier = scale * (key ? fminf(1.f, clip / (sqrtf(ss) + 1e-6f)) : rsqrtf(ss / 128.f + eps));
    float peak = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
    for (int step = 4; step; step /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, step));
    auto& block = reinterpret_cast<Fp4KvBlock*>(cache)[head * 4 + lane / 8];
    const uint16_t bits = f2h(peak * multiplier / 6.f);
    const float inv = bits ? multiplier / h2f(bits) : 0.f;
    if (!(lane & 7)) block.scale = bits;
    block.pairs[(lane & 7) * 2] = fp4_encode(x.x * inv) | (fp4_encode(x.y * inv) << 4);
    block.pairs[(lane & 7) * 2 + 1] = fp4_encode(x.z * inv) | (fp4_encode(x.w * inv) << 4);
}

// 1 warp (32 threads) per group across all q, k, v, r groups of T tokens
template <class KV, bool Sequences = false>
__global__ void prep_qkv_r_impl_kernel(float* q, const float* q_clip, const float* k_in, KV* kc, const float* v_in,
                                       KV* vc, float* r, const float* r_clip, int T, int H, int KVh, int D, int d_rel,
                                       float qk_clip, float rel_clip, float eps, const int* pos, float alpha,
                                       float n_floor, int pos0, int kv_cap, float k_scale, float v_scale,
                                       const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                                       int sink_k, float* sink_y, int q_ld = 0, int r_ld = 0, bool paged = false, bool fp4 = false,
                                       const int* slots = nullptr, size_t slot_stride = 0) {
    if (!Sequences && pos0_d) pos0 = *pos0_d;   // (a token of a replayed graph: its position read here)
    const int gid = blockIdx.x;
    const int per_tok = 2 * H + 2 * KVh;
    if (q_ld <= 0) q_ld = H * D;   // (q's and r's rows: a token's [H][D] and [H][d_rel] unless given)
    if (r_ld <= 0) r_ld = H * d_rel;
    if (gid >= T * per_tok) {
        // (the tokens' sinks, sink_y [T][H] = sink_w [H][sink_k] . sink_x [T][sink_k] - mv_f16_kernel's sums, a
        //  launch less)
        const int i = gid - T * per_tok, t = i / H, h = i % H, lane = threadIdx.x;
        if (!sink_w || t >= T) return;
        const uint16_t* w = sink_w + (size_t) h * sink_k;
        const float* sx = sink_x + (size_t) t * sink_k;
        float a = 0.0f;
#ifndef STRATA_USE_HIP
        if ((sink_k & 127) == 0 && ((uintptr_t)w & 7) == 0 && ((uintptr_t)sx & 15) == 0) {
            for (int j = lane * 4; j < sink_k; j += 128) {
                const uint2 wh = *(const uint2*)(w + j);
                const float2 w01 = u32_to_f2(wh.x), w23 = u32_to_f2(wh.y);
                const float4 xv = *(const float4*)(sx + j);
                a = fmaf(w01.x, xv.x, fmaf(w01.y, xv.y, fmaf(w23.x, xv.z, fmaf(w23.y, xv.w, a))));
            }
        } else
#endif
            for (int j = lane; j < sink_k; j += 32) a += h2f(w[j]) * sx[j];
        a = wsum(a);
        if (lane == 0) sink_y[(size_t) t * H + h] = a;
        return;
    }
    const int t = gid / per_tok, rem = gid % per_tok;
    if constexpr (Sequences) {
        pos0 = pos0_d[t] - t;
        kc += slots[t] * slot_stride;
        vc += slots[t] * slot_stride;
    }
    const float tau = pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[t] + 1.0f) / n_floor))) : 1.0f;
    if (rem < H) {
        const int h = rem;
        float* qp = q + (size_t) t * q_ld + (size_t) h * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp, q16 ? q16 + ((size_t) t * H + h) * D : nullptr, D,
                    qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* kp = k_in + ((size_t) t * KVh + kv) * D;
        if constexpr (sizeof(KV) == 1) {
            if (fp4) {
                const size_t head = paged ? kv_page_head(slot, kv, KVh) : (size_t)slot * KVh + kv;
                store_fp4_head(kp, kc, head, true, qk_clip, eps, k_scale);
                return;
            }
        }
        KV* kdst = kc + (paged ? kv_page_base(slot, kv, KVh) + (slot % 16) * 128 : ((size_t)slot * KVh + kv) * D);
        store_k_warp(kp, kdst, D, qk_clip, k_scale);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* vp = v_in + ((size_t) t * KVh + kv) * D;
        if constexpr (sizeof(KV) == 1) {
            if (fp4) {
                const size_t head = paged ? kv_page_head(slot, kv, KVh) : (size_t)slot * KVh + kv;
                store_fp4_head(vp, vc, head, false, qk_clip, eps, v_scale);
                return;
            }
        }
        KV* vdst = vc + (paged ? kv_page_base(slot, kv, KVh) + slot % 16 : ((size_t)slot * KVh + kv) * D);
        store_v_warp(vp, vdst, D, eps, v_scale, paged);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + (size_t) t * r_ld + (size_t) h * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

template <int NUM_WARPS>
__device__ __forceinline__ void attn_res_warp0_softmax(float sh_red[][kResMax * 4], float sc[3][kResMax], int n, int nq,
                                                       int C, float res_eps, int lane) {
    float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    if (lane < n) {
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            ss += sh_red[w][lane * 4 + 0];
            d0 += sh_red[w][lane * 4 + 1];
            d1 += sh_red[w][lane * 4 + 2];
            d2 += sh_red[w][lane * 4 + 3];
        }
    }
    const float r = (lane < n) ? rsqrtf(ss / (float) C + res_eps) : 0.0f;
    const float inv_sqrt_c = rsqrtf((float) C);
    const float raw[3] = {d0 * r * inv_sqrt_c, d1 * r * inv_sqrt_c, d2 * r * inv_sqrt_c};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) break;
        const float s_i = (lane < n) ? raw[qi] : -INFINITY;
        const float m = wmax(s_i);
        const float e_i = (lane < n) ? expf(s_i - m) : 0.0f;
        const float sum = wsum(e_i);
        if (lane < n) sc[qi][lane] = (e_i / sum) * r;
    }
}

constexpr int kResGroup = 4;
template <bool FUSED_RMS, int BLOCK_THREADS>
__global__ void attn_res_vec4_kernel(ResStates st, int n, const float* q, int nq, int T, int C, float res_eps,
                                     float rms_eps, float* out, RmsOuts outs, size_t q_step, size_t o_step) {
    q += blockIdx.y * q_step, out += blockIdx.y * o_step;   // (attn_res_jobs: a job a block of the launch)
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    float4 q0[3] = {}, q1[3] = {};
    if (d4_0 < c4) {
        q0[0] = __ldg((const float4*) q + d4_0);
        if (nq > 1) q0[1] = __ldg((const float4*) (q + C) + d4_0);
        if (nq > 2) q0[2] = __ldg((const float4*) (q + 2 * C) + d4_0);
    }
    if (d4_1 < c4) {
        q1[0] = __ldg((const float4*) q + d4_1);
        if (nq > 1) q1[1] = __ldg((const float4*) (q + C) + d4_1);
        if (nq > 2) q1[2] = __ldg((const float4*) (q + 2 * C) + d4_1);
    }
    // (kResGroup states' values asked for at once, then used in the states' order: a thread waits for a load before
    //  it issues the next one's, and a token's launch is those waits - the sums are the one-by-one loop's, bit for bit)
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const float4* x4 = (const float4*) (st.st[min(i0 + j, n - 1)] + (size_t) t * C);
            xa[j] = in0 ? __ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? __ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
            float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
            if (in0) {
                const float4 xv0 = xa[j];
                sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
                qd0 += xv0.x * q0[0].x + xv0.y * q0[0].y + xv0.z * q0[0].z + xv0.w * q0[0].w;
                if (nq > 1) qd1 += xv0.x * q0[1].x + xv0.y * q0[1].y + xv0.z * q0[1].z + xv0.w * q0[1].w;
                if (nq > 2) qd2 += xv0.x * q0[2].x + xv0.y * q0[2].y + xv0.z * q0[2].z + xv0.w * q0[2].w;
            }
            if (in1) {
                const float4 xv1 = xb[j];
                sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
                qd0 += xv1.x * q1[0].x + xv1.y * q1[0].y + xv1.z * q1[0].z + xv1.w * q1[0].w;
                if (nq > 1) qd1 += xv1.x * q1[1].x + xv1.y * q1[1].y + xv1.z * q1[1].z + xv1.w * q1[1].w;
                if (nq > 2) qd2 += xv1.x * q1[2].x + xv1.y * q1[2].y + xv1.z * q1[2].z + xv1.w * q1[2].w;
            }
            sum_sq = wsum(sum_sq);
            qd0 = wsum(qd0);
            if (nq > 1) qd1 = wsum(qd1);
            if (nq > 2) qd2 = wsum(qd2);
            if (lane == 0) {
                sh_red[warp][i * 4 + 0] = sum_sq;
                sh_red[warp][i * 4 + 1] = qd0;
                sh_red[warp][i * 4 + 2] = qd1;
                sh_red[warp][i * 4 + 3] = qd2;
            }
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();

    float4 o0[3] = {}, o1[3] = {};
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const float4* x4 = (const float4*) (st.st[min(i0 + j, n - 1)] + (size_t) t * C);
            xa[j] = in0 ? __ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? __ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                const float w_qi = sc[qi][i];
                if (in0) {
                    o0[qi].x += w_qi * xa[j].x;
                    o0[qi].y += w_qi * xa[j].y;
                    o0[qi].z += w_qi * xa[j].z;
                    o0[qi].w += w_qi * xa[j].w;
                }
                if (in1) {
                    o1[qi].x += w_qi * xb[j].x;
                    o1[qi].y += w_qi * xb[j].y;
                    o1[qi].z += w_qi * xb[j].z;
                    o1[qi].w += w_qi * xb[j].w;
                }
            }
        }
    }
    if constexpr (!FUSED_RMS) {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float4* y4 = (float4*) (out + ((size_t) qi * T + t) * C);
            if (d4_0 < c4) y4[d4_0] = o0[qi];
            if (d4_1 < c4) y4[d4_1] = o1[qi];
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float sq = 0.0f;
            if (d4_0 < c4) sq += o0[qi].x * o0[qi].x + o0[qi].y * o0[qi].y + o0[qi].z * o0[qi].z + o0[qi].w * o0[qi].w;
            if (d4_1 < c4) sq += o1[qi].x * o1[qi].x + o1[qi].y * o1[qi].y + o1[qi].z * o1[qi].z + o1[qi].w * o1[qi].w;
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float rs = rs_rms[qi];
            const float4* w4 = (const float4*) outs.w[qi];
            float4* y32_4 = (float4*) (outs.y32[qi] ? outs.y32[qi] + (size_t) t * C : nullptr);
            uint2* y16_4 = (uint2*) (outs.y16[qi] ? outs.y16[qi] + (size_t) t * C : nullptr);
            if (d4_0 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_0) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o0[qi].x * rs * ww.x, r1 = o0[qi].y * rs * ww.y;
                const float r2 = o0[qi].z * rs * ww.z, r3 = o0[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_0] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            if (d4_1 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_1) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o1[qi].x * rs * ww.x, r1 = o1[qi].y * rs * ww.y;
                const float r2 = o1[qi].z * rs * ww.z, r3 = o1[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_1] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
        }
    }
}

template <bool FUSED_RMS>
__global__ void attn_res_scalar_kernel(ResStates st, int n, const float* q, int nq, int T, int C, float res_eps,
                                       float rms_eps, float* out, RmsOuts outs, size_t q_step, size_t o_step) {
    q += blockIdx.y * q_step, out += blockIdx.y * o_step;
    constexpr int NUM_WARPS = kBlk / 32;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = 0; i < n; ++i) {
        const float* x = st.st[i] + (size_t) t * C;
        float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
        for (int d = tid; d < C; d += kBlk) {
            const float xv = x[d];
            sum_sq += xv * xv;
            qd0 += xv * q[d];
            if (nq > 1) qd1 += xv * q[C + d];
            if (nq > 2) qd2 += xv * q[2 * C + d];
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (nq > 1) qd1 = wsum(qd1);
        if (nq > 2) qd2 = wsum(qd2);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = qd1;
            sh_red[warp][i * 4 + 3] = qd2;
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();
    if constexpr (!FUSED_RMS) {
        for (int qi = 0; qi < nq; ++qi) {
            float* o = out + ((size_t) qi * T + t) * C;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                o[d] = a;
            }
        }
    } else {
        for (int qi = 0; qi < nq; ++qi) {
            float sq = 0.0f;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                sq += a * a;
            }
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
            for (int qi = 0; qi < nq; ++qi) {
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
        for (int qi = 0; qi < nq; ++qi) {
            const float rs = rs_rms[qi];
            const float* wq = outs.w[qi];
            uint16_t* y16 = outs.y16[qi] ? outs.y16[qi] + (size_t) t * C : nullptr;
            float* y32 = outs.y32[qi] ? outs.y32[qi] + (size_t) t * C : nullptr;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                const float r = a * rs * (wq ? wq[d] : 1.0f);
                if (y32) y32[d] = r;
                if (y16) y16[d] = f2h(r);
            }
        }
    }
}

__global__ void attn_probs_kernel(float* lg, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld,
                                  int d_rel, const float* proj, int extent, int window, const float* sink,
                                  int sink_ld) {
    __shared__ float sh[kBlk];
    const int i = blockIdx.x;
    float* row = lg + (size_t) i * nk;
    const float* ri = r + (size_t) i * r_ld;
    const int qp = qpos0 + i;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < nk; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = -INFINITY;
        if (dist >= 0 && dist < window) {
            l = row[j] * qk_scale;
            if (dist < extent) {
                const float* p = proj + (size_t) dist * d_rel;
                float b = 0.0f;
                for (int e = 0; e < d_rel; ++e) b += ri[e] * p[e];
                l += b;
            }
        }
        row[j] = l;
        mx = fmaxf(mx, l);
    }
    const float sk = sink[(size_t) i * sink_ld];
    const float m = fmaxf(bmax(mx, sh), sk);
    float a = 0.0f;
    for (int j = threadIdx.x; j < nk; j += kBlk) {
        const float e = expf(row[j] - m);   // (a masked key: exp(-inf) = 0)
        row[j] = e;
        a += e;
    }
    const float inv = 1.0f / (bsum(a, sh) + expf(sk - m));
    for (int j = threadIdx.x; j < nk; j += kBlk) row[j] *= inv;
}

__global__ void attn_probs_f16_kernel(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                                      float qk_scale, const float* r, int d_rel, const float* proj, int extent,
                                      int window, const float* sink) {
    __shared__ float sh[kBlk];
    __shared__ float rs[64];
    const int i = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* ri_g = r + ((size_t) i * H + h) * d_rel;
    if (d_rel <= 64) {
        for (int e = tid; e < d_rel; e += kBlk) rs[e] = ri_g[e];
        __syncthreads();
    }
    const int qp = qpos0 + i;
    const int j_lo = min(nk, max(0, qp - kpos0 - window + 1));
    const int j_hi = max(j_lo, min(nk, qp - kpos0 + 1));
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = __ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
            for (int e = 0; e < d_rel; ++e) b += (e < 64 ? rs[e] : ri_g[e]) * __ldg(p + e);
            l += b;
        }
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * expf(mx - l)) + 1.0f;
            mx = l;
        } else {
            sum += expf(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = fmaxf(bmax(mx, sh), sk);
    const float s_scaled = (mx == -INFINITY) ? 0.0f : sum * expf(mx - m);
    const float inv = 1.0f / (bsum(s_scaled, sh) + expf(sk - m));
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = __ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
            for (int e = 0; e < d_rel; ++e) b += (e < 64 ? rs[e] : ri_g[e]) * __ldg(p + e);
            l += b;
        }
        row_out[j] = f2h(expf(l - m) * inv);
    }
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

// attn_probs_f16 with the relative bias already in a table: bias [Tq * H rows][extent] (row i * H + h, column the
// distance), so an element costs a load instead of a d_rel-long dot product (done twice) - the prefill's hot kernel.
__global__ void attn_probs_bias_f16_kernel(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0,
                                           int kpos0, float qk_scale, const float* bias, int extent, int window,
                                           const float* sink) {
    __shared__ float sh[kBlk];
    const int i = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* brow = bias + ((size_t) i * H + h) * extent;
    const int qp = qpos0 + i;
    const int j_lo = min(nk, max(0, qp - kpos0 - window + 1));
    const int j_hi = max(j_lo, min(nk, qp - kpos0 + 1));
    const int j_b = max(j_lo, qp - kpos0 - extent + 1);   // from here on the distance is < extent
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = __ldg(row + j) * qk_scale;
        if (j >= j_b) l += __ldg(brow + (qp - kpos0 - j));
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * __expf(mx - l)) + 1.0f;
            mx = l;
        } else {
            sum += __expf(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = fmaxf(bmax(mx, sh), sk);
    const float s_scaled = (mx == -INFINITY) ? 0.0f : sum * __expf(mx - m);
    const float inv = 1.0f / (bsum(s_scaled, sh) + __expf(sk - m));
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = __ldg(row + j) * qk_scale;
        if (j >= j_b) l += __ldg(brow + (qp - kpos0 - j));
        row_out[j] = f2h(__expf(l - m) * inv);
    }
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

#ifndef STRATA_USE_HIP
// Prefill attention in one pass (FlashAttention-2 style, tensor cores via mma.sync m16n8k16, FP16 in / F32
// accumulate): the logits never leave registers.  A block: 64 queries (4 warps x 16) of one head against one split of
// its keys, in tiles of 64; the online softmax in the log2 domain.  Each split writes its unnormalised output and its
// (max, sum); attn_flash_combine merges the splits and the sink.  The relative bias from the table [Tq * H][extent].
constexpr int kFaBN = 64, kFaD = 128, kFaLd = kFaD + 8;   // (row pad: conflict-free fragments/ldmatrix)
constexpr float kLog2e = 1.4426950408889634f;

// one 64-key tile of K and V (rows kt .. kt+63; rows at or past k_end zero-filled) into smem, NT threads
template <int NT>
__device__ __forceinline__ void fa_load_tile(uint16_t* Ks, uint16_t* Vs, const uint16_t* kc, const uint16_t* vc,
                                             int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + row * kFaLd + col, kc + off, ok);
        cp_async16(Vs + row * kFaLd + col, vc + off, ok);
    }
}

// ... from an FP8 (E4M3) cache: the tile's bytes are copied as they are (cp.async, a tile ahead, as the FP16 tile)
// and turned into FP16 in shared memory once they have arrived.  (Turning them as they were loaded had every load
// wait for its bytes - a branch between it and the next - and the kernel ran at a sixth of the memory's speed.)
constexpr int kFaRaw = 2 * kFaBN * kFaD;   // the bytes of a tile: K's [kFaBN][kFaD], then V's
template <int NT>
__device__ __forceinline__ void fa_load_tile_raw(uint8_t* Rs, const uint8_t* kc, const uint8_t* vc, int kv_cap, int KD,
                                                 int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 16) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 3, col = (c & 7) * 16;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Rs + row * kFaD + col, kc + off, ok);
        cp_async16(Rs + kFaBN * kFaD + row * kFaD + col, vc + off, ok);
    }
}
// four E4M3 values as FP16: lo the first two (two halves a register), hi the other two.  A value is sign, 4 bits
// of exponent, 3 of mantissa, and as FP16 its magnitude is (mag << 7) + 0x2000: with a value the high byte of its
// half (x = sign 2^15 + mag 2^8) the FP16 is x - mag 2^7 + 0x2000.  A subnormal (exponent 0: under 1/64) is that
// less an amount by its mantissa, looked up for the four at once (a byte permutation) - no branch, the same cost
// whatever the values.  (0x7F, the NaN, comes out finite: f32_to_e4m3fn saturates, a cache has none.)
__device__ __forceinline__ void e4m3x4_h4(uint32_t w, uint32_t& lo, uint32_t& hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    lo = e4m3fnx2_to_h2_u32((uint16_t) w), hi = e4m3fnx2_to_h2_u32((uint16_t) (w >> 16));
#else
    const uint32_t t = w & 0x78787878u;
    uint32_t z = (t - 0x01010101u) & ~t & 0x80808080u;   // 0x80 a byte whose exponent is 0 ...
    z -= z >> 7;                                         // ... 0x7F there
    const uint32_t v = w & 0x07070707u, u = v | (v >> 4);
    // (the normal formula - the subnormal's FP16) / 0x80 by the mantissa: 64, 17, 10, 7, 4, 3, 2, 1
    const uint32_t dd = __byte_perm(0x070A1140u, 0x01020304u, __byte_perm(u, 0u, 0x4420u)) & z;
    const uint32_t x0 = __byte_perm(w, 0u, 0x1404u), x1 = __byte_perm(w, 0u, 0x3424u);
    const uint32_t y0 = __byte_perm(dd, 0u, 0x1404u), y1 = __byte_perm(dd, 0u, 0x3424u);
    lo = x0 - (((x0 & 0x7F007F00u) + y0) >> 1) + 0x20002000u;
    hi = x1 - (((x1 & 0x7F007F00u) + y1) >> 1) + 0x20002000u;
#endif
}
__device__ __forceinline__ uint4 e4m3x8_h8(uint2 p) {
    uint4 h;
    e4m3x4_h4(p.x, h.x, h.y);
    e4m3x4_h4(p.y, h.z, h.w);
    return h;
}
// the FP16 tile (Ks, Vs: [kFaBN][kFaLd]) of the bytes Rs, NT threads
template <int NT>
__device__ __forceinline__ void fa_tile_from_raw(uint16_t* Ks, uint16_t* Vs, const uint8_t* Rs, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const uint2 k8 = *(const uint2*) (Rs + row * kFaD + col);
        const uint2 v8 = *(const uint2*) (Rs + kFaBN * kFaD + row * kFaD + col);
        *(uint4*) (Ks + row * kFaLd + col) = e4m3x8_h8(k8);
        *(uint4*) (Vs + row * kFaLd + col) = e4m3x8_h8(v8);
    }
}

// The relative bias: a tile within the extent of a warp's queries has the warp copy its 16 rows x 64 distances of the
// table into shared memory (cp.async, coalesced, issued before the tile's Q K^T) - loading it element by element at
// the softmax cost ~3x the whole kernel in the sliding layers (every tile there is within the extent).
constexpr int kFaBld = 72;   // (a bias row in shared memory; 72: 2-way bank conflicts at most for the float2 reads)
constexpr int kFaNW = 4;     // warps a block (8: the same speed in a test, and past sm_89/sm_120's shared memory)
constexpr int kFaSmem = 4 * kFaBN * kFaLd * (int) sizeof(uint16_t) + kFaNW * 16 * kFaBld * (int) sizeof(float);
// Hopper (sm_90a) warpgroup MMA: the four warps of a 128-thread block as one warpgroup, 64 rows.
// D[64 x 64] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__device__ __forceinline__ void wgmma_m64n64k16_rs(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, %38;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    __trap();
#endif
}

// D[64 x 128] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__device__ __forceinline__ void wgmma_m64n128k16_rs(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, %70;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    __trap();
#endif
}

__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
// (keeps the compiler from moving an accumulator's use across the asynchronous MMA's issue or wait)
__device__ __forceinline__ void wg_fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_proxy_async() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
// shared-memory matrix descriptor with the 128-byte swizzle: start, leading / stride byte offsets
__device__ __forceinline__ uint64_t wg_desc(const void* p, uint32_t lbo, uint32_t sbo) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    return (uint64_t) ((a & 0x3FFFF) >> 4) | ((uint64_t) ((lbo & 0x3FFFF) >> 4) << 16) |
           ((uint64_t) ((sbo & 0x3FFFF) >> 4) << 32) | ((uint64_t) 1 << 62);
}
// A 64 x 128 tile (rows: keys, columns: dims), 1024-byte aligned, in the 128-byte-swizzle layout: two atoms of
// 64 rows x 64 columns (128 bytes a row), the 16-byte chunks of row r XOR-permuted by r % 8. K is then B K-major
// (SBO 1024 B between 8-row groups; a k-step 32 bytes further in the atom), V B MN-major (transposed: LBO 8192 B
// between the atoms along the dims, SBO 1024 B between 8-key groups).
constexpr int kWgTile = kFaBN * kFaD;   // halves a tile
__device__ __forceinline__ int wg_tile_off(int r, int c) {
    return (c >> 6) * (kFaBN * 64) + r * 64 + ((((c >> 3) & 7) ^ (r & 7)) << 3);
}
template <int NT>
__device__ __forceinline__ void fa_load_tile_wg(uint16_t* Ks, uint16_t* Vs, const uint16_t* kc, const uint16_t* vc,
                                                int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + wg_tile_off(row, col), kc + off, ok);
        cp_async16(Vs + wg_tile_off(row, col), vc + off, ok);
    }
}

// Four channels always lie in one 32-channel E2M1 block. MMA consumes
// the same pair of half2 registers for packed and native half caches.
__device__ __forceinline__ uint2 kv_half4(const Fp4KvBlock* cache, size_t element) {
    const auto& block = cache[element / 32];
    uint16_t codes;
    std::memcpy(&codes, block.pairs + (element % 32) / 2, sizeof(codes));
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const __half2 scale = __half2half2(__ushort_as_half(block.scale));
    auto pair = [&](uint8_t byte) {
        const __half2 values(__nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1));
        const __half2 scaled = __hmul2(values, scale);
        uint32_t bits;
        std::memcpy(&bits, &scaled, sizeof(bits));
        return bits;
    };
    return make_uint2(pair(codes), pair(codes >> 8));
#else
    const float scale = h2f(block.scale);
    return make_uint2(f2_to_u32(fp4_value(codes & 15) * scale, fp4_value((codes >> 4) & 15) * scale),
                      f2_to_u32(fp4_value((codes >> 8) & 15) * scale, fp4_value(codes >> 12) * scale));
#endif
}
__device__ __forceinline__ uint2 kv_half4(const uint16_t* cache, size_t element) {
    return __ldg(reinterpret_cast<const uint2*>(cache + element));
}
__device__ __forceinline__ uint4 kv_half8(const Fp4KvBlock* cache, size_t element) {
    const uint2 a = kv_half4(cache, element), b = kv_half4(cache, element + 4);
    return make_uint4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ uint4 kv_half8(const uint16_t* cache, size_t element) {
    return __ldg(reinterpret_cast<const uint4*>(cache + element));
}

// Two adjacent E2M1 blocks occupy nine aligned words. Eight lanes copy words 1..8
// to consecutive banks. The ninth word has its own plane and is broadcast at unpack.
constexpr int kFaFp4Pairs = kFaBN * kFaD / 64;
constexpr int kFaFp4Words = 9 * kFaFp4Pairs;  // one K or V tile
constexpr int kFaFp4Raw = 2 * kFaFp4Words * sizeof(uint32_t);

template<int NT>
__device__ __forceinline__ void fa_load_fp4(uint8_t* raw, const Fp4KvBlock* kc, const Fp4KvBlock* vc,
                                            int KV, int g, int kt, int end, int tid) {
    auto* k = reinterpret_cast<uint32_t*>(raw);
    auto* v = k + kFaFp4Words;
#pragma unroll
    for (int i = tid; i < 8 * kFaFp4Pairs; i += NT) {
        const int pair = i / 8, lane = i % 8, row = kt + pair / 2;
        const bool valid = row < end;
        const size_t off = valid ? ((size_t(row) * KV + g) * 2 + pair % 2) * 9 : 0;
        const auto* ks = reinterpret_cast<const uint32_t*>(kc) + off;
        const auto* vs = reinterpret_cast<const uint32_t*>(vc) + off;
        cp_async4(k + i, ks + 1 + lane, valid);
        cp_async4(v + i, vs + 1 + lane, valid);
        if (lane == 0) {
            cp_async4(k + 8 * kFaFp4Pairs + pair, ks, valid);
            cp_async4(v + 8 * kFaFp4Pairs + pair, vs, valid);
        }
    }
}

__device__ __forceinline__ uint32_t fa_fp4_half2(unsigned codes, uint32_t scale) {
    uint32_t halves;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const __half2 values(__nv_cvt_fp4x2_to_halfraw2(uint8_t(codes), __NV_E2M1));
    std::memcpy(&halves, &values, sizeof(halves));
#else
    // Half bits for |E2M1| are 0000, 3800, 3c00, 3e00, 4000, 4200, 4400, 4600.
    const unsigned selectors = ((codes & 7) << 4) | ((codes & 0x70) << 8);
    halves = __byte_perm(0x3e3c3800u, 0x46444240u, selectors)
           | ((codes & 8) << 12) | ((codes & 0x80) << 24);
#endif
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(halves) : "r"(halves), "r"(scale));
    return halves;
}

__device__ __forceinline__ uint4 fa_unpack_fp4(const uint32_t* raw, int i) {
    const uint32_t word = raw[i], first = raw[8 * kFaFp4Pairs + i / 8];
    const uint32_t previous = __shfl_up_sync(0xffffffff, word, 1, 8);
    const uint32_t middle = __shfl_sync(0xffffffff, word, 3, 8);
    // Word 0 starts with the first scale. Word 4 ends with the second scale.
    const uint32_t codes = (i & 4) ? word : __funnelshift_r((i & 7) ? previous : first, word, 16);
    const uint32_t scale = ((i & 4) ? middle >> 16 : first & 0xffff) * 0x10001u;
    return make_uint4(fa_fp4_half2(codes, scale), fa_fp4_half2(codes >> 8, scale),
                      fa_fp4_half2(codes >> 16, scale), fa_fp4_half2(codes >> 24, scale));
}

constexpr int kFaFp4BiasOffset = 2 * kFaBN * kFaLd + 2 * kFaFp4Raw / sizeof(uint16_t);

template <int NW, bool WG, class KVT = uint16_t>   // (KVT: the cache's values, FP16 or FP8 - an FP8 tile's bytes are
__global__ void __launch_bounds__(NW * 32, 2)        //  made FP16 in shared memory; the warpgroup path is FP16's)
attn_flash_f16_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                      const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                      const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                      float2* __restrict__ ml_part) {
    extern __shared__ __align__(16) uint16_t fa_sm[];
    constexpr bool packed = std::is_same_v<KVT, Fp4KvBlock>;
    constexpr bool F8 = sizeof(KVT) == 1;        // (FP8: one FP16 tile, and the bytes of two [2][kFaRaw] after it)
    uint16_t* Ks = fa_sm;                        // [2][kFaBN][kFaLd]
    uint16_t* Vs = fa_sm + ((F8 || packed) ? 1 : 2) * kFaBN * kFaLd;   // [2][kFaBN][kFaLd]
    uint8_t* Rs = (uint8_t*) (fa_sm + 2 * kFaBN * kFaLd);
    constexpr int kTileLd = WG ? kWgTile : (F8 || packed) ? 0 : kFaBN * kFaLd;   // (halves from one buffer to the other)
    float* Bw = (float*) (fa_sm + (packed ? kFaFp4BiasOffset : 4 * kFaBN * kFaLd)) + (threadIdx.x >> 5) * 16 * kFaBld;   // this warp's [16][kFaBld]
    if constexpr (WG) {   // [2][kWgTile] each, 1024-byte aligned (the swizzle's atoms), inside the same 68 KB
        uint16_t* a = (uint16_t*) (((uintptr_t) fa_sm + 1023) & ~(uintptr_t) 1023);
        Ks = a, Vs = a + 2 * kWgTile;
    }
    constexpr int kFaBM = 16 * NW;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gr = lane >> 2, tq = lane & 3;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    const int i0 = blockIdx.x * kFaBM;                     // the block's first query (index in the launch)
    const int i_last = min(Tq, i0 + kFaBM) - 1;
    // the block's keys: [k_lo, k_hi), cut into tiles; this split's tiles [t0, t1)
    const int k_hi = qpos0 + i_last + 1;
    const int k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kFaBN - 1) / kFaBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);

    // this warp's 16 queries: rows r0 = gr, r1 = gr + 8
    const int wi = i0 + warp * 16;
    const int ia = wi + gr, ib = wi + gr + 8;
    const int qa = qpos0 + ia, qb = qpos0 + ib;
    uint32_t qf[kFaD / 16][4];
    {
        const uint16_t* qra = q16 + (size_t) ia * q_ld + (size_t) h * kFaD;
        const uint16_t* qrb = q16 + (size_t) ib * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const int d = kk * 16 + tq * 2;
            qf[kk][0] = ia < Tq ? *(const uint32_t*) (qra + d) : 0u;
            qf[kk][1] = ib < Tq ? *(const uint32_t*) (qrb + d) : 0u;
            qf[kk][2] = ia < Tq ? *(const uint32_t*) (qra + d + 8) : 0u;
            qf[kk][3] = ib < Tq ? *(const uint32_t*) (qrb + d + 8) : 0u;
        }
    }
    float o[kFaD / 8][4];
#pragma unroll
    for (int j = 0; j < kFaD / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const int lo_a = qa - window + 1, lo_b = qb - window + 1;

    if (t0 < t1) {
        if constexpr (WG) fa_load_tile_wg<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else if constexpr (F8) fa_load_tile_raw<NW * 32>(Rs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else if constexpr (packed) fa_load_fp4<NW * 32>(Rs, kc, vc, KV, g, k_lo + t0 * kFaBN, k_hi, tid);
        else fa_load_tile<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        cp_async_commit();
    }
    for (int t = t0; t < t1; ++t) {
        const int buf = (t - t0) & 1;
        const int kt = k_lo + t * kFaBN;
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kFaBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kFaBN - 1) < extent;
        // cp.async groups in flight here: K/V(t), then this tile's bias, then K/V(t+1) (empty groups keep the count)
        if (any && near) {
#pragma unroll 4
            for (int rr = 0; rr < 16; ++rr) {
                const int i = wi + rr, qp = qpos0 + i;
                const float* brow = bias + ((size_t) i * H + h) * extent;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const int c = hf * 32 + lane, dist = qp - kt - c;
                    const bool ok = i < Tq && dist >= 0 && dist < extent;
                    cp_async4(Bw + rr * kFaBld + c, ok ? brow + dist : bias, ok);
                }
            }
        }
        cp_async_commit();
        if (t + 1 < t1) {
            if constexpr (WG)
                fa_load_tile_wg<NW * 32>(Ks + (buf ^ 1) * kTileLd, Vs + (buf ^ 1) * kTileLd, kc, vc, kv_cap,
                                         KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
            else if constexpr (F8)
                fa_load_tile_raw<NW * 32>(Rs + (buf ^ 1) * kFaRaw, kc, vc, kv_cap, KD, g, k_lo + (t + 1) * kFaBN, k_hi,
                                          tid);
            else if constexpr (packed)
                fa_load_fp4<NW * 32>(Rs + (buf ^ 1) * kFaFp4Raw, kc, vc, KV, g, kt + kFaBN, k_hi, tid);
            else
                fa_load_tile<NW * 32>(Ks + (buf ^ 1) * kFaBN * kFaLd, Vs + (buf ^ 1) * kFaBN * kFaLd, kc, vc, kv_cap,
                                      KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
        }
        cp_async_commit();
        cp_async_wait<2>();
        if constexpr (WG) fence_proxy_async();   // (the copies visible to the MMA's reads of shared memory)
        __syncthreads();
        if constexpr (packed) {
            const auto* raw = reinterpret_cast<const uint32_t*>(Rs + buf * kFaFp4Raw);
#pragma unroll
            for (int i = tid; i < kFaBN * kFaD / 8; i += NW * 32) {
                const int dst = (i / 16) * kFaLd + (i % 16) * 8;
                *(uint4*)(Ks + dst) = fa_unpack_fp4(raw, i);
                *(uint4*)(Vs + dst) = fa_unpack_fp4(raw + kFaFp4Words, i);
            }
            __syncthreads();
        } else if constexpr (F8) {   // (the tile's bytes are in: FP16 of them)
            fa_tile_from_raw<NW * 32>(Ks, Vs, Rs + buf * kFaRaw, tid);
            __syncthreads();
        }
        const uint16_t* K = Ks + buf * kTileLd;
        const uint16_t* V = Vs + buf * kTileLd;
        if (any || WG) {   // (the warpgroup's MMA takes all four warps)
            float s[kFaBN / 8][4];
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk)
                    wgmma_m64n64k16_rs<0>(&s[0][0], qf[kk], wg_desc(K + (kk >> 2) * (kFaBN * 64) + (kk & 3) * 16, 16, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                const uint16_t* kr = K + (j * 8 + (lane & 7)) * kFaLd + (lane >> 3) * 8;   // (two k-steps an ldmatrix)
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; kk += 2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, kr + kk * 16);
                    mma16816(s[j], qf[kk], b0, b1);
                    mma16816(s[j], qf[kk + 1], b2, b3);
                }
            }
            // logits (log2 units): scale, bias, mask - a tile inside every row's keys and past the bias's extent
            // needs only the scale
            cp_async_wait<1>();   // (this tile's bias)
            __syncwarp();
            const bool full = !near && kt + kFaBN - 1 <= qpos0 + wi && kt >= qpos0 + wi + 15 - window + 1 &&
                              wi + 15 < Tq;
            float mxa = -INFINITY, mxb = -INFINITY;
            if (full) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) {
                    s[j][0] *= sc, s[j][1] *= sc, s[j][2] *= sc, s[j][3] *= sc;
                    mxa = fmaxf(mxa, fmaxf(s[j][0], s[j][1]));
                    mxb = fmaxf(mxb, fmaxf(s[j][2], s[j][3]));
                }
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = j * 8 + tq * 2 + e, kp = kt + c;
                    const int da = qa - kp, db = qb - kp;
                    float va = s[j][e] * sc, vb = s[j][2 + e] * sc;
                    if (near) {   // (zero where out of the extent)
                        va += Bw[gr * kFaBld + c] * kLog2e;
                        vb += Bw[(gr + 8) * kFaBld + c] * kLog2e;
                    }
                    va = (da < 0 || kp < lo_a || ia >= Tq) ? -INFINITY : va;
                    vb = (db < 0 || kp < lo_b || ib >= Tq) ? -INFINITY : vb;
                    s[j][e] = va, s[j][2 + e] = vb;
                    mxa = fmaxf(mxa, va), mxb = fmaxf(mxb, vb);
                }
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
            const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
            const float ca = mna == -INFINITY ? 1.0f : exp2f(ma - mna);   // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            const float ua = mna == -INFINITY ? 0.0f : mna, ub = mnb == -INFINITY ? 0.0f : mnb;
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 8; ++j) o[j][0] *= ca, o[j][1] *= ca, o[j][2] *= cb, o[j][3] *= cb;
            // P (FP16) as the A fragments of P V, two 8-key tiles per 16-key step
            uint32_t pa[kFaBN / 16][4];
#pragma unroll
            for (int kk = 0; kk < kFaBN / 16; ++kk) {
                const float p00 = exp2f(s[2 * kk][0] - ua), p01 = exp2f(s[2 * kk][1] - ua);
                const float p02 = exp2f(s[2 * kk][2] - ub), p03 = exp2f(s[2 * kk][3] - ub);
                const float p10 = exp2f(s[2 * kk + 1][0] - ua), p11 = exp2f(s[2 * kk + 1][1] - ua);
                const float p12 = exp2f(s[2 * kk + 1][2] - ub), p13 = exp2f(s[2 * kk + 1][3] - ub);
                la += p00 + p01 + p10 + p11;
                lb += p02 + p03 + p12 + p13;
                pa[kk][0] = f2_to_u32(p00, p01);
                pa[kk][1] = f2_to_u32(p02, p03);
                pa[kk][2] = f2_to_u32(p10, p11);
                pa[kk][3] = f2_to_u32(p12, p13);
            }
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk)
                    wgmma_m64n128k16_rs<1>(&o[0][0], pa[kk], wg_desc(V + kk * 16 * 64, kFaBN * 128, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk) {
                    const uint16_t* vr = V + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kFaLd + (lane >> 4) * 8;
#pragma unroll
                    for (int j = 0; j < kFaD / 8; j += 2) {
                        uint32_t b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3, vr + j * 8);
                        mma16816(o[j], pa[kk], b0, b1);
                        mma16816(o[j + 1], pa[kk], b2, b3);
                    }
                }
            }
        }
        // Raw-cache paths overwrite the expanded tile after the next iteration's CTA barrier.
        if constexpr (!F8 && !packed) __syncthreads();
    }
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
    if (ia < Tq) {
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) *(float2*) (op + j * 8) = make_float2(o[j][0], o[j][1]);
        if (tq == 0) ml_part[((size_t) split * Tq + ia) * H + h] = make_float2(ma, la);
    }
    if (ib < Tq) {
        float* op = o_part + (((size_t) split * Tq + ib) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) *(float2*) (op + j * 8) = make_float2(o[j][2], o[j][3]);
        if (tq == 0) ml_part[((size_t) split * Tq + ib) * H + h] = make_float2(mb, lb);
    }
}

// the splits merged with the sink: out [Tq] rows of o_ld, head h at h * D; a block a (query, head), D threads
__global__ void attn_flash_combine_kernel(const float* __restrict__ o_part, const float2* __restrict__ ml_part,
                                          int n_split, int Tq, int H, const float* __restrict__ sink, float* out,
                                          int o_ld, float v_inv) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float m = sink[(size_t) i * H + h] * kLog2e;
    for (int s = 0; s < n_split; ++s) m = fmaxf(m, ml_part[((size_t) s * Tq + i) * H + h].x);
    float den = exp2f(sink[(size_t) i * H + h] * kLog2e - m), acc = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float2 ml = ml_part[((size_t) s * Tq + i) * H + h];
        if (ml.x == -INFINITY) continue;
        const float c = exp2f(ml.x - m);
        den += ml.y * c;
        acc += o_part[(((size_t) s * Tq + i) * H + h) * kFaD + d] * c;
    }
    out[(size_t) i * o_ld + (size_t) h * kFaD + d] = acc / den * v_inv;   // (v_inv: an FP8 cache's V scale)
}

// ---- one token's attention on the tensor cores (long contexts: the scalar kernels are bound by their arithmetic)
// A warp at a time and straight from the cache into the registers: no shared memory, so a dozen and more warps an
// SM run it, and a warp's wait for its rows is another warp's arithmetic - the kernel reads at the memory's speed,
// an FP8 cache too (its bytes made FP16 in the registers).  (attn_flash's way - tiles in shared memory, one block
// of four warps an SM, one of them computing - read an FP16 cache at the memory's speed and an FP8 one at a third
// of it.)
// A block's warps share a K / V head; a warp takes a run of its keys, 16 a step, for the R = H / Hkv <= 8 heads of
// that K / V head - a key's K and V rows are read once for them all:
//   Q K^T   C [16 keys][8 heads] += A [16 keys][16 dims] B [16 dims][8 heads].  A is K as it lies in the cache: a
//           lane loads 4 consecutive values of two keys, and the 16 dims of a step are taken in the order that makes
//           those the lane's fragment; B is the heads' q in the same order.
//   P V     C [16 dims][8 heads] += A [16 dims][16 keys] B [16 keys][8 heads].  A pairs two keys' values of a dim:
//           a lane loads 16 consecutive values of four keys and interleaves them; B is P, shuffled from the lanes
//           that hold a (key, head) logit to the lanes of that head's fragment.
// The online softmax in the log2 domain: a lane has the logits of two heads (the columns 2 tq, 2 tq + 1 of both
// C's), keeps their (max, sum) and 16 dims of their output.  A warp writes its unnormalised output and (max, sum);
// attn_decode_mma_combine merges the warps and the sink (and takes the output through the layer's gate: one launch
// less).  The relative bias r . proj is taken where a step is within the extent of the token, by the lanes that hold
// the logits.

constexpr int kWaNW = 4;    // warps a block
constexpr int kWaMin = 1;   // steps a warp before the next warp has keys (1: a short context or a window is bound by
                            // a warp's latency, not by the partials to merge)
// the keys of a token at qpos: steps of 16 from k_lo; of nw warps, the ones that have keys
__device__ __forceinline__ int wa_active(int qpos, int window, int nw, int& k_lo, int& n_steps) {
    k_lo = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    n_steps = (qpos + 1 - k_lo + 15) >> 4;
    return min(nw, (n_steps + kWaMin - 1) / kWaMin);
}

template <class KVT, bool Sparse = false, bool Sequences = false>
__global__ void __launch_bounds__(kWaNW * 32, 4)
attn_decode_mma_kernel(const uint16_t* __restrict__ q16, const KVT* __restrict__ kc, const KVT* __restrict__ vc,
                       int kv_cap, int H, int Hkv, const int* pos_d, int qpos, float qk_scale,
                       const float* __restrict__ r, const float* __restrict__ proj, int d_rel, bool rel4, int extent,
                       int window, int n_split, float* __restrict__ o_part, float2* __restrict__ ml_part, const int* pages = nullptr,
                       const int* slots = nullptr, size_t stride = 0, int ld = 0) {
    if constexpr (Sequences) {
        const size_t t = blockIdx.z, partial = t * n_split * kWaNW * H;
        q16 += t * H * kFaD;
        r += t * ld;
        kc += slots[t] * stride;
        vc += slots[t] * stride;
        pos_d += t;
        o_part += partial * kFaD;
        ml_part += partial;
    }
    constexpr bool F8 = sizeof(KVT) == 1;
    if (pos_d) qpos = *pos_d;   // (a token of a replayed graph: its position read here)
    const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const int g = blockIdx.x, R = H / Hkv, KD = Hkv * kFaD;
    const int w = blockIdx.y * kWaNW + (threadIdx.x >> 5);   // this warp of the K / V head's
    int k_lo, n_steps;
    const int last_offset = Sparse ? (kSparsePages - 1) * kSparsePageSize + qpos % kSparsePageSize : qpos;
    const int n_act = wa_active(last_offset, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    // the lane's part: head gr's q and P (B's column gr), the logits and the output of the heads qa and qb
    const int qa = 2 * tq, qb = qa + 1;
    const bool q_ok = gr < R, a_ok = qa < R, b_ok = qb < R;
    uint32_t qf[kFaD / 16][2];
    {
        const uint2* qr = (const uint2*) (q16 + (size_t) (g * R + (q_ok ? gr : 0)) * kFaD);
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const uint2 v = qr[kk * 4 + tq];   // (the dims 16 kk + 4 tq .. + 3)
            qf[kk][0] = q_ok ? v.x : 0u, qf[kk][1] = q_ok ? v.y : 0u;
        }
    }
    float o[kFaD / 16][4];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const size_t kg = (size_t) g * kFaD + tq * 4;    // the lane's values of a K row (+ 16 kk) ...
    const size_t vg = (size_t) g * kFaD + gr * 16;   // ... and of a V row
    const float* ra = r + (size_t) (g * R + (a_ok ? qa : 0)) * d_rel;   // (the bias: the heads' r)
    const float* rb = r + (size_t) (g * R + (b_ok ? qb : 0)) * d_rel;
    const int src = tq * 8 + (gr >> 1);   // the lane with head gr's logits of the key 2 tq (+ 4: of the key 2 tq + 1)
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;

    for (int st = st0; st < st1; ++st) {
        int kt = k_lo + st * 16;
        if constexpr (Sparse)
            kt = pages[g * kSparsePages + kt / kSparsePageSize] * kSparsePageSize + kt % kSparsePageSize;
        const int kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        // a key's row in the cache (a key past the token's: the token's row, its logits masked below)
        auto row = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        // Q K^T: the keys gr and gr + 8
        float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            const size_t p0 = kg + row(gr);
            const size_t p1 = kg + row(gr + 8);
            uint32_t ka[kFaD / 16][4];
            if constexpr (F8) {
                uint32_t w0[kFaD / 16], w1[kFaD / 16];
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    w0[kk] = __ldg((const uint32_t*) (kc + p0 + kk * 16));
                    w1[kk] = __ldg((const uint32_t*) (kc + p1 + kk * 16));
                }
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    e4m3x4_h4(w0[kk], ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1[kk], ka[kk][1], ka[kk][3]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint2 v0 = kv_half4(kc, p0 + kk * 16), v1 = kv_half4(kc, p1 + kk * 16);
                    ka[kk][0] = v0.x, ka[kk][2] = v0.y, ka[kk][1] = v1.x, ka[kk][3] = v1.y;
                }
            }
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) mma16816((kk & 1) ? sb : sa, ka[kk], qf[kk][0], qf[kk][1]);
        }
        // the logits (log2 units): scale, bias, mask
        const int k0 = kt + gr, k1 = k0 + 8;
        float c0 = (sa[0] + sb[0]) * sc, c1 = (sa[1] + sb[1]) * sc, c2 = (sa[2] + sb[2]) * sc, c3 = (sa[3] + sb[3]) * sc;
        if (qpos - (kt + 15) < extent) {   // (a step within the bias's extent of the token; zero past the extent)
            const int d0 = qpos - k0, d1 = qpos - k1;
            const bool e0 = d0 >= 0 && d0 < extent, e1 = d1 >= 0 && d1 < extent;
            const float* pj0 = proj + (size_t) (e0 ? d0 : 0) * d_rel;
            const float* pj1 = proj + (size_t) (e1 ? d1 : 0) * d_rel;
            float b0a = 0.0f, b0b = 0.0f, b1a = 0.0f, b1b = 0.0f;   // (key gr | gr + 8, head qa | qb)
            if (rel4) {
                for (int e = 0; e < d_rel; e += 4) {
                    const float4 x0 = __ldg((const float4*) (pj0 + e)), x1 = __ldg((const float4*) (pj1 + e));
                    const float4 ya = __ldg((const float4*) (ra + e)), yb = __ldg((const float4*) (rb + e));
                    b0a += x0.x * ya.x + x0.y * ya.y + x0.z * ya.z + x0.w * ya.w;
                    b0b += x0.x * yb.x + x0.y * yb.y + x0.z * yb.z + x0.w * yb.w;
                    b1a += x1.x * ya.x + x1.y * ya.y + x1.z * ya.z + x1.w * ya.w;
                    b1b += x1.x * yb.x + x1.y * yb.y + x1.z * yb.z + x1.w * yb.w;
                }
            } else {
                for (int e = 0; e < d_rel; ++e) {
                    const float x0 = pj0[e], x1 = pj1[e], ya = ra[e], yb = rb[e];
                    b0a += x0 * ya, b0b += x0 * yb, b1a += x1 * ya, b1b += x1 * yb;
                }
            }
            if (e0) c0 += b0a * kLog2e, c1 += b0b * kLog2e;
            if (e1) c2 += b1a * kLog2e, c3 += b1b * kLog2e;
        }
        c0 = (k0 > qpos || !a_ok) ? -INFINITY : c0;
        c1 = (k0 > qpos || !b_ok) ? -INFINITY : c1;
        c2 = (k1 > qpos || !a_ok) ? -INFINITY : c2;
        c3 = (k1 > qpos || !b_ok) ? -INFINITY : c3;
        float mxa = fmaxf(c0, c2), mxb = fmaxf(c1, c3);   // (over the lanes of the same heads: the same tq)
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
        }
        if (__ballot_sync(0xffffffffu, mxa > ma || mxb > mb)) {   // a new maximum (few steps have one): rescale
            const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
            const float ca = mna == -INFINITY ? 1.0f : exp2f(ma - mna);   // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][0] *= ca, o[j][1] *= cb, o[j][2] *= ca, o[j][3] *= cb;
        }
        const float ua = ma == -INFINITY ? 0.0f : ma, ub = mb == -INFINITY ? 0.0f : mb;
        const float p0 = exp2f(c0 - ua), p1 = exp2f(c1 - ub), p2 = exp2f(c2 - ua), p3 = exp2f(c3 - ub);
        la += p0 + p2, lb += p1 + p3;
        // P (FP16) to the lanes of its fragment: head gr's of the keys 2 tq, 2 tq + 1 and 2 tq + 8, 2 tq + 9
        const uint32_t plo = f2_to_u32(p0, p1), phi = f2_to_u32(p2, p3);   // (key gr, key gr + 8: heads qa | qb)
        const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
        const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
        const uint32_t pb0 = q_ok ? __byte_perm(alo, blo, psel) : 0u, pb1 = q_ok ? __byte_perm(ahi, bhi, psel) : 0u;
        // P V: the dims 16 gr .. + 15 of the keys 2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9; o[j]: the dims 16 gr + 2 j, + 1
        {
            const size_t v0 = vg + row(2 * tq);
            const size_t v1 = vg + row(2 * tq + 1);
            const size_t v8 = vg + row(2 * tq + 8);
            const size_t v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg((const uint4*) (vc + v0)), r1 = __ldg((const uint4*) (vc + v1));
                const uint4 r8 = __ldg((const uint4*) (vc + v8)), r9 = __ldg((const uint4*) (vc + v9));
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {   // (four bytes: two dims of two keys)
                    uint32_t a[4];
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x5140u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x5140u), a[2], a[3]);
                    mma16816(o[2 * i], a, pb0, pb1);
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x7362u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x7362u), a[2], a[3]);
                    mma16816(o[2 * i + 1], a, pb0, pb1);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 r0 = kv_half8(vc, v0 + i * 8), r1 = kv_half8(vc, v1 + i * 8);
                    const uint4 r8 = kv_half8(vc, v8 + i * 8), r9 = kv_half8(vc, v9 + i * 8);
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {__byte_perm(x0[jj], x1[jj], 0x5410u), __byte_perm(x0[jj], x1[jj], 0x7632u),
                                               __byte_perm(x8[jj], x9[jj], 0x5410u), __byte_perm(x8[jj], x9[jj], 0x7632u)};
                        mma16816(o[4 * i + jj], a, pb0, pb1);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, sh);
        lb += __shfl_xor_sync(0xffffffffu, lb, sh);
    }
    // the warp's partials: o_part [warp][H][D], ml_part [warp][H] = (max in log2 units, sum)
    if (a_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qa) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) *(float2*) (op + 2 * j) = make_float2(o[j][0], o[j][2]);
        if (gr == 0) ml_part[(size_t) w * H + g * R + qa] = make_float2(ma, la);
    }
    if (b_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qb) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) *(float2*) (op + 2 * j) = make_float2(o[j][1], o[j][3]);
        if (gr == 0) ml_part[(size_t) w * H + g * R + qb] = make_float2(mb, lb);
    }
}
// the warps merged with the sink: out [H][D] (times v_inv: an FP8 cache's V scale; times the sigmoid of gate [H][D]
// if given; out16: an FP16 copy); a block a head, D threads.  A warp's weight (the exp2 of its max less the overall
// one) is found once and kept in shared memory.
template<bool Sequences = false>
__global__ void attn_decode_mma_combine_kernel(const float* __restrict__ o_part, const float2* __restrict__ ml_part,
                                               int n_split, int H, const float* __restrict__ sink, float v_inv,
                                               const int* pos_d, int qpos, int window, const float* __restrict__ gate,
                                               float* out, uint16_t* out16, int ld = 0) {
    if constexpr (Sequences) {
        const size_t t = blockIdx.y, partial = t * n_split * kWaNW * H;
        pos_d += t;
        sink += t * H;
        gate += t * ld;
        out += t * H * kFaD;
        o_part += partial * kFaD;
        ml_part += partial;
    }
    extern __shared__ float cm_sm[];
    float* red = cm_sm;         // [kFaD] the block's reductions
    float* cw = cm_sm + kFaD;   // [n_split * kWaNW] the warps' weights
    if (pos_d) qpos = *pos_d;
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    const int h = blockIdx.x, d = threadIdx.x;
    const float sk = sink[h] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
    for (int s = d; s < n_act; s += kFaD) m = fmaxf(m, ml_part[(size_t) s * H + h].x);
    m = bmax(m, red);
    float den = d == 0 ? exp2f(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const float2 ml = ml_part[(size_t) s * H + h];
        const float c = ml.x == -INFINITY ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den += ml.y * c;
    }
    den = bsum(den, red);   // (and its barriers: the weights are all there)
    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) acc += o_part[((size_t) s * H + h) * kFaD + d] * cw[s];
    float res = acc / den * v_inv;
    if (gate) res = res * (1.0f / (1.0f + expf(-gate[(size_t) h * kFaD + d])));
    out[(size_t) h * kFaD + d] = res;
    if (out16) out16[(size_t) h * kFaD + d] = f2h(res);
}
#endif

template <bool WRITE_F16, bool WRITE_F32>
__global__ void gate_mul_impl_kernel(float* o, const float* g, size_t n, uint16_t* o16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (((uintptr_t) o & 15) == 0 && ((uintptr_t) g & 15) == 0 && (!WRITE_F16 || ((uintptr_t) o16 & 7) == 0)) {
        const size_t n4 = n >> 2;
        if (i < n4) {
            const float4 ov = ((const float4*) o)[i];
            const float4 gv = __ldg((const float4*) g + i);
            const float r0 = ov.x * (1.0f / (1.0f + expf(-gv.x)));
            const float r1 = ov.y * (1.0f / (1.0f + expf(-gv.y)));
            const float r2 = ov.z * (1.0f / (1.0f + expf(-gv.z)));
            const float r3 = ov.w * (1.0f / (1.0f + expf(-gv.w)));
            if (WRITE_F32) ((float4*) o)[i] = make_float4(r0, r1, r2, r3);
            if (WRITE_F16) ((uint2*) o16)[i] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
        const size_t tail = (n4 << 2) + i;
        if (tail < n) {
            const float r = o[tail] * (1.0f / (1.0f + expf(-g[tail])));
            if (WRITE_F32) o[tail] = r;
            if (WRITE_F16) o16[tail] = f2h(r);
        }
        return;
    }
    if (i < n) {
        const float r = o[i] * (1.0f / (1.0f + expf(-g[i])));
        if (WRITE_F32) o[i] = r;
        if (WRITE_F16) o16[i] = f2h(r);
    }
}

// ---- attention decode & multi-query decode (F32, FP16, or FP8 KV cache, optional kv_cap ring buffer)

template <class KV>
__device__ __forceinline__ float4 v_load4(const KV* vr, int lane);
template <>
__device__ __forceinline__ float4 v_load4<float>(const float* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 15) == 0) return __ldg((const float4*) (vr + o));
    return make_float4(vr[o], vr[o + 1], vr[o + 2], vr[o + 3]);
}
template <>
__device__ __forceinline__ float4 v_load4<uint16_t>(const uint16_t* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 7) == 0) {
        const uint2 v4 = __ldg((const uint2*) (vr + o));
        const float2 f0 = u32_to_f2(v4.x), f1 = u32_to_f2(v4.y);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
    return make_float4(h2f(vr[o]), h2f(vr[o + 1]), h2f(vr[o + 2]), h2f(vr[o + 3]));
}
template <>
__device__ __forceinline__ float4 v_load4<uint8_t>(const uint8_t* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 3) == 0) {
        const uint32_t v4 = __ldg((const uint32_t*) (vr + o));
        const float2 f0 = e4m3fnx2_to_f2((uint16_t) (v4 & 0xFFFFu));
        const float2 f1 = e4m3fnx2_to_f2((uint16_t) (v4 >> 16));
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
    return make_float4(e4m3fn_to_f32(vr[o]), e4m3fn_to_f32(vr[o + 1]), e4m3fn_to_f32(vr[o + 2]),
                       e4m3fn_to_f32(vr[o + 3]));
}

template <class KV>
__device__ __forceinline__ void attn_flash_range(const float* qh, const float* rh, const KV* kc, const KV* vc,
                                                 int KD, int g, int kv_cap, int k0, int j0, int j1, int qpos,
                                                 float qk_scale, const float* proj, int d_rel, int extent, int window,
                                                 float* sm_m, float* sm_s, float sm_o[][128], bool paged, bool fp4) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int o = lane * 4;
    const float4 q4 = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                                    : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
    const float r_lane = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
    float m_w = -INFINITY, s_w = 0.0f;
    float4 o_w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t page = paged ? kv_page_base(slot, g, KD / 128) : 0;
        const size_t head_off = paged ? page + (slot % 16) * 128 : (size_t)slot * KD + (size_t)g * 128;
        auto load = [&](const KV* cache) {
            if constexpr (sizeof(KV) == 1) {
                if (fp4) {
                    const size_t i = head_off + lane * 4;
                    const auto& block = reinterpret_cast<const Fp4KvBlock*>(cache)[i / 32];
                    const float scale = h2f(block.scale);
                    const unsigned lo = block.pairs[(i % 32) / 2];
                    const unsigned hi = block.pairs[(i % 32) / 2 + 1];
                    return make_float4(fp4_value(lo & 15) * scale, fp4_value(lo >> 4) * scale,
                                       fp4_value(hi & 15) * scale, fp4_value(hi >> 4) * scale);
                }
            }
            return v_load4(cache + head_off, lane);
        };
        const float4 k4 = load(kc);
        float l;
#ifdef STRATA_USE_HIP
        if constexpr (sizeof(KV) == 1) {
            // In the scalar E4M3 path, one ulp in two competing large scores can dominate a small difference of values.
            double a = (double)q4.x * k4.x + (double)q4.y * k4.y + (double)q4.z * k4.z + (double)q4.w * k4.w;
            a *= qk_scale;
            if (dist < extent && lane < d_rel) a += (double)r_lane * __ldg(proj + (size_t)dist * d_rel + lane);
#pragma unroll
            for (int delta = 16; delta; delta /= 2) a += __shfl_xor(a, delta, 32);
            l = (float)a;
        } else
#endif
        {
            const float a = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            l = wsum(a) * qk_scale;
            if (dist < extent) {
                const float b = (lane < d_rel) ? r_lane * __ldg(proj + (size_t)dist * d_rel + lane) : 0.0f;
                l += wsum(b);
            }
        }
        if (l > -INFINITY) {
            float4 v4;
            if constexpr (sizeof(KV) == 1) {
                if (paged && !fp4) {
                    const KV* v = vc + page + slot % 16 + lane * 4 * 16;
                    v4 = make_float4(e4m3fn_to_f32(v[0]), e4m3fn_to_f32(v[16]), e4m3fn_to_f32(v[32]), e4m3fn_to_f32(v[48]));
                } else v4 = load(vc);
            } else v4 = load(vc);
            const float m_new = fmaxf(m_w, l);
            const float alpha = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            const float beta = expf(l - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
            o_w.x = o_w.x * alpha + v4.x * beta;
            o_w.y = o_w.y * alpha + v4.y * beta;
            o_w.z = o_w.z * alpha + v4.z * beta;
            o_w.w = o_w.w * alpha + v4.w * beta;
        }
    }
    if (lane == 0) {
        sm_m[warp] = m_w;
        sm_s[warp] = s_w;
    }
    *(float4*) (&sm_o[warp][o]) = o_w;
    __syncthreads();
}

template <class KV>
__device__ __forceinline__ void attn_split_row(const float* q, const float* r, const KV* kc, const KV* vc,
                                               int kv_cap, int H, int Hkv, int k0, int nk, int qpos, float qk_scale,
                                               const float* proj, int d_rel, int extent, int window, float* part,
                                               bool paged, bool fp4) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + (size_t)h * 128;
    const float* rh = r + (size_t)h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m,
                     sm_s, sm_o, paged, fp4);
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = expf(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + ((size_t)h * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}

template <class KV>
__global__ void attn_split_kernel(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int H,
                                  int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel,
                                  int extent, int window, float* part, const int* pos_d, bool paged = false, bool fp4 = false) {
    if (pos_d) {
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int)gridDim.z - k0;
    }
    const size_t row = (size_t)blockIdx.z * H;
    attn_split_row(q + row * 128, r + row * d_rel, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0 + (int)blockIdx.z,
                   qk_scale, proj, d_rel, extent, window, part + row * gridDim.y * 130, paged, fp4);
}

#ifdef STRATA_USE_HIP
__global__ void sequence_fp4_split(const float* q, const float* r, int ld, const uint8_t* kc, const uint8_t* vc,
                                   size_t stride, const int* slots, const int* pos, int H, int Hkv, float scale,
                                   const float* proj, int dr, int extent, int window, float* part, bool paged, int ring) {
    const int t = blockIdx.z, last = pos[t];
    const int first = max(0, last + 1 - window);
    const size_t cache = (size_t)slots[t] * stride;
    // Each row can see keys only through its own position, including packed verify rows sharing a slot.
    attn_split_row(q + (size_t)t * ld, r + (size_t)t * ld, kc + cache, vc + cache, ring, H, Hkv, first,
                   last + 1 - first, last, scale, proj, dr, extent, window, part + (size_t)t * H * gridDim.y * 130,
                   paged, true);
}
#endif

__global__ void attn_combine_kernel(const float* sink, const float* part, int H, int P, float v_inv_scale, float* out,
                                     const float* gate = nullptr, uint16_t* out16 = nullptr, int o_ld = 0, int gate_ld = 0) {
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x;
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
    for (int p = 0; p < P; ++p) M = fmaxf(M, pp[(size_t) p * 130]);
    float den = expf(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    const size_t index = (size_t)qi * (o_ld ? o_ld : H * 128) + h * 128 + tid;
    float value = (o / den) * v_inv_scale;
    if (gate) value *= 1.0f / (1.0f + expf(-gate[gate_ld ? (size_t)qi * gate_ld + h * 128 + tid : index]));
    out[index] = value;
    if (out16) out16[index] = f2h(value);
}

#ifdef STRATA_USE_HIP
// Compute one exponential per split. Each feature then combines the same partials in the scalar kernel's order.
__global__ void rdna3_combine_splits(const float* sink, const float* partials, int splits, float inv,
                                     float* out, const float* gate, uint16_t* out16, int gate_ld = 0) {
    const int head = blockIdx.x + blockIdx.y * gridDim.x, tid = threadIdx.x;
    if (gate_ld) gate += blockIdx.y * (gate_ld - gridDim.x * 128);
    const float* data = partials + (size_t)head * splits * 130;
    __shared__ float wave_max[4];
    extern __shared__ float2 weights[];
    const float sk = sink[head];
    float m = sk;
    for (int p = tid; p < splits; p += 128) m = fmaxf(m, data[p * 130]);
#pragma unroll
    for (int d = 16; d; d /= 2) m = fmaxf(m, __shfl_xor(m, d, 32));
    if (!(tid & 31)) wave_max[tid / 32] = m;
    __syncthreads();
    m = fmaxf(fmaxf(wave_max[0], wave_max[1]), fmaxf(wave_max[2], wave_max[3]));
    for (int p = tid; p < splits; p += 128) {
        const float mp = data[p * 130];
        weights[p] = make_float2(mp == -INFINITY ? 0.f : expf(mp - m), data[p * 130 + 1]);
    }
    __syncthreads();
    float sum = expf(sk - m), y = 0.f;
    for (int p = 0; p < splits; ++p) {
        const float2 w = weights[p];
        if (w.x) {
            sum += w.y * w.x;
            y += data[p * 130 + 2 + tid] * w.x;
        }
    }
    const int index = head * 128 + tid;
    float value = (y / sum) * inv;
    if (gate) value *= 1.f / (1.f + expf(-gate[index]));
    out[index] = value;
    if (out16) out16[index] = f2h(value);
}
#endif

void combine_token_splits(const float* sink, const float* partials, int heads, int splits, float inv, float* out,
                          const float* gate, uint16_t* out16, cudaStream_t stream) {
#ifdef STRATA_USE_HIP
    static const bool shared = getenv("IK2_SHARED_COMBINE") && atoi(getenv("IK2_SHARED_COMBINE"));
    if (shared && splits >= 16 && splits <= 1024) {
        rdna3_combine_splits<<<heads, 128, splits * sizeof(float2), stream>>>(sink, partials, splits, inv, out, gate, out16);
        return;
    }
#endif
    attn_combine_kernel<<<heads, 128, 0, stream>>>(sink, partials, heads, splits, inv, out, gate, out16);
}

template <class KV>
__global__ void attn_decode_kernel(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc,
                                   int kv_cap, int H, int Hkv, int k0, int nk, int qpos, float qk_scale,
                                   float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* out, bool paged = false, bool fp4 = false) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel, extent, window, sm_m, sm_s,
                     sm_o, paged, fp4);
    const float sk = sink[h];
    const float M = fmaxf(sk, fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3])));
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    out[(size_t) h * 128 + tid] = (o / den) * v_inv_scale;
}

__device__ __forceinline__ uint32_t unpack_scaled_pair(uint16_t bytes, float scale) {
    if (scale == 1.0f) return e4m3fnx2_to_h2_u32(bytes);
    const float2 v = e4m3fnx2_to_f2(bytes);
    const uint32_t result = f2_to_u32(v.x * scale, v.y * scale);
#if defined(__HIP_PLATFORM_AMD__)
    // A packed conversion on gfx1100 can canonicalize -0. Preserve the product's two sign bits.
    const uint32_t signs = ((uint32_t) (bytes & 0x8000u) << 16) | ((uint32_t) (bytes & 0x80u) << 8);
    return (result & 0x7fff7fffu) | (signs ^ ((float_to_bits(scale) >> 31) * 0x80008000u));
#else
    return result;
#endif
}

__device__ __forceinline__ uint16_t unpack_scaled_byte(uint8_t byte, float scale) {
    if (scale == 1.0f) return e4m3fn_to_f16(byte);
    const uint16_t result = f2h(e4m3fn_to_f32(byte) * scale);
#if defined(__HIP_PLATFORM_AMD__)
    return (result & 0x7fffu) | (((uint16_t)byte << 8 ^ (float_to_bits(scale) >> 16)) & 0x8000u);
#else
    return result;
#endif
}

__global__ void unpack_kv_fp8_to_f16_vec4_kernel(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk,
                                                 int KD, float k_inv_scale, float v_inv_scale, uint16_t* k_out16,
                                                 uint16_t* v_out16) {
    const int kd4 = KD >> 2;
    const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (d4 >= kd4) return;
    for (int j = blockIdx.y; j < nk; j += gridDim.y) {
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src_off = (size_t) slot * KD;
        const size_t dst_off = (size_t) j * KD;
        if (k_out16) {
            const uint32_t k4 = __ldg((const uint32_t*) (kc + src_off) + d4);
            ((uint2*) (k_out16 + dst_off))[d4] = make_uint2(unpack_scaled_pair((uint16_t) k4, k_inv_scale),
                                                           unpack_scaled_pair((uint16_t) (k4 >> 16), k_inv_scale));
        }
        if (v_out16) {
            const uint32_t v4 = __ldg((const uint32_t*) (vc + src_off) + d4);
            ((uint2*) (v_out16 + dst_off))[d4] = make_uint2(unpack_scaled_pair((uint16_t) v4, v_inv_scale),
                                                           unpack_scaled_pair((uint16_t) (v4 >> 16), v_inv_scale));
        }
    }
}

__global__ void unpack_kv_fp8_to_f16_kernel(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int KD,
                                            float k_inv_scale, float v_inv_scale, uint16_t* k_out16,
                                            uint16_t* v_out16, bool paged, bool fp4) {
    const size_t total = (size_t) nk * KD;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t) gridDim.x * blockDim.x) {
        const int j = (int) (idx / (size_t) KD);
        const int d = (int) (idx % (size_t) KD);
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src = (size_t)slot * KD + d;
        const size_t page = paged ? kv_page_base(slot, d / 128, KD / 128) : 0;
        const size_t k = paged ? page + (slot % 16) * 128 + d % 128 : src;
        const size_t v = paged ? page + (d % 128) * 16 + slot % 16 : src;
        if (k_out16) k_out16[idx] = fp4 ? f2h(fp4_read(kc, k) * k_inv_scale) : unpack_scaled_byte(kc[k], k_inv_scale);
        if (v_out16) v_out16[idx] = fp4 ? f2h(fp4_read(vc, k) * v_inv_scale) : unpack_scaled_byte(vc[v], v_inv_scale);
    }
}

// ---- the block path: a few tokens at once (2 .. 8, a speculative step's verify window) on the resident path

// the keys of a window of T tokens at pos0 .. pos0 + T - 1: steps of 16 from the first token's k_lo to the last token
#ifndef STRATA_USE_HIP
__device__ __forceinline__ int wa_active_block(int pos0, int T, int window, int nw, int& k_lo, int& n_steps) {
    k_lo = window < (1 << 30) ? max(0, pos0 - window + 1) : 0;
    n_steps = (pos0 + T - k_lo + 15) >> 4;
    return min(nw, (n_steps + kWaMin - 1) / kWaMin);
}

// attn_decode_mma_kernel for the T tokens of a window: the B columns are (token, head) pairs of the K / V head -
// column c is token c / R, head g R + c % R - so that the tokens' queries read a key's K and V rows once; NT tiles of
// 8 columns (T R <= 8 NT).  A column's keys: its token's (causal, its own window); the bias by its own distance.
template <class KVT, int NT>
__global__ void __launch_bounds__(kWaNW * 32, NT == 1 ? 4 : 2)
attn_block_mma_kernel(const uint16_t* __restrict__ q16, const KVT* __restrict__ kc, const KVT* __restrict__ vc,
                      int kv_cap, int H, int Hkv, const int* __restrict__ pos_d, int T, float qk_scale,
                      const float* __restrict__ r, int r_ld, const float* __restrict__ proj, int d_rel, bool rel4,
                      int extent, int window, int n_split, float* __restrict__ o_part, float2* __restrict__ ml_part) {
    constexpr bool F8 = sizeof(KVT) == 1;
    const int pos0 = *pos_d;
    const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const int g = blockIdx.x, R = H / Hkv, KD = Hkv * kFaD, NC = T * R;
    const int w = blockIdx.y * kWaNW + (threadIdx.x >> 5);
    int k_lo, n_steps;
    const int n_act = wa_active_block(pos0, T, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    const int qlast = pos0 + T - 1;
    uint32_t qf[NT][kFaD / 16][2];
    int qpa[NT], qpb[NT];             // the positions of the lane's columns a = 2 tq, b = 2 tq + 1 (-1: none)
    const float* ra[NT];
    const float* rb[NT];
    bool q_ok[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int cq = j * 8 + gr, ca = j * 8 + 2 * tq, cb = ca + 1;
        q_ok[j] = cq < NC;
        const int tqq = q_ok[j] ? cq / R : 0, hq = q_ok[j] ? cq % R : 0;
        const uint2* qr = (const uint2*) (q16 + ((size_t) tqq * H + g * R + hq) * kFaD);
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const uint2 v = qr[kk * 4 + tq];
            qf[j][kk][0] = q_ok[j] ? v.x : 0u, qf[j][kk][1] = q_ok[j] ? v.y : 0u;
        }
        const bool a_ok = ca < NC, b_ok = cb < NC;
        qpa[j] = a_ok ? pos0 + ca / R : -1;
        qpb[j] = b_ok ? pos0 + cb / R : -1;
        ra[j] = r + (size_t) (a_ok ? ca / R : 0) * r_ld + (size_t) (g * R + (a_ok ? ca % R : 0)) * d_rel;
        rb[j] = r + (size_t) (b_ok ? cb / R : 0) * r_ld + (size_t) (g * R + (b_ok ? cb % R : 0)) * d_rel;
    }
    float o[NT][kFaD / 16][4];
    float ma[NT], mb[NT], la[NT], lb[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        ma[j] = mb[j] = -INFINITY, la[j] = lb[j] = 0.0f;
#pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) o[j][jj][0] = o[j][jj][1] = o[j][jj][2] = o[j][jj][3] = 0.0f;
    }
    const float sc = qk_scale * kLog2e;
    const size_t kg = (size_t) g * kFaD + tq * 4;
    const size_t vg = (size_t) g * kFaD + gr * 16;
    const int src = tq * 8 + (gr >> 1);
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;
    const bool windowed = window < (1 << 30);

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16, kmax = qlast - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        auto row = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        uint32_t ka[kFaD / 16][4];
        {
            const size_t p0 = kg + row(gr);
            const size_t p1 = kg + row(gr + 8);
            if constexpr (F8) {
                uint32_t w0[kFaD / 16], w1[kFaD / 16];
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    w0[kk] = __ldg((const uint32_t*) (kc + p0 + kk * 16));
                    w1[kk] = __ldg((const uint32_t*) (kc + p1 + kk * 16));
                }
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    e4m3x4_h4(w0[kk], ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1[kk], ka[kk][1], ka[kk][3]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint2 v0 = kv_half4(kc, p0 + kk * 16), v1 = kv_half4(kc, p1 + kk * 16);
                    ka[kk][0] = v0.x, ka[kk][2] = v0.y, ka[kk][1] = v1.x, ka[kk][3] = v1.y;
                }
            }
        }
        const int k0 = kt + gr, k1 = k0 + 8;
        const bool near = qlast - (kt + 15) < extent;
        uint32_t pb0[NT], pb1[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) mma16816((kk & 1) ? sb : sa, ka[kk], qf[j][kk][0], qf[j][kk][1]);
            float c0 = (sa[0] + sb[0]) * sc, c1 = (sa[1] + sb[1]) * sc, c2 = (sa[2] + sb[2]) * sc,
                  c3 = (sa[3] + sb[3]) * sc;
            const int pa = qpa[j], pb = qpb[j];
            if (near) {
                const int da0 = pa - k0, da1 = pa - k1, db0 = pb - k0, db1 = pb - k1;
                const bool ea0 = pa >= 0 && da0 >= 0 && da0 < extent, ea1 = pa >= 0 && da1 >= 0 && da1 < extent;
                const bool eb0 = pb >= 0 && db0 >= 0 && db0 < extent, eb1 = pb >= 0 && db1 >= 0 && db1 < extent;
                const float* pa0 = proj + (size_t) (ea0 ? da0 : 0) * d_rel;
                const float* pa1 = proj + (size_t) (ea1 ? da1 : 0) * d_rel;
                const float* pb0p = proj + (size_t) (eb0 ? db0 : 0) * d_rel;
                const float* pb1p = proj + (size_t) (eb1 ? db1 : 0) * d_rel;
                float b0a = 0.0f, b1a = 0.0f, b0b = 0.0f, b1b = 0.0f;
                if (rel4) {
                    for (int e = 0; e < d_rel; e += 4) {
                        const float4 ya = __ldg((const float4*) (ra[j] + e)), yb = __ldg((const float4*) (rb[j] + e));
                        const float4 x0 = __ldg((const float4*) (pa0 + e)), x1 = __ldg((const float4*) (pa1 + e));
                        const float4 z0 = __ldg((const float4*) (pb0p + e)), z1 = __ldg((const float4*) (pb1p + e));
                        b0a += x0.x * ya.x + x0.y * ya.y + x0.z * ya.z + x0.w * ya.w;
                        b1a += x1.x * ya.x + x1.y * ya.y + x1.z * ya.z + x1.w * ya.w;
                        b0b += z0.x * yb.x + z0.y * yb.y + z0.z * yb.z + z0.w * yb.w;
                        b1b += z1.x * yb.x + z1.y * yb.y + z1.z * yb.z + z1.w * yb.w;
                    }
                } else {
                    for (int e = 0; e < d_rel; ++e) {
                        b0a += pa0[e] * ra[j][e], b1a += pa1[e] * ra[j][e];
                        b0b += pb0p[e] * rb[j][e], b1b += pb1p[e] * rb[j][e];
                    }
                }
                if (ea0) c0 += b0a * kLog2e;
                if (eb0) c1 += b0b * kLog2e;
                if (ea1) c2 += b1a * kLog2e;
                if (eb1) c3 += b1b * kLog2e;
            }
            // (causal and windowed by the column's own token)
            const bool va0 = pa >= 0 && k0 <= pa && (!windowed || k0 > pa - window);
            const bool vb0 = pb >= 0 && k0 <= pb && (!windowed || k0 > pb - window);
            const bool va1 = pa >= 0 && k1 <= pa && (!windowed || k1 > pa - window);
            const bool vb1 = pb >= 0 && k1 <= pb && (!windowed || k1 > pb - window);
            c0 = va0 ? c0 : -INFINITY, c1 = vb0 ? c1 : -INFINITY, c2 = va1 ? c2 : -INFINITY, c3 = vb1 ? c3 : -INFINITY;
            float mxa = fmaxf(c0, c2), mxb = fmaxf(c1, c3);
#pragma unroll
            for (int sh = 4; sh < 32; sh <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
            }
            if (__ballot_sync(0xffffffffu, mxa > ma[j] || mxb > mb[j])) {
                const float mna = fmaxf(ma[j], mxa), mnb = fmaxf(mb[j], mxb);
                const float ca = mna == -INFINITY ? 1.0f : exp2f(ma[j] - mna);
                const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb[j] - mnb);
                ma[j] = mna, mb[j] = mnb;
                la[j] *= ca, lb[j] *= cb;
#pragma unroll
                for (int jj = 0; jj < kFaD / 16; ++jj) o[j][jj][0] *= ca, o[j][jj][1] *= cb, o[j][jj][2] *= ca, o[j][jj][3] *= cb;
            }
            const float ua = ma[j] == -INFINITY ? 0.0f : ma[j], ub = mb[j] == -INFINITY ? 0.0f : mb[j];
            const float p0 = exp2f(c0 - ua), p1 = exp2f(c1 - ub), p2 = exp2f(c2 - ua), p3 = exp2f(c3 - ub);
            la[j] += p0 + p2, lb[j] += p1 + p3;
            const uint32_t plo = f2_to_u32(p0, p1), phi = f2_to_u32(p2, p3);
            const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
            const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
            pb0[j] = q_ok[j] ? __byte_perm(alo, blo, psel) : 0u, pb1[j] = q_ok[j] ? __byte_perm(ahi, bhi, psel) : 0u;
        }
        {
            const size_t v0 = vg + row(2 * tq);
            const size_t v1 = vg + row(2 * tq + 1);
            const size_t v8 = vg + row(2 * tq + 8);
            const size_t v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg((const uint4*) (vc + v0)), r1 = __ldg((const uint4*) (vc + v1));
                const uint4 r8 = __ldg((const uint4*) (vc + v8)), r9 = __ldg((const uint4*) (vc + v9));
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x5140u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x5140u), a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816(o[j][2 * i], a, pb0[j], pb1[j]);
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x7362u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x7362u), a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816(o[j][2 * i + 1], a, pb0[j], pb1[j]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 r0 = kv_half8(vc, v0 + i * 8), r1 = kv_half8(vc, v1 + i * 8);
                    const uint4 r8 = kv_half8(vc, v8 + i * 8), r9 = kv_half8(vc, v9 + i * 8);
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {__byte_perm(x0[jj], x1[jj], 0x5410u), __byte_perm(x0[jj], x1[jj], 0x7632u),
                                               __byte_perm(x8[jj], x9[jj], 0x5410u), __byte_perm(x8[jj], x9[jj], 0x7632u)};
#pragma unroll
                        for (int j = 0; j < NT; ++j) mma16816(o[j][4 * i + jj], a, pb0[j], pb1[j]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            la[j] += __shfl_xor_sync(0xffffffffu, la[j], sh);
            lb[j] += __shfl_xor_sync(0xffffffffu, lb[j], sh);
        }
        const int ca = j * 8 + 2 * tq, cb = ca + 1;
        // the warp's partials: o_part [warp][T][H][D], ml_part [warp][T][H]
        if (ca < NC) {
            const size_t ix = (size_t) w * T * H + (size_t) (ca / R) * H + g * R + ca % R;
            float* op = o_part + ix * kFaD + gr * 16;
#pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) *(float2*) (op + 2 * jj) = make_float2(o[j][jj][0], o[j][jj][2]);
            if (gr == 0) ml_part[ix] = make_float2(ma[j], la[j]);
        }
        if (cb < NC) {
            const size_t ix = (size_t) w * T * H + (size_t) (cb / R) * H + g * R + cb % R;
            float* op = o_part + ix * kFaD + gr * 16;
#pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) *(float2*) (op + 2 * jj) = make_float2(o[j][jj][1], o[j][jj][3]);
            if (gr == 0) ml_part[ix] = make_float2(mb[j], lb[j]);
        }
    }
}
// attn_decode_mma_combine_kernel for a window: a block a (head, token); sink [T][H]; gate rows gate_ld apart;
// out, out16 [T][H][D]
__global__ void attn_block_combine_kernel(const float* __restrict__ o_part, const float2* __restrict__ ml_part,
                                          int n_split, int H, int T, const float* __restrict__ sink, float v_inv,
                                          const int* pos_d, int window, const float* __restrict__ gate, int gate_ld,
                                          float* out, uint16_t* out16) {
    extern __shared__ float cm_sm[];
    float* red = cm_sm;
    float* cw = cm_sm + kFaD;
    int k_lo, n_steps;
    const int n_act = wa_active_block(*pos_d, T, window, n_split * kWaNW, k_lo, n_steps);
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, TH = T * H, ix = t * H + h;
    const float sk = sink[ix] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
    for (int s = d; s < n_act; s += kFaD) m = fmaxf(m, ml_part[(size_t) s * TH + ix].x);
    m = bmax(m, red);
    float den = d == 0 ? exp2f(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const float2 ml = ml_part[(size_t) s * TH + ix];
        const float c = ml.x == -INFINITY ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den += ml.y * c;
    }
    den = bsum(den, red);
    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) acc += o_part[((size_t) s * TH + ix) * kFaD + d] * cw[s];
    float res = acc / den * v_inv;
    if (gate) res = res * (1.0f / (1.0f + expf(-gate[(size_t) t * gate_ld + (size_t) h * kFaD + d])));
    out[(size_t) ix * kFaD + d] = res;
    if (out16) out16[(size_t) ix * kFaD + d] = f2h(res);
}
#endif

}  // namespace

template <bool FUSED_RMS>
void launch_attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps, float rms_eps,
                     float* out, const RmsOuts& outs, cudaStream_t s, int jobs = 1) {
    if (T <= 0) return;
    const size_t q_step = (size_t) nq * C, o_step = (size_t) nq * T * C;   // (a job's queries, its outputs)
    const dim3 g((unsigned) T, (unsigned) jobs);
    if (n < 1 || n > kResMax || nq < 1 || nq > 3) {
        std::fprintf(stderr, "ik2::k::attn_res: n %d, nq %d\n", n, nq);
        std::abort();
    }
    bool vec4 = ((C & 3) == 0) && (((uintptr_t) q & 15) == 0);
    for (int i = 0; i < n && vec4; ++i) vec4 &= (((uintptr_t) states.st[i] & 15) == 0);
    if constexpr (!FUSED_RMS) {
        vec4 &= (((uintptr_t) out & 15) == 0);
    } else {
        for (int qi = 0; qi < nq && vec4; ++qi) {
            vec4 &= (!outs.w[qi] || (((uintptr_t) outs.w[qi] & 15) == 0)) &&
                    (!outs.y16[qi] || (((uintptr_t) outs.y16[qi] & 7) == 0)) &&
                    (!outs.y32[qi] || (((uintptr_t) outs.y32[qi] & 15) == 0));
        }
    }
    if (vec4 && C == 4608) {
        attn_res_vec4_kernel<FUSED_RMS, 576><<<g, 576, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                               q_step, o_step);
    } else if (vec4 && C <= 4096) {
        attn_res_vec4_kernel<FUSED_RMS, 512><<<g, 512, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                               q_step, o_step);
    } else {
        attn_res_scalar_kernel<FUSED_RMS><<<g, kBlk, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                             q_step, o_step);
    }
    check(FUSED_RMS ? "attn_res_rms_f16" : "attn_res");
}

void attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float eps, float* out,
              cudaStream_t s) {
    launch_attn_res<false>(states, n, q, nq, T, C, eps, 0.0f, out, RmsOuts{}, s);
}
void attn_res_jobs(const ResStates& states, int n, const float* q, int nq, int jobs, int C, float eps, float* out,
                   cudaStream_t s) {
    launch_attn_res<false>(states, n, q, nq, 1, C, eps, 0.0f, out, RmsOuts{}, s, jobs);
}

void attn_res_rms_f16(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps,
                      float rms_eps, const RmsOuts& outs, cudaStream_t s) {
    launch_attn_res<true>(states, n, q, nq, T, C, res_eps, rms_eps, nullptr, outs, s);
}

void attn_probs(float* lg, int Tq, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld, int d_rel,
                const float* proj, int extent, int window, const float* sink, int sink_ld, cudaStream_t s) {
    if (Tq <= 0 || nk <= 0) return;
    attn_probs_kernel<<<Tq, kBlk, 0, s>>>(lg, nk, qpos0, kpos0, qk_scale, r, r_ld, d_rel, proj, extent, window, sink,
                                           sink_ld);
    check("attn_probs");
}
void attn_probs_bias_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                         float qk_scale, const float* bias, int extent, int window, const float* sink, cudaStream_t s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    attn_probs_bias_f16_kernel<<<dim3((unsigned) Tq, (unsigned) H), kBlk, 0, s>>>(lg, out16, H, Tq, nk, qpos0, kpos0,
                                                                                   qk_scale, bias, extent, window, sink);
    check("attn_probs_bias_f16");
}
// One wave chooses a page for one temporal bin and KV head. Sample two keys
// per candidate and keep the strongest dot product across the grouped queries.
__global__ void select_sparse_pages(const uint16_t* q, const uint8_t* kc, int group, int Hkv,
                                   const int* pos, int* pages) {
    const int slot = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const int total = *pos / kSparsePageSize + 1;
    int* selected = pages + head * kSparsePages + slot;
    if (slot < kSparseFirst || slot >= kSparsePages - kSparseRecent) {
        if (!lane) *selected = slot < kSparseFirst ? slot : total - kSparsePages + slot;
        return;
    }
    constexpr int bins = kSparsePages - kSparseFirst - kSparseRecent;
    const int bin = slot - kSparseFirst, middle = total - kSparseFirst - kSparseRecent;
    const int lo = kSparseFirst + int((int64_t)bin * middle / bins);
    const int hi = kSparseFirst + int((int64_t)(bin + 1) * middle / bins);
    float best = -INFINITY;
    int chosen = lo;
    for (int page = lo; page < hi; ++page) {
        for (int sample = 16; sample < kSparsePageSize; sample += 32) {
            const size_t row = (size_t)(page * kSparsePageSize + sample) * Hkv + head;
            uint32_t bytes;
            std::memcpy(&bytes, kc + row * 128 + lane * 4, sizeof(bytes));
            const float2 a = e4m3fnx2_to_f2((uint16_t)bytes), b = e4m3fnx2_to_f2(bytes >> 16);
            for (int h = 0; h < group; ++h) {
                const uint16_t* x = q + (head * group + h) * 128 + lane * 4;
                float dot = h2f(x[0])*a.x + h2f(x[1])*a.y + h2f(x[2])*b.x + h2f(x[3])*b.y;
                for (int shift = 16; shift; shift >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, shift);
                if (!lane && dot > best) { best = dot; chosen = page; }
            }
        }
    }
    if (!lane) *selected = chosen;
}

#ifdef STRATA_USE_HIP
#include "ik2_attention_rdna3.cuh"
template <class KV>
void rdna3_flash(const uint16_t* q, int q_ld, const KV* kc, const KV* vc, Fp8KvLayout layout, int H, int Hkv, int D,
                 int T, int pos, float scale, float inv, const float* bias, int extent, int window, const float* sink,
                 float* out, int o_ld, float* scratch, size_t scratch_floats, int splits, cudaStream_t stream) {
    const int ring = layout.ring_capacity;
    if (T <= 0) return;
    const size_t per_split = (size_t)T * H * 130;
    if (D != 128 || Hkv <= 0 || H % Hkv != 0 || scratch_floats < per_split) {
        std::fprintf(stderr, "rdna3 flash attention: incompatible shapes or workspace\n");
        std::abort();
    }
    splits = std::max(1, std::min<int>(splits, scratch_floats / per_split));
    static const int waves = getenv("IK2_FLASH_WAVES") ? atoi(getenv("IK2_FLASH_WAVES")) : 1;
    auto launch = [&]<int Waves>() {
        constexpr int queries = Waves * 16;
        const dim3 grid(H, splits, (T + queries - 1) / queries);
        if constexpr (sizeof(KV) == 1) {
            if (layout.fp4) {
                rdna3_attn_part<true, false, Waves, (Waves > 1), false, true><<<grid, Waves * 32, 0, stream>>>(
                    q, nullptr, kc, vc, ring, H, Hkv, nullptr, pos, scale, nullptr, 0, extent, window, scratch, T, q_ld, bias, 0, nullptr, layout.paged);
                return;
            }
            if (layout.paged) {
                rdna3_attn_part<true, false, Waves, (Waves > 1), true><<<grid, Waves * 32, 0, stream>>>(
                    q, nullptr, kc, vc, ring, H, Hkv, nullptr, pos, scale, nullptr, 0, extent, window, scratch, T, q_ld, bias);
                return;
            }
        }
        rdna3_attn_part<true, false, Waves><<<grid, Waves * 32, 0, stream>>>(
            q, nullptr, kc, vc, ring, H, Hkv, nullptr, pos, scale, nullptr, 0, extent, window, scratch, T, q_ld, bias);
    };
    switch (waves) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 4: launch.template operator()<4>(); break;
        case 8: launch.template operator()<8>(); break;
        default: std::fprintf(stderr, "IK2_FLASH_WAVES expects 1, 2, 4 or 8\n"); std::abort();
    }
    check("rdna3 prefill attention");
    attn_combine_kernel<<<dim3(H, T), 128, 0, stream>>>(sink, scratch, H, splits, inv, out, nullptr, nullptr, o_ld);
    check("rdna3 prefill attention combine");
}
#endif
#ifndef STRATA_USE_HIP
// the Hopper kernel (wgmma) on a 9.x GPU of a build with sm_90a; IK2_WGMMA=0: the mma.sync one everywhere
bool fa_wgmma() {
    static const int on = [] {
        if (getenv("IK2_WGMMA") && !std::atoi(getenv("IK2_WGMMA"))) return 0;
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major == 9 ? 1 : 0;
    }();
    return on != 0;
}
template <int NW, bool WG, class KVT>
void attn_flash_launch(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV,
                       int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window, int n_split,
                       float* o_part, float2* ml_part, cudaStream_t s) {
    constexpr int smem = std::is_same_v<KVT, Fp4KvBlock>
        ? kFaFp4BiasOffset * sizeof(uint16_t) + NW * 16 * kFaBld * sizeof(float) : kFaSmem;
    static bool attr[16] = {};   // (per device: the attribute is a device's, and the layer split runs on several)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 16 && !attr[dev]) {
        cudaFuncSetAttribute(attn_flash_f16_kernel<NW, WG, KVT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        check("attn_flash (shared memory)");
        attr[dev] = true;
    }
    const dim3 grid((unsigned) ((Tq + 16 * NW - 1) / (16 * NW)), (unsigned) H, (unsigned) n_split);
    attn_flash_f16_kernel<NW, WG, KVT><<<grid, NW * 32, smem, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0,
                                                                     qk_scale, bias, extent, window, n_split, o_part,
                                                                     ml_part);
    check("attn_flash");
}
#endif
void attn_flash_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window,
                    const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                    cudaStream_t s) {
#ifdef STRATA_USE_HIP
    rdna3_flash(q16, q_ld, kc, vc, kv_cap, H, KV, D, Tq, qpos0, qk_scale, 1.0f, bias, extent, window,
                sink, out, o_ld, scratch, scratch_floats, n_split, s);
#else
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_f16: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFaD);
    if (fa_wgmma())
        attn_flash_launch<kFaNW, true, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                 window, n_split, o_part, ml_part, s);
    else
        attn_flash_launch<kFaNW, false, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                  window, n_split, o_part, ml_part, s);
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFaD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, 1.0f);
    check("attn_flash_combine");
#endif
}
void attn_flash_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, Fp8KvLayout layout, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, float v_inv_scale, const float* bias, int extent,
                    int window, const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats,
                    int n_split, cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
#ifdef STRATA_USE_HIP
    rdna3_flash(q16, q_ld, kc, vc, layout, H, KV, D, Tq, qpos0, qk_scale, v_inv_scale, bias, extent, window,
                sink, out, o_ld, scratch, scratch_floats, n_split, s);
#else
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_fp8: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFaD);
    auto launch = [&](auto keys, auto values) {
        attn_flash_launch<kFaNW, false>(q16, q_ld, keys, values, kv_cap, H, KV, Tq, qpos0, qk_scale, bias,
                                       extent, window, n_split, o_part, ml_part, s);
    };
    if (layout.fp4) launch(reinterpret_cast<const Fp4KvBlock*>(kc), reinterpret_cast<const Fp4KvBlock*>(vc));
    else launch(kc, vc);
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFaD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, v_inv_scale);
    check("attn_flash_combine");
#endif
}
int attn_flash_splits(int Tq, int H, int nk, int n_sm) {
#ifdef STRATA_USE_HIP
    static const int waves = getenv("IK2_FLASH_WAVES") ? std::max(1, atoi(getenv("IK2_FLASH_WAVES"))) : 1;
    const int rows = 16 * waves;
    const int groups = (Tq + rows - 1) / rows * H;
    return std::max(1, std::min({16, (nk + 127) / 128, (n_sm * 8 + groups - 1) / groups}));
#else
    const int blocks = (Tq + 16 * kFaNW - 1) / (16 * kFaNW) * H;
    const int want = (6 * n_sm + blocks - 1) / blocks;   // (about three waves of two blocks an SM)
    const int most = std::max(1, (nk + 16 * kFaBN - 1) / (16 * kFaBN));   // (>= 1024 keys a split)
    return std::max(1, std::min({want, most, 16}));
#endif
}
void attn_probs_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0, float qk_scale,
                    const float* r, int d_rel, const float* proj, int extent, int window, const float* sink,
                    cudaStream_t s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    attn_probs_f16_kernel<<<dim3((unsigned) Tq, (unsigned) H), kBlk, 0, s>>>(lg, out16, H, Tq, nk, qpos0, kpos0,
                                                                              qk_scale, r, d_rel, proj, extent, window,
                                                                              sink);
    check("attn_probs_f16");
}

void gate_mul(float* o, const float* g, size_t n, cudaStream_t s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0);
    gate_mul_impl_kernel<false, true><<<vec4 ? grid((n + 3) >> 2) : grid(n), kBlk, 0, s>>>(o, g, n, nullptr);
    check("gate_mul");
}
void gate_mul_f16(float* o, const float* g, size_t n, uint16_t* o16, bool write_f32, cudaStream_t s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0) &&
                      (!o16 || ((uintptr_t) o16 & 7) == 0);
    const unsigned nb = vec4 ? grid((n + 3) >> 2) : grid(n);
    if (o16 && write_f32) gate_mul_impl_kernel<true, true><<<nb, kBlk, 0, s>>>(o, g, n, o16);
    else if (o16) gate_mul_impl_kernel<true, false><<<nb, kBlk, 0, s>>>(o, g, n, o16);
    else if (write_f32) gate_mul_impl_kernel<false, true><<<nb, kBlk, 0, s>>>(o, g, n, nullptr);
    check("gate_mul_f16");
}

void prep_qkv_r_f16(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    cudaStream_t s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        pos0, kv_cap, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    check("prep_qkv_r_f16");
}

void prep_qkv_r_fp8(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, Fp8KvLayout layout,
                    float k_scale, float v_scale, cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        pos0, kv_cap, k_scale, v_scale, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0, 0, layout.paged, layout.fp4);
    check("prep_qkv_r_fp8");
}

template <class KV>
void attn_multi_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, Fp8KvLayout layout,
                     int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, float v_inv_scale,
                     const float* proj, int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi: d_rel %d\n", d_rel);
        std::abort();
    }
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    float* part = scratch;
    attn_split_kernel<KV><<<dim3(H, P, nq), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj,
                                                          d_rel, extent, window, part, nullptr, layout.paged, layout.fp4);
    check("attn_multi_split");
    attn_combine_kernel<<<dim3(H, nq), 128, 0, s>>>(sink, part, H, P, v_inv_scale, out);
    check("attn_multi_combine");
}

void attn_multi(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int nq, int H,
                int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                float* scratch, float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, 0, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                    int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                    Fp8KvLayout layout, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, layout, nq, H, Hkv, k0, nk, qpos0, qk_scale, v_inv_scale, proj, d_rel, extent,
                    window, scratch, out, s);
}

template <class KV>
void attn_decode_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, Fp8KvLayout layout, int H,
                      int Hkv, int D, int k0, int nk, int qpos, float qk_scale, float v_inv_scale, const float* proj,
                      int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
    if (D != 128 || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode: D %d, d_rel %d\n", D, d_rel);
        std::abort();
    }
    // Flash-Decode splits over chunks of ~64 keys, up to 128 blocks per head (partials written directly to scratch)
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (P == 1) {
        attn_decode_kernel<KV><<<H, 128, 0, s>>>(q, r, sink, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale,
                                                   v_inv_scale, proj, d_rel, extent, window, out, layout.paged, layout.fp4);
        check("attn_decode");
        return;
    }
    float* part = scratch;
    attn_split_kernel<KV><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel,
                                                      extent, window, part, nullptr, layout.paged, layout.fp4);
    check("attn_decode_split");
    attn_combine_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, part, H, P, v_inv_scale, out);
    check("attn_decode_combine");
}

void attn_decode(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int H, int Hkv,
                 int D, int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                 float* scratch, float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, 0, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                     int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                     Fp8KvLayout layout, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                     float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, layout, H, Hkv, D, k0, nk, qpos, qk_scale, v_inv_scale, proj, d_rel, extent,
                     window, scratch, out, s);
}

void prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       uint16_t* q16, cudaStream_t s, const uint16_t* sink_w, const float* sink_x, int sink_k,
                       float* sink_y) {
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) (2 * H + 2 * KV + (sink_w ? H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, 1.0f, 1.0f, pos, q16, sink_w, sink_x, sink_k, sink_y);
    check("prep_qkv_r_f16_at");
}
void prep_qkv_r_fp8_at(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, Fp8KvLayout layout,
                       float k_scale, float v_scale, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                       const float* sink_x, int sink_k, float* sink_y) {
    const int kv_cap = layout.ring_capacity;
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) (2 * H + 2 * KV + (sink_w ? H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, k_scale, v_scale, pos, q16, sink_w, sink_x, sink_k, sink_y, 0, 0, layout.paged, layout.fp4);
    check("prep_qkv_r_fp8_at");
}

#ifdef STRATA_USE_HIP
static bool use_subwave_decode() {
    static const bool enabled = getenv("IK2_DECODE_SUBWAVE") && atoi(getenv("IK2_DECODE_SUBWAVE"));
    return enabled;
}

template <bool Sequences, class KV>
void launch_subwave_decode(const float* q, const float* r, const KV* kc, const KV* vc, Fp8KvLayout layout,
                           int rows, int H, int Hkv, const int* pos, int splits, float scale, const float* proj,
                           int dr, int extent, int window, float* scratch, cudaStream_t stream,
                           const int* slots = nullptr, size_t stride = 0, int ld = 0) {
    auto launch = [&]<bool Paged, bool Packed>() {
        rdna3_decode_split<KV, Paged, Sequences, Packed><<<dim3(H, splits, rows), 128, 0, stream>>>(
            q, r, kc, vc, layout.ring_capacity, H, Hkv, pos, scale, proj, dr, extent, window, scratch, slots, stride, ld);
    };
    if constexpr (sizeof(KV) == 1) {
        if (layout.fp4) {
            if (layout.paged) launch.template operator()<true, true>();
            else launch.template operator()<false, true>();
            return;
        }
        if (layout.paged) {
            launch.template operator()<true, false>();
            return;
        }
    }
    launch.template operator()<false, false>();
}
#endif

template <class KV>
void split_token_attention(const float* q, const float* r, const KV* kc, const KV* vc, Fp8KvLayout layout,
                           int H, int Hkv, const int* pos, int splits, float scale, const float* proj,
                           int dr, int extent, int window, float* scratch, cudaStream_t stream) {
    const int ring = layout.ring_capacity;
#ifdef STRATA_USE_HIP
    if (use_subwave_decode()) {
        launch_subwave_decode<false>(q, r, kc, vc, layout, 1, H, Hkv, pos, splits, scale, proj, dr, extent,
                                     window, scratch, stream);
        check("rdna3_decode_split");
        return;
    }
#endif
    attn_split_kernel<KV><<<dim3(H, splits), 128, 0, stream>>>(q, r, kc, vc, ring, H, Hkv, 0, 0, 0, scale,
                                                             proj, dr, extent, window, scratch, pos, layout.paged, layout.fp4);
    check("attn_split_kernel");
}

void sequence_attention_quantized(float* q, float* r, float* gate, int ld, const float* qclip, const float* rclip,
                            const float* keys, const float* values, uint8_t* kc, uint8_t* vc, Fp8KvLayout layout,
                            size_t stride, const int* slots, const int* pos, int T, int H, int KV, int dr,
                            float qcap, float rcap, float eps, float alpha, float floor, float kscale, float vscale,
                            const float* rel, int extent, int window, int splits, const float* sink,
                            float* scratch, float* out, cudaStream_t stream, uint16_t* q16) {
    const int ring = layout.ring_capacity;
    prep_qkv_r_impl_kernel<uint8_t, true><<<T * (2 * H + 2 * KV), 32, 0, stream>>>(
        q, qclip, keys, kc, values, vc, r, rclip, T, H, KV, 128, dr, qcap, rcap, eps,
        ring ? nullptr : pos, alpha, floor, 0, ring, kscale, vscale, pos, q16, nullptr, nullptr,
        0, nullptr, ld, ld, layout.paged, layout.fp4, slots, stride);
    check("sequence KV append");
    const float scale = (1.f / sqrtf(128.f)) / kscale;
#ifndef STRATA_USE_HIP
    const dim3 grid(KV, splits, T);
    const size_t partials = (size_t)T * splits * kWaNW * H;
    auto* ml = reinterpret_cast<float2*>(scratch + partials * kFaD);
    const bool rel4 = dr % 4 == 0 && ld % 4 == 0 && (((uintptr_t)r | (uintptr_t)rel) & 15) == 0;
    auto launch = [&](auto* k, auto* v) {
        using Element = typename std::remove_pointer<decltype(k)>::type;
        attn_decode_mma_kernel<Element, false, true><<<grid, kWaNW * 32, 0, stream>>>(
            q16, k, v, ring, H, KV, pos, 0, scale, r, rel, dr, rel4, extent, window, splits, scratch, ml,
            nullptr, slots, stride / sizeof(Element), ld);
    };
    if (layout.fp4) launch(reinterpret_cast<Fp4KvBlock*>(kc), reinterpret_cast<Fp4KvBlock*>(vc));
    else launch(kc, vc);
    check("sequence MMA attention");
    attn_decode_mma_combine_kernel<true><<<dim3(H, T), kFaD, (kFaD + splits * kWaNW) * sizeof(float), stream>>>(
        scratch, ml, splits, H, sink, 1.f / vscale, pos, 0, window, gate, out, nullptr, ld);
#else
    if (layout.fp4 && !use_subwave_decode())
        sequence_fp4_split<<<dim3(H, splits, T), 128, 0, stream>>>(q, r, ld, kc, vc, stride, slots, pos, H, KV,
                                                              scale, rel, dr, extent, window, scratch, layout.paged, ring);
    else
        launch_subwave_decode<true>(q, r, kc, vc, layout, T, H, KV, pos, splits, scale, rel, dr, extent,
                                    window, scratch, stream, slots, stride, ld);
    check("sequence attention");
    static const bool shared = getenv("IK2_SHARED_COMBINE") && atoi(getenv("IK2_SHARED_COMBINE"));
    if (shared && splits >= 16)
        rdna3_combine_splits<<<dim3(H, T), 128, splits * sizeof(float2), stream>>>(
            sink, scratch, splits, 1.f / vscale, out, gate, nullptr, ld);
    else
        attn_combine_kernel<<<dim3(H, T), 128, 0, stream>>>(
            sink, scratch, H, splits, 1.f / vscale, out, gate, nullptr, 0, ld);
#endif
    check("sequence attention combine");
}

void attn_decode_f16_at(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                        int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s, const float* gate, uint16_t* out16) {
    split_token_attention(q, r, kc, vc, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel, extent, window, scratch, s);
    combine_token_splits(sink, scratch, H, P, 1.0f, out, gate, out16, s);
    check("attn_decode_f16_at (combine)");
}
void attn_decode_fp8_at(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                        Fp8KvLayout layout, int H, int Hkv, const int* pos, int P, float qk_scale, float v_inv_scale,
                        const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                        cudaStream_t s, const float* gate, uint16_t* out16) {
    split_token_attention(q, r, kc, vc, layout, H, Hkv, pos, P, qk_scale, proj, d_rel, extent, window, scratch, s);
    combine_token_splits(sink, scratch, H, P, v_inv_scale, out, gate, out16, s);
    check("attn_decode_fp8_at (combine)");
}


void sparse_decode(const uint16_t* q, const float* r, const float* sink, const uint8_t* kc,
                         const uint8_t* vc, int H, int Hkv, const int* pos, float qscale, float v_inv,
                         const float* proj, int dr, int extent, int* pages, float* scratch, float* out,
                         const float* gate, uint16_t* out16, cudaStream_t stream) {
    constexpr int splits = 64;
    select_sparse_pages<<<dim3(kSparsePages, Hkv), 32, 0, stream>>>(q, kc, H / Hkv, Hkv, pos, pages);
    check("select_sparse_pages");
#ifdef STRATA_USE_HIP
    rdna3_attn_part<false, false, 1, true, false, false, true><<<dim3(Hkv, splits), 32, 0, stream>>>(
        q, r, kc, vc, 0, H, Hkv, pos, 0, qscale, proj, dr, extent, 1 << 30, scratch, 1, H * 128, nullptr, 0, pages);
    check("rdna3 sparse attention");
    combine_token_splits(sink, scratch, H, splits, v_inv, out, gate, out16, stream);
#else
    constexpr int warps = splits * kWaNW;
    float2* ml = reinterpret_cast<float2*>(scratch + size_t(warps) * H * kFaD);
    const bool rel4 = dr % 4 == 0 && (((uintptr_t)r | (uintptr_t)proj) & 15) == 0;
    attn_decode_mma_kernel<uint8_t, true><<<dim3(Hkv, splits), kWaNW * 32, 0, stream>>>(
        q, kc, vc, 0, H, Hkv, pos, 0, qscale, r, proj, dr, rel4, extent, 1 << 30, splits, scratch, ml, pages);
    check("sparse attention");
    // Even a partial last page leaves more than one tile for every warp.
    attn_decode_mma_combine_kernel<<<H, kFaD, (kFaD + warps) * sizeof(float), stream>>>(
        scratch, ml, splits, H, sink, v_inv, nullptr, kSparsePages * kSparsePageSize - 1, 1 << 30, gate, out, out16);
#endif
    check("sparse attention combine");
}

int attn_decode_mma_blocks(int Hkv) {
#ifdef STRATA_USE_HIP
    return std::max(1, 256 / std::max(Hkv, 1));
#else
    int dev = 0, n_sm = 0, occ16 = 0, occ8 = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ16, attn_decode_mma_kernel<uint16_t>, kWaNW * 32, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, attn_decode_mma_kernel<uint8_t>, kWaNW * 32, 0);
    check("attn_decode_mma (occupancy)");
    return std::max(1, n_sm * std::min(occ16, occ8) / std::max(1, Hkv));
#endif
}
size_t attn_decode_mma_scratch(int H, int n_split) {
#ifdef STRATA_USE_HIP
    return (size_t)H * n_split * 130;
#else
    return (size_t) n_split * kWaNW * H * (kFaD + 2);
#endif
}
template <class KV>
void attn_decode_mma_impl(const uint16_t* q16, const float* r, const float* sink, const KV* kc, const KV* vc, Fp8KvLayout layout,
                          int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale, float v_inv,
                          const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                          const float* gate, uint16_t* out16, cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
#ifdef STRATA_USE_HIP
    if (Hkv <= 0 || H % Hkv || H / Hkv > 16 || n_split < 1) {
        std::fprintf(stderr, "rdna3 attention: unsupported head geometry or split count\n");
        std::abort();
    }
    static const bool staged = getenv("IK2_ATTN_STAGED") && atoi(getenv("IK2_ATTN_STAGED"));
    if (layout.fp4)
        rdna3_attn_part<false, false, 1, true, false, true><<<dim3(Hkv, n_split), 32, 0, s>>>(q16, r, kc, vc, kv_cap, H, Hkv, pos, qpos,
                                                      qk_scale, proj, d_rel, extent, window, scratch, 1, H * 128, nullptr, 0, nullptr, layout.paged);
    else if (layout.paged)
        rdna3_attn_part<false, false, 1, false, true><<<dim3(Hkv, n_split), 32, 0, s>>>(q16, r, kc, vc, kv_cap, H, Hkv, pos, qpos,
                                                      qk_scale, proj, d_rel, extent, window, scratch, 1, H * 128, nullptr);
    else if (staged)
        rdna3_attn_part<false, false, 1, true><<<dim3(Hkv, n_split), 32, 0, s>>>(q16, r, kc, vc, kv_cap, H, Hkv, pos, qpos,
                                                      qk_scale, proj, d_rel, extent, window, scratch, 1, H * 128, nullptr);
    else
        rdna3_attn_part<false><<<dim3(Hkv, n_split), 32, 0, s>>>(q16, r, kc, vc, kv_cap, H, Hkv, pos, qpos,
                                                      qk_scale, proj, d_rel, extent, window, scratch, 1, H * 128, nullptr);
    check("rdna3_attn_part");
    combine_token_splits(sink, scratch, H, n_split, v_inv, out, gate, out16, s);
    check("rdna3 attention combine");
#else
    if (H % Hkv || H / Hkv > 8 || n_split < 1 || (kv_cap > 0 && kv_cap < 16)) {
        std::fprintf(stderr, "ik2::k::attn_decode_mma: %d heads on %d, %d splits, a cache of %d\n", H, Hkv, n_split,
                     kv_cap);
        std::abort();
    }
    float* o_part = scratch;
    float2* ml_part = (float2*) (o_part + (size_t) n_split * kWaNW * H * kFaD);
    const bool rel4 = d_rel % 4 == 0 && (((uintptr_t) r | (uintptr_t) proj) & 15) == 0;
    auto launch = [&](auto keys, auto values) {
        attn_decode_mma_kernel<<<dim3(Hkv, n_split), kWaNW * 32, 0, s>>>(
            q16, keys, values, kv_cap, H, Hkv, pos, qpos, qk_scale, r, proj, d_rel, rel4, extent, window,
            n_split, o_part, ml_part);
    };
    if (layout.fp4) launch(reinterpret_cast<const Fp4KvBlock*>(kc), reinterpret_cast<const Fp4KvBlock*>(vc));
    else launch(kc, vc);
    check("attn_decode_mma");
    attn_decode_mma_combine_kernel<<<(unsigned) H, kFaD, (size_t) (kFaD + n_split * kWaNW) * sizeof(float), s>>>(
        o_part, ml_part, n_split, H, sink, v_inv, pos, qpos, window, gate, out, out16);
    check("attn_decode_mma (combine)");
#endif
}
void attn_decode_mma_f16(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                         const float* gate, uint16_t* out16, cudaStream_t s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, kv_cap, H, Hkv, pos, qpos, n_split, qk_scale, 1.0f, proj, d_rel, extent,
                         window, scratch, out, gate, out16, s);
}
void attn_decode_mma_fp8(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                         Fp8KvLayout layout, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                         float* out, const float* gate, uint16_t* out16, cudaStream_t s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, layout, H, Hkv, pos, qpos, n_split, qk_scale, v_inv_scale, proj, d_rel,
                         extent, window, scratch, out, gate, out16, s);
}

void unpack_kv_fp8_to_f16(const uint8_t* kc, const uint8_t* vc, Fp8KvLayout layout, int k0, int nk, int KD,
                          float k_inv_scale, float v_inv_scale, uint16_t* k_out16, uint16_t* v_out16,
                          cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
    if (nk <= 0 || KD <= 0) return;
    if (!layout.paged && !layout.fp4 && (KD & 3) == 0 && (!k_out16 || ((((uintptr_t) kc & 3) == 0) && (((uintptr_t) k_out16 & 7) == 0))) &&
        (!v_out16 || ((((uintptr_t) vc & 3) == 0) && (((uintptr_t) v_out16 & 7) == 0)))) {
        const int kd4 = KD >> 2;
        const int bx = std::min(256, kd4);
        const dim3 g((unsigned) ((kd4 + bx - 1) / bx), (unsigned) std::min(nk, 65535));
        unpack_kv_fp8_to_f16_vec4_kernel<<<g, (unsigned) bx, 0, s>>>(kc, vc, kv_cap, k0, nk, KD, k_inv_scale,
                                                                     v_inv_scale, k_out16, v_out16);
        check("unpack_kv_fp8_to_f16");
        return;
    }
    const size_t total = (size_t) nk * KD;
    const unsigned blocks = (unsigned) std::min<size_t>((total + kBlk - 1) / kBlk, 65535u);
    unpack_kv_fp8_to_f16_kernel<<<blocks, kBlk, 0, s>>>(kc, vc, kv_cap, k0, nk, KD, k_inv_scale, v_inv_scale, k_out16,
                                                        v_out16, layout.paged, layout.fp4);
    check("unpack_kv_fp8_to_f16");
}

#ifndef STRATA_USE_HIP
template <int NT, class KV>
static void attn_block_launch(const uint16_t* q16, const float* r, int r_ld, const float* sink, const KV* kc,
                              const KV* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split,
                              float qk_scale, float v_inv, const float* proj, int d_rel, int extent, int window,
                              float* scratch, float* out, const float* gate, int gate_ld, uint16_t* out16,
                              cudaStream_t s) {
    float* o_part = scratch;
    float2* ml_part = (float2*) (o_part + (size_t) n_split * kWaNW * T * H * kFaD);
    const bool rel4 = d_rel % 4 == 0 && r_ld % 4 == 0 && (((uintptr_t) r | (uintptr_t) proj) & 15) == 0;
    attn_block_mma_kernel<KV, NT><<<dim3((unsigned) Hkv, (unsigned) n_split), kWaNW * 32, 0, s>>>(
        q16, kc, vc, kv_cap, H, Hkv, pos, T, qk_scale, r, r_ld, proj, d_rel, rel4, extent, window, n_split, o_part,
        ml_part);
    check("attn_block_mma");
    attn_block_combine_kernel<<<dim3((unsigned) H, (unsigned) T), kFaD,
                                (size_t) (kFaD + n_split * kWaNW) * sizeof(float), s>>>(
        o_part, ml_part, n_split, H, T, sink, v_inv, pos, window, gate, gate_ld, out, out16);
    check("attn_block_mma (combine)");
}
#endif
template <class KV>
static void attn_block_impl(const uint16_t* q16, const float* r, int r_ld, const float* sink, const KV* kc,
                            const KV* vc, Fp8KvLayout layout, int H, int Hkv, const int* pos, int T, int n_split,
                            float qk_scale, float v_inv, const float* proj, int d_rel, int extent, int window,
                            float* scratch, float* out, const float* gate, int gate_ld, uint16_t* out16,
                            cudaStream_t s) {
    const int kv_cap = layout.ring_capacity;
#ifdef STRATA_USE_HIP
    if (Hkv <= 0 || H % Hkv || T < 1 || T > 8 || n_split < 1 || (kv_cap && kv_cap < T)) {
        std::fprintf(stderr, "RDNA3 block attention: incompatible heads, window or splits\n");
        std::abort();
    }
    const int columns = T * (H / Hkv);
    if (layout.fp4)
        rdna3_attn_part<false, true, 1, false, false, true><<<dim3(Hkv, n_split, (columns + 15) / 16), 32, 0, s>>>(
        q16, r, kc, vc, kv_cap, H, Hkv, pos, 0, qk_scale, proj, d_rel, extent, window,
        scratch, T, H * 128, nullptr, r_ld, nullptr, layout.paged);
    else if (layout.paged)
        rdna3_attn_part<false, true, 1, false, true><<<dim3(Hkv, n_split, (columns + 15) / 16), 32, 0, s>>>(
        q16, r, kc, vc, kv_cap, H, Hkv, pos, 0, qk_scale, proj, d_rel, extent, window,
        scratch, T, H * 128, nullptr, r_ld);
    else
        rdna3_attn_part<false, true><<<dim3(Hkv, n_split, (columns + 15) / 16), 32, 0, s>>>(
        q16, r, kc, vc, kv_cap, H, Hkv, pos, 0, qk_scale, proj, d_rel, extent, window,
        scratch, T, H * 128, nullptr, r_ld);
    check("rdna3 block attention");
    attn_combine_kernel<<<dim3(H, T), 128, 0, s>>>(sink, scratch, H, n_split, v_inv, out, gate, out16,
                                                   H * 128, gate_ld);
    check("rdna3 block attention combine");
#else
    const int R = H / Hkv, NC = T * R;
    if (H % Hkv || T < 1 || NC > 24 || n_split < 1 || (kv_cap > 0 && kv_cap < 16 + T)) {
        std::fprintf(stderr, "ik2::k::attn_block: %d heads on %d, %d tokens, %d splits, a cache of %d\n", H, Hkv, T,
                     n_split, kv_cap);
        std::abort();
    }
    auto launch = [&](auto keys, auto values) {
        if (NC <= 8)
            attn_block_launch<1>(q16, r, r_ld, sink, keys, values, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
        else if (NC <= 16)
            attn_block_launch<2>(q16, r, r_ld, sink, keys, values, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
        else
            attn_block_launch<3>(q16, r, r_ld, sink, keys, values, kv_cap, H, Hkv, pos, T, n_split, qk_scale, v_inv, proj,
                                 d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
    };
    if (layout.fp4) launch(reinterpret_cast<const Fp4KvBlock*>(kc), reinterpret_cast<const Fp4KvBlock*>(vc));
    else launch(kc, vc);
#endif
}
void attn_block_f16(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint16_t* kc,
                    const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                    const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s) {
    attn_block_impl<uint16_t>(q16, r, r_ld, sink, kc, vc, kv_cap, H, Hkv, pos, T, n_split, qk_scale, 1.0f, proj,
                              d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
}
void attn_block_fp8(const uint16_t* q16, const float* r, int r_ld, const float* sink, const uint8_t* kc,
                    const uint8_t* vc, Fp8KvLayout layout, int H, int Hkv, const int* pos, int T, int n_split, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, const float* gate, int gate_ld, uint16_t* out16, cudaStream_t s) {
    attn_block_impl<uint8_t>(q16, r, r_ld, sink, kc, vc, layout, H, Hkv, pos, T, n_split, qk_scale, v_inv_scale,
                             proj, d_rel, extent, window, scratch, out, gate, gate_ld, out16, s);
}
size_t attn_block_scratch(int H, int T, int n_split) {
#ifdef STRATA_USE_HIP
    return (size_t)T * H * n_split * 130;
#else
    return (size_t) n_split * kWaNW * T * H * (kFaD + 2);
#endif
}

void prep_qkv_r_fp8_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in,
                          uint8_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, Fp8KvLayout layout, float k_scale, float v_scale, uint16_t* q16, cudaStream_t s,
                          const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y) {
    const int kv_cap = layout.ring_capacity;
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) (T * (2 * H + 2 * KV) + (sink_w ? T * H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, k_scale, v_scale, pos, q16, sink_w, sink_x, sink_k, sink_y, q_ld, r_ld, layout.paged, layout.fp4);
    check("prep_qkv_r_fp8_block");
}
void prep_qkv_r_f16_block(float* q, int q_ld, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in,
                          uint16_t* vc, float* r, int r_ld, const float* r_clip, int T, int H, int KV, int D, int d_rel,
                          float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale, float alpha,
                          float n_floor, int kv_cap, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                          const float* sink_x, int sink_k, float* sink_y) {
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) (T * (2 * H + 2 * KV) + (sink_w ? T * H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, 1.0f, 1.0f, pos, q16, sink_w, sink_x, sink_k, sink_y, q_ld, r_ld);
    check("prep_qkv_r_f16_block");
}
void attn_res_jobs_block(const ResStates& states, int n, const float* q, int nq, int jobs, int T, int C, float eps,
                         float* out, cudaStream_t s) {
    launch_attn_res<false>(states, n, q, nq, T, C, eps, 0.0f, out, RmsOuts{}, s, jobs);
}

}  // namespace ik2::k
