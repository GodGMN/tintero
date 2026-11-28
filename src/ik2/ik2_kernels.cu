#include "ik2_kernel_common.cuh"

namespace ik2::k {
namespace {

template <bool WRITE_F16, bool WRITE_F32>
__global__ void rms_impl_kernel(const float* x, const float* w, int D, float eps, uint16_t* y16, float* y32) {
    __shared__ float sh[kBlk];
    const float* xr = x + (size_t) blockIdx.x * D;
    uint16_t* y16r = WRITE_F16 ? y16 + (size_t) blockIdx.x * D : nullptr;
    float* y32r = WRITE_F32 ? y32 + (size_t) blockIdx.x * D : nullptr;
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (!WRITE_F32 || ((uintptr_t) y32r & 15) == 0) &&
                      (!WRITE_F16 || ((uintptr_t) y16r & 7) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const float4* x4 = (const float4*) xr;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        const float4* w4 = (const float4*) w;
        float4* y32_4 = (float4*) y32r;
        uint2* y16_4 = (uint2*) y16r;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            if (WRITE_F32) y32_4[d4] = make_float4(r0, r1, r2, r3);
            if (WRITE_F16) y16_4[d4] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
    } else {
        for (int d = threadIdx.x; d < D; d += kBlk) a += xr[d] * xr[d];
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        for (int d = threadIdx.x; d < D; d += kBlk) {
            const float r = xr[d] * rs * (w ? w[d] : 1.0f);
            if (WRITE_F32) y32r[d] = r;
            if (WRITE_F16) y16r[d] = f2h(r);
        }
    }
}
// rms of up to three vectors, each with its own weights, in one launch (a block a vector; rms_impl_kernel's sums)
__global__ void rms_multi_kernel(RmsJobs jobs, int D, float eps) {
    __shared__ float sh[kBlk];
    const float* xr = jobs.x[blockIdx.x];
    const float* w = jobs.w[blockIdx.x];
    float* y32r = jobs.y[blockIdx.x];
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (((uintptr_t) y32r & 15) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const float4* x4 = (const float4*) xr;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        const float4* w4 = (const float4*) w;
        float4* y32_4 = (float4*) y32r;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            y32_4[d4] = make_float4(r0, r1, r2, r3);
        }
    } else {
        for (int d = threadIdx.x; d < D; d += kBlk) a += xr[d] * xr[d];
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        for (int d = threadIdx.x; d < D; d += kBlk) y32r[d] = xr[d] * rs * (w ? w[d] : 1.0f);
    }
}

__global__ void l2clip_kernel(const float* x, const float* scale, int G, int D, float th, const int* pos, float alpha,
                              float n_floor, float* y) {
    __shared__ float sh[128];
    const int row = blockIdx.x / G, g = blockIdx.x % G;
    const float* xr = x + (size_t) blockIdx.x * D;
    float* yr = y + (size_t) blockIdx.x * D;
    const float* sc = scale ? scale + (size_t) g * D : nullptr;
    float a = 0.0f;
    for (int d = threadIdx.x; d < D; d += 128) {
        const float v = xr[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sqrtf(bsum(a, sh));
    float f = fminf(1.0f, th / (nrm + 1e-6f));
    if (pos) f *= 1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[row] + 1.0f) / n_floor));
    for (int d = threadIdx.x; d < D; d += 128) yr[d] = xr[d] * (sc ? sc[d] : 1.0f) * f;
}

constexpr int kConvFusedMaxT = 16;
constexpr int kConvTileT = 16;

// Single-launch channel-mapped conv4 for small T (T <= kConvFusedMaxT): y must not alias x.
__global__ void conv4_fused_kernel(const float* x, float* hist, const float* w, const float* dyn, int heads, int T,
                                   int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = 0; t < T; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
    hist[c] = h0;
    hist[(size_t) C + c] = h1;
    hist[(size_t) 2 * C + c] = h2;
}

// Channel-mapped tiled conv4 for large T (prefill): reads w and head once per tile, no hist writes.
__global__ void conv4_tiled_kernel(const float* x, const float* hist, const float* w, const float* dyn, int heads,
                                   int T, int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = blockIdx.y * kConvTileT;
    const int t1 = min(T, t0 + kConvTileT);
    auto xp_at = [&](int r) -> float { return r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c]; };
    float h0 = xp_at(t0), h1 = xp_at(t0 + 1), h2 = xp_at(t0 + 2);
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = t0; t < t1; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
}

__global__ void conv4_hist_kernel(const float* x, float* hist, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int j = 0; j < 3; ++j) {   // (row T + j of xp; in place: it reads rows past the ones written)
        const int r = T + j;
        hist[(size_t) j * C + c] = r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
}

__global__ void add_kernel(const float* a, const float* b, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void to_f16_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h(x[i]);
}

__global__ void to_f16_limbs_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t hi = f2h(x[i]);
    y[i] = hi, y[n + i] = f2h(x[i] - h2f(hi));
}
__global__ void softcap_kernel(float* x, size_t n, float cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// The loop's message gates, up to three (a layer's q, k and v inputs) in two launches that spread over the SMs.
// sc [n][T][2 R + 2]: per gate and token, the two halves of each g before the norms (D's x columns . x, its m columns
// . m), then the two sums of squares.
struct LoopGateArgs {
    const float *x[3], *m[3];
    const void *D[3], *F[3], *G[3];
    float* y[3];
};
constexpr int kLoopRankMax = 128;
template <bool Compact> __device__ __forceinline__ float gate_weight(const void* p, size_t i) {
    if constexpr (Compact)
        return __uint_as_float((uint32_t)((const uint16_t*)p)[i] << 16);
    else
        return ((const float*)p)[i];
}

// (1) a block a rank (128 threads): its two dots; rank 0's block also stores the sums of squares
template <bool Compact> __global__ void loop_gate_dots_kernel(LoopGateArgs a, int T, int C, int R, float* sc) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x, j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, warp = tid >> 5;
    const float* xr = a.x[j] + (size_t) t * C;
    const float* mr = a.m[j] + (size_t) t * C;
    const size_t row = (size_t)r * 2 * C;
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += 128) {
        const float xv = xr[c], mv = mr[c];
        ax += gate_weight<Compact>(a.D[j], row + c) * xv, am += gate_weight<Compact>(a.D[j], row + C + c) * mv;
        sx += xv * xv, sm += mv * mv;
    }
    ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
    if ((tid & 31) == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    __syncthreads();
    if (tid == 0) {
        float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
        o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}
// (2) y = (1 - tanh(F g)) x + tanh(G g) m: a block is kBlk channels of a token, a thread a channel; F and G by rank
// ([R][C]), so that a warp's threads read side by side.  y may be x (x's sums were taken in (1)).
template <int Rank, bool Compact>
__global__ void loop_gate_apply_kernel(LoopGateArgs a, int T, int C, int R, float eps, const float* __restrict__ sc) {
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
    for (int i = tid; i < R; i += blockDim.x) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    __syncthreads();
    const int c = (int)blockIdx.x * blockDim.x + tid;
    if (c >= C) return;

    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < (Rank ? Rank : R); ++r) {
        const float gv = sh_g[r];
        f += gate_weight<Compact>(a.F[j], (size_t)r * C + c) * gv,
            gn += gate_weight<Compact>(a.G[j], (size_t)r * C + c) * gv;
    }
    const size_t i = (size_t) t * C + c;
    a.y[j][i] = (1.0f - tanhf(f)) * a.x[j][i] + tanhf(gn) * a.m[j][i];
}
#ifdef STRATA_USE_HIP
#include "ik2_loop_rdna3.cuh"
#endif

// The gates of a chunk of tokens: their two products (g = D [rms(x); rms(m)], then F g and G g) are matrix products
// taken outside, and these are what is left.  (1) g [T][R] = a1 / rms(x) + a2 / rms(m), a1 = x D_x^T and
// a2 = m D_m^T; a block a token.
__global__ void loop_gate_mix_kernel(const float* __restrict__ x, const float* __restrict__ m,
                                     const float* __restrict__ a1, const float* __restrict__ a2, int C, int R, float eps,
                                     float* __restrict__ g) {
    __shared__ float sh[kBlk];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t) t * C;
    const float* mr = m + (size_t) t * C;
    float sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += kBlk) {
        const float xv = xr[c], mv = mr[c];
        sx += xv * xv, sm += mv * mv;
    }
    sx = bsum(sx, sh), sm = bsum(sm, sh);
    if (tid < R) {
        const float rx = rsqrtf(sx / (float) C + eps), rm = rsqrtf(sm / (float) C + eps);
        g[(size_t) t * R + tid] = rx * a1[(size_t) t * R + tid] + rm * a2[(size_t) t * R + tid];
    }
}
// (2) y = (1 - tanh f) x + tanh(gn) m, f = g F and gn = g G; y may be x
__global__ void loop_gate_blend_kernel(const float* x, const float* m, const float* __restrict__ f,
                                       const float* __restrict__ gn, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (1.0f - tanhf(f[i])) * x[i] + tanhf(gn[i]) * m[i];
}

// ---- speculative decoding

__global__ void conv_hist_after_kernel(const float* old, const float* x, int T, int C, float* hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
        const int r = T + j;   // row r of [old; x]
        v[j] = r < 3 ? old[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
    for (int j = 0; j < 3; ++j) hist[(size_t) j * C + c] = v[j];
}

__global__ void to_bf16_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf_rn(x[i]);
}

// a few tokens' short conv (conv4_fused_kernel's arithmetic) with row strides; the history before and the inputs
// kept for a rollback (rec_old [3][C], rec_x [T][C]: conv_hist_after's) when given
template<bool MultiSlot>
__global__ void conv_block_kernel(const float* __restrict__ x, int x_ld, float* hist, const float* __restrict__ w,
                                  const float* __restrict__ dyn, int d_ld, int heads, int T, int C,
                                  const float* __restrict__ res, int res_ld, float* y, int y_ld, float* rec_old,
                                  float* rec_x, const int* slots, int record_stride) {
    if constexpr (MultiSlot) {
        const int first = blockIdx.y * T, slot = slots[first];
        x += (size_t)first * x_ld;
        if (dyn) dyn += (size_t)first * d_ld;
        if (res) res += (size_t)first * res_ld;
        y += (size_t)first * y_ld;
        hist += (size_t)slot * 3 * C;
        rec_old += (size_t)slot * 3 * C;
        rec_x += (size_t)slot * record_stride * C;
    }
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    if (rec_old) rec_old[c] = h0, rec_old[(size_t) C + c] = h1, rec_old[(size_t) 2 * C + c] = h2;
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = dyn ? dyn + (size_t) (c / (C / heads)) * 4 : nullptr;
    for (int t = 0; t < T; ++t) {
        const float xn = x[(size_t) t * x_ld + c];
        if (rec_x) rec_x[(size_t) t * C + c] = xn;
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xn;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_ld;
            a += dc[0] * xn + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (res) a += res[(size_t) t * res_ld + c];
        y[(size_t) t * y_ld + c] = a;
        h0 = h1, h1 = h2, h2 = xn;
    }
    hist[c] = h0, hist[(size_t) C + c] = h1, hist[(size_t) 2 * C + c] = h2;
}

// every short conv's history as after the first n of a window's tokens, all the layers' convs in one launch
__global__ void rollback_block_kernel(const ConvRec* __restrict__ recs, int n) {
    const ConvRec rc = recs[blockIdx.y];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < rc.C; c += gridDim.x * blockDim.x) {
        float v[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int r = n + j;
            v[j] = r < 3 ? rc.old[(size_t) r * rc.C + c] : rc.x[(size_t) (r - 3) * rc.C + c];
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) rc.hist[(size_t) j * rc.C + c] = v[j];
    }
}

}  // namespace

__global__ void project_out_kernel(float* __restrict__ y, const float* __restrict__ r, float s, float c, int D,
                                   const float* __restrict__ srow, bool keep_norm) {
    float* row = y + (size_t) blockIdx.x * D;
    __shared__ float part[256], part2[256];
    float d = 0.0f, n2 = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) d += row[i] * r[i], n2 += row[i] * row[i];
    part[threadIdx.x] = d;
    part2[threadIdx.x] = n2;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w], part2[threadIdx.x] += part2[threadIdx.x + w];
        __syncthreads();
    }
    const float k = (srow ? srow[blockIdx.x] * s : s) * (part[0] - c);
    float g = 1.0f;   // |y|^2 after: |y|^2 - 2 k (r . y) + k^2 (r unit)
    if (keep_norm) {
        const float after = part2[0] - 2.0f * k * part[0] + k * k;
        g = after > 1e-30f ? sqrtf(part2[0] / after) : 1.0f;
    }
    for (int i = threadIdx.x; i < D; i += blockDim.x) row[i] = (row[i] - k * r[i]) * g;
}
void project_out(float* y, const float* r, float s, float c, int T, int D, cudaStream_t st, const float* srow,
                 bool keep_norm) {
    if (T <= 0) return;
    project_out_kernel<<<T, 256, 0, st>>>(y, r, s, c, D, srow, keep_norm);
    check("project_out");
}
void rms(const float* x, const float* w, int rows, int D, float eps, float* y, cudaStream_t s) {
    if (rows <= 0) return;
    rms_impl_kernel<false, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, nullptr, y);
    check("rms");
}
void rms_multi(const RmsJobs& jobs, int n, int D, float eps, cudaStream_t s) {
    if (n <= 0) return;
    if (n > 3) {
        std::fprintf(stderr, "ik2::k::rms_multi: %d vectors\n", n);
        std::abort();
    }
    rms_multi_kernel<<<(unsigned) n, kBlk, 0, s>>>(jobs, D, eps);
    check("rms_multi");
}
void rms_f16(const float* x, const float* w, int rows, int D, float eps, uint16_t* y16, float* y32, cudaStream_t s) {
    if (rows <= 0) return;
    if (y16 && y32) rms_impl_kernel<true, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, y16, y32);
    else if (y16) rms_impl_kernel<true, false><<<rows, kBlk, 0, s>>>(x, w, D, eps, y16, nullptr);
    else if (y32) rms_impl_kernel<false, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, nullptr, y32);
    check("rms_f16");
}

void l2clip(const float* x, const float* scale, int rows, int G, int D, float th, const int* pos, float alpha,
            float n_floor, float* y, cudaStream_t s) {
    if (rows <= 0) return;
    l2clip_kernel<<<rows * G, 128, 0, s>>>(x, scale, G, D, th, pos, alpha, n_floor, y);
    check("l2clip");
}

void conv4_res(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C,
               const float* residual, float* y, cudaStream_t s) {
    if (T <= 0) return;
    if (T <= kConvFusedMaxT) {
        conv4_fused_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(x, hist, w, dyn, heads, T, C, residual, y);
        check("conv4_res");
        return;
    }
    const dim3 g((unsigned) ((C + 127) / 128), (unsigned) ((T + kConvTileT - 1) / kConvTileT));
    conv4_tiled_kernel<<<g, 128, 0, s>>>(x, hist, w, dyn, heads, T, C, residual, y);
    check("conv4_res");
    conv4_hist_kernel<<<grid(C), kBlk, 0, s>>>(x, hist, T, C);
    check("conv4_hist");
}

void conv4(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C, float* y,
           cudaStream_t s) {
    conv4_res(x, hist, w, dyn, heads, T, C, nullptr, y, s);
}

void add(const float* a, const float* b, size_t n, float* y, cudaStream_t s) {
    if (!n) return;
    add_kernel<<<grid(n), kBlk, 0, s>>>(a, b, n, y);
    check("add");
}
void to_f16(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_f16_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_f16");
}

void to_f16_limbs(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_f16_limbs_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_f16_limbs");
}
void softcap(float* x, size_t n, float cap, cudaStream_t s) {
    if (!n) return;
    softcap_kernel<<<grid(n), kBlk, 0, s>>>(x, n, cap);
    check("softcap");
}

void loop_gates(const LoopGate* gt, int n, int T, int C, int R, float eps, float* sc, cudaStream_t s, bool sequences) {
    if (T <= 0 || n <= 0) return;
    if (n > 3 || R > kLoopRankMax || T > 65535) {
        std::fprintf(stderr, "ik2::k::loop_gates: %d gates, rank %d, %d tokens\n", n, R, T);
        std::abort();
    }
    bool compact = true;
    for (int j = 0; j < n; ++j)
        compact = compact && gt[j].compact[0] && gt[j].compact[1] && gt[j].compact[2];
    LoopGateArgs a{};
    for (int j = 0; j < n; ++j) {
        a.x[j] = gt[j].x;
        a.m[j] = gt[j].m;
        a.y[j] = gt[j].y;
        a.D[j] = compact ? (const void*)gt[j].compact[0] : gt[j].D;
        a.F[j] = compact ? (const void*)gt[j].compact[1] : gt[j].F;
        a.G[j] = compact ? (const void*)gt[j].compact[2] : gt[j].G;
    }
    static const int tile = getenv("IK2_GATE_TILE") ? atoi(getenv("IK2_GATE_TILE")) : 0;
#ifdef STRATA_USE_HIP
    bool aligned = true;
    for (int j = 0; j < n; ++j) {
        aligned &= !(((uintptr_t)a.x[j] | (uintptr_t)a.m[j]) & 15) && !((uintptr_t)a.D[j] & 7);
    }
    if (tile == 256 && (T == 1 || sequences) && R == 64 && C % 64 == 0 && compact && aligned) {
        rdna3_gate_dots<<<dim3(R, n, T), 128, 0, s>>>(a, C, sc);
        check("rdna3_gate_dots");
        rdna3_gate_apply<<<dim3(C / 64, n, T), 256, 0, s>>>(a, C, eps, sc);
        check("rdna3_gate_apply");
        return;
    }
    if ((tile == 256 || tile == 0) && T >= 2 && T <= 8 && R == 64 && C % 64 == 0 && compact && aligned) {
        const dim3 dg(R, n), ag(C / 64, n);
        switch (T) {
            case 2: rdna3_gate_dots_block<2><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<2><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 3: rdna3_gate_dots_block<3><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<3><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 4: rdna3_gate_dots_block<4><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<4><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 5: rdna3_gate_dots_block<5><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<5><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 6: rdna3_gate_dots_block<6><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<6><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 7: rdna3_gate_dots_block<7><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<7><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
            case 8: rdna3_gate_dots_block<8><<<dg, 128, 0, s>>>(a, n, C, sc); rdna3_gate_apply_block<8><<<ag, 256, 0, s>>>(a, n, C, eps, sc); break;
        }
        check("rdna3_gate_block");
        return;
    }
#endif
    const dim3 dots(R, n, T);
    if (compact)
        loop_gate_dots_kernel<true><<<dots, 128, 0, s>>>(a, T, C, R, sc);
    else
        loop_gate_dots_kernel<false><<<dots, 128, 0, s>>>(a, T, C, R, sc);
    check("loop_gates (dots)");
    const bool fixed = T == 1 && R == 64 && (tile == 32 || tile == 64 || tile == 128);
    const int threads = fixed ? tile : kBlk;
    const dim3 apply((C + threads - 1) / threads, n, T);
    if (fixed && compact)
        loop_gate_apply_kernel<64, true><<<apply, threads, 0, s>>>(a, T, C, R, eps, sc);
    else if (fixed)
        loop_gate_apply_kernel<64, false><<<apply, threads, 0, s>>>(a, T, C, R, eps, sc);
    else if (compact)
        loop_gate_apply_kernel<0, true><<<apply, threads, 0, s>>>(a, T, C, R, eps, sc);
    else
        loop_gate_apply_kernel<0, false><<<apply, threads, 0, s>>>(a, T, C, R, eps, sc);
    check("loop_gates (apply)");
}
void loop_gate(const float* x, const float* m, const float* D, const float* F, const float* G, int T, int C, int R,
               float eps, float* sc, float* y, cudaStream_t s) {
    const LoopGate g{x, m, D, F, G, y};
    loop_gates(&g, 1, T, C, R, eps, sc, s);
}

void loop_gate_mix(const float* x, const float* m, const float* a1, const float* a2, int T, int C, int R, float eps,
                   float* g, cudaStream_t s) {
    if (T <= 0) return;
    if (R > kBlk) {
        std::fprintf(stderr, "ik2::k::loop_gate_mix: rank %d\n", R);
        std::abort();
    }
    loop_gate_mix_kernel<<<(unsigned) T, kBlk, 0, s>>>(x, m, a1, a2, C, R, eps, g);
    check("loop_gate_mix");
}
void loop_gate_blend(const float* x, const float* m, const float* f, const float* gn, size_t n, float* y,
                     cudaStream_t s) {
    if (!n) return;
    loop_gate_blend_kernel<<<grid(n), kBlk, 0, s>>>(x, m, f, gn, n, y);
    check("loop_gate_blend");
}

void conv_hist_after(const float* old, const float* x, int T, int C, float* hist, cudaStream_t s) {
    conv_hist_after_kernel<<<grid(C), kBlk, 0, s>>>(old, x, T, C, hist);
    check("conv_hist_after");
}
void to_bf16(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_bf16_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_bf16");
}

void conv_block(const float* x, int x_ld, float* hist, const float* w, const float* dyn, int d_ld, int heads, int T,
                int C, const float* res, int res_ld, float* y, int y_ld, float* rec_old, float* rec_x,
                cudaStream_t s) {
    if (T <= 0) return;
    conv_block_kernel<false><<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(x, x_ld, hist, w, dyn, d_ld, heads > 0 ? heads : 1,
                                                                   T, C, res, res_ld, y, y_ld, rec_old, rec_x, nullptr, 0);
    check("conv_block");
}
void conv_windows(const int* slots, int rows, int window, int record_stride, const float* x, int ldx,
                  MvConv cv, float* old, float* inputs, cudaStream_t s) {
    conv_block_kernel<true><<<dim3((cv.n + 127) / 128, rows / window), 128, 0, s>>>(
        x, ldx, cv.hist, cv.w, cv.dyn, cv.heads * 4, cv.heads, window, cv.n, cv.res, cv.n,
        cv.y, cv.n, old, inputs, slots, record_stride);
    check("conv_windows");
}
void rollback_block(const ConvRec* recs, int n_recs, int n, cudaStream_t s) {
    if (n_recs <= 0) return;
    rollback_block_kernel<<<dim3(8, (unsigned) n_recs), 256, 0, s>>>(recs, n);
    check("rollback_block");
}

}  // namespace ik2::k
