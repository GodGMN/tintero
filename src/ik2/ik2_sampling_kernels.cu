#include "ik2_kernel_common.cuh"

namespace ik2::k {
namespace {

// ---- sampling: the largest logits found on the device - the host gets a thousand or two candidates and not the
// vocabulary (copying 200,000 logits out and sorting them was 0.8 ms a token).  A histogram of the values a block,
// the bin from which the bins hold K values at least, the values of those bins gathered.
constexpr int kCandBins = 1024;
__device__ __forceinline__ int cand_bin(float x, float lo, float inv_w) {
    return min(kCandBins - 1, max(0, (int) ((x - lo) * inv_w)));
}
__global__ void cand_hist_kernel(const float* __restrict__ x, int n, float lo, float inv_w, int per,
                                 int* __restrict__ part) {
    __shared__ int h[kCandBins];
    for (int j = threadIdx.x; j < kCandBins; j += blockDim.x) h[j] = 0;
    __syncthreads();
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) atomicAdd(&h[cand_bin(x[i], lo, inv_w)], 1);
    __syncthreads();
    for (int j = threadIdx.x; j < kCandBins; j += blockDim.x) part[(size_t) blockIdx.x * kCandBins + j] = h[j];
}
// a thread a bin (kCandBins threads): head[1] = the bin from which the bins hold K values (0 if they all hold
// fewer), head[0] = 0 (the gathered count starts there)
__global__ void cand_threshold_kernel(const int* __restrict__ part, int n_part, int K, int* head) {
    __shared__ int cnt[kCandBins];
    __shared__ int wcnt[kCandBins / 32];
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    int c = 0;
    for (int b = 0; b < n_part; ++b) c += part[(size_t) b * kCandBins + j];
    cnt[j] = c;
    int ws = c;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
    if (lane == 0) wcnt[warp] = ws;
    __syncthreads();
    int above = 0;   // the values of the bins over this one
    for (int w = warp + 1; w < kCandBins / 32; ++w) above += wcnt[w];
    for (int i = j + 1; i < (warp + 1) * 32; ++i) above += cnt[i];
    if (above < K && above + c >= K) head[1] = j;
    if (j == 0) {
        head[0] = 0;
        if (above + c < K) head[1] = 0;
    }
}
__global__ void cand_gather_kernel(const float* __restrict__ x, int n, float lo, float inv_w, int cap, int* head,
                                   int* idx, float* val) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    if (cand_bin(v, lo, inv_w) < head[1]) return;
    const int at = atomicAdd(head, 1);
    if (at < cap) idx[at] = i, val[at] = v;
}

constexpr int kArgmaxMaxPartials = 1024;
__device__ float g_argmax_v[kArgmaxMaxPartials];
__device__ int g_argmax_i[kArgmaxMaxPartials];

__device__ __forceinline__ void argmax_better(float v, int i, float& best, int& idx) {
    if (v > best || (v == best && i < idx)) best = v, idx = i;
}

__device__ __forceinline__ void argmax_block_reduce(float& best, int& idx) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        argmax_better(ob, oi, best, idx);
    }
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) bv[warp] = best, bi[warp] = idx;
    __syncthreads();
    if (warp == 0) {
        best = (lane < nw) ? bv[lane] : -INFINITY;
        idx = (lane < nw) ? bi[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            argmax_better(ob, oi, best, idx);
        }
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_sum_part_kernel(const float* a, const float* b, const float* c, int n, int* out) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) a & 15) == 0) && (!b || ((uintptr_t) b & 15) == 0) && (!c || ((uintptr_t) c & 15) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        const float4* a4 = (const float4*) a;
        const float4* b4 = (const float4*) b;
        const float4* c4 = (const float4*) c;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const float4 va = __ldg(a4 + v4);
            const float4 vb = b4 ? __ldg(b4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 vc = c4 ? __ldg(c4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const int base = v4 << 2;
            argmax_better(va.x + vb.x + vc.x, base + 0, best, idx);
            argmax_better(va.y + vb.y + vc.y, base + 1, best, idx);
            argmax_better(va.z + vb.z + vc.z, base + 2, best, idx);
            argmax_better(va.w + vb.w + vc.w, base + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    } else {
        for (int v = gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[blockIdx.x] = best, g_argmax_i[blockIdx.x] = idx;
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_rows_part_kernel(const float* x, int ld, int n_cols, int* out) {
    const int r = blockIdx.y;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    const float* row = x + (size_t) r * ld;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    if (((uintptr_t) row & 15) == 0) {
        const int n4 = n_cols >> 2;
        const float4* r4 = (const float4*) row;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const float4 f = __ldg(r4 + v4);
            const int base = v4 << 2;
            argmax_better(f.x, base + 0, best, idx);
            argmax_better(f.y, base + 1, best, idx);
            argmax_better(f.z, base + 2, best, idx);
            argmax_better(f.w, base + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n_cols; v += gstride) argmax_better(__ldg(row + v), v, best, idx);
    } else {
        for (int v = gtid; v < n_cols; v += gstride) argmax_better(__ldg(row + v), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) out[r] = idx;
        else g_argmax_v[r * gridDim.x + blockIdx.x] = best, g_argmax_i[r * gridDim.x + blockIdx.x] = idx;
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_gemv_f16_part_kernel(const uint16_t* W, int ld, const float* x, const float* base,
                                            const uint16_t* bias_f16, int R, int n, bool use_smem, int* out) {
    extern __shared__ float sx[];
    if (use_smem) {
        for (int r = threadIdx.x; r < R; r += blockDim.x) sx[r] = x[r];
        __syncthreads();
    }
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) W & 7) == 0) && ((ld & 3) == 0) &&
                      (!base || ((uintptr_t) base & 15) == 0) && (!bias_f16 || ((uintptr_t) bias_f16 & 7) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const int v = v4 << 2;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld) {
                const float xr = use_smem ? sx[r] : __ldg(x + r);
                const uint2 w4 = __ldg((const uint2*) wr);
                const float2 f0 = u32_to_f2(w4.x), f1 = u32_to_f2(w4.y);
                d0 = fmaf(f0.x, xr, d0);
                d1 = fmaf(f0.y, xr, d1);
                d2 = fmaf(f1.x, xr, d2);
                d3 = fmaf(f1.y, xr, d3);
            }
            if (base) {
                const float4 bs = __ldg((const float4*) (base + v));
                d0 += bs.x, d1 += bs.y, d2 += bs.z, d3 += bs.w;
            }
            if (bias_f16) {
                const uint2 b4 = __ldg((const uint2*) (bias_f16 + v));
                const float2 f0 = u32_to_f2(b4.x), f1 = u32_to_f2(b4.y);
                d0 += f0.x, d1 += f0.y, d2 += f1.x, d3 += f1.y;
            }
            argmax_better(d0, v + 0, best, idx);
            argmax_better(d1, v + 1, best, idx);
            argmax_better(d2, v + 2, best, idx);
            argmax_better(d3, v + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld)
                d = fmaf(h2f(__ldg(wr)), use_smem ? sx[r] : __ldg(x + r), d);
            if (base) d += __ldg(base + v);
            if (bias_f16) d += h2f(__ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    } else {
        for (int v = gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld)
                d = fmaf(h2f(__ldg(wr)), use_smem ? sx[r] : __ldg(x + r), d);
            if (base) d += __ldg(base + v);
            if (bias_f16) d += h2f(__ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[blockIdx.x] = best, g_argmax_i[blockIdx.x] = idx;
    }
}

__global__ void argmax_final_kernel(int n_partials, int stride_partials, int* out) {
    const int r = blockIdx.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < n_partials; j += blockDim.x)
        argmax_better(g_argmax_v[r * stride_partials + j], g_argmax_i[r * stride_partials + j], best, idx);
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) out[r] = idx;
}

}  // namespace

constexpr int kCandPer = 8192;   // values a block of the histogram
size_t cand_scratch(int n) { return (size_t) ((n + kCandPer - 1) / kCandPer) * kCandBins; }
void cand_topk(const float* x, int n, float lo, float hi, int K, int cap, int* scratch, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const float inv_w = (float) kCandBins / (hi - lo);
    const int n_part = (n + kCandPer - 1) / kCandPer;
    cand_hist_kernel<<<(unsigned) n_part, kBlk, 0, s>>>(x, n, lo, inv_w, kCandPer, scratch);
    check("cand_topk (histogram)");
    cand_threshold_kernel<<<1, kCandBins, 0, s>>>(scratch, n_part, K, out);
    check("cand_topk (threshold)");
    cand_gather_kernel<<<grid((size_t) n), kBlk, 0, s>>>(x, n, lo, inv_w, cap, out, out + 2, (float*) (out + 2 + cap));
    check("cand_topk (gather)");
}

void argmax_sum(const float* a, const float* b, const float* c, int n, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const int B = std::min(256, std::max(1, (n + 1023) / 1024));
    if (B == 1) {
        argmax_sum_part_kernel<true><<<1, 256, 0, s>>>(a, b, c, n, out);
        check("argmax_sum");
        return;
    }
    argmax_sum_part_kernel<false><<<B, 256, 0, s>>>(a, b, c, n, out);
    check("argmax_sum_part");
    argmax_final_kernel<<<1, 256, 0, s>>>(B, 1, out);
    check("argmax_sum_final");
}
void argmax_rows(const float* x, int ld, int n_rows, int n_cols, int* out, cudaStream_t s) {
    if (n_rows <= 0 || n_cols <= 0) return;
    const int max_b = std::max(1, kArgmaxMaxPartials / n_rows);
    const int B = std::min(max_b, std::min(256, std::max(1, (n_cols + 1023) / 1024)));
    if (B == 1) {
        argmax_rows_part_kernel<true><<<dim3(1, (unsigned) n_rows), 256, 0, s>>>(x, ld, n_cols, out);
        check("argmax_rows");
        return;
    }
    argmax_rows_part_kernel<false><<<dim3((unsigned) B, (unsigned) n_rows), 256, 0, s>>>(x, ld, n_cols, out);
    check("argmax_rows_part");
    argmax_final_kernel<<<(unsigned) n_rows, 256, 0, s>>>(B, B, out);
    check("argmax_rows_final");
}
void argmax_gemv_f16(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias_f16,
                     int R, int n, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const int n4 = std::max(1, n >> 2);
    const int B = std::min(512, std::max(1, (n4 + 127) / 128));
    const bool use_smem = R <= 4096;
    const size_t smem = use_smem ? (size_t) R * sizeof(float) : 0;
    if (B == 1) {
        argmax_gemv_f16_part_kernel<true><<<1, 128, smem, s>>>(W, ld, x, base, bias_f16, R, n, use_smem, out);
        check("argmax_gemv_f16");
        return;
    }
    argmax_gemv_f16_part_kernel<false><<<B, 128, smem, s>>>(W, ld, x, base, bias_f16, R, n, use_smem, out);
    check("argmax_gemv_f16_part");
    argmax_final_kernel<<<1, 256, 0, s>>>(B, 1, out);
    check("argmax_gemv_f16_final");
}

}  // namespace ik2::k
