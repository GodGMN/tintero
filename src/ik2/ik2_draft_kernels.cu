#include "ik2_kernel_common.cuh"

namespace ik2::k {
namespace {

__global__ void mix_rms_kernel(const float* x, int n_tap, size_t tap_stride, int C, const float* alpha, const float* w,
                               float eps, float* out) {
    __shared__ float sh[kBlk];
    const int row = blockIdx.x;
    float a = 0.0f;
    for (int c = threadIdx.x; c < C; c += kBlk) {
        float m = 0.0f;
        for (int t = 0; t < n_tap; ++t) m += alpha[t] * x[(size_t) t * tap_stride + (size_t) row * C + c];
        out[(size_t) row * C + c] = m;
        a += m * m;
    }
    const float r = rsqrtf(bsum(a, sh) / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += kBlk) out[(size_t) row * C + c] *= r * w[c];
}

__global__ void rope_neox_kernel(float* x, int heads, const int* pos, float theta) {
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i < 64
    const float inv = 1.0f / powf(theta, (float) (2 * i) / 128.0f);
    const float ang = (float) pos[row] * inv;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float* p = x + ((size_t) row * heads + h) * 128;
    const float a = p[i], b = p[i + 64];
    p[i] = a * cs - b * sn;
    p[i + 64] = b * cs + a * sn;
}

// a block a (head, query): the logits a key a warp, softmax, the values
// the drafter's attention, flash-decoding style: a block a (head, key chunk) for all the block's queries (nq <= 8) -
// a key's K and V rows read once for them - its unnormalised output and (max, sum) a query, then merged.  Keys: the
// context [k_lo, k_hi) (rows of a ring of cap: (k_lo + j) % cap) and the block's own nq (kb, vb).
constexpr int kDraftQ = 8;
__device__ __forceinline__ float ld_f(const float* p) { return *p; }
__device__ __forceinline__ float ld_f(const uint16_t* p) { return h2f(*p); }
template <class KVT, bool row_window>
__global__ void draft_attn_online_part_kernel(const float* __restrict__ q, const KVT* __restrict__ kc,
                                              const KVT* __restrict__ vc, const int* __restrict__ d_P, int k_lo_arg,
                                              int k_hi_arg, int window, const float* __restrict__ kb,
                                              const float* __restrict__ vb, int nq, int H, int Hkv, float scale,
                                              float* __restrict__ part_o, float2* __restrict__ part_ml, int cap, const int* slots = nullptr) {
    if constexpr (row_window) {
        const int i = blockIdx.z;
        const size_t cache = slots ? size_t(slots[i]) * cap * Hkv * 128 : 0;
        kc += cache;
        vc += cache;
        q += i * nq * H * 128;
        kb += i * nq * Hkv * 128;
        vb += i * nq * Hkv * 128;
        d_P += i * nq;
        const int partial = i * gridDim.y * H * kDraftQ;
        part_o += partial * 128;
        part_ml += partial;
    }
    __shared__ float qs[kDraftQ][128];
    __shared__ float wm[4];
    __shared__ float wl[4];
    __shared__ float wo[4][128];
    const int h = blockIdx.x, sp = blockIdx.y, S = gridDim.y, g = h / (H / Hkv);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int k_hi = d_P ? *d_P : k_hi_arg;
    const int k_lo = d_P ? max(0, k_hi - window) : k_lo_arg;
    const int KD = Hkv * 128, nc = k_hi - k_lo, nk = nc + nq;
    const int per = (nk + S - 1) / S;
    const int j0 = min(nk, sp * per), j1 = min(nk, j0 + per), n = j1 - j0;
    for (int qi = 0; qi < nq; ++qi) qs[qi][tid] = q[((size_t) qi * H + h) * 128 + tid];
    __syncthreads();
    float mx[kDraftQ], sm[kDraftQ], o0[kDraftQ], o1[kDraftQ], o2[kDraftQ], o3[kDraftQ];
#pragma unroll
    for (int qi = 0; qi < kDraftQ; ++qi) {
        mx[qi] = -INFINITY;
        sm[qi] = 0.0f;
        o0[qi] = 0.0f; o1[qi] = 0.0f; o2[qi] = 0.0f; o3[qi] = 0.0f;
    }
    for (int jj = warp; jj < n; jj += 4) {
        const int j = j0 + jj;
        float k0, k1, k2, k3, v0, v1, v2, v3;
        if (j < nc) {
            const size_t off = (size_t) ((k_lo + j) % cap) * KD + (size_t) g * 128;
            const KVT* kr = kc + off;
            const KVT* vr = vc + off;
            k0 = ld_f(kr + lane); k1 = ld_f(kr + lane + 32); k2 = ld_f(kr + lane + 64); k3 = ld_f(kr + lane + 96);
            v0 = ld_f(vr + lane); v1 = ld_f(vr + lane + 32); v2 = ld_f(vr + lane + 64); v3 = ld_f(vr + lane + 96);
        } else {
            const size_t off = (size_t) (j - nc) * KD + (size_t) g * 128;
            const float* kr = kb + off;
            const float* vr = vb + off;
            k0 = kr[lane]; k1 = kr[lane + 32]; k2 = kr[lane + 64]; k3 = kr[lane + 96];
            v0 = vr[lane]; v1 = vr[lane + 32]; v2 = vr[lane + 64]; v3 = vr[lane + 96];
        }
#pragma unroll
        for (int qi = 0; qi < kDraftQ; ++qi) {
            if (qi >= nq) break;
            if (row_window && j < nc && k_lo + j < k_hi + qi - window) continue;
            float a = qs[qi][lane] * k0 + qs[qi][lane + 32] * k1 + qs[qi][lane + 64] * k2 + qs[qi][lane + 96] * k3;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            a *= scale;
            const float m_new = fmaxf(mx[qi], a);
            const float alpha = mx[qi] == -INFINITY ? 0.0f : expf(mx[qi] - m_new);
            const float beta = expf(a - m_new);
            sm[qi] = sm[qi] * alpha + beta;
            mx[qi] = m_new;
            o0[qi] = fmaf(o0[qi], alpha, v0 * beta);
            o1[qi] = fmaf(o1[qi], alpha, v1 * beta);
            o2[qi] = fmaf(o2[qi], alpha, v2 * beta);
            o3[qi] = fmaf(o3[qi], alpha, v3 * beta);
        }
    }
    for (int qi = 0; qi < nq; ++qi) {
        if (lane == 0) wm[warp] = mx[qi], wl[warp] = sm[qi];
        wo[warp][lane] = o0[qi];
        wo[warp][lane + 32] = o1[qi];
        wo[warp][lane + 64] = o2[qi];
        wo[warp][lane + 96] = o3[qi];
        __syncthreads();
        const float bm = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
        const float c0 = wm[0] == -INFINITY ? 0.0f : expf(wm[0] - bm);
        const float c1 = wm[1] == -INFINITY ? 0.0f : expf(wm[1] - bm);
        const float c2 = wm[2] == -INFINITY ? 0.0f : expf(wm[2] - bm);
        const float c3 = wm[3] == -INFINITY ? 0.0f : expf(wm[3] - bm);
        if (tid == 0)
            part_ml[((size_t) sp * H + h) * kDraftQ + qi] =
                make_float2(bm, wl[0] * c0 + wl[1] * c1 + wl[2] * c2 + wl[3] * c3);
        part_o[(((size_t) sp * H + h) * kDraftQ + qi) * 128 + tid] =
            wo[0][tid] * c0 + wo[1][tid] * c1 + wo[2][tid] * c2 + wo[3][tid] * c3;
        __syncthreads();
    }
}
__global__ void draft_attn_merge_kernel(const float* part_o, const float2* part_ml, int S, int H, float* out) {
    const int partial = blockIdx.z * S * H * kDraftQ;
    part_o += partial * 128;
    part_ml += partial;
    out += blockIdx.z * gridDim.y * H * 128;
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x;
    float m = -INFINITY;
    for (int sp = 0; sp < S; ++sp) m = fmaxf(m, part_ml[((size_t) sp * H + h) * kDraftQ + qi].x);
    float den = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < S; ++sp) {
        const float2 ml = part_ml[((size_t) sp * H + h) * kDraftQ + qi];
        const float c = ml.x == -INFINITY ? 0.0f : expf(ml.x - m);
        den += ml.y * c;
        acc += part_o[(((size_t) sp * H + h) * kDraftQ + qi) * 128 + tid] * c;
    }
    out[((size_t) qi * H + h) * 128 + tid] = acc / den;
}

__global__ void silu_mul_bf16_kernel(const float* gu, int T, int F, uint16_t* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * F) return;
    const size_t t = i / F, f = i % F;
    const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = f2bf_rn(g / (1.0f + expf(-g)) * u);
}

__global__ void rnn_inputs_kernel(const float* state, const uint16_t* emb, const int* prev, const float* hidden,
                                  const float* f, int R, int C, uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int W = 2 * R + 2 * C;
    if (i >= W) return;
    float v;
    if (i < R) v = state[i];
    else if (i < 2 * R) v = bf2f(emb[(size_t) (*prev) * R + (i - R)]);
    else if (i < 2 * R + C) v = hidden[i - 2 * R];
    else v = f[i - 2 * R - C];
    out[i] = f2bf_rn(v);
}

__global__ void rnn_gates_kernel(const float* proj, const float* bias, int R, float* state, float* latent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float gr = proj[i] + bias[i], cr = proj[R + i] + bias[R + i], orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + expf(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(cr);
    latent[i] = tanhf(orw);
}

}  // namespace

void mix_rms(const float* x, int n_tap, size_t tap_stride, int rows, int C, const float* alpha, const float* w,
             float eps, float* out, cudaStream_t s) {
    if (rows <= 0) return;
    mix_rms_kernel<<<rows, kBlk, 0, s>>>(x, n_tap, tap_stride, C, alpha, w, eps, out);
    check("mix_rms");
}
void rope_neox(float* x, int rows, int heads, const int* pos, float theta, cudaStream_t s) {
    if (rows <= 0) return;
    rope_neox_kernel<<<dim3(rows, heads), 64, 0, s>>>(x, heads, pos, theta);
    check("rope_neox");
}
template <class KVT, bool row_window = false>
static void draft_attn_online_impl(const float* q, const KVT* kc, const KVT* vc, const int* d_P, int k_lo, int k_hi,
                                   int window, const float* kb, const float* vb, int nq, int H, int Hkv, float scale,
                                   int S, float* scratch, float* out, cudaStream_t s, int cap, int active = 1, const int* slots = nullptr) {
    if (nq > kDraftQ) {
        std::fprintf(stderr, "ik2::k::draft_attn: %d queries (at most %d)\n", nq, kDraftQ);
        std::abort();
    }
    float* part_o = scratch;
    float2* part_ml = (float2*) (part_o + (size_t) active * S * H * kDraftQ * 128);
    draft_attn_online_part_kernel<KVT, row_window><<<dim3((unsigned) H, (unsigned) S, active), 128, 0, s>>>(
        q, kc, vc, d_P, k_lo, k_hi, window, kb, vb, nq, H, Hkv, scale, part_o, part_ml, cap > 0 ? cap : (1 << 30), slots);
    check("draft_attn_online (chunks)");
    draft_attn_merge_kernel<<<dim3((unsigned) H, (unsigned) nq, active), 128, 0, s>>>(part_o, part_ml, S, H, out);
    check("draft_attn_online (merge)");
}
void draft_attn(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb, const float* vb,
                int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s, int cap) {
    const int nk = k_hi - k_lo + nq;
    const int S = std::max(1, std::min(64, (nk + 255) / 256));
    draft_attn_online_impl<float>(q, kc, vc, nullptr, k_lo, k_hi, 0, kb, vb, nq, H, Hkv, scale, S, scratch, out, s, cap);
}
void draft_attn_f16(const float* q, const uint16_t* kc, const uint16_t* vc, int k_lo, int k_hi, const float* kb,
                    const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s,
                    int cap) {
    const int nk = k_hi - k_lo + nq;
    const int S = std::max(1, std::min(64, (nk + 255) / 256));
    draft_attn_online_impl<uint16_t>(q, kc, vc, nullptr, k_lo, k_hi, 0, kb, vb, nq, H, Hkv, scale, S, scratch, out, s, cap);
}
void draft_attn_graph(const float* q, const float* kc, const float* vc, const int* d_P, int window, const float* kb,
                      const float* vb, int nq, int H, int Hkv, float scale, int S, float* scratch, float* out,
                      cudaStream_t s, int cap) {
    draft_attn_online_impl<float>(q, kc, vc, d_P, 0, 0, window, kb, vb, nq, H, Hkv, scale, S, scratch, out, s, cap);
}
void dflash2_attn(const float* q, const float* kc, const float* vc, const int* d_P, int window, const float* kb,
                      const float* vb, int nq, int H, int Hkv, float scale, int S, float* scratch, float* out,
                      cudaStream_t s, int cap, int active, const int* slots) {
    draft_attn_online_impl<float, true>(q, kc, vc, d_P, 0, 0, window, kb, vb, nq, H, Hkv, scale, S, scratch, out, s, cap, active, slots);
}
void draft_attn_f16_graph(const float* q, const uint16_t* kc, const uint16_t* vc, const int* d_P, int window,
                          const float* kb, const float* vb, int nq, int H, int Hkv, float scale, int S, float* scratch,
                          float* out, cudaStream_t s, int cap) {
    draft_attn_online_impl<uint16_t>(q, kc, vc, d_P, 0, 0, window, kb, vb, nq, H, Hkv, scale, S, scratch, out, s, cap);
}
void silu_mul_bf16(const float* gu, int T, int F, uint16_t* out, cudaStream_t s) {
    silu_mul_bf16_kernel<<<grid((size_t) T * F), kBlk, 0, s>>>(gu, T, F, out);
    check("silu_mul_bf16");
}
void rnn_inputs(const float* state, const uint16_t* emb, const int* prev, const float* hidden, const float* f, int R,
                int C, uint16_t* out, cudaStream_t s) {
    rnn_inputs_kernel<<<grid((size_t) 2 * R + 2 * C), kBlk, 0, s>>>(state, emb, prev, hidden, f, R, C, out);
    check("rnn_inputs");
}
void rnn_gates(const float* proj, const float* bias, int R, float* state, float* latent, cudaStream_t s) {
    rnn_gates_kernel<<<grid(R), kBlk, 0, s>>>(proj, bias, R, state, latent);
    check("rnn_gates");
}

}  // namespace ik2::k
