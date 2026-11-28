#include "ik2_engine_internal.cuh"

namespace ik2::engine {

__global__ void gather_f16_16_kernel(const uint16_t* __restrict__ x16, const int* __restrict__ idx, int n, int C,
                                     uint16_t* __restrict__ y16) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint4* src = (const uint4*) (x16 + (size_t) idx[row] * C);
    uint4* dst = (uint4*) (y16 + (size_t) row * C);
    const int n4 = C >> 3;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) dst[i] = src[i];
}

// out[idx[row]] += wr[row] * y[row] over rows (one expert's tokens: each token at most once, so no two rows collide)
__global__ void scatter_add_rows_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                        const float* __restrict__ wr, int C, float* __restrict__ out, int ld) {
    const int row = blockIdx.x;
    const float w = wr[row];
    const float4* src = (const float4*) (y + (size_t) row * C);
    float4* dst = (float4*) (out + (size_t) idx[row] * ld);
    for (int i = threadIdx.x; i < (C >> 2); i += blockDim.x) {
        const float4 a = src[i];
        float4 b = dst[i];
        b.x += w * a.x, b.y += w * a.y, b.z += w * a.z, b.w += w * a.w;
        dst[i] = b;
    }
}

// Y [T] rows (ldy) = rows [r0, r0 + n) of a dense matrix over X: a fused projection's part that takes its own input
Mat mat_rows(const Mat& m, int r0, int n) {   // the rows [r0, r0 + n) of a matrix
    Mat p = m;
    p.N = n;
    if (m.f16) p.f16 = m.f16 + (size_t) r0 * m.K;
    if (m.q8) p.q8 = m.q8 + (size_t) r0 * m.K, p.q8d = m.q8d + (size_t) r0 * (m.K >> 5);
    if (m.raw) p.raw = m.raw + (size_t) r0 * (m.K / k::block_values(m.type)) * k::block_bytes(m.type);
    return p;
}
void matmul_rows(const Mat& m, int r0, int n, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
    matmul(mat_rows(m, r0, n), x32, x16, T, Y, ldy);
}

// A looped layer's message gates over a chunk of tokens.  A gate is two matrix products - g = D [rms(x); rms(m)],
// then F g and G g - and k::loop_gates' kernels take them a token at a time: over a prefill chunk that was a quarter
// of the prefill.  Here the products go through cuBLAS and the kernels are what is left around them.
constexpr int kLoopGemmMin = 32;   // tokens from which the products are cuBLAS's
void loop_gates_chunk(Work& W, const k::LoopGate* gt, int n, int T, int C, int R, float eps) {
    if (T < kLoopGemmMin) {
        k::loop_gates(gt, n, T, C, R, eps, W.lp_g, g_s);
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    float *a1 = W.lp_a, *a2 = a1 + (size_t) T * R, *g = a2 + (size_t) T * R;
    float *f = W.lp_f, *gn = f + (size_t) T * C;
    for (int j = 0; j < n; ++j) {
        // a1 [T][R] = x D_x^T, a2 = m D_m^T (D [R][2 C]: its rows' halves)
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D, 2 * C, gt[j].x, C, &zero, a1, R));
        CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, R, T, C, &one, gt[j].D + C, 2 * C, gt[j].m, C, &zero, a2, R));
        k::loop_gate_mix(gt[j].x, gt[j].m, a1, a2, T, C, R, eps, g, g_s);
        // f [T][C] = g F, gn = g G (F, G [R][C])
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].F, C, g, R, &zero, f, C));
        CB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, C, T, R, &one, gt[j].G, C, g, R, &zero, gn, C));
        k::loop_gate_blend(gt[j].x, gt[j].m, f, gn, (size_t) T * C, gt[j].y, g_s);
    }
}

// One layer's attention + post-attention conv + residual into r_out ([T][C]).
// h_in ([T][C]) is unmodified and acts as the residual input; when l % c.block == 0, it is saved to blk(nb).
// msg: a looped core's later pass - the layer's messages [4][T][C] (q's, k's, v's, the MoE's), which gate its q, k and
// v inputs before their norms (so a sliding layer too projects each from its own input).
void run_layer_attn(const Model& M, Layer& Ly, int l, int T, int pos0, const float* h_in, const int* pos_p, int nb,
                    size_t blk_off, float* r_out, Work& W, const float* msg = nullptr) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + ((size_t) i * W.blk_ld + blk_off) * C; };

    const bool need_x32 = g_dense_q8 && T == 1;
    const uint16_t* xk16 = W.x16;
    const uint16_t* xv16 = W.x16;
    const float *in_q = h_in, *in_k = h_in, *in_v = h_in;
    if (msg) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        float* y32[3] = {W.xq, W.xk, W.xv};
        k::LoopGate gt[3];
        for (int j = 0; j < 3; ++j)
            gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                     Ly.lp_forget[j], Ly.lp_gain[j], y32[j]};
        loop_gates_chunk(W, gt, 3, T, C, c.rank, 1e-6f);
        for (int j = 0; j < 3; ++j)
            k::rms_f16(y32[j], nw[j], T, C, c.eps, y16[j], need_x32 ? y32[j] : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    } else if (l > 0) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = h_in;
        k::RmsOuts outs{};
        if (Ly.sliding) {
            outs.w[0] = Ly.attn_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = need_x32 ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = in_k = in_v = W.xq;
        } else {
            outs.w[0] = Ly.q_in_norm, outs.w[1] = Ly.k_in_norm, outs.w[2] = Ly.v_in_norm;
            outs.y16[0] = W.x16, outs.y16[1] = W.xk16, outs.y16[2] = W.xv16;
            if (need_x32) outs.y32[0] = W.xq, outs.y32[1] = W.xk, outs.y32[2] = W.xv;
            k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            in_q = W.xq, in_k = W.xk, in_v = W.xv;
            xk16 = W.xk16, xv16 = W.xv16;
        }
    } else if (Ly.sliding) {
        k::rms_f16(in_q, Ly.attn_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        in_q = in_k = in_v = W.xq;
    } else {
        k::rms_f16(in_q, Ly.q_in_norm, T, C, c.eps, W.x16, need_x32 ? W.xq : nullptr, g_s);
        k::rms_f16(in_k, Ly.k_in_norm, T, C, c.eps, W.xk16, need_x32 ? W.xk : nullptr, g_s);
        k::rms_f16(in_v, Ly.v_in_norm, T, C, c.eps, W.xv16, need_x32 ? W.xv : nullptr, g_s);
        in_q = W.xq, in_k = W.xk, in_v = W.xv;
        xk16 = W.xk16, xv16 = W.xv16;
    }
    const int HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    float *q_p = W.q, *k_p = W.kk, *v_p = W.vv, *r_p = W.rr, *gate_p = W.gate, *sink_p = W.sink;
    float *kd_p = W.dyn, *vd_p = W.dyn;
    if (Ly.sliding) {
        if (Ly.wqkvrg.N > 0) {
            if (msg) {   // the fused rows by the input they take: q | k | v | r, gate (and the sink)
                const int N = Ly.wqkvrg.N;
                matmul_rows(Ly.wqkvrg, 0, HD, in_q, W.x16, T, W.qkv, N);
                matmul_rows(Ly.wqkvrg, HD, KD, in_k, xk16, T, W.qkv + HD, N);
                matmul_rows(Ly.wqkvrg, HD + KD, KD, in_v, xv16, T, W.qkv + HD + KD, N);
                matmul_rows(Ly.wqkvrg, HD + 2 * KD, N - HD - 2 * KD, in_q, W.x16, T, W.qkv + HD + 2 * KD, N);
            } else {
                matmul(Ly.wqkvrg, in_q, W.x16, T, W.qkv, Ly.wqkvrg.N);
            }
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                k_p = q_p + HD;
                v_p = k_p + KD;
                r_p = v_p + KD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T,
                                    {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.kk, KD}, {W.vv, KD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
        conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
        k::conv4(k_p, Ly.h_k, Ly.sc_k, nullptr, 0, T, KD, W.kconv, g_s);
        k::conv4(v_p, Ly.h_v, Ly.sc_v, nullptr, 0, T, KD, W.vconv, g_s);
    } else {
        if (Ly.wqrg.N > 0) {
            matmul(Ly.wqrg, in_q, W.x16, T, W.qkv, Ly.wqrg.N);
            const bool has_sink = !g_dense_q8;
            if (T == 1) {
                q_p = W.qkv;
                r_p = q_p + HD;
                gate_p = r_p + HR;
                if (has_sink) sink_p = gate_p + HD;
            } else {
                if (has_sink)
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}, {W.sink, c.H}}, g_s);
                else
                    unpack_segments(W.qkv, T, {{W.q, HD}, {W.rr, HR}, {W.gate, HD}}, g_s);
            }
            if (!has_sink) gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        } else {
            matmul(Ly.wq, in_q, W.x16, T, q_p, HD);
            matmul(Ly.wr, in_q, W.x16, T, r_p, HR);
            matmul(Ly.wgate, in_q, W.x16, T, gate_p, HD);
            gemm16(Ly.sink16, c.H, C, W.x16, T, sink_p, c.H);
        }
        if (!g_dense_q8) {
            const int off_k = T == 1 ? Ly.wqrg.N : 0;
            const int off_v = T == 1 ? Ly.wqrg.N + Ly.wk_dyn.N : 0;
            matmul(Ly.wk_dyn, in_k, xk16, T, W.qkv + off_k, Ly.wk_dyn.N);
            if (T == 1) {
                k_p = W.qkv + off_k;
                kd_p = k_p + KD;
            } else {
                unpack_segments(W.qkv + off_k, T, {{W.kk, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv_dyn, in_v, xv16, T, W.qkv + off_v, Ly.wv_dyn.N);
            if (T == 1) {
                v_p = W.qkv + off_v;
                vd_p = v_p + KD;
            } else {
                unpack_segments(W.qkv + off_v, T, {{W.vv, KD}, {W.dyn, c.KV * 4}}, g_s);
            }
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        } else {
            matmul(Ly.wk, in_k, xk16, T, k_p, KD);
            gemm16(Ly.sc_k_dyn16, c.KV * 4, C, xk16, T, kd_p, c.KV * 4);
            conv_rec(Ly, 0, k_p, Ly.h_k, T, KD);
            k::conv4(k_p, Ly.h_k, Ly.sc_k, kd_p, c.KV, T, KD, W.kconv, g_s);

            matmul(Ly.wv, in_v, xv16, T, v_p, KD);
            gemm16(Ly.sc_v_dyn16, c.KV * 4, C, xv16, T, vd_p, c.KV * 4);
            conv_rec(Ly, 1, v_p, Ly.h_v, T, KD);
            k::conv4(v_p, Ly.h_v, Ly.sc_v, vd_p, c.KV, T, KD, W.vconv, g_s);
        }
    }
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f;
    const float v_scale = 32.0f;
    const float k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
    // Long prefill revisits a whole global K/V span. Migrate it in bulk before
    // the flash kernel so warps never have to fault it in one page at a time.
    if (g_kv_managed && !Ly.sliding && T > 8 && pos0 + T > 196608) {
        const size_t bytes = (size_t)(pos0 + T) * kv_row_bytes(KD, Ly.sliding);
        CU(cudaMemPrefetchAsync(Ly.kc8 ? (void*)Ly.kc8 : (void*)Ly.kc, bytes, Ly.dev, g_s));
        CU(cudaMemPrefetchAsync(Ly.kc8 ? (void*)Ly.vc8 : (void*)Ly.vc, bytes, Ly.dev, g_s));
    }
    if (Ly.kc8) {
        k::prep_qkv_r_fp8(q_p, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.fp8_layout(), k_scale, v_scale, g_s);
    } else {
        k::prep_qkv_r_f16(q_p, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, Ly.r_clip, T, c.H, c.KV, c.D, c.d_rel,
                          c.qk_clip, c.rel_clip, c.eps, Ly.sliding ? nullptr : pos_p, c.alpha, c.n_floor, pos0,
                          Ly.kv_cap, g_s);
    }

    const int extent = Ly.sliding ? c.extent_swa : c.extent;
    const int window = Ly.sliding ? c.window : (1 << 30);
    const int k0 = Ly.sliding ? std::max(0, pos0 - c.window + 1) : 0;
    const int nk = pos0 + T - k0;
    const float one = 1.0f, zero = 0.0f;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const int attn_kv_cap = Ly.sliding ? Ly.kv_cap : 0;
    if (T == 1) {
        if (Ly.kc8) {
            k::attn_decode_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, Ly.fp8_layout(), c.H, c.KV, c.D, k0, nk, pos0,
                               qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_decode_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, k0, nk, pos0, qk_scale,
                               Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else if (T <= 8 && g_group && g_group_attn) {
        if (Ly.kc8) {
            k::attn_multi_fp8(q_p, r_p, sink_p, Ly.kc8, Ly.vc8, Ly.fp8_layout(), T, c.H, c.KV, k0, nk, pos0,
                              qk_scale * k_inv, v_inv, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        } else {
            k::attn_multi_f16(q_p, r_p, sink_p, Ly.kc, Ly.vc, attn_kv_cap, T, c.H, c.KV, k0, nk, pos0, qk_scale,
                              Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, g_s);
        }
    } else {
        k::to_f16(q_p, (size_t) T * HD, W.x16, g_s);
        uint16_t* lg16 = W.lg16;
        const int GQA = c.H / c.KV;
        // softmax rows (with the relative bias and the sink) -> lg16: the bias as one small GEMM (K = d_rel) first
        auto attn_probs = [&](int Tq, int nk_s, int ts, int k0_s) {
            if (W.relb && extent <= W.relb_ext) {
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                k::attn_probs_bias_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, W.relb, extent, window,
                                       sink_p + (size_t) ts * c.H, g_s);
            } else {
                k::attn_probs_f16(W.lg, lg16, c.H, Tq, nk_s, pos0 + ts, k0_s, qk_scale, r_p + (size_t) ts * HR,
                                  c.d_rel, Ly.rel_proj, extent, window, sink_p + (size_t) ts * c.H, g_s);
            }
        };
        if (g_flash && c.D == 128 && W.relb && extent <= W.relb_ext) {
            if (!g_n_sm) CU(cudaDeviceGetAttribute(&g_n_sm, cudaDevAttrMultiProcessorCount, Ly.dev));
            for (int ts = 0; ts < T; ts += Work::kRelbRows) {
                const int Tq = std::min(Work::kRelbRows, T - ts);
                const float one_f = 1.0f, zero_f = 0.0f;
                CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, extent, Tq * c.H, c.d_rel, &one_f, Ly.rel_proj,
                                CUDA_R_32F, c.d_rel, r_p + (size_t) ts * HR, CUDA_R_32F, c.d_rel, &zero_f, W.relb,
                                CUDA_R_32F, extent, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
                const int k_lo = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
                const int ns = k::attn_flash_splits(Tq, c.H, pos0 + ts + Tq - k_lo, g_n_sm);
                if (Ly.kc8)
                    k::attn_flash_fp8(W.x16 + (size_t) ts * HD, HD, Ly.kc8, Ly.vc8, Ly.fp8_layout(), c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale * k_inv, v_inv, W.relb, extent, window,
                                      sink_p + (size_t) ts * c.H, W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
                else
                    k::attn_flash_f16(W.x16 + (size_t) ts * HD, HD, Ly.kc, Ly.vc, attn_kv_cap, c.H, c.KV, c.D, Tq,
                                      pos0 + ts, qk_scale, W.relb, extent, window, sink_p + (size_t) ts * c.H,
                                      W.o + (size_t) ts * HD, HD, W.lg, W.lg_cap, ns, g_s);
            }
        } else
        for (int ts = 0; ts < T;) {
            const int k0_s = Ly.sliding ? std::max(0, pos0 + ts - c.window + 1) : 0;
            const int nk_ub = pos0 + T - k0_s;
            int Tq = std::max(1, std::min(T - ts, (int) (W.lg_cap / ((size_t) c.H * nk_ub))));
            if (W.relb) Tq = std::min(Tq, Work::kRelbRows);
            const int nk_s = pos0 + ts + Tq - k0_s;
            const long long qn = (long long) Tq * nk_s;
            if (Ly.kc8) {
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, Ly.fp8_layout(), k0_s + k_off, nk_c, KD, k_inv, v_inv, W.kv_k16,
                                            W.kv_v16, g_s);
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Kp = W.kv_k16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, nk_c, Tq, c.D, &one, Kp, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + k_off, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int k_off = 0; k_off < nk_s; k_off += W.kv_tmp_cap) {
                    const int nk_c = std::min(W.kv_tmp_cap, nk_s - k_off);
                    if (nk_s > W.kv_tmp_cap) {
                        k::unpack_kv_fp8_to_f16(Ly.kc8, Ly.vc8, Ly.fp8_layout(), k0_s + k_off, nk_c, KD, k_inv, v_inv,
                                                W.kv_k16, W.kv_v16, g_s);
                    }
                    const float beta = k_off == 0 ? zero : one;
                    for (int g = 0; g < c.KV; ++g) {
                        const uint16_t* Vp = W.kv_v16 + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, nk_c, &one, Vp, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + k_off, CUDA_R_16F, nk_s, qn, &beta,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            } else {
                const int start = attn_kv_cap > 0 ? (k0_s % attn_kv_cap) : k0_s;
                const int n1 = attn_kv_cap > 0 ? std::min(nk_s, attn_kv_cap - start) : nk_s;
                const int n2 = nk_s - n1;
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Kp1 = Ly.kc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n1, Tq, c.D, &one, Kp1, CUDA_R_16F, KD, 0,
                        W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                        W.lg + (size_t) g * GQA * qn, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Kp2 = Ly.kc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n2, Tq, c.D, &one, Kp2, CUDA_R_16F, KD, 0,
                            W.x16 + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_16F, HD, (long long) c.D, &zero,
                            W.lg + (size_t) g * GQA * qn + n1, CUDA_R_32F, nk_s, qn, GQA, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
                    }
                }
                attn_probs(Tq, nk_s, ts, k0_s);
                for (int g = 0; g < c.KV; ++g) {
                    const uint16_t* Vp1 = Ly.vc + (size_t) start * KD + (size_t) g * c.D;
                    CB(cublasGemmStridedBatchedEx(
                        g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n1, &one, Vp1, CUDA_R_16F, KD, 0,
                        lg16 + (size_t) g * GQA * qn, CUDA_R_16F, nk_s, qn, &zero,
                        W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    if (n2 > 0) {
                        const uint16_t* Vp2 = Ly.vc + (size_t) g * c.D;
                        CB(cublasGemmStridedBatchedEx(
                            g_blas, CUBLAS_OP_N, CUBLAS_OP_N, c.D, Tq, n2, &one, Vp2, CUDA_R_16F, KD, 0,
                            lg16 + (size_t) g * GQA * qn + n1, CUDA_R_16F, nk_s, qn, &one,
                            W.o + (size_t) ts * HD + (size_t) g * GQA * c.D, CUDA_R_32F, HD, (long long) c.D, GQA,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    }
                }
            }
            ts += Tq;
        }
    }
    k::gate_mul_f16(W.o, gate_p, (size_t) T * HD, W.x16, (g_dense_q8 && T == 1) || Ly.attn_lora.rank, g_s);
    matmul(Ly.wo, W.o, W.x16, T, W.tmp, C);
    if (Ly.attn_lora.rank) add_attention_lora(Ly, W.o, T, W.attn_lora_tmp, W.tmp);
    conv_rec(Ly, 2, W.tmp, Ly.h_attn, T, C);
    if (l % c.block == 0) {
        CU(cudaMemcpyAsync(blk(nb), h_in, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        k::conv4(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, r_out, g_s);
    } else {
        k::conv4_res(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, T, C, h_in, r_out, g_s);
    }
    abl(r_out, T, C, 1);
}

// A layer-major prefill pass of many tokens uses nearly every expert of a layer, so the ones not in VRAM can be
// copied in before the layer's routing is known: while the previous layer's experts and this layer's attention run.
// The copy stream then has no idle gap between layers. Slots are taken at the current tick, which keeps the previous
// layer's slots (still being computed) from being evicted; only done when the pool holds both layers with room.
void prefetch_layer(Model& M, int l, Work& W, ExpertCache& EC, int T, const Work* Wpop = nullptr, int own = -1, int ahead = 1) {
    const int pr = ((l % Work::kPfRing) + Work::kPfRing) % Work::kPfRing;
    W.pf_layers[pr] = -1;
    if (getenv("IK2_PREFETCH") && !std::atoi(getenv("IK2_PREFETCH"))) return;
    if (!g_overlap || g_prof || l >= M.c.n_layer || (!dual() && M.L[l].dev != g_cur)) return;
    const Work& U = Wpop ? *Wpop : W;   // (whose picks so far: under EP, GPU 0's)
    const int E = M.c.E, pi = EC.layer_pool[l];
    if (pi < 0 || EC.pools[pi].n_active < (ahead + 1) * E + 64) return;
    Layer& Ly = M.L[l];
    const uint64_t t = EC.tick + (uint64_t) ahead;   // (the tick its layer will run in: kept till then)
    W.pf_bs[pr].assign(E, -1);
    // Only the experts this pass will likely use: P(used) = 1 - (1 - p_e)^(T K) from the prefills so far (p_e the
    // expert's share of the layer's tokens); with little history yet, all of them (IK2_PF_P: the threshold)
    const bool known = !U.use_tot.empty() && U.use_tot[l] >= (uint64_t) 16 * E;
    // (0.999, not 0.5: a request's tokens are alike, so the independent-picks estimate is far too high - at 0.5 a
    // fifth of the prefetched experts went unused; 500 / 800 new tokens 4.2 / 4.3 -> 3.5 / 3.6 s, 21.8k unchanged)
    const double thr = getenv("IK2_PF_P") ? std::atof(getenv("IK2_PF_P")) : 0.999;
    const double picks = (double) T * M.c.K;
    int n = 0;
    for (int e = 0; e < E; ++e) {
        if (EC.has(l, e) || (own >= 0 && ep_owner(l, e) != own)) continue;
        if (known) {
            const double pe = (U.use[(size_t) l * E + e] + 0.5) / (U.use_tot[l] + 0.5 * E);
            if (1.0 - std::pow(1.0 - pe, picks) < thr) continue;
        }
        uint8_t* dst = EC.get(Ly, l, e, t, g_cs, nullptr, false, true);
        EC.copy_in(Ly, e, dst, g_cs);
        W.pf_bs[pr][e] = n / Work::kMissBatch;
        if (++n % Work::kMissBatch == 0) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch - 1], g_cs));
    }
    if (n % Work::kMissBatch) CU(cudaEventRecord(W.pf_evs[pr][n / Work::kMissBatch], g_cs));
    W.pf_layers[pr] = l;
}

// Each token sums its experts in ascending expert order, exactly as the scalar scatter loop.
__global__ void combine_prefill_groups(const float* y, const int* rows, const float* weights, int C, int K,
                                       float* out) {
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < C / 4; j += blockDim.x) {
        float4 sum = make_float4(0, 0, 0, 0);
        for (int k = 0; k < K; ++k) {
            const int row = rows[t * K + k];
            const float w = weights[row];
            const float4 v = ((const float4*)(y + (size_t)row * C))[j];
            sum.x += w * v.x;
            sum.y += w * v.y;
            sum.z += w * v.z;
            sum.w += w * v.w;
        }
        ((float4*)(out + (size_t)t * C))[j] = sum;
    }
}

#ifndef STRATA_USE_HIP
bool blackwell_prefill_enabled() {
    static const bool enabled = !getenv("IK2_BLACKWELL_PREFILL") || atoi(getenv("IK2_BLACKWELL_PREFILL"));
    return enabled && allow_short_batch_products() && k::use_blackwell_nvfp4();
}
#endif

void run_prefill_moe(const Model& M, Layer& Ly, int l, int T, const uint16_t* xq16, const int* top, const float* w,
                     const float* r_in, float* h_out, Work& W, ExpertCache& EC, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C, F = c.F, S = T * c.K;
    const double te0 = now();
    if (g_seq_done + 1 < g_seq_next) {
        CU(cudaStreamSynchronize(g_cs));
        g_seq_done = g_seq_next - 1;
    }
    std::fill(W.cnt.begin(), W.cnt.end(), 0);
    for (int i = 0; i < S; ++i) ++W.cnt[top[i]];
    if (W.use.empty()) W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
    const int prl = l % Work::kPfRing;
    const bool pf_here = W.pf_layers[prl] == l && (int) W.pf_bs[prl].size() == c.E;
    for (int e = 0; e < c.E; ++e) {
        W.use[(size_t) l * c.E + e] += W.cnt[e];
        const int k = W.cnt[e];
        if (k) ++g_cnt_hist[k == 1 ? 0 : k == 2 ? 1 : k < 8 ? 2 : 3];
        const bool pfd = pf_here && W.pf_bs[prl][e] >= 0;
        if (k) ++g_src_hist[pfd ? 1 : EC.has(l, e) ? 0 : 2];
        else if (pfd) ++g_src_hist[3];
    }
    W.use_tot[l] += S;
    W.start[0] = 0;
    for (int e = 0; e < c.E; ++e) W.start[e + 1] = W.start[e] + W.cnt[e];
    for (int e = 0; e < c.E; ++e) W.fill[e] = W.start[e];

    const int ps = l & 1;
    int* gidx = W.gidx_h[ps];
    int* inv = W.inv_h[ps];
    float* sg = W.sg_h[ps];
    float* su = W.su_h[ps];
    float* wt = W.wt_h[ps];
    float* wtr = W.wtr_h[ps];
    for (int i = 0; i < S; ++i) {
        const int e = top[i], row = W.fill[e]++;
        gidx[row] = i / c.K;
        inv[i] = row;
        sg[row] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
        su[row] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
        wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
        wtr[row] = wt[i];
    }

    // the experts on a GPU: d -1 all of them (one GPU), else the ones it owns (IK2_EP), into X.tmp
    auto exec = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
        const uint64_t tick = ++XC.tick;
        const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
        X.hit_exps.clear();
        X.miss_exps.clear();
        const double tu = now();
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) {
                X.slot_ptrs[e] = nullptr;
                continue;
            }
            if (XC.has(l, e)) {
                X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                X.hit_exps.push_back(e);
            } else {
                X.slot_ptrs[e] = nullptr;
                X.miss_exps.push_back(e);
            }
        }
        for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);

        const int n_miss = (int) X.miss_exps.size();
        const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
        for (int b = 0; b < n_batches; ++b) {
            const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
            for (int m = b0; m < b1; ++m) {
                const int e = X.miss_exps[m];
                XC.copy_in(Ly, e, X.slot_ptrs[e], cst);
            }
            if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
        }
        XC.t_up += now() - tu;
        g_pf.lap(g_pf.upload);

#ifndef STRATA_USE_HIP
        const bool blackwell = k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, F) &&
                               blackwell_prefill_enabled();
        auto* packed_x = reinterpret_cast<k::BlackwellActivation*>(X.ex16);
        auto* packed_h = reinterpret_cast<k::BlackwellActivation*>(X.gu32);
        if (blackwell) k::blackwell_pack_rows(xin, T, C, packed_x, g_s);
#else
        constexpr bool blackwell = false;
#endif
        static const bool batch_prefill = getenv("IK2_BATCH_PREFILL") && atoi(getenv("IK2_BATCH_PREFILL"));
        const bool batch = !blackwell && batch_prefill && d == -1 && !Ly.tab_h.empty() && n_miss == 0 && T >= 64 && C % 8 == 0 &&
                           (size_t)S * (2 * std::max(C, F) + 4 * C) <= X.lg_cap * 4 &&
                           (size_t)S * 2 * F * sizeof(float) <= X.gu32_bytes;
        if (batch) {
            for (int t = 0; t < T; ++t)
                std::sort(inv + t * c.K, inv + (t + 1) * c.K);
        }
        CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.inv, inv, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
        auto gate_up_product = [&](int e, const uint16_t* input, float* output) {
            const uint8_t* blob = X.slot_ptrs[e];
            if (Ly.ty[0] == Ly.ty[1]) {
                k::decode_f16(Ly.ty[0], blob, (size_t)2 * F * C, X.w16, g_s);
            } else {
                k::decode_f16(Ly.ty[0], blob, (size_t)F * C, X.w16, g_s);
                k::decode_f16(Ly.ty[1], blob + Ly.eb[0], (size_t)F * C, X.w16 + (size_t)F * C, g_s);
            }
            gemm16(X.w16, 2 * F, C, input, W.cnt[e], output, 2 * F);
        };
        auto down_product = [&](int e, const uint16_t* hidden, float* output) {
            const uint8_t* down = X.slot_ptrs[e] + Ly.eb[0] + Ly.eb[1];
            k::decode_f16(Ly.ty[2], down, (size_t)C * F, X.w16, g_s);
            gemm16(X.w16, C, F, hidden, W.cnt[e], output, C);
        };
        if (batch) {
            // Attention has finished. Its temporary output is free until the next visit.
            uint16_t* sorted_input = (uint16_t*)X.lg;
            float* sorted_out = (float*)(sorted_input + (size_t)S * std::max(C, F));
            gather_f16_16_kernel<<<S, 256, 0, g_s>>>(xin, X.gidx, S, C, sorted_input);
            for (int e : X.hit_exps) {
                const size_t offset = W.start[e];
                gate_up_product(e, sorted_input + offset * C, X.gu32 + offset * 2 * F);
            }
            k::swiglu_cap_f32(X.gu32, S, F, X.sg, X.su, c.gate_cap, c.up_cap, sorted_input, g_s);
            for (int e : X.hit_exps) {
                const size_t offset = W.start[e];
                down_product(e, sorted_input + offset * F, sorted_out + offset * C);
            }
            combine_prefill_groups<<<T, 256, 0, g_s>>>(sorted_out, X.inv, X.wtr, C, c.K, X.tmp);
            X.pf_layers[l % Work::kPfRing] = -1;
            return;
        }
        CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));

        // Each expert: its tokens gathered, gate/up, SwiGLU, down, then added into tmp with their weights. The experts run
        // in index order whatever is cached (each waiting for its copies), so the sums' order and the result do not
        // depend on the cache. Only [T][C] of scratch: no [T * K][C] copies of the inputs or the outputs.
        auto run_exp = [&](int e) {
            const int cnt_e = W.cnt[e], st_e = W.start[e];
#ifndef STRATA_USE_HIP
            if (blackwell) {
                k::BlackwellExpertBatch batch{X.slot_ptrs[e], nullptr, nullptr, X.gidx + st_e, nullptr, cnt_e, true};
                k::blackwell_expert_product(true, batch, Ly.eb[0], C, F, xin, packed_x, X.sg + st_e, X.su + st_e,
                                            c.gate_cap, c.up_cap, X.h16, nullptr, g_s, packed_h);
                k::blackwell_expert_product(false, batch, Ly.eb[0] + Ly.eb[1], C, F, X.h16, packed_h,
                                            nullptr, nullptr, 0, 0, nullptr, X.y32, g_s);
            } else
#endif
            {
                gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
                gate_up_product(e, X.ex16, X.gu32);
                k::swiglu_cap_f32(X.gu32, cnt_e, F, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                down_product(e, X.h16, X.y32);
            }
            scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, C, X.tmp, C);
        };
        const bool pf = X.pf_layers[l % Work::kPfRing] == l;
        X.miss_pos.assign(c.E, -1);
        for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
        int pf_waited = -1, miss_waited = -1;
        for (int e = 0; e < c.E; ++e) {
            if (!W.cnt[e] || (d >= 0 && ep_owner(l, e) != d)) continue;
            if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
            if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
            run_exp(e);
        }
        X.pf_layers[l % Work::kPfRing] = -1;
    };
    if (g_ep) {   // GPU 1's experts on GPU 1 (the input over, its sums back), GPU 0's meanwhile
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        Work& X1 = dev_work(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        exec(1, X1, dev_cache(1), X1.xq16_sup);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        exec(0, W, EC, xq16);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpyPeerAsync(W.y32, 0, X1.tmp, 1, (size_t) T * C * 4, g_s));
        k::add(W.tmp, W.y32, (size_t) T * C, W.tmp, g_s);
    } else if (g_tp) {
        // tensor-parallel experts: (1) each GPU its half of every expert's h into tp_h (GPU 1 with the input copied
        // over), (2) the halves swapped, (3) each GPU its half of the output rows, GPU 1's columns copied to GPU 0
        const int Fd = F / 2, Cd = C / 2;
        auto tp_gu = [&](int d, Work& X, ExpertCache& XC, const uint16_t* xin) {
            const uint64_t tick = ++XC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            X.hit_exps.clear();
            X.miss_exps.clear();
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) {
                    X.slot_ptrs[e] = nullptr;
                    continue;
                }
                if (XC.has(l, e)) {
                    X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
                    X.hit_exps.push_back(e);
                } else {
                    X.slot_ptrs[e] = nullptr;
                    X.miss_exps.push_back(e);
                }
            }
            for (int e : X.miss_exps) X.slot_ptrs[e] = XC.get(Ly, l, e, tick, cst, nullptr, false, true);
            const int n_miss = (int) X.miss_exps.size();
            const int n_batches = (n_miss + Work::kMissBatch - 1) / Work::kMissBatch;
            for (int b = 0; b < n_batches; ++b) {
                const int b0 = b * Work::kMissBatch, b1 = std::min(n_miss, b0 + Work::kMissBatch);
                for (int m = b0; m < b1; ++m) XC.copy_in(Ly, X.miss_exps[m], X.slot_ptrs[X.miss_exps[m]], cst);
                if (cst != g_s) CU(cudaEventRecord(X.miss_ev[b], cst));
            }
            CU(cudaMemcpyAsync(X.gidx, gidx, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.sg, sg, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.su, su, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            CU(cudaMemcpyAsync(X.wtr, wtr, (size_t) S * 4, cudaMemcpyHostToDevice, g_s));
            size_t o0, b0g, o1, b1u;
            tp_part(Ly, d, 0, o0, b0g);
            tp_part(Ly, d, 1, o1, b1u);
            const bool pf = X.pf_layers[l % Work::kPfRing] == l;
            X.miss_pos.assign(c.E, -1);
            for (int m = 0; m < n_miss; ++m) X.miss_pos[X.miss_exps[m]] = m;
            int pf_waited = -1, miss_waited = -1;
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                if (pf && X.pf_bs[l % Work::kPfRing][e] >= 0 && X.pf_bs[l % Work::kPfRing][e] > pf_waited)
                    CU(cudaStreamWaitEvent(g_s, X.pf_evs[l % Work::kPfRing][pf_waited = X.pf_bs[l % Work::kPfRing][e]], 0));
                if (X.miss_pos[e] >= 0 && cst != g_s && X.miss_pos[e] / Work::kMissBatch > miss_waited)
                    CU(cudaStreamWaitEvent(g_s, X.miss_ev[miss_waited = X.miss_pos[e] / Work::kMissBatch], 0));
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                const uint8_t* sl = X.slot_ptrs[e];
                gather_f16_16_kernel<<<cnt_e, 256, 0, g_s>>>(xin, X.gidx + st_e, cnt_e, C, X.ex16);
                if (Ly.ty[0] == Ly.ty[1]) {
                    k::decode_f16(Ly.ty[0], sl, (size_t) 2 * Fd * C, X.w16, g_s);
                } else {
                    k::decode_f16(Ly.ty[0], sl, (size_t) Fd * C, X.w16, g_s);
                    k::decode_f16(Ly.ty[1], sl + b0g, (size_t) Fd * C, X.w16 + (size_t) Fd * C, g_s);
                }
                gemm16(X.w16, 2 * Fd, C, X.ex16, cnt_e, X.gu32, 2 * Fd);
                k::swiglu_cap_f32(X.gu32, cnt_e, Fd, X.sg + st_e, X.su + st_e, c.gate_cap, c.up_cap, X.h16, g_s);
                CU(cudaMemcpy2DAsync(X.tp_h + (size_t) st_e * F + (size_t) d * Fd, (size_t) F * 2, X.h16, (size_t) Fd * 2,
                                     (size_t) Fd * 2, cnt_e, cudaMemcpyDeviceToDevice, g_s));
            }
            X.pf_layers[l % Work::kPfRing] = -1;
            CU(cudaEventRecord(g_tp_h[d], g_s));
        };
        auto tp_dn = [&](int d, Work& X, const Work& O) {
            // the other GPU's half of h in, then this GPU's rows of down for every expert
            CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
            CU(cudaMemcpy2DAsync(X.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                 (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
            CU(cudaMemsetAsync(X.tmp, 0, (size_t) T * C * 4, g_s));
            size_t o, b0g, b1u;
            tp_part(Ly, d, 0, o, b0g);
            tp_part(Ly, d, 1, o, b1u);
            for (int e = 0; e < c.E; ++e) {
                if (!W.cnt[e]) continue;
                const int cnt_e = W.cnt[e], st_e = W.start[e];
                k::decode_f16(Ly.ty[2], X.slot_ptrs[e] + b0g + b1u, (size_t) Cd * F, X.w16, g_s);
                gemm16(X.w16, Cd, F, X.tp_h + (size_t) st_e * F, cnt_e, X.y32, Cd);
                scatter_add_rows_kernel<<<cnt_e, 256, 0, g_s>>>(X.y32, X.gidx + st_e, X.wtr + st_e, Cd,
                                                               X.tmp + (size_t) d * Cd, C);
            }
        };
        Work& X1 = dev_work(1);
        CU(cudaEventRecord(g_ep_x, g_s));
        use_dev(1);
        CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
        CU(cudaMemcpyPeerAsync(X1.xq16_sup, 1, xq16, 0, (size_t) T * C * 2, g_s));
        tp_gu(1, X1, dev_cache(1), X1.xq16_sup);
        use_dev(0);
        tp_gu(0, W, EC, xq16);
        use_dev(1);
        tp_dn(1, X1, W);
        CU(cudaEventRecord(g_ep_done, g_s));
        use_dev(0);
        tp_dn(0, W, X1);
        CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
        CU(cudaMemcpy2DAsync(W.tmp + Cd, (size_t) C * 4, X1.tmp + Cd, (size_t) C * 4, (size_t) Cd * 4, T,
                             cudaMemcpyDefault, g_s));
    } else {
        exec(-1, W, EC, xq16);
    }
    g_pf.lap(g_pf.experts);
    t_exp += now() - te0;
    conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
    k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, r_in, h_out, g_s);
    abl(h_out, T, C, 2);
}

void run_head(const Model& M, int T, bool all_logits, OutMode out_mode, int nb, const float* h_in, Work& W,
              std::vector<float>& logits_out, std::vector<int>* argmax_out) {
    const Config& c = M.c;
    const int C = c.C;
    const size_t TC = (size_t) T * C;
    g_pf.start();
    if (out_mode != OutMode::kNone || g_taps) {
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
        st.st[nb] = h_in;
        if (out_mode != OutMode::kNone && (all_logits || T == 1)) {
            k::RmsOuts outs{};
            outs.w[0] = M.out_norm;
            outs.y16[0] = W.x16;
            outs.y32[0] = (g_taps || g_dense_q8) ? W.xq : nullptr;
            k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        } else {
            k::attn_res(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, W.tmp, g_s);
            k::rms(W.tmp, M.out_norm, T, C, c.eps, W.xq, g_s);
            if (g_taps)
                CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            if (out_mode != OutMode::kNone) k::to_f16(W.xq + (size_t) (T - 1) * C, (size_t) C, W.x16, g_s);
        }
        if (out_mode != OutMode::kNone) {
            const int t0 = all_logits ? 0 : T - 1, nt = T - t0;
            if (out_mode == OutMode::kArgmax) {
                argmax_out->resize(nt);
                for (int i = 0; i < nt;) {
                    const int remaining = nt - i;
                    const int n = remaining >= 32 ? std::min(W.head_rows, remaining / 16 * 16) : std::min(16, remaining);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    if (n == 1) k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d + i, g_s);
                    else k::argmax_rows(W.logits, c.n_rows, n, c.n_vocab, W.argmax_d + i, g_s);
                    i += n;
                }
                CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) nt * 4, cudaMemcpyDeviceToHost, g_s));
                CU(cudaStreamSynchronize(g_s));
                for (int i = 0; i < nt; ++i) (*argmax_out)[i] = W.argmax_h[i];
            } else {
                const size_t start = logits_out.size();
                logits_out.resize(start + (size_t)nt * c.n_vocab);
                for (int i = 0; i < nt; i += 16) {
                    const int n = std::min(16, nt - i);
                    matmul(M.out_w, W.xq + (size_t) (t0 + i) * C, W.x16 + (size_t) i * C, n, W.logits, c.n_rows);
                    k::softcap(W.logits, (size_t) n * c.n_rows, c.logit_cap, g_s);
                    CU(cudaMemcpy2DAsync(logits_out.data() + start + (size_t)i * c.n_vocab, (size_t) c.n_vocab * 4, W.logits,
                                         (size_t) c.n_rows * 4, (size_t) c.n_vocab * 4, n, cudaMemcpyDeviceToHost,
                                         g_s));
                }
                CU(cudaStreamSynchronize(g_s));
            }
        }
    }
    g_pf.lap(g_pf.rest);
}

#ifndef STRATA_USE_HIP
bool resident_expert_prefill(const Config& c, const Layer& ly, int T, Work& w, double& t_exp) {
    const int C = c.C, F = c.F, S = T * c.K;
    if (!g_fast || T < 2 || !ly.tab_blob || c.E > 1024 ||
        !k::experts_mma_fits(ly.ty[0], ly.ty[1], ly.ty[2], C, F) ||
        !blackwell_prefill_enabled()) return false;
    const int groups = k::blackwell_group_capacity(S, c.E);
    const size_t hidden_values = size_t(S) * F;
    if (groups > 65535 || k::blackwell_group_bytes(groups) > size_t(2) * F * C * sizeof(uint16_t) ||
        size_t(S) * C > w.lg_cap ||
        hidden_values * sizeof(uint16_t) + hidden_values / 64 * sizeof(k::BlackwellActivation) > w.gu32_bytes) return false;

    auto batch = k::blackwell_prefill_groups(w.route_d, (const float*)(w.route_d + S), S, c.K, c.E, ly.tab_blob,
                                             ly.tab_sc[0], ly.tab_sc[1], ly.tab_sc[2], w.w16,
                                             w.sg, w.su, w.wtr, w.inv, g_s);
    g_pf.lap(g_pf.route);
    const double te0 = now();
    auto* packed_x = reinterpret_cast<k::BlackwellActivation*>(w.ex16);
    auto* hidden = reinterpret_cast<uint16_t*>(w.gu32);
    auto* packed_h = reinterpret_cast<k::BlackwellActivation*>(hidden + hidden_values);
    k::blackwell_pack_rows(w.x16, T, C, packed_x, g_s);
    k::blackwell_expert_product(true, batch, ly.eb[0], C, F, w.x16, packed_x, w.sg, w.su,
                                c.gate_cap, c.up_cap, hidden, nullptr, g_s, packed_h);
    k::blackwell_expert_product(false, batch, ly.eb[0] + ly.eb[1], C, F, hidden, packed_h,
                                nullptr, nullptr, 0, 0, nullptr, w.lg, g_s);
    combine_prefill_groups<<<T, 256, 0, g_s>>>(w.lg, w.inv, w.wtr, C, c.K, w.tmp);
    g_pf.lap(g_pf.experts);
    t_exp += now() - te0;
    return true;
}
#else
bool resident_expert_prefill(const Config& c, const Layer& ly, int T, Work& w, double&) {
    static const bool enabled = getenv("IK2_PREFILL_WMMA") && atoi(getenv("IK2_PREFILL_WMMA"));
    const int S = T * c.K, C = c.C, F = c.F;
    static const int fixed_rows = [] {
        const char* rows = getenv("IK2_PREFILL_TILE");
        if (!rows) return 16;
        if (!std::strcmp(rows, "auto")) return 0;
        const int n = atoi(rows);
        if (n == 16 || n == 32 || n == 64) return n;
        die("IK2_PREFILL_TILE expects auto, 16, 32 or 64");
    }();
    const int tile_rows = fixed_rows ? fixed_rows : S <= 8 * c.E ? 16 : S <= 16 * c.E ? 32 : 64;
    const int capacity = k::expert_tile_capacity(S, c.E, tile_rows);
    const int block = ly.ty[0] == k::kNVFP4 ? 64 : 32;
    if (g_prof || !enabled || !g_fast || T < 2 || !ly.tab_blob || c.E > 1024 ||
        ly.ty[0] != ly.ty[1] || ly.ty[0] != ly.ty[2] || (ly.ty[0] != k::kIQ4_NL && ly.ty[0] != k::kQ8_0 && ly.ty[0] != k::kNVFP4) ||
        C % block || F % block || (size_t)capacity * sizeof(k::ExpertTile) > (size_t)T * F * sizeof(uint16_t) ||
        (size_t)S * std::max(C, F) * sizeof(uint16_t) > w.lg_cap * sizeof(float) ||
        (size_t)S * std::max(2 * F, C) * sizeof(float) > w.gu32_bytes) return false;
    k::ExpertTiles tiles{reinterpret_cast<k::ExpertTile*>(w.h16), capacity, tile_rows};
    static const bool fuse_swiglu = getenv("IK2_PREFILL_SWIGLU") && atoi(getenv("IK2_PREFILL_SWIGLU"));
    const bool fused = fuse_swiglu && tiles.rows == 64;
    // The down projection overwrites the gate/up workspace after SwiGLU has consumed it.
    auto* hidden = reinterpret_cast<uint16_t*>(w.lg);
    k::rdna3_expert_groups(w.route_d, (const float*)(w.route_d + S), S, c.K, c.E, ly.tab_sc[0], ly.tab_sc[1],
                           ly.tab_sc[2], tiles, w.gidx, w.inv, w.sg, w.su, w.wtr, g_s);
    if (fused) {
        k::rdna3_expert_swiglu(ly.ty[0], ly.tab_blob, tiles, w.x16, C, F, w.sg, w.su, c.gate_cap, c.up_cap, hidden, w.gidx, g_s);
    } else {
        gather_f16_16_kernel<<<S, 256, 0, g_s>>>(w.x16, w.gidx, S, C, hidden);
        k::rdna3_expert_gemm(ly.ty[0], ly.tab_blob, 0, tiles, hidden, C, 2 * F, w.gu32, g_s);
        k::swiglu_cap_f32(w.gu32, S, F, w.sg, w.su, c.gate_cap, c.up_cap, hidden, g_s);
    }
    k::rdna3_expert_gemm(ly.ty[2], ly.tab_blob, ly.eb[0] + ly.eb[1], tiles, hidden, F, C, w.gu32, g_s);
    combine_prefill_groups<<<T, 256, 0, g_s>>>(w.gu32, w.inv, w.wtr, C, c.K, w.tmp);
    return true;
}
#endif

void forward(Model& M, const int* tok, int T, int pos0, bool all_logits, OutMode out_mode, const std::string& dump,
             std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp, bool generation) {
    k::enable_grouped_tensor_products(allow_short_batch_products());
    const Config& c = M.c;
    const int C = c.C;
    // A one-token prompt chunk uses prompt ablation, even just after a masked decode row.
    const bool prompt_ablation = !generation && !g_abl_r.empty() && (!g_abl_decode || g_abl_tokens >= 0);
    if (g_fast && !prompt_ablation && T == 1 && dump.empty() && !g_prof && !g_rec && !g_trace && !host_profile() && g_img.at(pos0) < 0) {
        fast_token(M, tok[0], pos0, out_mode, logits_out, argmax_out, generation);
        return;
    }
    if (out_mode == OutMode::kCand) out_mode = OutMode::kLogits, g_cand.n = -1;   // (this path: the logits)
    use_dev(M.L[0].dev);
    Work* Wp = &dev_work(g_cur);
    ExpertCache* ECp = &dev_cache(g_cur);
#define W (*Wp)
#define EC (*ECp)
    // A headless chunk can return with its kernels queued. Its caller will refill the same
    // pinned token positions and embeddings, so release those buffers once both copies finish.
    cudaEvent_t input_copy = out_mode == OutMode::kNone ? W.prefill_inputs_copied : nullptr;
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int t = 0; t < T; ++t)
            k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, C, W.emb_h + (size_t) t * C);
        CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) T * C * 4, cudaMemcpyHostToDevice, g_s));
        k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
        place_pictures(W.h, T, pos0, C, g_s);
        abl(W.h, T, C, 0);
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
        if (input_copy) CU(cudaEventRecord(input_copy, g_s));
    }
    int nb = 0;
    const size_t TC = (size_t) T * C;
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    // Record the last token once per completed layer visit, including fused HIP prefill.
    static const char* capture_path = std::getenv("IK2_CAPTURE_PREFILL_RESIDUALS");
    static const char* capture_attn_path = std::getenv("IK2_CAPTURE_ATTN_RESIDUALS");   // the state after a visit's attention
    const bool capture = capture_path && T > 1 && out_mode != OutMode::kNone;
    auto capture_visit = [&](int visit) {
        const int32_t header[] = {visit, pos0 + T - 1, C};
        std::vector<float> last(C);
        CU(cudaMemcpyAsync(last.data(), W.h + (size_t)(T - 1) * C, C * sizeof(float), cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        FILE* file = std::fopen(capture_path, "ab");
        if (!file || std::fwrite(header, sizeof(header), 1, file) != 1 ||
            std::fwrite(last.data(), sizeof(float), C, file) != (size_t) C || std::fclose(file))
            die("cannot write prefill residual capture");
    };
    auto dump_f = [&](const std::string& name, const float* d, size_t n) {
        if (dump.empty()) return;
        std::vector<float> h(n);
        CU(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dump + "/" + name + ".f32").c_str(), "wb");
        if (!f) die("cannot write in " + dump);
        std::fwrite(h.data(), 4, n, f);
        std::fclose(f);
    };

    int nb_pre = 0, n_bank = 0;   // (a looped core: the blocks before it, the states of the pass before's bank)
    for (int vis = 0; vis < c.n_visit; ++vis) {
        g_abl_vis = vis;
        if (capture && vis) capture_visit(vis - 1);
        const int l = M.V[vis].l;   // (the layer: its weights, its experts in the cache; the state is the visit's)
        Layer& Ly = bind_visit(M, vis);
        if (Ly.dev != g_cur) {   // the next GPU's layers: the stream and the block states go along
            const int from = g_cur;
            Work& F = dev_work(from);
            Work& Tn = dev_work(Ly.dev);
            CU(cudaStreamSynchronize(g_dev[from].s));
            CU(cudaMemcpyPeer(Tn.h, Ly.dev, F.h, from, TC * 4));
            for (int i = 0; i < nb; ++i)
                CU(cudaMemcpyPeer(Tn.blocks + (size_t) i * Tn.blk_ld * C, Ly.dev, F.blocks + (size_t) i * F.blk_ld * C,
                                  from, TC * 4));
            CU(cudaMemcpyPeer(Tn.pos, Ly.dev, F.pos, from, (size_t) T * 4));
            use_dev(Ly.dev);
            Wp = &dev_work(g_cur);
            ECp = &dev_cache(g_cur);
        }
        g_pf.start();
        // A looped core.  Its first pass: the stream it is given is kept.  A later pass starts again from that stream
        // and the blocks before the core, and reads the pass before through messages: that pass's bank is the blocks
        // it added after its first (the first is the kept stream) and its final stream.
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TC * 4, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, TC * 4, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
            }
        }
        if (M.V[vis].pass > 0) {   // the layer's messages: q's, k's, v's, the MoE's
            k::ResStates st{};
            for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
            // (two queries a launch: the kernel takes at most three)
            k::attn_res(st, n_bank, Ly.lp_query, 2, T, C, 1e-6f, W.lp_msg, g_s);
            k::attn_res(st, n_bank, Ly.lp_query + (size_t) 2 * C, 2, T, C, 1e-6f, W.lp_msg + 2 * TC, g_s);
            msg = W.lp_msg;
        }
        // (the drafter's taps: the stream at the start of layers 5, 11, .., 41 - and of a looped model's visits 5, 23
        // and 41, its mtp_config's target_layer_ids)
        if (g_taps) {
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, TC * 4, cudaMemcpyDeviceToDevice, g_s));
        }
        run_layer_attn(M, Ly, l, T, pos0, W.h, W.pos, nb, 0, W.r, W, msg);
        if (capture_attn_path && T > 1 && out_mode != OutMode::kNone) {
            const int32_t header[] = {vis, pos0 + T - 1, C};
            std::vector<float> last(C);
            CU(cudaMemcpyAsync(last.data(), W.r + (size_t)(T - 1) * C, C * sizeof(float), cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            FILE* file = std::fopen(capture_attn_path, "ab");
            if (!file || std::fwrite(header, sizeof(header), 1, file) != 1 ||
                std::fwrite(last.data(), sizeof(float), C, file) != (size_t) C || std::fclose(file))
                die("cannot write attention residual capture");
        }
        if (!dump.empty() && Ly.attn_lora.rank) {
            const std::string prefix = "blk." + std::to_string(vis) + ".lora.";
            dump_f(prefix + "input", W.o, (size_t) T * Ly.wo.K);
            dump_f(prefix + "output", W.tmp, TC);
        }
        dump_f("blk." + std::to_string(vis) + ".attn_out", W.r, TC);
        if (l % c.block == 0) ++nb;
        // the MoE
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {   // (its message gates the input before the norm)
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq};
                loop_gates_chunk(W, &gt, 1, T, C, c.rank, 1e-6f);
                k::rms_f16(W.xq, Ly.ffn_norm, T, C, c.eps, W.x16, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm;
                outs.y16[0] = W.x16;
                outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.x16, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        if (dump.empty() && !g_trace && !host_profile() && resident_expert_prefill(c, Ly, T, W, t_exp)) {
            if (g_pop_device) k::count_routes(rtop, S, device_counts(l), g_s);
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            abl(W.h, T, C, 2);
            g_pf.lap(g_pf.rest);
            continue;
        }
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        const bool cpu_lane = g_pool && g_overlap && !g_prof && T <= 8 && Ly.ty[0] == k::kNVFP4 &&
                              Ly.ty[1] == k::kNVFP4 && Ly.ty[2] == k::kNVFP4;
        if (cpu_lane) CU(cudaMemcpyAsync(W.x_h, W.xq, TC * 4, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top = W.route_h;
        const float* w = (const float*) (W.route_h + S);
        pop_count(l, top, S);
        g_pf.lap(g_pf.route);
        if (!dump.empty()) {   // the experts picked, [T][K] int32 (compare with the reference's blk.L.picks)
            FILE* f = std::fopen((dump + "/blk." + std::to_string(vis) + ".picks.i32").c_str(), "wb");
            if (f) std::fwrite(top, 4, (size_t) S, f), std::fclose(f);
        }
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top, 4, S, g_trace);
        }
        if (g_tp && T == 1 && g_overlap && !g_prof) {   // tensor-parallel experts, one token
            const double te0 = now();
            const int F = c.F, Fd = F / 2, Cd = C / 2;
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            const uint8_t* canon[2][16];
            // (1) each GPU: its halves in (hits run first), its half of h for every pick into tp_h (row = the pick)
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int ord[16], nhit = 0, nmis = 0, mis[16];
                for (int i = 0; i < S; ++i) {
                    bool missed = false;
                    canon[d][i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    if (missed) mis[nmis++] = i;
                    else ord[nhit++] = i;
                }
                for (int j = 0; j < nmis; ++j) ord[nhit + j] = mis[j];
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const uint8_t* blob[16];
                int zeros[16] = {}, orow[16];
                float sgv[16], suv[16];
                for (int j = 0; j < S; ++j) {
                    const int i = ord[j], e = top[i];
                    blob[j] = canon[d][i], orow[j] = i;
                    sgv[j] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                    suv[j] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                }
                std::memcpy(pk, blob, S * sizeof(void*));
                std::memcpy(pk + 512, zeros, S * 4);
                std::memcpy(pk + 768, orow, S * 4);
                std::memcpy(pk + 1024, sgv, S * 4);
                std::memcpy(pk + 1280, suv, S * 4);
                std::memcpy(pk + 1536, canon[d], S * sizeof(void*));
                CU(cudaMemcpyAsync(pk_d, pk, 1600, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const int* por = (const int*) (pk_d + 768);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) {
                    CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                    CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                }
                size_t o, b0g;
                tp_part(Ly, d, 0, o, b0g);
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu_rows(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], b0g, Wd.xq, C, Fd,
                                                   psg + p0, psu + p0, c.gate_cap, c.up_cap, Wd.tp_h + (size_t) d * Fd,
                                                   por + p0, F, g_s);
                }
                CU(cudaEventRecord(g_tp_h[d], g_s));
                Wd.tp_pk = pk_d;
            }
            // (2) each GPU: the other's half of h in, its rows of down; GPU 1's rows to GPU 0, combined there
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                const Work& O = dev_work(1 - d);
                CU(cudaStreamWaitEvent(g_s, g_tp_h[1 - d], 0));
                CU(cudaMemcpy2DAsync(Wd.tp_h + (size_t) (1 - d) * Fd, (size_t) F * 2, O.tp_h + (size_t) (1 - d) * Fd,
                                     (size_t) F * 2, (size_t) Fd * 2, S, cudaMemcpyDefault, g_s));
                size_t o, b0g, b1u;
                tp_part(Ly, d, 0, o, b0g);
                tp_part(Ly, d, 1, o, b1u);
                k::experts_down((const uint8_t* const*) (Wd.tp_pk + 1536), S, Ly.ty[2], b0g + b1u, Wd.tp_h, Cd, F,
                                Wd.y32, g_s);
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) S * Cd * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = i;
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1, c.K, Cd, W.tmp,
                                   false, g_s);
                k::moe_combine_f32(W.y32 + (size_t) 8 * C, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), 1,
                                   c.K, Cd, W.tmp + Cd, false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            abl(W.h, T, C, 2);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_ep && T == 1 && g_overlap && !g_prof) {
            const double te0 = now();
            const int F = c.F;
            int row_of[16], n_d[2] = {0, 0};
            const uint8_t* blob_d[2][16];
            float sg_d[2][16], su_d[2][16];
            int dev_of[16];
            for (int i = 0; i < S; ++i) dev_of[i] = ep_owner(l, top[i]);
            // GPU 1 first (it has the farther to go): x over, its picks (hits first, then the copied ones), y back
            CU(cudaEventRecord(g_ep_x, g_dev[0].s));
            for (int d = 1; d >= 0; --d) {
                use_dev(d);
                Work& Wd = dev_work(d);
                ExpertCache& Ed = dev_cache(d);
                const uint64_t tick = ++Ed.tick;
                int hit[16], mis[16], nhit = 0, nmis = 0;
                const uint8_t* where[16];
                for (int i = 0; i < S; ++i) {
                    if (dev_of[i] != d) continue;
                    bool missed = false;
                    where[i] = Ed.get(Ly, l, top[i], tick, g_cs, &missed, true, false);
                    (missed ? mis[nmis++] : hit[nhit++]) = i;
                }
                CU(cudaEventRecord(g_ep_cp[d], g_cs));
                int n = 0;
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < (pass ? nmis : nhit); ++j) {
                        const int i = pass ? mis[j] : hit[j], e = top[i];
                        blob_d[d][n] = where[i];
                        sg_d[d][n] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                        su_d[d][n] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                        row_of[i] = n++;
                    }
                n_d[d] = n;
                if (d == 1) CU(cudaStreamWaitEvent(g_s, g_ep_x, 0));
                if (!n) continue;
                const int ps = Wd.pack_slot;
                Wd.pack_slot = (Wd.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = Wd.pack_h[ps];
                uint8_t* pk_d = Wd.pack_d[ps];
                const int zeros[16] = {};
                std::memcpy(pk, blob_d[d], n * sizeof(void*));
                std::memcpy(pk + 512, zeros, n * 4);
                std::memcpy(pk + 1024, sg_d[d], n * 4);
                std::memcpy(pk + 1280, su_d[d], n * 4);
                CU(cudaMemcpyAsync(pk_d, pk, 1536, cudaMemcpyHostToDevice, g_s));
                const auto* pb = (const uint8_t* const*) pk_d;
                const int* ptk = (const int*) (pk_d + 512);
                const float* psg = (const float*) (pk_d + 1024);
                const float* psu = (const float*) (pk_d + 1280);
                if (d == 1) CU(cudaMemcpyPeerAsync(Wd.xq, 1, dev_work(0).xq, 0, (size_t) C * 4, g_s));
                for (int part = 0; part < 2; ++part) {
                    const int p0 = part ? nhit : 0, m = part ? nmis : nhit;
                    if (part) CU(cudaStreamWaitEvent(g_s, g_ep_cp[d], 0));
                    if (!m) continue;
                    k::experts_gate_up_swiglu(pb + p0, ptk + p0, m, Ly.ty[0], Ly.ty[1], Ly.eb[0], Wd.xq, C, F, psg + p0,
                                              psu + p0, c.gate_cap, c.up_cap, Wd.h16 + (size_t) p0 * F, g_s);
                    k::experts_down(pb + p0, m, Ly.ty[2], Ly.eb[0] + Ly.eb[1], Wd.h16 + (size_t) p0 * F, C, F,
                                    Wd.y32 + (size_t) p0 * C, g_s);
                }
                if (d == 1) {
                    CU(cudaMemcpyPeerAsync(dev_work(0).y32 + (size_t) 8 * C, 0, Wd.y32, 1, (size_t) n * C * 4, g_s));
                    CU(cudaEventRecord(g_ep_done, g_s));
                }
            }
            // on GPU 0 (current): the combine over both GPUs' rows (GPU 1's from row 8)
            if (n_d[1]) CU(cudaStreamWaitEvent(g_s, g_ep_done, 0));
            {
                const int ps = W.pack_slot;
                W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
                uint8_t* pk = W.pack_h[ps];
                uint8_t* pk_d = W.pack_d[ps];
                int id[16];
                float wt[16];
                for (int i = 0; i < S; ++i) {
                    id[i] = (dev_of[i] ? 8 : 0) + row_of[i];
                    wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][top[i]]);
                }
                std::memcpy(pk + 768, id, S * 4);
                std::memcpy(pk + 1536, wt, S * 4);
                CU(cudaMemcpyAsync(pk_d + 768, pk + 768, 1024, cudaMemcpyHostToDevice, g_s));
                k::moe_combine_f32(W.y32, (const int*) (pk_d + 768), (const float*) (pk_d + 1536), T, c.K, C, W.tmp,
                                   false, g_s);
            }
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            abl(W.h, T, C, 2);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_fast && T > 1 && S <= W.rg_slots && !Ly.tab_h.empty()) {
            // Every expert in VRAM and a prompt of a few tokens (a chat turn): the experts straight over their
            // blocks, an expert's blocks read once for up to kGroupMax of the tokens that picked it.  (The GEMM path
            // below decodes a whole expert to FP16 for the two or three tokens it has: a 47-token prompt took 0.34 s.)
            const double te0 = now();
            const int GM = W.rg_groups;
            const size_t o_n = (size_t) GM * 8, o_tok = o_n + (size_t) GM * 4, o_row = o_tok + (size_t) GM * k::kGroupMax * 4,
                         o_sg = o_row + (size_t) GM * k::kGroupMax * 4, o_su = o_sg + (size_t) W.rg_slots * 4,
                         o_wt = o_su + (size_t) W.rg_slots * 4, o_id = o_wt + (size_t) W.rg_slots * 4;
            auto** gb = (const uint8_t**) W.rg_h;
            int *gn = (int*) (W.rg_h + o_n), *gtok = (int*) (W.rg_h + o_tok), *grow = (int*) (W.rg_h + o_row);
            float *sg = (float*) (W.rg_h + o_sg), *su = (float*) (W.rg_h + o_su), *wt = (float*) (W.rg_h + o_wt);
            int* id = (int*) (W.rg_h + o_id);
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            int G = 0;
            for (int i = 0; i < S; ++i) {   // slot i (token i / K, one of its picks): row i of h16 and y32
                const int e = top[i];
                int g = W.gi_stamp[e] == stp ? W.gi_idx[e] : -1;
                if (g < 0 || gn[g] == k::kGroupMax) {   // the expert's first group, or its next
                    g = G++;
                    W.gi_stamp[e] = stp, W.gi_idx[e] = g;
                    gb[g] = Ly.tab_h[e], gn[g] = 0;
                }
                gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                grow[(size_t) g * k::kGroupMax + gn[g]] = i;
                ++gn[g];
                sg[i] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[i] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                id[i] = i;
            }
            // (the stream was synchronised after the routing: the copy of the last layer's tables is done)
            CU(cudaMemcpyAsync(W.rg_d, W.rg_h, W.rg_bytes, cudaMemcpyHostToDevice, g_s));
            k::experts_gate_up_swiglu_g((const uint8_t* const*)W.rg_d, (const int*)(W.rg_d + o_n),
                                        (const int*)(W.rg_d + o_tok), (const int*)(W.rg_d + o_row), G, Ly.ty[0],
                                        Ly.ty[1], Ly.eb[0], W.xq, C, c.F, (const float*)(W.rg_d + o_sg),
                                        (const float*)(W.rg_d + o_su), c.gate_cap, c.up_cap, W.h16, g_s, W.x16);
            k::experts_down_g((const uint8_t* const*) W.rg_d, (const int*) (W.rg_d + o_n), (const int*) (W.rg_d + o_row),
                              G, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            k::moe_combine_f32(W.y32, (const int*) (W.rg_d + o_id), (const float*) (W.rg_d + o_wt), T, c.K, C, W.tmp,
                               false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            abl(W.h, T, C, 2);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        if (g_mv && S <= 64) {
            const double te0 = now();
            // the token path: each slot (token t's pick i) its expert's blob straight from the cache
            // The slots already in VRAM first, then the ones copied in now: the first group's experts run while the
            // second's blocks are on their way (the copies on a stream of their own).
            const uint8_t* blob[64];
            int tk[64], id[64];
            float sg[64], su[64], wt[64];
            const uint64_t tick = ++EC.tick;
            const cudaStream_t cst = g_overlap && !g_prof ? g_cs : g_s;
            const uint8_t* where[64];
            char late[64] = {};
            int slot_of[64];
            if (++W.stamp == 0) {
                std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                W.stamp = 1;
            }
            const uint32_t stp = W.stamp;
            lazy_poll();
            double acc_up = 0, acc_cpu = 0;
            int urgent[64], lazy[64];
            int n_urgent = 0, n_lazy = 0;
            for (int i = 0; i < S; ++i) {
                const int e = top[i];
                if (W.seen_stamp[e] == stp) {   // a second token of this call on an expert missing in it
                    late[i] = W.seen_late[e];
                    where[i] = EC.has(l, e) ? EC.get(Ly, l, e, tick, cst, nullptr, false, false) : nullptr;
                    slot_of[i] = where[i] ? EC.last_handle : -1;
                    continue;
                }
                if (EC.has(l, e)) {
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    if (EC.slot_sseq(EC.last_handle) > g_seq_done) late[i] = 1;   // still arriving from a lazy copy
                    continue;
                }
                // a miss: the CPU's or the link's, whichever is free first
                if (cpu_lane && acc_cpu + g_t_cpu <= acc_up + g_t_up) {
                    late[i] = 2, acc_cpu += g_t_cpu;
                    if (g_cpu_lazy) {   // a slot for its blocks, copied in afterwards
                        where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                        slot_of[i] = EC.last_handle;
                        lazy[n_lazy++] = i;
                    } else {            // no slot taken: nothing in VRAM is evicted for it
                        ++EC.misses;
                        where[i] = nullptr, slot_of[i] = -1;
                    }
                } else {
                    late[i] = 1, acc_up += g_t_up;
                    where[i] = EC.get(Ly, l, e, tick, cst, nullptr, false, false);
                    slot_of[i] = EC.last_handle;
                    urgent[n_urgent++] = i;
                }
                W.seen_stamp[e] = stp;
                W.seen_late[e] = late[i];
            }
            int nh = 0, nw = 0;
            for (int i = 0; i < S; ++i) nh += late[i] == 0, nw += late[i] == 1;
            for (int ui = 0; ui < n_urgent; ++ui) {
                const int i = urgent[ui];
                EC.copy_in(Ly, top[i], (uint8_t*) where[i], cst);
            }
            if (nw && cst != g_s) CU(cudaEventRecord(g_ev, g_cs));
            if (n_lazy > 0) {
                const uint64_t seq = g_seq_next++;
                if (seq > g_seq_done + kRing - 1) {   // the ring's entry is still an unfinished batch's
                    CU(cudaStreamSynchronize(g_cs));
                    g_seq_done = seq - 1;
                }
                for (int li = 0; li < n_lazy; ++li) {
                    const int i = lazy[li];
                    EC.copy_in(Ly, top[i], (uint8_t*) where[i], g_cs);
                    EC.slot_sseq(slot_of[i]) = seq;
                }
                CU(cudaEventRecord(g_ring[seq % kRing], g_cs));
            }
            const int ps = W.pack_slot;
            W.pack_slot = (W.pack_slot + 1) % Work::kPackRing;
            uint8_t* pk = W.pack_h[ps];
            uint8_t* pk_d = W.pack_d[ps];
            float* yh = W.y_h[ps];
            int ph = 0, pm = nh, pc = nh + nw;
            ik2::cpu::ExpertJob jobs[64];
            const int n_jobs = S - nh - nw;
            const double tc0 = now();
            for (int i = 0; i < S; ++i) {
                const int e = top[i], p = late[i] == 2 ? pc++ : late[i] ? pm++ : ph++;
                blob[p] = where[i];
                tk[p] = i / c.K;
                id[i] = p;
                sg[p] = Ly.scale[0].empty() ? 1.0f : Ly.scale[0][e];
                su[p] = Ly.scale[1].empty() ? 1.0f : Ly.scale[1][e];
                wt[i] = w[i] * (Ly.scale[2].empty() ? 1.0f : Ly.scale[2][e]);
                if (late[i] == 2) {
                    auto& J = jobs[p - nh - nw];
                    J.gate = Ly.host[0] + (size_t) e * Ly.stride[0];
                    J.up = Ly.host[1] + (size_t) e * Ly.stride[1];
                    J.down = Ly.host[2] + (size_t) e * Ly.stride[2];
                    J.x = W.x_h + (size_t) (i / c.K) * C;
                    J.sg = sg[p], J.su = su[p];
                    J.y = yh + (size_t) (p - nh - nw) * C;
                    ik2::cpu::post_expert(*g_pool, &J, C, c.F, c.gate_cap, c.up_cap);
                }
            }
            g_n_cpu += n_jobs, g_n_wait += nw;
            // the groups: a part's (0: in VRAM, 1: waited for) slots by expert, each expert's blocks read once
            int ng[2] = {0, 0};
            const uint8_t* gb[64];
            int gn[64];
            int gtok[64 * k::kGroupMax];
            int grow[64 * k::kGroupMax];
            int G = 0;
            for (int part = 0; part < 2; ++part) {
                if (++W.stamp == 0) {
                    std::fill(W.seen_stamp.begin(), W.seen_stamp.end(), 0);
                    std::fill(W.gi_stamp.begin(), W.gi_stamp.end(), 0);
                    W.stamp = 1;
                }
                const uint32_t gstp = W.stamp;
                for (int i = 0; i < S; ++i) {
                    if (late[i] != part) continue;
                    const int e = top[i];
                    int g;
                    if (W.gi_stamp[e] != gstp) {
                        W.gi_stamp[e] = gstp;
                        g = G++;
                        W.gi_idx[e] = g;
                        gb[g] = where[i];
                        gn[g] = 0;
                        ++ng[part];
                    } else {
                        g = W.gi_idx[e];
                    }
                    gtok[(size_t) g * k::kGroupMax + gn[g]] = i / c.K;
                    grow[(size_t) g * k::kGroupMax + gn[g]] = id[i];
                    ++gn[g];
                }
            }
            g_pf.lap(g_pf.upload);
            // one copy of the tables from pinned memory: [64] pointers, then tok, inv, sg, su, wt [64] each
            std::memcpy(pk, blob, S * sizeof(void*));
            std::memcpy(pk + 512, tk, S * 4);
            std::memcpy(pk + 768, id, S * 4);
            std::memcpy(pk + 1024, sg, S * 4);
            std::memcpy(pk + 1280, su, S * 4);
            std::memcpy(pk + 1536, wt, S * 4);
            // ... and the groups: [64] pointers, n [64], tok [64][8], row [64][8]
            std::memcpy(pk + 1792, gb, G * sizeof(void*));
            std::memcpy(pk + 2304, gn, G * 4);
            std::memcpy(pk + 2560, gtok, (size_t) G * k::kGroupMax * 4);
            std::memcpy(pk + 4608, grow, (size_t) G * k::kGroupMax * 4);
            CU(cudaMemcpyAsync(pk_d, pk, 6656, cudaMemcpyHostToDevice, g_s));
            const auto* pgb = (const uint8_t* const*) (pk_d + 1792);
            const int* pgn = (const int*) (pk_d + 2304);
            const int* pgt = (const int*) (pk_d + 2560);
            const int* pgr = (const int*) (pk_d + 4608);
            const auto* pb = (const uint8_t* const*) pk_d;
            const int* ptk = (const int*) (pk_d + 512);
            const int* pid = (const int*) (pk_d + 768);
            const float* psg = (const float*) (pk_d + 1024);
            const float* psu = (const float*) (pk_d + 1280);
            const float* pwt = (const float*) (pk_d + 1536);
            const int F = c.F;
            for (int part = 0; part < 2; ++part) {
                const int p0 = part ? nh : 0, n = part ? nw : nh;
                if (part && cst != g_s) CU(cudaStreamWaitEvent(g_s, g_ev, 0));
                if (!n) continue;
                if (g_group && g_group_exp && T > 1) {
                    const int g0 = part ? ng[0] : 0, gc = ng[part];
                    k::experts_gate_up_swiglu_g(pgb + g0, pgn + g0, pgt + (size_t) g0 * k::kGroupMax,
                                                pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                                W.xq, C, F, psg, psu, c.gate_cap, c.up_cap, W.h16, g_s);
                    k::experts_down_g(pgb + g0, pgn + g0, pgr + (size_t) g0 * k::kGroupMax, gc, Ly.ty[2],
                                      Ly.eb[0] + Ly.eb[1], W.h16, C, F, W.y32, g_s);
                    continue;
                }
                k::experts_gate_up_swiglu(pb + p0, ptk + p0, n, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, F, psg + p0,
                                          psu + p0, c.gate_cap, c.up_cap, W.h16 + (size_t) p0 * F, g_s);
                k::experts_down(pb + p0, n, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16 + (size_t) p0 * F, C, F,
                                W.y32 + (size_t) p0 * C, g_s);
            }
            if (n_jobs > 0) {
                g_pool->wait();
                g_t_cpu = 0.9 * g_t_cpu + 0.1 * (now() - tc0) / n_jobs;
                CU(cudaMemcpyAsync(W.y32 + (size_t) (nh + nw) * C, yh, (size_t) n_jobs * C * 4, cudaMemcpyHostToDevice,
                                   g_s));
            }
            k::moe_combine_f32(W.y32, pid, pwt, T, c.K, C, W.tmp, false, g_s);
            g_pf.lap(g_pf.experts);
            t_exp += now() - te0;
            conv_rec(Ly, 3, W.tmp, Ly.h_mlp, T, C);
            k::conv4_res(W.tmp, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, T, C, W.r, W.h, g_s);
            abl(W.h, T, C, 2);
            g_pf.lap(g_pf.rest);
            dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
            continue;
        }
        run_prefill_moe(M, Ly, l, T, W.x16, top, w, W.r, W.h, W, EC, t_exp);
        g_pf.lap(g_pf.rest);
        dump_f("blk." + std::to_string(vis) + ".out", W.h, TC);
    }
    if (capture) capture_visit(c.n_visit - 1);
    run_head(M, T, all_logits, out_mode, nb, W.h, W, logits_out, argmax_out);
    if (input_copy) CU(cudaEventSynchronize(input_copy));
#undef W
#undef EC
}

// the scratch of a T-token pass, carved from base (or only its size, base null)
size_t big_carve(const Config& c, int T, size_t gu32_have, size_t lg_want, size_t lg_have, Work* W, uint8_t* base) {
    const size_t C = c.C, F = c.F, S = (size_t) T * c.K, nblk = c.n_layer / c.block + 1;
    const size_t Tr = std::max<size_t>(64, T);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        uint8_t* p = base ? base + off : nullptr;
        off += (bytes + 255) / 256 * 256;
        return p;
    };
    float* h = (float*) take(T * C * 4);
    float* r = (float*) take(T * C * 4);
    float* tmp = (float*) take(T * C * 4);
    float* blocks = (float*) take(nblk * T * C * 4);
    uint16_t* xq = (uint16_t*) take(T * C * 2);
    int* pos = (int*) take((size_t) T * 4);
    float* rlog = (float*) take((size_t) T * c.E * 4);
    int* route = (int*) take(S * 8);
    uint16_t* ex16 = (uint16_t*) take(T * C * 2);
    uint16_t* h16 = (uint16_t*) take(Tr * F * 2);
    const bool new_gu = (size_t) T * 2 * F * 4 > gu32_have;
    float* gu32 = new_gu ? (float*) take((size_t) T * 2 * F * 4) : nullptr;
    float* y32 = (float*) take(Tr * C * 4);
    float* sg = (float*) take(S * 4);
    float* su = (float*) take(S * 4);
    float* wt = (float*) take(S * 4);
    float* wtr = (float*) take(S * 4);
    int* gidx = (int*) take(S * 4);
    int* inv = (int*) take(S * 4);
    // the attention's logits and probabilities, room for kAttnRows queries at the prompt's end (lg_want of each)
    const bool new_lg = lg_want > lg_have;
    float* lg = new_lg ? (float*) take(lg_want * 4) : nullptr;
    uint16_t* lg16 = new_lg ? (uint16_t*) take(lg_want * 2) : nullptr;
    if (W && base) {
        W->h = h, W->r = r, W->tmp = tmp, W->blocks = blocks, W->xq16_sup = xq, W->pos = pos, W->rlog = rlog;
        W->route_d = route, W->ex16 = ex16, W->h16 = h16, W->y32 = y32;
        W->sg = sg, W->su = su, W->wt = wt, W->wtr = wtr, W->gidx = gidx, W->inv = inv;
        if (new_gu) W->gu32 = gu32, W->qkv = gu32, W->lg16 = (uint16_t*) gu32;
        if (new_lg) W->lg = lg, W->lg16 = lg16, W->lg_cap = lg_want;
        W->blk_ld = (size_t) T;
        W->T_sup = T;
    }
    return off;
}
// Tokens a pass for a prompt of n new tokens: W.T_sup, or more with a lent scratch (then big_end gives it back).
int big_begin(const Model& M, Work& W, ExpertCache& EC, int n, int ctx_end, BigScratch& bs) {
    bs.pool = -1;
    if (W.T_sup <= 0 || dual() || (getenv("IK2_BORROW") && !std::atoi(getenv("IK2_BORROW")))) return W.T_sup;
    const Config& c = M.c;
    // the attention's scratch for 256 queries against the whole context: few, large GEMMs (the default ~400 MB left
    // room for only ~20 queries a step at 100k tokens) - also lent for a short prompt at the end of a long context
    const size_t lg_want = (g_flash && !byte_cache(false) && c.D == 128) ? 0 : (size_t) c.H * 256 * (size_t) ctx_end;
    const bool long_ctx = lg_want > W.lg_cap;
    if (n <= W.T_sup && !long_ctx) return W.T_sup;
    int p = -1;
    size_t most = 0;
    for (size_t i = 0; i < EC.pools.size(); ++i) {
        const size_t b = (size_t) EC.pools[i].n_active * EC.pools[i].slot_bytes;
        if (b > most) most = b, p = (int) i;
    }
    if (p < 0) return W.T_sup;
    const auto& P = EC.pools[p];
    const int k_max = P.n_active - (2 * c.E + 80);   // (prefetch_layer still needs two layers' slots)
    if (k_max <= 0) return W.T_sup;
    const size_t avail = (size_t) k_max * P.slot_bytes;
    const size_t lg_bytes = long_ctx ? (lg_want * 4 + 255) / 256 * 256 + lg_want * 2 + 256 : 0;
    if (lg_bytes >= avail) return W.T_sup;
    int T = W.T_sup;
    if (n > W.T_sup) {
        const size_t per_tok = big_carve(c, 1 << 14, W.gu32_bytes, 0, 0, nullptr, nullptr) >> 14;
        const int T_cap = (int) std::min<size_t>((size_t) W.T_host, (avail - lg_bytes) / (per_tok + 1));
        if (T_cap > W.T_sup) {
            const int passes = (n + T_cap - 1) / T_cap;
            T = (n + passes - 1) / passes;
            while (T > W.T_sup && big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) > avail) T -= 256;
            if (T < W.T_sup) T = W.T_sup;
        }
    }
    if (T == W.T_sup && !long_ctx) return W.T_sup;
    const size_t bytes = T > W.T_sup ? big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, nullptr, nullptr) : lg_bytes;
    const int k = (int) ((bytes + P.slot_bytes - 1) / P.slot_bytes);
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    g_seq_done = g_seq_next - 1;
    uint8_t* base = EC.lend(p, k);
    if (!base) return W.T_sup;
    bs.h = W.h, bs.r = W.r, bs.tmp = W.tmp, bs.blocks = W.blocks, bs.gu32 = W.gu32, bs.qkv = W.qkv, bs.y32 = W.y32;
    bs.sg = W.sg, bs.su = W.su, bs.wt = W.wt, bs.wtr = W.wtr, bs.rlog = W.rlog, bs.xq16_sup = W.xq16_sup;
    bs.ex16 = W.ex16, bs.h16 = W.h16, bs.pos = W.pos, bs.route_d = W.route_d, bs.gidx = W.gidx, bs.inv = W.inv;
    bs.blk_ld = W.blk_ld, bs.T_sup = W.T_sup;
    bs.lg = W.lg, bs.lg16 = W.lg16, bs.lg_cap = W.lg_cap;
    if (T > W.T_sup) {
        big_carve(c, T, W.gu32_bytes, lg_want, W.lg_cap, &W, base);
    } else {   // only the attention's scratch
        W.lg = (float*) base;
        W.lg16 = (uint16_t*) (base + (lg_want * 4 + 255) / 256 * 256);
        W.lg_cap = lg_want;
    }
    bs.pool = p;
    return T;
}
void big_end(Work& W, ExpertCache& EC, BigScratch& bs) {
    if (bs.pool < 0) return;
    CU(cudaStreamSynchronize(g_cs));
    CU(cudaStreamSynchronize(g_s));
    W.h = bs.h, W.r = bs.r, W.tmp = bs.tmp, W.blocks = bs.blocks, W.gu32 = bs.gu32, W.qkv = bs.qkv, W.y32 = bs.y32;
    W.sg = bs.sg, W.su = bs.su, W.wt = bs.wt, W.wtr = bs.wtr, W.rlog = bs.rlog, W.xq16_sup = bs.xq16_sup;
    W.ex16 = bs.ex16, W.h16 = bs.h16, W.pos = bs.pos, W.route_d = bs.route_d, W.gidx = bs.gidx, W.inv = bs.inv;
    W.blk_ld = bs.blk_ld, W.T_sup = bs.T_sup;
    W.lg = bs.lg, W.lg16 = bs.lg16, W.lg_cap = bs.lg_cap;
    EC.give_back(bs.pool);
    bs.pool = -1;
}

// Sweeps each layer once across up to W.T_sup tokens (sub-chunked attention for KV/attention scratch, then one combined MoE pass).
void forward_prefill_sup(Model& M, const int* tok, int T, int pos0, int sub_chunk, OutMode out_mode,
                         std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp) {
    const Config& c = M.c;
    const int C = c.C;
    use_dev(M.L[0].dev);
    Work& W = dev_work(g_cur);
    ExpertCache& EC = dev_cache(g_cur);

    // Embed and norm all T tokens in sub-chunks of <= W.Tm
    {
        const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
        for (int off = 0; off < T; off += W.Tm) {
            const int ts = std::min(W.Tm, T - off);
            for (int t = 0; t < ts; ++t)
                k::decode_host(M.embd_type, M.embd + (size_t) tok[off + t] * rb, C, W.emb_h + (size_t) t * C);
            CU(cudaMemcpyAsync(W.tmp, W.emb_h, (size_t) ts * C * 4, cudaMemcpyHostToDevice, g_s));
            k::rms(W.tmp, M.embd_norm, ts, C, c.eps, W.h + (size_t) off * C, g_s);
            place_pictures(W.h + (size_t) off * C, ts, pos0 + off, C, g_s);
            abl(W.h + (size_t) off * C, ts, C, 0);
            if (off + ts < T) CU(cudaStreamSynchronize(g_s));   // (W.emb_h reused on next iteration)
        }
        for (int t = 0; t < T; ++t) W.pos_h[t] = pos0 + t;
        CU(cudaMemcpyAsync(W.pos, W.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    }

    // (enough tokens that a layer uses most of its experts; below, the misses copied after the routing cost less than
    // prefetching experts that turn out unused - a 119-token turn took 3.4 s with 2E, 1.9 s without)
    const bool pf = (size_t) T * c.K >= (size_t) 8 * c.E;
    EC.scan = pf && !(getenv("IK2_SCAN") && !std::atoi(getenv("IK2_SCAN")));
    if (dual()) dev_cache(1).scan = EC.scan;
    // (under EP each GPU prefetches the experts it owns, by GPU 0's picks so far)
    // One layer ahead: two or three layers in flight cost more than they hide - their slots squeeze the cache (2x RTX
    // PRO 6000, a 21.8k prompt: copies 244 -> 328 GiB, 8.05 -> 9.73 s); the next layer's prefetch issued before this
    // layer's routing round trip instead of after its experts are queued was slower too
    const int depth = 1;
    auto prefetch = [&](int ll, int ahead) {
        if (!dual()) {
            prefetch_layer(M, ll, W, EC, T, nullptr, -1, ahead);
            return;
        }
        use_dev(1);
        prefetch_layer(M, ll, dev_work(1), dev_cache(1), T, &W, g_tp ? -1 : 1, ahead);
        use_dev(0);
        prefetch_layer(M, ll, W, EC, T, nullptr, g_tp ? -1 : 0, ahead);
    };
    if (pf)
        for (int ll = 0; ll < depth; ++ll) prefetch(ll, ll + 1);
    int nb = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {   // (no loop here: a looped model's prompts go through forward())
        g_abl_vis = vis;
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        g_pf.start();
        for (int off = 0; off < T; off += sub_chunk) {
            const int ts = std::min(sub_chunk, T - off);
            run_layer_attn(M, Ly, l, ts, pos0 + off, W.h + (size_t) off * C, W.pos + off, nb, off,
                           W.r + (size_t) off * C, W);
        }
        if (l % c.block == 0) ++nb;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + (size_t) i * W.blk_ld * C;
            st.st[nb] = W.r;
            k::RmsOuts outs{};
            outs.w[0] = Ly.ffn_norm;
            outs.y16[0] = W.xq16_sup;
            k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        }
        g_pf.lap(g_pf.attn);
        gemm16(Ly.router16, c.E, C, W.xq16_sup, T, W.rlog, c.E);
        const int S = T * c.K;
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
        k::route(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        CU(cudaMemcpyAsync(W.route_h, W.route_d, (size_t) S * 8, cudaMemcpyDeviceToHost, g_s));
        CU(cudaStreamSynchronize(g_s));
        const int* top_h = W.route_h;
        const float* w_h = (const float*) (W.route_h + S);
        pop_count(l, top_h, S);
        g_pf.lap(g_pf.route);
        if (g_trace) {
            const int hdr[3] = {l, T, c.K};
            std::fwrite(hdr, 4, 3, g_trace);
            std::fwrite(top_h, 4, S, g_trace);
        }
        run_prefill_moe(M, Ly, l, T, W.xq16_sup, top_h, w_h, W.r, W.h, W, EC, t_exp);
        if (pf) prefetch(l + depth, depth);
        g_pf.lap(g_pf.rest);
    }
    EC.scan = false;
    if (dual()) dev_cache(1).scan = false;

    if (out_mode != OutMode::kNone) {
        g_pf.start();
        const int last = T - 1;
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = W.blocks + ((size_t) i * W.blk_ld + last) * C;
        st.st[nb] = W.h + (size_t) last * C;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm;
        outs.y16[0] = W.x16;
        outs.y32[0] = g_dense_q8 ? W.xq : nullptr;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        matmul(M.out_w, W.xq, W.x16, 1, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax) {
            argmax_out->resize(1);
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
            (*argmax_out)[0] = W.argmax_h[0];
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            const size_t at = logits_out.size();
            logits_out.resize(at + (size_t) c.n_vocab);
            CU(cudaMemcpyAsync(logits_out.data() + at, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
            CU(cudaStreamSynchronize(g_s));
        }
        g_pf.lap(g_pf.rest);
    } else {
        CU(cudaStreamSynchronize(g_s));
    }
}

}  // namespace ik2::engine
