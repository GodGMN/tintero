#include "ik2_engine_internal.cuh"

namespace ik2::engine {


// ---- the resident token path
// When the cache holds every expert of the model (Inkling-2-Mini on a 24 GB card) none is ever missing or evicted, so
// an expert's blocks are found on the device (Layer::tab_blob) and a token needs nothing from the host between its
// embedding and its logits: no routing round trip a layer, no table upload.  The position is read on the device too
// (W.pos), so a token's launches do not change from one token to the next and are replayed as one CUDA graph.  The
// dense weights are the GGUF's blocks (IK2_DENSE=q8, the default when the experts fit), one product a launch.
bool g_fast = false;        // set at load: every expert resident, one GPU, the dense weights as blocks (IK2_FAST=0: no)
bool g_fast_want = false;
bool gpu_served = false;
ServedExperts served;
#include "ik2_served_kernels.cuh"
bool g_sparse_decode = false;
bool g_graph = true;        // IK2_GRAPH=0: the resident path's launches one by one
bool g_block = true;        // IK2_BLOCK=0: a verify window through forward() instead of the block path
int g_attn_mma = 1;         // IK2_ATTN_MMA=0 uses scalar kernels; 2 uses tensor cores only without a window.
int g_mma_split = 1;        // its blocks a K / V head (k::attn_decode_mma_blocks: the device filled once)
bool tier_active = false;
bool stage_cold = false;
FastGraph g_fgraph[2][4]; // by OutMode

HostExpertPlan host_plan;

void prepare_host_experts(Model& model, size_t bytes_needed, int depth) {
#if defined(IK2_HOST_AVX2) && !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
    static int threads = getenv("IK2_DEEP_CPU_THREADS") ? std::max(0, atoi(getenv("IK2_DEEP_CPU_THREADS"))) : 0;
    if (!threads || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") ||
        !__builtin_cpu_supports("f16c"))
        return;
    if (model.c.C % 64 || model.c.F % 64) return;
    const int limit = getenv("IK2_DEEP_CPU_MIN") ? std::max(0, atoi(getenv("IK2_DEEP_CPU_MIN"))) : 524288;
    host_plan.forced = getenv("IK2_DEEP_CPU_TEST") && atoi(getenv("IK2_DEEP_CPU_TEST"));
    const auto& c = model.c;
    if ((!host_plan.forced && (depth < limit || !bytes_needed))) {
        if (host_plan.active.empty()) return;
        bytes_needed = 0;
    }
    struct Choice {
        int layer;
        double visits_per_byte;
        size_t bytes;
        bool q8;
    };
    std::vector<Choice> choices;
    for (int l = 0; l < c.n_layer; ++l) {
        const auto& layer = model.L[l];
        const bool q8 = layer.ty[0] == k::kQ8_0;
        if (layer.ty[1] != layer.ty[0] || layer.ty[2] != layer.ty[0] || (!q8 && layer.ty[0] != k::kNVFP4)) continue;
        const size_t bytes = (size_t)c.E * (layer.eb[0] + layer.eb[1] + layer.eb[2]);
        const int visits = (l >= c.core0 && l < c.core1) ? c.n_pass : 1;
        choices.push_back({l, (double)visits / bytes, bytes, q8});
    }
    std::stable_sort(choices.begin(), choices.end(),
                     [](const Choice& a, const Choice& b) { return a.visits_per_byte < b.visits_per_byte; });
    std::vector<uint8_t> active(c.n_layer);
    size_t placed = 0;
    for (const auto& choice : choices) {
        if (!host_plan.forced && placed >= bytes_needed) break;
        active[choice.layer] = 1;
        placed += choice.bytes;
    }
    if (active == host_plan.active) return;
    auto& buffers = host_plan.buffers;
    if (placed && !buffers.picks) {
        CU(cudaHostAlloc(&buffers.picks, c.K * sizeof(int), cudaHostAllocPortable));
        CU(cudaHostAlloc(&buffers.input, c.C * sizeof(float), cudaHostAllocPortable));
        CU(cudaHostAlloc(&buffers.gate_up, c.K * 2 * c.F * sizeof(float), cudaHostAllocPortable));
        CU(cudaHostAlloc(&buffers.hidden, c.K * c.F * sizeof(uint16_t), cudaHostAllocPortable));
        CU(cudaHostAlloc(&buffers.output, c.K * c.C * sizeof(float), cudaHostAllocPortable));
        host_plan.packed.resize(c.n_layer);
    }
    const double start = now();
    try {
        for (const auto& choice : choices) {
            const int l = choice.layer;
            if (!active[l] || host_plan.packed[l]) continue;
            auto& layer = model.L[l];
            host_plan.packed[l] =
                ik2::cpu::pack_layer(c.C, c.F, c.K, c.E, layer.host, layer.stride, threads, choice.q8, buffers);
        }
    } catch (const std::bad_alloc&) {
        threads = 0;
        active.assign(c.n_layer, 0);
        std::fprintf(stderr, "host expert allocation failed, using GPU staging\n");
    }
    // A captured host node owns a stable pointer to each selected packed layer.
    // Synchronize before replacing the graph for a different placement.
    CU(cudaStreamSynchronize(g_s));
    for (auto& graph : g_fgraph[1]) {
        if (graph.exec) CU(cudaGraphExecDestroy(graph.exec));
        if (graph.graph) CU(cudaGraphDestroy(graph.graph));
        graph = {};
    }
    host_plan.active = std::move(active);
    if (!threads) host_plan.packed.clear();
    std::fprintf(stderr, "deep CPU: %zu layers, %d threads, placement %.2f s\n",
                 (size_t)std::count(host_plan.active.begin(), host_plan.active.end(), uint8_t(1)), threads,
                 now() - start);
#endif
}

// one token's launches: its embedding row in W.emb_h and its position in W.pos_h[0] (pinned host; copied first)
// Stage only remote expert blobs with long, coalesced PCIe reads.
__global__ void stage_selected_experts(const uint8_t* const* table, const uint8_t* remote, const int* indices,
                                       const uint8_t** selected, size_t nvec, uint4* scratch) {
    const int slot = blockIdx.y, e = indices[slot];
    if (!remote[e])
        return;
    const uint4* src = (const uint4*)table[e];
    uint4* dst = scratch + (size_t)slot * nvec;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += blockDim.x * gridDim.x)
        dst[i] = src[i];
    if (!blockIdx.x && !threadIdx.x)
        selected[slot] = (const uint8_t*)dst;
}

void decode_attention(const Config& c, const Layer& Ly, int attention_bucket, float* q_p, float* r_p,
                      float* gate_p, float* kconv, float* vconv, const int* pos, uint16_t* x16,
                      const uint16_t* sink_after_projection, const float* xq, float* sink, float* lg, float* o,
                      int* sparse_pages = nullptr) {
    const int C = c.C, HD = c.H * c.D;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    // the attention's key splits: fixed, so that the launches are the same at every position
    const int P_sl = std::min(128, std::max(1, (c.window + 63) / 64));
    constexpr int full_splits[] = {128, 16, 32, 64, 128, 256, 512};
    const int P_full = full_splits[attention_bucket];
    const int extent = Ly.sliding ? c.extent_swa : c.extent;
    const int window = Ly.sliding ? c.window : (1 << 30);
    const int P = Ly.sliding ? P_sl : P_full;
    // the attention on the tensor cores, from an FP16 copy of q (IK2_ATTN_MMA=0: the scalar kernels - a layer
    // without a window, its keys the whole context, is bound by their arithmetic)
#ifdef STRATA_USE_HIP
    const float* split_gate = gate_p;
    uint16_t* split_half = x16;
#else
    const float* split_gate = nullptr;
    uint16_t* split_half = nullptr;
#endif
    const bool mma = g_attn_mma == 1 || ((g_attn_mma == 2 || attention_bucket >= 3) && !Ly.sliding);
    const int ring = Ly.sliding ? Ly.kv_cap : 0;
    int ns = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : (attention_bucket ? P_full : g_mma_split);
#ifdef STRATA_USE_HIP
    if (Ly.paged_kv && attention_bucket == 5) ns = 128;
#endif
    uint16_t* q16 = mma ? x16 : nullptr;
    if (Ly.kc8) {
        k::prep_qkv_r_fp8_at(q_p, Ly.q_clip, kconv, Ly.kc8, vconv, Ly.vc8, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                             c.qk_clip, c.rel_clip, c.eps, pos, !Ly.sliding, c.alpha, c.n_floor, Ly.fp8_layout(),
                             k_scale, v_scale, q16, g_s, sink_after_projection, xq, C, sink);
        if (sparse_pages && !Ly.sliding) {
            k::sparse_decode(q16, r_p, sink, Ly.kc8, Ly.vc8, c.H, c.KV, pos, qk_scale / k_scale,
                             1.f / v_scale, Ly.rel_proj, c.d_rel, extent, sparse_pages, lg, o, gate_p, x16, g_s);
            return;
        }
        if (mma)
            k::attn_decode_mma_fp8(q16, r_p, sink, Ly.kc8, Ly.vc8, Ly.fp8_layout(), c.H, c.KV, pos, 0, ns,
                                   qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, lg,
                                   o, gate_p, x16, g_s);
        else
            k::attn_decode_fp8_at(q_p, r_p, sink, Ly.kc8, Ly.vc8, Ly.fp8_layout(), c.H, c.KV, pos, P,
                                  qk_scale / k_scale, 1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, lg,
                                  o, g_s, split_gate, split_half);
    } else {
        k::prep_qkv_r_f16_at(q_p, Ly.q_clip, kconv, Ly.kc, vconv, Ly.vc, r_p, Ly.r_clip, c.H, c.KV, c.D, c.d_rel,
                             c.qk_clip, c.rel_clip, c.eps, pos, !Ly.sliding, c.alpha, c.n_floor, Ly.kv_cap, q16,
                             g_s, sink_after_projection, xq, C, sink);
        if (mma)
            k::attn_decode_mma_f16(q16, r_p, sink, Ly.kc, Ly.vc, ring, c.H, c.KV, pos, 0, ns, qk_scale,
                                   Ly.rel_proj, c.d_rel, extent, window, lg, o, gate_p, x16, g_s);
        else
            k::attn_decode_f16_at(q_p, r_p, sink, Ly.kc, Ly.vc, Ly.sliding ? Ly.kv_cap : 0, c.H, c.KV, pos, P,
                                  qk_scale, Ly.rel_proj, c.d_rel, extent, window, lg, o, g_s, split_gate, split_half);
    }
    if (!mma && !split_gate) k::gate_mul_f16(o, gate_p, (size_t)HD, x16, true, g_s);
}

void fast_token_launches(Model& M, Work& W, OutMode out_mode, int attention_bucket, bool sparse) {
    AblDecodeScope abl_decode;
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel;
    const size_t Cb = (size_t) C * 4;
    CU(cudaMemcpyAsync(W.tmp, W.emb_h, Cb, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, 1, C, c.eps, W.h, g_s);
    abl(W.h, 1, C, 0);
    CU(cudaMemcpyAsync(W.pos, W.pos_h, 4, cudaMemcpyHostToDevice, g_s));
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        g_abl_vis = vis;
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        // a looped core's passes (as forward()'s)
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, blk(i), Cb, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, Cb, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
                // the pass's messages, every core layer's at once: the bank stays as it is until the pass ends, and a
                // layer's launch of its own was 7 us on one SM (a job a block: the same sums, 24 launches less a token)
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * C;
                k::attn_res_jobs(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), C, 1e-6f, W.lp_msg_all, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = W.lp_msg_all + (size_t) (l - c.core0) * 4 * C;   // (q, k | v, MoE)
        if (g_taps) {   // (the drafter's taps of the token: row 0)
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
        }
        // the attention's inputs, normed, in F32: W.xq, W.xk, W.xv (sep: q, k and v each have their own)
        float* x3[3] = {W.xq, W.xk, W.xv};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t)(Ly.n_res_q == 3 ? j : 0) * C,
                             msg + (size_t)j * C,
                             Ly.lp_down[j],
                             Ly.lp_forget[j],
                             Ly.lp_gain[j],
                             x3[j],
                             {Ly.lp_bf[j][0], Ly.lp_bf[j][1], Ly.lp_bf[j][2]}};
                k::loop_gates(gt, 3, 1, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms_multi(k::RmsJobs{{x3[0], x3[1], x3[2]}, {nw[0], nw[1], nw[2]}, {x3[0], x3[1], x3[2]}}, 3, C, c.eps,
                             g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, 1, C, 1e-6f, c.eps, outs, g_s);
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms(W.h, nw[j], 1, C, c.eps, x3[j], g_s);
            }
        }
        // the projections, straight over the blocks
        float *q_p, *k_p, *v_p, *r_p, *gate_p, *kd_p = nullptr, *vd_p = nullptr;
        const uint16_t* sink_after_projection = Ly.sink16;
        k::MvF16Jobs side{};
#ifdef STRATA_USE_HIP
        static const bool fuse_projections = getenv("IK2_DECODE_PROJ") && atoi(getenv("IK2_DECODE_PROJ"));
        if (fuse_projections && C == 2304 && (Ly.sliding ? Ly.wqkvrg.q8 : Ly.wqrg.q8)) {
            side.job[0] = {Ly.sink16, W.xq, W.sink, c.H};
            sink_after_projection = nullptr;
            if (!Ly.sliding) {
                side.job[1] = {Ly.sc_k_dyn16, W.xk, W.dyn, c.KV * 4};
                side.job[2] = {Ly.sc_v_dyn16, W.xv, W.dyn + c.KV * 4, c.KV * 4};
            }
        }
#endif
        if (Ly.sliding) {
            const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
            q_p = W.qkv, k_p = q_p + HD, v_p = k_p + KD, r_p = v_p + KD, gate_p = r_p + HR;
            if (m.q8) {
                const k::MvAlt key = sep ? k::MvAlt{W.xk, HD, KD} : k::MvAlt{};
                const k::MvAlt value = sep ? k::MvAlt{W.xv, HD + KD, KD} : k::MvAlt{};
                k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, W.xq, W.qkv,
                               {HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                               {HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv}, g_s, key, value, side);
            } else if (!sep) {
                mv1c(m, W.xq, W.qkv, {HD, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv},
                     {HD + KD, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
            } else {
                mv1(mat_rows(m, 0, HD), W.xq, W.qkv);
                mv1c(mat_rows(m, HD, KD), W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, nullptr, 0, nullptr, W.kconv});
                mv1c(mat_rows(m, HD + KD, KD), W.xv, v_p,
                     k::MvConv{0, KD, Ly.h_v, Ly.sc_v, nullptr, 0, nullptr, W.vconv});
                mv1(mat_rows(m, HD + 2 * KD, m.N - HD - 2 * KD), W.xq, W.qkv + HD + 2 * KD);
            }
        } else {
            const Mat& m = Ly.wqrg;     // q | r | gate
            q_p = W.qkv, r_p = q_p + HD, gate_p = r_p + HR;
            k_p = W.kk, v_p = W.vv, kd_p = W.dyn, vd_p = W.dyn + (size_t) c.KV * 4;
            if (side.job[0].rows) {
                k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, W.xq, W.qkv, {}, {}, g_s, {}, {}, side);
            } else {
                mv1(m, W.xq, W.qkv);
                k::mv_f16(Ly.sc_k_dyn16, c.KV * 4, C, W.xk, kd_p, g_s);
                k::mv_f16(Ly.sc_v_dyn16, c.KV * 4, C, W.xv, vd_p, g_s);
            }
            mv1c(Ly.wk, W.xk, k_p, k::MvConv{0, KD, Ly.h_k, Ly.sc_k, kd_p, c.KV, nullptr, W.kconv});
            mv1c(Ly.wv, W.xv, v_p, k::MvConv{0, KD, Ly.h_v, Ly.sc_v, vd_p, c.KV, nullptr, W.vconv});
        }
        // the clips, the token's K and V into the cache at its position, the attention
        decode_attention(c, Ly, attention_bucket, q_p, r_p, gate_p, W.kconv, W.vconv, W.pos,
                         W.x16, sink_after_projection, W.xq, W.sink, W.lg, W.o, sparse ? W.sparse_pages : nullptr);
        // the output projection with its short conv (and the stream added, but where a block starts: there the
        // stream joins the blocks and starts again)
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, Cb, cudaMemcpyDeviceToDevice, g_s));
        if (Ly.attn_lora.rank) {
            mv1(Ly.wo, W.o, W.tmp);
            add_attention_lora(Ly, W.o, 1, W.attn_lora_tmp, W.tmp);
            k::conv4_res(W.tmp, Ly.h_attn, Ly.sc_attn, nullptr, 0, 1, C, first ? nullptr : W.h, W.r, g_s);
        } else {
            mv1c(Ly.wo, W.o, W.tmp, k::MvConv{0, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, first ? nullptr : W.h, W.r});
        }
        abl(W.r, 1, C, 1);
        if (first) ++nb;
        // the MoE: its input, the router and the picks' tables on the device, the experts
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq,
                                     msg + (size_t)3 * C,
                                     Ly.lp_down[3],
                                     Ly.lp_forget[3],
                                     Ly.lp_gain[3],
                                     W.xq,
                                     {Ly.lp_bf[3][0], Ly.lp_bf[3][1], Ly.lp_bf[3][2]}};
                k::loop_gates(&gt, 1, 1, C, c.rank, 1e-6f, W.lp_g, g_s);
                k::rms(W.xq, Ly.ffn_norm, 1, C, c.eps, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + c.K);
        static const int limb = getenv("IK2_EXPERT_LIMB") ? atoi(getenv("IK2_EXPERT_LIMB")) : 0;
        const bool iq4_gate = Ly.ty[0] == k::kIQ4_NL && Ly.ty[1] == k::kIQ4_NL;
        const bool q8_gate = Ly.ty[0] == k::kQ8_0 && Ly.ty[1] == k::kQ8_0;
        const bool fast_gate =
            C % 64 == 0 && (((limb & 1) && Ly.ty[0] == k::kNVFP4 && Ly.ty[1] == k::kNVFP4) || ((limb & 4) && q8_gate) || ((limb & 16) && iq4_gate));
        const bool fast_down =
            c.F % 64 == 0 && (((limb & 2) && Ly.ty[2] == k::kNVFP4) || ((limb & 8) && Ly.ty[2] == k::kQ8_0) || ((limb & 32) && Ly.ty[2] == k::kIQ4_NL));
#ifdef STRATA_USE_HIP
        static const bool exp_mma = getenv("IK2_EXPERT_MMA") && atoi(getenv("IK2_EXPERT_MMA"));
#else
        static const bool exp_mma = !getenv("IK2_EXPERT_MMA") || atoi(getenv("IK2_EXPERT_MMA"));
#endif
        const bool exp_tc = exp_mma && !fast_gate && !fast_down && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F);
        k::mv_f16(Ly.router16, c.E, C, W.xq, W.rlog, g_s);
        k::route_token(W.rlog, Ly.probs_b, c.E, c.K, c.route_scale, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1],
                       Ly.tab_sc[2], rtop, rw, W.blob, W.tok, W.inv, W.sg, W.su, W.wt, g_s, W.xq, C,
                       fast_gate ? W.gu32 : nullptr, q8_gate || iq4_gate, exp_tc ? (uint16_t*) W.gu32 : nullptr, device_counts(l));

        if (gpu_served) {
            const auto& layer = served.layers[l];
            auto reserve = Ly.tab_blob + c.E + M.V[vis].pass * c.K;
            int* records = served.records + vis * c.K;
            copy_served_experts<<<dim3(64, c.K), 256, 0, g_s>>>(rtop, Ly.tab_blob, layer.cold, reserve, layer.bytes, records);
            publish_served_experts<<<1, c.K, 0, g_s>>>(records, c.K, Ly.tab_blob, reserve, W.blob);
        }

#ifdef IK2_HOST_AVX2
        if ((tier_active || host_plan.forced) && !host_plan.active.empty() && host_plan.active[l]) {
            auto& buf = host_plan.buffers;
            auto* layer = host_plan.packed[l].get();
            // Keep routing and the nonlinear activation on the GPU. Only the
            // exact-order products run on the CPU, with activations crossing PCIe.
            CU(cudaMemcpyAsync(buf.picks, rtop, c.K * sizeof(int), cudaMemcpyDeviceToHost, g_s));
            CU(cudaMemcpyAsync(buf.input, W.xq, C * sizeof(float), cudaMemcpyDeviceToHost, g_s));
            CU(cudaLaunchHostFunc(g_s, ik2::cpu::gate_up_rows, layer));
            CU(cudaMemcpyAsync(W.gu32, buf.gate_up, c.K * 2 * c.F * sizeof(float), cudaMemcpyHostToDevice, g_s));
            k::swiglu_cap_f32(W.gu32, c.K, c.F, W.sg, W.su, c.gate_cap, c.up_cap, W.h16, g_s);
            CU(cudaMemcpyAsync(buf.hidden, W.h16, c.K * c.F * sizeof(uint16_t), cudaMemcpyDeviceToHost, g_s));
            CU(cudaLaunchHostFunc(g_s, ik2::cpu::down_rows, layer));
            CU(cudaMemcpyAsync(W.y32, buf.output, c.K * C * sizeof(float), cudaMemcpyHostToDevice, g_s));
        } else
#endif
        {
            const size_t exbytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
            if (stage_cold && tier_active && exbytes % 16 == 0 && exbytes * c.K <= W.lg_cap * sizeof(float))
                stage_selected_experts<<<dim3(128, c.K), 256, 0, g_s>>>(Ly.tab_blob, Ly.tier_cold, rtop, W.blob,
                                                                        exbytes / 16, (uint4*)W.lg);
            if (exp_tc) {
                k::experts_gate_up_swiglu_mma(W.blob, c.K, Ly.eb[0], (const uint16_t*) W.gu32, C, c.F, W.sg, W.su,
                                              c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_down_mma(W.blob, c.K, Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            } else {
            if (fast_gate)
                k::expert_limb_mv(true, W.blob, Ly.eb[0], C, c.F, c.K, W.xq, W.h16, W.y32, W.sg, W.su, c.gate_cap,
                                  c.up_cap, W.gu32, g_s, Ly.ty[0], true);
            else
                k::experts_gate_up_swiglu(W.blob, W.tok, c.K, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq, C, c.F, W.sg, W.su,
                                          c.gate_cap, c.up_cap, W.h16, g_s);
            if (fast_down && C % 2 == 0)
                k::expert_limb_mv(false, W.blob, Ly.eb[0] + Ly.eb[1], C, c.F, c.K, W.xq, W.h16, W.y32, W.sg, W.su,
                                  c.gate_cap, c.up_cap, W.gu32, g_s, Ly.ty[2], false);
            else
                k::experts_down(W.blob, c.K, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
            }
        }
        k::moe_combine_conv(W.y32, W.inv, W.wt, c.K, C, Ly.h_mlp, Ly.sc_mlp, W.r, W.h, g_s);
        abl(W.h, 1, C, 2);
    }
    if (out_mode == OutMode::kNone) return;
    {   // the head
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, 1, C, 1e-6f, c.eps, outs, g_s);
        if (g_taps)
            CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, Cb, cudaMemcpyDeviceToDevice, g_s));
        mv1(M.out_w, W.xq, W.logits);
        if (out_mode == OutMode::kArgmax) {
            k::argmax_sum(W.logits, nullptr, nullptr, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, 4, cudaMemcpyDeviceToHost, g_s));
        } else if (out_mode == OutMode::kCand) {   // (the softcap keeps the logits within +- logit_cap)
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            k::cand_topk(W.logits, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap, W.cand_part, W.cand_d, g_s);
            CU(cudaMemcpyAsync(W.cand_h, W.cand_d, (size_t) (2 + 2 * kCandCap) * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) c.n_rows, c.logit_cap, g_s);
            CU(cudaMemcpyAsync(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
}

// one token through the resident path: its logits appended to logits_out (kLogits) or its argmax (kArgmax)
int decode_bucket(int pos) {
    return g_attn_mma != 3 ? 0 : pos < 1024 ? 1 : pos < 8192 ? 2 : pos < 32768 ? 3 :
                                 pos < 131072 ? 4 : pos < 262144 ? 5 : 6;
}

void fast_token(Model& M, int tok, int pos, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out, bool generation) {
    const Config& c = M.c;
    Work& W = dev_work(0);
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) tok * rb, c.C, W.emb_h);
    W.pos_h[0] = pos;
    const int attention_bucket = decode_bucket(pos);
    const bool sparse = generation && g_sparse_decode && pos >= 32768;
    if (sparse) {
        static bool reported = false;
        if (!reported) std::fprintf(stderr, "sparse global decode active at position %d\n", pos);
        reported = true;
    }
    if (!g_graph) {
        fast_token_launches(M, W, out_mode, attention_bucket, sparse);
    } else {
        FastGraph& G =
            g_fgraph[(tier_active && (stage_cold || !host_plan.active.empty())) || host_plan.forced][(int)out_mode];
#ifdef STRATA_USE_HIP
        // A chat server can alternate between short and long requests on successive turns.
        static FastGraph previous[4];
        static const bool keep_pair = getenv("IK2_GRAPH_PAIR") && atoi(getenv("IK2_GRAPH_PAIR"));
        if (keep_pair && G.exec && (G.attention_bucket != attention_bucket || G.sparse != sparse) && !g_managed_experts &&
            !tier_active && !host_plan.forced && !stage_cold)
            std::swap(G, previous[(int)out_mode]);
#endif
        if (G.exec && (G.attention_bucket != attention_bucket || G.sparse != sparse)) {
            CU(cudaGraphExecDestroy(G.exec));
            CU(cudaGraphDestroy(G.graph));
            G = {};
        }
        if (!G.exec) {   // the first token of this kind: its launches recorded
            G.attention_bucket = attention_bucket;
            G.sparse = sparse;
            CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
            fast_token_launches(M, W, out_mode, attention_bucket, sparse);
            CU(cudaStreamEndCapture(g_s, &G.graph));
            CU(cudaGraphInstantiate(&G.exec, G.graph, 0));
        }
        CU(cudaGraphLaunch(G.exec, g_s));
    }
    if (gpu_served)
        CU(cudaMemcpyAsync(served.published, served.records, c.n_visit * c.K * sizeof(int), cudaMemcpyDeviceToHost, g_s));
    CU(cudaStreamSynchronize(g_s));
    if (gpu_served) served.finish(M, W);
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(1, W.argmax_h[0]);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
    } else if (out_mode == OutMode::kCand) {
        if (W.cand_h[0] <= kCandCap) {
            g_cand.n = W.cand_h[0], g_cand.idx = W.cand_h + 2, g_cand.val = (const float*) (W.cand_h + 2 + kCandCap);
        } else {   // (more than the buffer holds - logits flat past belief: the logits themselves, this once)
            CU(cudaMemcpy(W.logits_h, W.logits, (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost));
            logits_out.insert(logits_out.end(), W.logits_h, W.logits_h + c.n_vocab);
            g_cand.n = -1;
        }
    }
}


// ---- the block path: the T tokens (2 .. 8) of a speculative step's verify window through the resident path in one
// go - fast_token_launches for several tokens: every position read from the device, the experts grouped on the
// device (an expert's blocks read once for the tokens that picked it), the attention a key's rows read once for the
// window's queries, the convs' histories and inputs kept for a rollback - and its launches replayed as one CUDA graph
// a window size.  The dense products are the tensor cores' (q8_tile, the prompt path's arithmetic).
BlockBufs g_blk;
FastGraph g_bgraph[kBlockMax + 1][4][3];   // by window size, OutMode, and attention depth bucket

void block_init(Model& M) {
    if (g_blk.ready) return;
    const Config& c = M.c;
    const int C = c.C, KD = c.KV * c.D, S = kBlockMax * c.K;
    CU(cudaHostAlloc(&g_blk.emb_h, (size_t) kBlockMax * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.pos_h, (size_t) kBlockMax * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.logits_h, (size_t) kBlockMax * c.n_vocab * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&g_blk.cand_h, (size_t) kBlockMax * (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    g_blk.cand_d = dalloc<int>((size_t) kBlockMax * (2 + 2 * kCandCap));
    g_blk.blob = dalloc<const uint8_t*>(S);
    g_blk.n = dalloc<int>(S), g_blk.inv = dalloc<int>(S);
    g_blk.tok = dalloc<int>((size_t) S * k::kGroupMax), g_blk.row = dalloc<int>((size_t) S * k::kGroupMax);
    g_blk.sg = dalloc<float>(S), g_blk.su = dalloc<float>(S), g_blk.wt = dalloc<float>(S);
    if (c.n_pass > 1) g_blk.lp_msg = dalloc<float>((size_t) (c.core1 - c.core0) * 4 * kBlockMax * C);
    std::vector<k::ConvRec> recs;
    for (auto& Vi : M.V) {
        float* hist[4] = {Vi.h_k, Vi.h_v, Vi.h_attn, Vi.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) {
            if (!Vi.rec_old[i] || !Vi.rec_x[i]) die("the block path: the convs' records are not allocated");
            recs.push_back({Vi.rec_old[i], Vi.rec_x[i], hist[i], cc[i]});
        }
    }
    g_blk.n_recs = (int) recs.size();
    g_blk.recs = dalloc<k::ConvRec>(recs.size());
    CU(cudaMemcpy(g_blk.recs, recs.data(), recs.size() * sizeof(k::ConvRec), cudaMemcpyHostToDevice));
#ifndef STRATA_USE_HIP
    if (g_taps && getenv("IK2_BLACKWELL_NVFP4") && !atoi(getenv("IK2_BLACKWELL_NVFP4"))) {
        g_blk.zero_slots = dalloc<int>(kBlockMax);
        CU(cudaMemset(g_blk.zero_slots, 0, kBlockMax * sizeof(int)));
    }
#endif
    g_blk.ready = true;
}

// rows [r0, r0 + n) of a planar Q8_0 matrix times x16 [T][K] (FP16): y [T][ldy] (the tensor cores, q8_tile)
void tile_rows(const Mat& m, int r0, int n, const uint16_t* x16, int T, float* y, int ldy) {
    if (!m.q8) die("the block path: a dense matrix that is not planar Q8_0");
    k::q8_tile(m.q8 + (size_t) r0 * m.K, m.q8d + (size_t) r0 * (m.K >> 5), n, m.K, x16, T, y, ldy, g_s);
}

Visit sequence_visit(Visit v, const Layer& ly, const Config& c, int slot) {
    const size_t kd = c.KV * c.D;
    const size_t cap = ly.paged_kv ? ((size_t)v.kv_cap + 15) / 16 * 16 : v.kv_cap;
    if (v.kc8) {
        const size_t off = slot * cap * kv_row_bytes(kd, ly.sliding);
        v.kc8 += off;
        v.vc8 += off;
    } else {
        v.kc += slot * cap * kd;
        v.vc += slot * cap * kd;
    }
    v.h_k += slot * 3 * kd;
    v.h_v += slot * 3 * kd;
    v.h_attn += slot * 3 * c.C;
    v.h_mlp += slot * 3 * c.C;
    for (int i = 0; i < 4; ++i) {
        const size_t offset = slot * (i < 2 ? kd : c.C);
        if (v.rec_old[i]) v.rec_old[i] += 3 * offset;
        if (v.rec_x[i]) v.rec_x[i] += kBlockMax * offset;
    }
    return v;
}

void sequence_product(const Mat& m, const float* x, int T, float* y, int ld) {
    k::mv_q8p_sequences(m.q8, m.q8d, m.N, m.K, x, T, y, ld, g_s);
}

void sequence_conv(const SequenceRows& rows, const float* x, int ld, k::MvConv cv, const Visit& visit, int which) {
    const int count = rows.slots.size();
    if (rows.window)
        k::conv_windows(rows.device_slots, count, rows.window, kBlockMax, x, ld, cv, visit.rec_old[which], visit.rec_x[which], g_s);
    else
        k::conv_sequences(rows.device_slots, count, x, ld, cv, g_s);
}

void sequence_attention(Model& M, Work& W, int vis, bool sep, const SequenceRows& rows) {
    const Config& c = M.c;
    Layer& ly = bind_visit(M, vis);
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, T = (int)rows.slots.size();
    const Mat& m = ly.sliding ? ly.wqkvrg : ly.wqrg;
    const int N = m.N;
    auto product = [&](const Mat& weight, const float* x, const uint16_t* half, float* out, int ld) {
        if (rows.native_fp4) tile_rows(weight, 0, weight.N, half, T, out, ld);
        else sequence_product(weight, x, T, out, ld);
    };
    if (ly.sliding && sep) {
        product(mat_rows(m, 0, HD), W.xq, W.x16, W.qkv, N);
        product(mat_rows(m, HD, KD), W.xk, W.xk16, W.qkv + HD, N);
        product(mat_rows(m, HD + KD, KD), W.xv, W.xv16, W.qkv + HD + KD, N);
        product(mat_rows(m, HD + 2 * KD, N - HD - 2 * KD), W.xq, W.x16, W.qkv + HD + 2 * KD, N);
    } else product(m, W.xq, W.x16, W.qkv, N);
    float* kd = nullptr;
    float* vd = nullptr;
    const float* keys = W.qkv + HD;
    const float* values = W.qkv + HD + KD;
    int ld = N;
    if (!ly.sliding) {
        const int D = c.KV * 4;
        kd = W.dyn;
        vd = kd + T * D;
        k::mv_f16_rows(ly.sc_k_dyn16, D, C, W.xk, C, T, kd, D, g_s);
        k::mv_f16_rows(ly.sc_v_dyn16, D, C, W.xv, C, T, vd, D, g_s);
        product(ly.wk, W.xk, W.xk16, W.kk, KD);
        product(ly.wv, W.xv, W.xv16, W.vv, KD);
        keys = W.kk;
        values = W.vv;
        ld = KD;
    }
    sequence_conv(rows, keys, ld, {0, KD, ly.h_k, ly.sc_k, kd, c.KV, nullptr, W.kconv}, M.V[vis], 0);
    sequence_conv(rows, values, ld, {0, KD, ly.h_v, ly.sc_v, vd, c.KV, nullptr, W.vconv}, M.V[vis], 1);
    k::mv_f16_rows(ly.sink16, c.H, C, W.xq, C, T, W.sink, c.H, g_s);

#ifdef STRATA_USE_HIP
    // Different depth buckets can require different split counts.
    const int bucket = rows.buckets.front();
    const bool same_split = ly.sliding || std::all_of(rows.buckets.begin(), rows.buckets.end(),
                                                      [bucket](int b) { return b == bucket; });
    const bool scalar = g_attn_mma != 1 && (ly.sliding || (g_attn_mma != 2 && bucket < 3));
    static const bool grouped = getenv("IK2_DECODE_SUBWAVE") && atoi(getenv("IK2_DECODE_SUBWAVE"));
    const bool use_batch = grouped && same_split && scalar;
    const int splits = ly.sliding ? std::min(128, (c.window + 63) / 64) : bucket == 1 ? 16 : bucket == 2 ? 32 : 128;
    uint16_t* q16 = nullptr;
    const size_t scratch = (size_t)T * c.H * splits * 130;
#else
    const bool use_batch = rows.window && g_attn_mma == 1;
    const int splits = ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
    uint16_t* q16 = W.x16;
    const size_t scratch = (size_t)T * k::attn_decode_mma_scratch(c.H, splits);
#endif
    if (use_batch && ly.kc8 && c.D == 128 && scratch <= W.lg_cap) {
        const size_t cap = ly.paged_kv ? ((size_t)ly.kv_cap + 15) / 16 * 16 : ly.kv_cap;
        float* r = W.qkv + HD + (ly.sliding ? 2 * KD : 0);
        k::sequence_attention_quantized(W.qkv, r, r + c.H * c.d_rel, N, ly.q_clip, ly.r_clip, W.kconv, W.vconv,
                                      ly.kc8, ly.vc8, ly.fp8_layout(), cap * kv_row_bytes(KD, ly.sliding),
                                      rows.device_slots, W.pos, T, c.H, c.KV, c.d_rel, c.qk_clip, c.rel_clip,
                                      c.eps, c.alpha, c.n_floor, c.qk_clip <= 28.f ? 16.f : 1.f, 32.f, ly.rel_proj,
                                      ly.sliding ? c.extent_swa : c.extent, ly.sliding ? c.window : 1 << 30,
                                      splits, W.sink, W.lg, W.o, g_s, q16);
        return;
    }
    for (int t = 0; t < T; ++t) {
        const Visit v = sequence_visit(M.V[vis], ly, c, rows.slots[t]);
        ly.kc = v.kc; ly.vc = v.vc; ly.kc8 = v.kc8; ly.vc8 = v.vc8;
        float* q = W.qkv + t * N;
        float* r = q + HD + (ly.sliding ? 2 * KD : 0);
        decode_attention(c, ly, rows.buckets[t], q, r, r + c.H * c.d_rel, W.kconv + t * KD,
                         W.vconv + t * KD, W.pos + t, W.x16 + t * HD, nullptr, W.xq + t * C,
                         W.sink + t * c.H, W.lg, W.o + t * HD);
    }
}

#ifndef STRATA_USE_HIP
bool blackwell_verify_enabled() {
    static const bool enabled = getenv("IK2_BLACKWELL_VERIFY") && atoi(getenv("IK2_BLACKWELL_VERIFY"));
    return enabled && k::use_blackwell_nvfp4();
}
#endif

void fast_block_launches(Model& M, Work& W, int T, OutMode out_mode, BlockBufs& buf,
                         const SequenceRows* rows, int attn_bkt) {
    AblDecodeScope abl_decode;
    const Config& c = M.c;
    const int C = c.C, HD = c.H * c.D, KD = c.KV * c.D, HR = c.H * c.d_rel, S = T * c.K;
    const size_t TC = (size_t) T * C, TCb = TC * 4;
    const float qk_scale = 1.0f / std::sqrt((float) c.D);
    const float k_scale = c.qk_clip <= 28.0f ? 16.0f : 1.0f, v_scale = 32.0f;
    CU(cudaMemcpyAsync(W.tmp, buf.emb_h, TCb, cudaMemcpyHostToDevice, g_s));
    k::rms(W.tmp, M.embd_norm, T, C, c.eps, W.h, g_s);
    abl(W.h, T, C, 0);
    CU(cudaMemcpyAsync(W.pos, buf.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, g_s));
    auto blk = [&](int i) { return W.blocks + (size_t) i * W.blk_ld * C; };
    int nb = 0, nb_pre = 0, n_bank = 0;
    for (int vis = 0; vis < c.n_visit; ++vis) {
        g_abl_vis = vis;
        const int l = M.V[vis].l;
        Layer& Ly = bind_visit(M, vis);
        const float* msg = nullptr;
        if (c.n_pass > 1 && l == c.core0) {
            if (M.V[vis].pass == 0) {
                nb_pre = nb;
                CU(cudaMemcpyAsync(W.lp_pre, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
            } else {
                n_bank = 0;
                for (int i = nb_pre + 1; i < nb; ++i, ++n_bank)
                    CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, blk(i), TCb, cudaMemcpyDeviceToDevice, g_s));
                CU(cudaMemcpyAsync(W.lp_bank + (size_t) n_bank * TC, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
                ++n_bank;
                CU(cudaMemcpyAsync(W.h, W.lp_pre, TCb, cudaMemcpyDeviceToDevice, g_s));
                nb = nb_pre;
                k::ResStates st{};
                for (int i = 0; i < n_bank; ++i) st.st[i] = W.lp_bank + (size_t) i * TC;
                k::attn_res_jobs_block(st, n_bank, M.lp_query_all, 2, 2 * (c.core1 - c.core0), T, C, 1e-6f,
                                       buf.lp_msg, g_s);
            }
        }
        if (M.V[vis].pass > 0) msg = buf.lp_msg + (size_t) (l - c.core0) * 4 * TC;   // [4][T][C]: q, k, v, MoE
        if (g_taps) {
            const int tap = draft_tap(c, l, vis);
            if (tap >= 0)
                CU(cudaMemcpyAsync(g_taps + (size_t) tap * g_taps_ld * C, W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
        }
        float* x3[3] = {W.xq, W.xk, W.xv};
        uint16_t* y16[3] = {W.x16, W.xk16, W.xv16};
        const float* nw[3] = {Ly.sliding ? Ly.attn_norm : Ly.q_in_norm, Ly.sliding ? Ly.attn_norm : Ly.k_in_norm,
                              Ly.sliding ? Ly.attn_norm : Ly.v_in_norm};
        bool sep = !Ly.sliding;
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.h;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, W.lp_in, g_s);
                k::LoopGate gt[3];
                for (int j = 0; j < 3; ++j)
                    gt[j] = {W.lp_in + (size_t) (Ly.n_res_q == 3 ? j : 0) * TC, msg + (size_t) j * TC, Ly.lp_down[j],
                             Ly.lp_forget[j], Ly.lp_gain[j], x3[j], {Ly.lp_bf[j][0], Ly.lp_bf[j][1], Ly.lp_bf[j][2]}};
                k::loop_gates(gt, 3, T, C, c.rank, 1e-6f, W.lp_g, g_s, rows != nullptr);
                for (int j = 0; j < 3; ++j) k::rms_f16(x3[j], nw[j], T, C, c.eps, y16[j], x3[j], g_s);
                sep = true;
            } else if (l > 0) {
                k::RmsOuts outs{};
                for (int j = 0; j < Ly.n_res_q; ++j) outs.w[j] = nw[j], outs.y16[j] = y16[j], outs.y32[j] = x3[j];
                k::attn_res_rms_f16(st, nb + 1, Ly.res_score, Ly.n_res_q, T, C, 1e-6f, c.eps, outs, g_s);
            } else {
                for (int j = 0; j < (sep ? 3 : 1); ++j) k::rms_f16(W.h, nw[j], T, C, c.eps, y16[j], x3[j], g_s);
            }
        }
        if (rows) {
            sequence_attention(M, W, vis, sep, *rows);
        } else {
            // the projections (the tensor cores), the k and v short convs (their histories and inputs recorded)
            float *q_p, *r_p, *gate_p;
            int q_ld, r_ld, gate_ld;
            if (Ly.sliding) {
                const Mat& m = Ly.wqkvrg;   // q | k | v | r | gate
                const int N = m.N;
                if (!sep) {
                    tile_rows(m, 0, N, W.x16, T, W.qkv, N);
                } else {
#ifdef STRATA_USE_HIP
                    if (m.q8 && T >= 1 && T <= 8 && m.K % 128 == 0 && ((uintptr_t)m.q8 & 3) == 0 &&
                        ((uintptr_t)W.x16 & 7) == 0 && ((uintptr_t)W.xk16 & 7) == 0 && ((uintptr_t)W.xv16 & 7) == 0) {
                        k::rdna3_q8_tile_qkv_sep(m.q8, m.q8d, N, m.K, HD, KD, W.x16, W.xk16, W.xv16, T, W.qkv, N, g_s);
                    } else {
                        tile_rows(m, 0, HD, W.x16, T, W.qkv, N);
                        tile_rows(m, HD, KD, W.xk16, T, W.qkv + HD, N);
                        tile_rows(m, HD + KD, KD, W.xv16, T, W.qkv + HD + KD, N);
                        tile_rows(m, HD + 2 * KD, N - HD - 2 * KD, W.x16, T, W.qkv + HD + 2 * KD, N);
                    }
#else
                    tile_rows(m, 0, HD, W.x16, T, W.qkv, N);
                    tile_rows(m, HD, KD, W.xk16, T, W.qkv + HD, N);
                    tile_rows(m, HD + KD, KD, W.xv16, T, W.qkv + HD + KD, N);
                    tile_rows(m, HD + 2 * KD, N - HD - 2 * KD, W.x16, T, W.qkv + HD + 2 * KD, N);
#endif
                }
                q_p = W.qkv, r_p = W.qkv + HD + 2 * KD, gate_p = r_p + HR;
                q_ld = r_ld = gate_ld = N;
                k::conv_block(W.qkv + HD, N, Ly.h_k, Ly.sc_k, nullptr, 0, 0, T, KD, nullptr, 0, W.kconv, KD, Ly.rec_old[0],
                              Ly.rec_x[0], g_s);
                k::conv_block(W.qkv + HD + KD, N, Ly.h_v, Ly.sc_v, nullptr, 0, 0, T, KD, nullptr, 0, W.vconv, KD,
                              Ly.rec_old[1], Ly.rec_x[1], g_s);
            } else {
                const Mat& m = Ly.wqrg;     // q | r | gate
                const int N = m.N;
                tile_rows(m, 0, N, W.x16, T, W.qkv, N);
                q_p = W.qkv, r_p = W.qkv + HD, gate_p = r_p + HR;
                q_ld = r_ld = gate_ld = N;
                const int DL = c.KV * 4;
                float *kd = W.dyn, *vd = W.dyn + (size_t) T * DL;
                k::mv_f16_rows(Ly.sc_k_dyn16, DL, C, W.xk, C, T, kd, DL, g_s);
                k::mv_f16_rows(Ly.sc_v_dyn16, DL, C, W.xv, C, T, vd, DL, g_s);
                tile_rows(Ly.wk, 0, KD, W.xk16, T, W.kk, KD);
                tile_rows(Ly.wv, 0, KD, W.xv16, T, W.vv, KD);
                k::conv_block(W.kk, KD, Ly.h_k, Ly.sc_k, kd, DL, c.KV, T, KD, nullptr, 0, W.kconv, KD, Ly.rec_old[0],
                              Ly.rec_x[0], g_s);
                k::conv_block(W.vv, KD, Ly.h_v, Ly.sc_v, vd, DL, c.KV, T, KD, nullptr, 0, W.vconv, KD, Ly.rec_old[1],
                              Ly.rec_x[1], g_s);
            }
            // the clips, the window's K and V into the cache at its positions, the attention of its queries
            const int extent = Ly.sliding ? c.extent_swa : c.extent;
            const int window = Ly.sliding ? c.window : (1 << 30);
            const int ring = Ly.sliding ? Ly.kv_cap : 0;
            const int ns_full = Ly.sliding ? std::min(g_mma_split, std::max(1, (c.window + 63) / 64)) : g_mma_split;
            const int ns = attn_bkt == 0 ? std::min(ns_full, 2) : (attn_bkt == 1 ? std::min(ns_full, 8) : ns_full);
            uint16_t* q16 = W.x16;
            if (Ly.kc8) {
                k::prep_qkv_r_fp8_block(q_p, q_ld, Ly.q_clip, W.kconv, Ly.kc8, W.vconv, Ly.vc8, r_p, r_ld, Ly.r_clip, T, c.H,
                                        c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha,
                                        c.n_floor, Ly.fp8_layout(), k_scale, v_scale, q16, g_s, Ly.sink16, W.xq, C, W.sink);
                k::attn_block_fp8(q16, r_p, r_ld, W.sink, Ly.kc8, Ly.vc8, Ly.fp8_layout(), c.H, c.KV, W.pos, T, ns, qk_scale / k_scale,
                                  1.0f / v_scale, Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, gate_ld, W.x16,
                                  g_s);
            } else {
                k::prep_qkv_r_f16_block(q_p, q_ld, Ly.q_clip, W.kconv, Ly.kc, W.vconv, Ly.vc, r_p, r_ld, Ly.r_clip, T, c.H,
                                        c.KV, c.D, c.d_rel, c.qk_clip, c.rel_clip, c.eps, W.pos, !Ly.sliding, c.alpha,
                                        c.n_floor, Ly.kv_cap, q16, g_s, Ly.sink16, W.xq, C, W.sink);
                k::attn_block_f16(q16, r_p, r_ld, W.sink, Ly.kc, Ly.vc, ring, c.H, c.KV, W.pos, T, ns, qk_scale,
                                  Ly.rel_proj, c.d_rel, extent, window, W.lg, W.o, gate_p, gate_ld, W.x16, g_s);
            }
        }
        // the output projection and its short conv (the stream added, but where a block starts)
        const bool first = l % c.block == 0;
        if (first) CU(cudaMemcpyAsync(blk(nb), W.h, TCb, cudaMemcpyDeviceToDevice, g_s));
        if (rows && rows->native_fp4) {
            k::to_f16(W.o, (size_t)T * HD, W.x16, g_s);
            tile_rows(Ly.wo, 0, C, W.x16, T, W.tmp, C);
        } else if (rows) sequence_product(Ly.wo, W.o, T, W.tmp, C);
        else tile_rows(Ly.wo, 0, C, W.x16, T, W.tmp, C);
        if (Ly.attn_lora.rank) add_attention_lora(Ly, W.o, T, W.attn_lora_tmp, W.tmp);
        if (rows) sequence_conv(*rows, W.tmp, C,
                                    {0, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, first ? nullptr : W.h, W.r}, M.V[vis], 2);
        else k::conv_block(W.tmp, C, Ly.h_attn, Ly.sc_attn, nullptr, 0, 0, T, C, first ? nullptr : W.h, C, W.r, C,
                      Ly.rec_old[2], Ly.rec_x[2], g_s);
        abl(W.r, T, C, 1);
        if (first) ++nb;
        // the MoE: its input, the router, the picks grouped by expert on the device, the experts
        {
            k::ResStates st{};
            for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
            st.st[nb] = W.r;
            if (msg) {
                k::attn_res(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, W.xq, g_s);
                const k::LoopGate gt{W.xq, msg + (size_t) 3 * TC, Ly.lp_down[3], Ly.lp_forget[3], Ly.lp_gain[3], W.xq,
                                     {Ly.lp_bf[3][0], Ly.lp_bf[3][1], Ly.lp_bf[3][2]}};
                k::loop_gates(&gt, 1, T, C, c.rank, 1e-6f, W.lp_g, g_s, rows != nullptr);
                if (rows && !buf.row) k::rms(W.xq, Ly.ffn_norm, T, C, c.eps, W.xq, g_s);
                else k::rms_f16(W.xq, Ly.ffn_norm, T, C, c.eps, W.x16, W.xq, g_s);
            } else {
                k::RmsOuts outs{};
                outs.w[0] = Ly.ffn_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
                k::attn_res_rms_f16(st, nb + 1, Ly.ffn_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
            }
        }
        int* rtop = W.route_d;
        float* rw = (float*) (W.route_d + S);
#ifdef STRATA_USE_HIP
        if (fused_router(c.E, c.K, T)) {
            k::rdna3_project_route(Ly.router16, W.xq, C, T, Ly.probs_b, c.route_scale, rtop, rw, g_s);
        } else
#endif
        {
            k::mv_f16_rows(Ly.router16, c.E, C, W.xq, C, T, W.rlog, c.E, g_s);
            k::route_rows(W.rlog, Ly.probs_b, T, c.E, c.K, c.route_scale, rtop, rw, g_s);
        }
        if (g_pop_device) k::count_routes(rtop, S, device_counts(l), g_s);
#ifdef STRATA_USE_HIP
        if (!rows && buf.n && buf.row && S <= 64) {
            k::group_route(rtop, rw, S, c.K, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1], Ly.tab_sc[2], buf.blob, buf.n,
                           buf.tok, buf.row, buf.sg, buf.su, buf.wt, buf.inv, g_s);
            const bool rdna_gu = Ly.ty[0] == Ly.ty[1] &&
                                 ((Ly.ty[0] == k::kNVFP4 && C % 64 == 0 && ((uintptr_t)W.x16 & 15) == 0) ||
                                  ((Ly.ty[0] == k::kQ8_0 || Ly.ty[0] == k::kIQ4_NL) && C % 32 == 0 && ((uintptr_t)W.x16 & 7) == 0));
            if (rdna_gu)
                k::rdna3_experts_block_gu(Ly.ty[0], buf.blob, buf.n, buf.tok, buf.row, S, T, Ly.eb[0], C, c.F,
                                          W.x16, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s);
            else
                k::experts_gate_up_swiglu_g(buf.blob, buf.n, buf.tok, buf.row, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq,
                                            C, c.F, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s);
            const bool rdna_dn = ((Ly.ty[2] == k::kNVFP4 && c.F % 64 == 0 && ((uintptr_t)W.h16 & 15) == 0) ||
                                  ((Ly.ty[2] == k::kQ8_0 || Ly.ty[2] == k::kIQ4_NL) && c.F % 32 == 0 && ((uintptr_t)W.h16 & 7) == 0));
            if (rdna_dn)
                k::rdna3_experts_block_down(Ly.ty[2], buf.blob, buf.n, buf.row, S, T, Ly.eb[0] + Ly.eb[1], C, c.F,
                                            W.h16, W.y32, g_s);
            else
                k::experts_down_g(buf.blob, buf.n, buf.row, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32,
                                  g_s);
#else
        if (rows && rows->native_fp4 && blackwell_verify_enabled() &&
            k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F)) {
            auto batch = k::blackwell_prefill_groups(rtop, rw, S, c.K, c.E, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1],
                                                     Ly.tab_sc[2], W.w16, buf.sg, buf.su, buf.wt, buf.inv, g_s);
            auto* x = reinterpret_cast<k::BlackwellActivation*>(W.ex16);
            auto* h = reinterpret_cast<k::BlackwellActivation*>(W.gu32);
            k::blackwell_pack_rows(W.x16, T, C, x, g_s);
            k::blackwell_expert_product(true, batch, Ly.eb[0], C, c.F, W.x16, x, buf.sg, buf.su,
                                        c.gate_cap, c.up_cap, W.h16, nullptr, g_s, h);
            k::blackwell_expert_product(false, batch, Ly.eb[0] + Ly.eb[1], C, c.F, W.h16, h,
                                        nullptr, nullptr, 0, 0, nullptr, W.y32, g_s);
        } else if (!rows || (buf.row && S <= k::kGroupRouteSlots)) {
            k::group_route(rtop, rw, S, c.K, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1], Ly.tab_sc[2], buf.blob, buf.n,
                           buf.tok, buf.row, buf.sg, buf.su, buf.wt, buf.inv, g_s);
            // Each distinct expert needs one pick, and each additional chunk needs eight.
            const int distinct = std::min(c.E, S);
            const int groups = rows ? distinct + (S - distinct) / k::kGroupMax : S;
            if (rows && rows->window && Ly.ty[0] == k::kQ8_0 && Ly.ty[1] == k::kQ8_0 &&
                Ly.ty[2] == k::kQ8_0 && !rows->native_fp4) {
                k::experts_q8_verify(buf.blob, buf.n, buf.tok, buf.row, groups, T, Ly.eb[0], Ly.eb[0] + Ly.eb[1],
                                     W.xq, C, c.F, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, W.y32, g_s);
            } else if ((!rows || W.batch_group_mma || rows->native_fp4) && (Ly.ty[0] == k::kQ8_0 || Ly.ty[0] == k::kIQ4_NL) &&
                Ly.ty[1] == Ly.ty[0] && Ly.ty[2] == Ly.ty[0] && C % 32 == 0 && c.F % 32 == 0) {
                uint16_t* limbs = (uint16_t*) W.gu32;
                k::f16_limbs_rows(W.xq, T, C, limbs, g_s);
                k::experts_block_gu_linear(Ly.ty[0], buf.blob, buf.n, buf.tok, buf.row, groups, T, Ly.eb[0], C, c.F, limbs,
                                           buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s);
                k::experts_block_down_linear(Ly.ty[2], buf.blob, buf.n, buf.row, groups, Ly.eb[0] + Ly.eb[1], C, c.F,
                                             W.h16, W.y32, g_s);
            } else if (!rows && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F) &&
                       blackwell_verify_enabled()) {
                k::BlackwellExpertBatch batch{nullptr, buf.blob, buf.n, buf.tok, buf.row, groups, false};
                auto* packed = reinterpret_cast<k::BlackwellActivation*>(W.gu32);
                k::blackwell_pack_rows(W.x16, T, C, packed, g_s);
                k::blackwell_expert_product(true, batch, Ly.eb[0], C, c.F, W.x16, packed, buf.sg, buf.su,
                                            c.gate_cap, c.up_cap, W.h16, nullptr, g_s);
                k::blackwell_pack_rows(W.h16, S, c.F, packed, g_s);
                k::blackwell_expert_product(false, batch, Ly.eb[0] + Ly.eb[1], C, c.F, W.h16, packed,
                                            nullptr, nullptr, 0, 0, nullptr, W.y32, g_s);
            } else if ((!rows || (rows->window && !rows->native_fp4)) && k::experts_mma_fits(Ly.ty[0], Ly.ty[1], Ly.ty[2], C, c.F)) {
                const int mask = rows ? 0 : k::block_nvfp4_mma_mask();
                if (mask & 1) {
                    k::experts_gate_up_swiglu_g(buf.blob, buf.n, buf.tok, buf.row, groups, Ly.ty[0], Ly.ty[1], Ly.eb[0],
                                                W.xq, C, c.F, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s, W.x16);
                } else {
                    uint16_t* limbs = (uint16_t*) W.gu32;
                    k::f16_limbs_rows(W.xq, T, C, limbs, g_s);
                    k::experts_block_gu(buf.blob, buf.n, buf.tok, buf.row, groups, T, Ly.eb[0], C, c.F, limbs, buf.sg,
                                        buf.su, c.gate_cap, c.up_cap, W.h16, g_s);
                }
                if (mask & 2)
                    k::experts_down_g(buf.blob, buf.n, buf.row, groups, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
                else
                    k::experts_block_down(buf.blob, buf.n, buf.row, groups, Ly.eb[0] + Ly.eb[1], C, c.F, W.h16, W.y32, g_s);
            } else {
                k::experts_gate_up_swiglu_g(buf.blob, buf.n, buf.tok, buf.row, groups, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq,
                                            C, c.F, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s, rows ? W.x16 : nullptr);
                k::experts_down_g(buf.blob, buf.n, buf.row, groups, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32,
                                  g_s);
            }
#endif
        } else {
            k::route_gather(rtop, rw, Ly.tab_blob, Ly.tab_sc[0], Ly.tab_sc[1], Ly.tab_sc[2], S, c.K,
                            buf.blob, buf.tok, buf.inv, buf.sg, buf.su, buf.wt, g_s, rows && c.E == 128);
            k::experts_gate_up_swiglu(buf.blob, buf.tok, S, Ly.ty[0], Ly.ty[1], Ly.eb[0], W.xq,
                                      C, c.F, buf.sg, buf.su, c.gate_cap, c.up_cap, W.h16, g_s);
            k::experts_down(buf.blob, S, Ly.ty[2], Ly.eb[0] + Ly.eb[1], W.h16, C, c.F, W.y32, g_s);
        }
        k::moe_combine_f32(W.y32, buf.inv, buf.wt, T, c.K, C, W.tmp, false, g_s);
        if (rows) sequence_conv(*rows, W.tmp, C,
                                    {0, C, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, W.r, W.h}, M.V[vis], 3);
        else k::conv_block(W.tmp, C, Ly.h_mlp, Ly.sc_mlp, nullptr, 0, 0, T, C, W.r, C, W.h, C, Ly.rec_old[3], Ly.rec_x[3],
                      g_s);
        abl(W.h, T, C, 2);
    }
    if (out_mode == OutMode::kNone) return;
    {   // the head: every token's logits
        k::ResStates st{};
        for (int i = 0; i < nb; ++i) st.st[i] = blk(i);
        st.st[nb] = W.h;
        k::RmsOuts outs{};
        outs.w[0] = M.out_norm, outs.y16[0] = W.x16, outs.y32[0] = W.xq;
        k::attn_res_rms_f16(st, nb + 1, M.out_res_score, 1, T, C, 1e-6f, c.eps, outs, g_s);
        if (g_taps)
            CU(cudaMemcpyAsync(g_taps + (size_t) g_final_tap * g_taps_ld * C, W.xq, TCb, cudaMemcpyDeviceToDevice, g_s));
        if (rows && !rows->native_fp4) sequence_product(M.out_w, W.xq, T, W.logits, c.n_rows);
        else tile_rows(M.out_w, 0, c.n_rows, W.x16, T, W.logits, c.n_rows);
        if (out_mode == OutMode::kArgmax || rows) {
            // A mixed batch still takes greedy rows' argmax before the sampler's softcap.
            k::argmax_rows(W.logits, c.n_rows, T, c.n_vocab, W.argmax_d, g_s);
            CU(cudaMemcpyAsync(W.argmax_h, W.argmax_d, (size_t) T * 4, cudaMemcpyDeviceToHost, g_s));
            if (out_mode == OutMode::kArgmax) return;
        }
        if (out_mode == OutMode::kCand) {   // (each row's candidates, as the token path's)
            const size_t cs = 2 + 2 * (size_t) kCandCap;
            k::softcap(W.logits, (size_t) T * c.n_rows, c.logit_cap, g_s);
            for (int t = 0; t < T; ++t)
                k::cand_topk(W.logits + (size_t) t * c.n_rows, c.n_vocab, -c.logit_cap, c.logit_cap, kCandK, kCandCap,
                             W.cand_part, buf.cand_d + t * cs, g_s);
            CU(cudaMemcpyAsync(buf.cand_h, buf.cand_d, (size_t) T * cs * 4, cudaMemcpyDeviceToHost, g_s));
        } else {
            k::softcap(W.logits, (size_t) T * c.n_rows, c.logit_cap, g_s);
            for (int t = 0; t < T; ++t)
                CU(cudaMemcpyAsync(buf.logits_h + (size_t) t * c.n_vocab, W.logits + (size_t) t * c.n_rows,
                                   (size_t) c.n_vocab * 4, cudaMemcpyDeviceToHost, g_s));
        }
    }
}

// The window tok [T] at positions pos0 .. pos0 + T - 1 through the block path: argmax_out [T] (kArgmax) or logits_out
// [T][n_vocab] appended (kLogits).  The convs' records then hold the window (block_rollback takes it back).
void fast_block(Model& M, const int* tok, int T, int pos0, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out) {
    const Config& c = M.c;
    if (T < 1 || T > kBlockMax) die("fast_block: a window of 1 .. 8 tokens");
    if (!g_blk.ready) block_init(M);
    Work& W = dev_work(0);
    const size_t rb = (size_t) c.C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    for (int t = 0; t < T; ++t) {
        k::decode_host(M.embd_type, M.embd + (size_t) tok[t] * rb, c.C, g_blk.emb_h + (size_t) t * c.C);
        g_blk.pos_h[t] = pos0 + t;
    }
    const int attn_bkt = (pos0 + T) < 256 ? 0 : ((pos0 + T) < 1024 ? 1 : 2);
    SequenceRows rows;
    if (g_blk.zero_slots) {
        rows.window = T;
        rows.slots.assign(T, 0);
        rows.device_slots = g_blk.zero_slots;
        for (int t = 0; t < T; ++t) rows.buckets.push_back(decode_bucket(pos0 + t));
    }
    if (!g_graph || (g_blk.zero_slots && g_attn_mma == 3)) {
        fast_block_launches(M, W, T, out_mode, g_blk, g_blk.zero_slots ? &rows : nullptr, attn_bkt);
    } else {
        FastGraph& G = g_bgraph[T][(int) out_mode][attn_bkt];
        if (!G.exec) {
            CU(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeGlobal));
            fast_block_launches(M, W, T, out_mode, g_blk, g_blk.zero_slots ? &rows : nullptr, attn_bkt);
            CU(cudaStreamEndCapture(g_s, &G.graph));
            CU(cudaGraphInstantiate(&G.exec, G.graph, 0));
        }
        CU(cudaGraphLaunch(G.exec, g_s));
    }
    CU(cudaStreamSynchronize(g_s));
    if (out_mode == OutMode::kArgmax) {
        argmax_out->assign(W.argmax_h, W.argmax_h + T);
    } else if (out_mode == OutMode::kLogits) {
        logits_out.insert(logits_out.end(), g_blk.logits_h, g_blk.logits_h + (size_t) T * c.n_vocab);
    }
}
// row t's candidates of the last kCand window (logit, token); false: more than kept (the row's logits then)
bool block_cands(const int* candidates, int t, std::vector<std::pair<float, int>>& cd) {
    const int* r = candidates + (size_t) t * (2 + 2 * kCandCap);
    if (r[0] > kCandCap) return false;
    const int* idx = r + 2;
    const float* val = (const float*) (r + 2 + kCandCap);
    cd.resize(r[0]);
    for (int i = 0; i < r[0]; ++i) cd[i] = {val[i], idx[i]};
    return true;
}
// every conv's history as after the first n tokens of the last window (fast_block's records)
void block_rollback(int n) {
    k::rollback_block(g_blk.recs, g_blk.n_recs, n, g_s);
}


// ---- speculative decoding on the resident path: the drafter's tokens verified as one window (fast_block)

// the serve loop's sampling distribution over candidates cd (logit, token): softmax(logit / temp) over the top_p
// mass of the min(top_k, kCandK) largest (ties: the lower token first) - pick_top's - as (token, probability)
std::vector<std::pair<int, double>> sample_dist(std::vector<std::pair<float, int>>& cd, const SampleCfg& sc) {
    const int kk = std::min((int) cd.size(), sc.top_k > 0 ? std::min(sc.top_k, kCandK) : kCandK);
    std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first || (a.first == b.first && a.second < b.second);
                      });
    std::vector<double> pr(kk);
    double z = 0;
    for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / sc.temp));
    double acc = 0;
    int keep = kk;
    for (int j = 0; j < kk; ++j) {
        acc += pr[j] / z;
        if (acc >= sc.top_p) {
            keep = j + 1;
            break;
        }
    }
    double zk = 0;
    for (int j = 0; j < keep; ++j) zk += pr[j];
    std::vector<std::pair<int, double>> out(keep);
    for (int j = 0; j < keep; ++j) out[j] = {cd[j].second, pr[j] / zk};
    return out;
}

ik2::draft::Draws draft_draws(double temperature, int count, std::mt19937_64& rng) {
    ik2::draft::Draws draws{};
    draws.temperature = temperature;
    for (int i = 0; i < count; ++i) draws.uniform[i] = std::uniform_real_distribution<double>(0, 1)(rng);
    return draws;
}

// One speculative step: the anchor (the last chosen token, not run yet) at position P, B - 1 drafts after it, the B
// verified as one window.  next: the tokens chosen after the anchor - the accepted drafts and the target's own next
// token (the last: not run yet, the next step's anchor), cut after a stop token.  The anchor and next but its last
// are then in the cache, the convs' histories taken back to them and the drafter's context written for them.
void spec_step(Model& M, ik2::draft::Drafter* drafter, int anchor, int P, int B, const SampleCfg& sc,
               const std::vector<int>& stops, std::vector<int>& next) {
    const Config& c = M.c;
    static float *raw = nullptr, *anchor_h = nullptr;
    if (!raw) raw = dalloc<float>(c.C), anchor_h = dalloc<float>(c.C);
    embed_row(M, anchor, raw, anchor_h);
    int dr[kBlockMax];
    const bool sample = sc.temp > 0 && ik2::draft::probabilistic(drafter);
    ik2::draft::Draws draws;
    ik2::draft::DraftMass mass[kBlockMax-1];
    if (sample) draws = draft_draws(sc.temp, B-1, *sc.rng);
    ik2::draft::propose(drafter, anchor, anchor_h, P, nullptr, c.n_rows, c.n_vocab, dr, B - 1,
                        sample ? &draws : nullptr, sample ? mass : nullptr);
    int tk[kBlockMax];
    tk[0] = anchor;
    for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
    next.clear();
    std::vector<float> lo;
    if (sc.temp <= 0) {
        std::vector<int> g;
        fast_block(M, tk, B, P, OutMode::kArgmax, lo, &g);
        int n = 0;
        while (n < B - 1 && dr[n] == g[n]) ++n;
        for (int j = 0; j < n; ++j) next.push_back(dr[j]);
        next.push_back(g[n]);
    } else {
        fast_block(M, tk, B, P, OutMode::kCand, lo, nullptr);
        for (int j = 0; j < B; ++j) {
            std::vector<std::pair<float, int>> cd;
            if (!block_cands(g_blk.cand_h, j, cd)) {   // (more candidates than kept: the row's logits)
                std::vector<float> row((size_t) c.n_vocab);
                CU(cudaMemcpy(row.data(), dev_work(0).logits + (size_t) j * c.n_rows, (size_t) c.n_vocab * 4,
                              cudaMemcpyDeviceToHost));
                cd.resize(c.n_vocab);
                for (int v = 0; v < c.n_vocab; ++v) cd[v] = {row[v], v};
            }
            auto dist = sample_dist(cd, sc);
            if (j == B - 1) {
                next.push_back(sample_from(dist, *sc.rng));
                break;
            }
            if (sample) {
                next.push_back(verify_draft(dist, dr[j], mass[j], *sc.rng));
                if (next.back() != dr[j]) break;
                continue;
            }
            double pd = 0;
            for (const auto& e : dist)
                if (e.first == dr[j]) pd = e.second;
            if (std::uniform_real_distribution<double>(0, 1)(*sc.rng) < pd) {
                next.push_back(dr[j]);
            } else {
                next.push_back(sample_from(dist, *sc.rng, dr[j]));
                break;
            }
        }
    }
    for (size_t j = 0; j + 1 < next.size(); ++j)   // (a stop among the accepted: it is the last)
        if (std::find(stops.begin(), stops.end(), next[j]) != stops.end()) {
            next.resize(j + 1);
            break;
        }
    const int m = (int) next.size();   // (the anchor and next[0 .. m - 2] are kept: m positions)
    if (m < B) block_rollback(m);
    ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, m, P);
    ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, m - 1);
}

// every short conv's history as after the first n of the verify window's tokens
void rollback(Model& M, int n) {
    const int C = M.c.C, KD = M.c.KV * M.c.D;
    for (int vis = 0; vis < (int) M.V.size(); ++vis) {
        Layer& Ly = bind_visit(M, vis);
        float* hist[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
        const int cc[4] = {KD, KD, C, C};
        for (int i = 0; i < 4; ++i) k::conv_hist_after(Ly.rec_old[i], Ly.rec_x[i], n, cc[i], hist[i], g_s);
    }
}

// Cold prefill sweeps all experts. Keep its physical GPU allocation until an
// actual deep decode needs the room, then change the backing of the pools. The
// GPU pointer tables themselves are stable, including inside captured graphs.
int tier_depth = 0;
std::map<uint8_t*, bool> tier_host_groups;

void replace_expert_pools(Model& M, bool managed) {
    ExpertCache& ec = dev_cache(0);
    CU(cudaStreamSynchronize(g_s));
    CU(cudaStreamSynchronize(g_cs));
    if (!managed) {
        // Give back the global pages first so regular cudaMalloc has room.
        const size_t row_bytes = kv_row_bytes((size_t)M.c.KV * M.c.D, false);
        for (auto& v : M.V) {
            if (M.L[v.l].sliding) continue;
            const size_t bytes = (size_t)std::min(tier_depth, v.kv_cap) * row_bytes;
            if (!bytes) continue;
            CU(cudaMemPrefetchAsync(v.kc8 ? (void*)v.kc8 : (void*)v.kc, bytes, cudaCpuDeviceId, g_s));
            CU(cudaMemPrefetchAsync(v.vc8 ? (void*)v.vc8 : (void*)v.vc, bytes, cudaCpuDeviceId, g_s));
        }
        CU(cudaStreamSynchronize(g_s));
    }
    for (auto& pool : ec.pools) {
        const size_t bytes = (size_t)pool.n_slots * pool.slot_bytes;
        CU(cudaFree(pool.data));
        if (managed)
            CU(cudaMallocManaged(&pool.data, bytes));
        else
            pool.data = dalloc<uint8_t>(bytes);
        for (int i = 0; i < pool.n_slots; ++i) {
            const int owner = pool.owner[i];
            if (owner < 0) continue;
            const int l = owner / M.c.E, e = owner % M.c.E;
            const Layer& ly = M.L[l];
            auto* dst = pool.data + (size_t)i * pool.slot_bytes;
            if (managed) {
                // First touch on the CPU. Copy only the serialized tensor bytes.
                for (int part = 0; part < 3; ++part) {
                    std::memcpy(dst, ly.host[part] + (size_t)e * ly.stride[part], ly.eb[part]);
                    dst += ly.eb[part];
                }
            } else
                ec.copy_in(ly, e, dst, g_cs);
        }
        if (managed) CU(cudaMemAdvise(pool.data, bytes, cudaMemAdviseSetAccessedBy, 0));
    }
    CU(cudaStreamSynchronize(g_cs));
    for (int l = 0; l < M.c.n_layer; ++l) {
        Layer& ly = M.L[l];
        const auto& pool = ec.pools[ec.layer_pool[l]];
        for (int e = 0; e < M.c.E; ++e) {
            const int slot = ec.loc[(size_t)l * M.c.E + e];
            ly.tab_h[e] = pool.data + (size_t)slot * pool.slot_bytes;
        }
        CU(cudaMemcpy(ly.tab_blob, ly.tab_h.data(), ly.tab_h.size() * sizeof(void*), cudaMemcpyHostToDevice));
    }
    if (!managed) place_tails(false);
    tier_host_groups.clear();
    tier_active = managed;
}

// For deep decode, eight full K/V visits can be many times larger than the
// routed weights read in one step. Keep the live K/V on the GPU and move the
// least used expert ranges to host memory. The graph keeps the same pointers.
void place_decode(Model& M, int depth) {
    if (!g_managed_experts || !g_fast) return;
    Work& W = dev_work(0);
    ExpertCache& ec = dev_cache(0);
    const int elements = M.c.KV * M.c.D;
    const int token_bytes = kv_row_bytes(elements, false);
    size_t live = 0;
    for (const auto& v : M.V) {
        if (!M.L[v.l].sliding) live += (size_t)2 * std::min(v.kv_cap, depth + kPlacementInterval) * token_bytes;
    }
    if (live > g_kv_room + parked_bytes && parked_bytes) place_tails(true);
    const size_t room = g_kv_room + (tails_parked ? parked_bytes : 0);
    const size_t need = live > room ? live - room : 0;
    struct Group {
        uint8_t* ptr;
        size_t bytes;
        double score;
    };
    prepare_host_experts(M, need, depth);
    if (!need) {
        if (tier_active) replace_expert_pools(M, false);
        return;
    }
    const bool just_converted = !tier_active;
    if (just_converted) replace_expert_pools(M, true);
    tier_depth = depth;
    std::vector<Group> groups;
    for (auto& pool : ec.pools) {
        // These slot sizes end on half an OS page. Two slots have disjoint
        // page ranges, and preserve more of the routing profile than large groups.
        const int span = pool.slot_bytes % 4096 == 0 ? 1 : 4096 / std::gcd(pool.slot_bytes, size_t(4096));
        for (int i = 0; i < pool.n_slots; i += span) {
            const int n = std::min(span, pool.n_slots - i);
            uint64_t score = 0;
            for (int j = 0; j < n; ++j) {
                const int idx = pool.owner[i + j];
                if (idx < 0) continue;
                const int l = idx / M.c.E;
                if (!host_plan.active.empty() && host_plan.active[l]) continue;
                if (!host_plan.active.empty()) ++score;
                score += W.use.empty() ? ((l >= M.c.core0 && l < M.c.core1) ? M.c.n_pass : 1) : W.use[idx];
            }
            groups.push_back({pool.data + i * pool.slot_bytes, n * pool.slot_bytes, (double)score / n});
        }
    }
    std::stable_sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) { return a.score < b.score; });
    size_t cold = 0;
    for (auto& group : groups) {
        const bool host = cold < need;
        if (host) cold += group.bytes;
        if (!just_converted && (tier_host_groups.count(group.ptr) != 0) == host) continue;
        const int loc = host ? cudaCpuDeviceId : 0;
        CU(cudaMemAdvise(group.ptr, group.bytes, cudaMemAdviseSetPreferredLocation, loc));
        CU(cudaMemPrefetchAsync(group.ptr, group.bytes, loc, g_s));
        if (host)
            tier_host_groups[group.ptr] = true;
        else
            tier_host_groups.erase(group.ptr);
    }
    if (stage_cold) {
        std::vector<uint8_t> flags(M.c.E);
        for (int l = 0; l < M.c.n_layer; ++l) {
            auto& ly = M.L[l];
            auto& pool = ec.pools[ec.layer_pool[l]];
            const size_t span = 4096 / std::gcd(pool.slot_bytes, size_t(4096));
            for (int e = 0; e < M.c.E; ++e) {
                const size_t slot = ec.loc[(size_t)l * M.c.E + e];
                flags[e] = tier_host_groups.count(pool.data + (slot / span) * span * pool.slot_bytes) != 0;
            }
            CU(cudaMemcpy(ly.tier_cold, flags.data(), flags.size(), cudaMemcpyHostToDevice));
        }
    }
    if (cold) {
        for (auto& v : M.V) {
            if (M.L[v.l].sliding) continue;
            const size_t bytes = (size_t)std::min(depth, v.kv_cap) * token_bytes;
            if (!bytes) continue;
            CU(cudaMemPrefetchAsync(v.kc8 ? (void*)v.kc8 : (void*)v.kc, bytes, 0, g_s));
            CU(cudaMemPrefetchAsync(v.vc8 ? (void*)v.vc8 : (void*)v.vc, bytes, 0, g_s));
        }
    }
    CU(cudaStreamSynchronize(g_s));
    std::fprintf(stderr, "decode placement: depth %d, %.2f GiB of expert weights on host\n", depth,
                 cold / 1073741824.0);
}

// A repeated suffix proposes tokens already present earlier in this request. Every proposed
// token is checked against the full target; the first mismatch supplies the next anchor.
int suffix_window = 0;
int verify_suffix(Model& M, const std::vector<int>& hist, int anchor, int remaining, const std::vector<int>& stops,
                  std::vector<int>& next, double& t_exp, SuffixStats& stats, const RowSampler& sample) {
    constexpr int match = 12;
    const int end = (int)hist.size();
    if (!g_fast || !suffix_window || remaining < match || end < 2 * match) return 0;
    int src = end - match;
    const int floor = std::max(match - 1, end - 65536);
    for (; src >= floor; --src) {
        if (hist[src] == anchor &&
            std::equal(hist.begin() + src - match + 1, hist.begin() + src, hist.end() - match + 1))
            break;
    }
    if (src < floor) return 0;
    int matched = match;
    while (matched < suffix_window && src >= matched && hist[src - matched] == hist[end - matched])
        ++matched;
    const int n = std::min({suffix_window, remaining, end - src, 2 * matched - 8});
#ifdef STRATA_USE_HIP
    const bool use_block = g_block && n >= 2 && n <= kBlockMax && suffix_window <= kBlockMax;
#else
    constexpr bool use_block = false;
#endif
    if (!use_block && n < match) return 0;
    std::vector<float> logits;
    std::vector<int> predicted;
    ++stats.forwards;
    stats.offered += n - 1;
    const auto mode = sample ? OutMode::kLogits : OutMode::kArgmax;
    if (use_block) fast_block(M, hist.data() + src, n, end, mode, logits, &predicted);
    else {
        g_rec = true;
        forward(M, hist.data() + src, n, end, true, mode, std::string(), logits, &predicted, t_exp);
        g_rec = false;
    }
    // Sample from the target once per committed next token. No random numbers
    // are consumed for positions beyond a rejected proposal.
    for (int i = 0; i < n; ++i) {
        const int picked = sample ? sample(logits.data() + (size_t)i * M.c.n_vocab) : predicted[i];
        next.push_back(picked);
        if (g_cancel || i + 1 == n || picked != hist[src + i + 1]) break;
        ++stats.accepted;
        if (std::find(stops.begin(), stops.end(), picked) != stops.end()) break;
    }
    const int used = (int)next.size();
    stats.inputs += used;
    if (used < n) {
        if (use_block) block_rollback(used);
        else rollback(M, used);
    }
    return used;
}

// the embedding row of token t, normed as the model's input, into out (device [C])
void embed_row(Model& M, int t, float* raw, float* out) {
    const int C = M.c.C;
    float* e = dev_work(M.L[0].dev).emb_h;
    const size_t rb = (size_t) C / k::block_values(M.embd_type) * k::block_bytes(M.embd_type);
    k::decode_host(M.embd_type, M.embd + (size_t) t * rb, C, e);
    CU(cudaMemcpyAsync(raw, e, (size_t) C * 4, cudaMemcpyHostToDevice, g_s));
    k::rms(raw, M.embd_norm, 1, C, M.c.eps, out, g_s);
}

}  // namespace ik2::engine
