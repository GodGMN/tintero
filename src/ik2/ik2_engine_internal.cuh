#pragma once

// Private host state and operations shared by startup, prefill and resident decode.

#include "ik2_usage.hpp"
#include "ik2_cpu.hpp"
#include "ik2_cpu_deep.hpp"
#include "ik2_draft_sampling.hpp"
#include "ik2_kernels.cuh"
#include "ik2_lora.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if !defined(STRATA_USE_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
static inline cudaError_t cudaMemPrefetchAsync_compat(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0) {
    cudaMemLocation loc{};
    if (dstDevice < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = dstDevice;
    }
    return cudaMemPrefetchAsync(devPtr, count, loc, 0, stream);
}
#define cudaMemPrefetchAsync cudaMemPrefetchAsync_compat

static inline cudaError_t cudaMemAdvise_compat(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) {
    cudaMemLocation loc{};
    if (device < 0) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    return cudaMemAdvise(devPtr, count, advice, loc);
}
#define cudaMemAdvise cudaMemAdvise_compat
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <functional>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include "ik2_served_policy.hpp"
#include <memory>
#include <string>
#include <thread>
#include <csignal>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ik2::engine {

namespace k = ik2::k;

#define CU(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "tintero: %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)
#define CB(x)                                                                                              \
    do {                                                                                                   \
        cublasStatus_t e_ = (x);                                                                           \
        if (e_ != CUBLAS_STATUS_SUCCESS) {                                                                 \
            std::fprintf(stderr, "tintero: %s: cuBLAS status %d (line %d)\n", #x, (int) e_, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

[[noreturn]] void die(const std::string& m);
double now();
void lazy_poll();

// ---- the model

struct Config {
    int n_layer, C, H, KV, D, E, K, F, window, d_rel, extent, extent_swa, block, n_vocab, n_rows;
    // A looped model (Inkling-2-Mini): the layers core0 .. core1 - 1 (whole blocks) run n_pass times a token, so a
    // token is n_visit layer visits (n_layer without a loop).  rank: the rank of the loop's message gates.
    int n_visit = 0, n_pass = 1, core0 = 0, core1 = 0, rank = 0;
    float eps, route_scale, qk_clip, rel_clip, alpha, n_floor, gate_cap, up_cap, logit_cap;
    std::vector<bool> sliding;   // by layer
};

// a dense matrix as the GGUF holds it, in VRAM: its blocks [N][K]
struct Mat {
    // (IK2_DENSE=q8 with Q8_0 blocks and K a multiple of 128: the blocks taken apart - the int8 values [N][K] and the
    //  blocks' FP16 scales [N][K / 32] - which a warp reads 128 values an instruction; raw is then null)
    int8_t* q8 = nullptr;
    uint16_t* q8d = nullptr;
    uint8_t* raw = nullptr;     // IK2_DENSE=q8: the blocks, one token's products straight over them
    uint16_t* f16 = nullptr;    // else (the default): an FP16 copy, every product a cuBLAS GEMM - measured faster on an
    int type = 0, N = 0, K = 0; // RTX PRO 6000 (16 against 21 ms a token), for 4 GiB more VRAM
};

// GGUF stores A as [input, rank] and B as [rank, output], fastest dimension first.
struct AttentionLora {
    float *a = nullptr, *b = nullptr;
    int rank = 0;
    float scale = 0;
};

struct Layer {
    int dev = 0;   // the GPU it runs on (--gpus: the layers in equal runs)
    bool sliding;
    // FP16 [out][in] (fused projections on !g_dense_q8, or when quant types match on g_dense_q8)
    Mat wq, wk, wv, wr, wgate, wo;
    AttentionLora attn_lora;
    Mat wqkvrg, wqrg, wk_dyn, wv_dyn;
    uint16_t *sink16 = nullptr;                    // F16 [H][C] (used in g_dense_q8; fused in !g_dense_q8)
    uint16_t *router16 = nullptr;                  // F16 [E][C]
    float *probs_b;                                // F32 [E]
    float *attn_norm, *q_in_norm, *k_in_norm, *v_in_norm, *ffn_norm;
    float *q_clip, *r_clip, *rel_proj;             // [H][D], [H][d_rel], [extent][d_rel]
    float *res_score, *ffn_res_score;              // [nq][C], [C]
    int n_res_q;
    float *sc_k, *sc_v, *sc_attn, *sc_mlp;         // [C'][4]
    uint16_t *sc_k_dyn16 = nullptr, *sc_v_dyn16 = nullptr;   // F16 [KV * 4][C] (used in g_dense_q8; fused in !g_dense_q8)
    // the experts: types, bytes an expert, host blocks, scales (host)
    int ty[3];
    size_t eb[3];            // bytes of a role's part of an expert
    size_t stride[3];        // bytes between two experts' parts in host[r] (eb, or 2 eb in a fused gate_up, or slot_bytes)
    const uint8_t* host[3];
    bool host_contig = false;
    size_t slot_bytes = 0;
    std::vector<float> scale[3];                   // empty: 1
    // a looped core's layer: its later passes' message gates by module (0 q's, 1 k's, 2 v's, 3 the MoE's) -
    // down [rank][2 C], forget and gain [rank][C] - and the four messages' queries [4][C] in that order (F32)
    float *lp_down[4] = {}, *lp_forget[4] = {}, *lp_gain[4] = {}, *lp_query = nullptr;
    uint16_t* lp_bf[4][3] = {};
    // the resident token path (every expert in VRAM): where each expert's blocks are, and the experts' scales, on the
    // device ([E] each; a scale's null: 1)
    uint8_t* tier_cold = nullptr;
    const uint8_t** tab_blob = nullptr;
    float* tab_sc[3] = {};
    std::vector<const uint8_t*> tab_h;             // (tab_blob on the host: a short prompt's groups)
    // the state: of the visit being run (bind_visit)
    uint16_t *kc = nullptr, *vc = nullptr;         // FP16 [kv_cap][KV * D] (IK2_KV=f16)
    uint8_t *kc8 = nullptr, *vc8 = nullptr;        // E4M3 or packed E2M1 storage. kv_row_bytes() gives the bytes per token.
    int kv_cap = 0;
    bool paged_kv = false;
    bool fp4_kv = false;
    k::Fp8KvLayout fp8_layout() const {
#ifdef STRATA_USE_HIP
        if (paged_kv) return k::Fp8KvLayout::global_pages(fp4_kv);
        if (fp4_kv && sliding) return k::Fp8KvLayout::fp4_ring(kv_cap);
#endif
        if (fp4_kv) return k::Fp8KvLayout::global_fp4();
        return sliding ? kv_cap : 0;
    }
    float *h_k, *h_v, *h_attn, *h_mlp;             // conv histories [3][C']
    float *rec_old[4] = {}, *rec_x[4] = {};        // (speculative) the convs' histories and inputs of a verify window
};

// A layer visit and its own state.  Without a loop a visit is a layer; a looped core's layer is visited once a pass,
// each time with that visit's K / V cache and conv histories (the weights, and the experts in the cache, are the
// layer's).  bind_visit() points the layer's state at a visit's before it runs.
struct Visit {
    int l = 0;        // the layer
    int pass = 0;     // the core's pass (0: its first, or a layer outside the core)
    uint16_t *kc = nullptr, *vc = nullptr;
    uint8_t *kc8 = nullptr, *vc8 = nullptr;
    int kv_cap = 0;
    float *h_k = nullptr, *h_v = nullptr, *h_attn = nullptr, *h_mlp = nullptr;
    float *rec_old[4] = {}, *rec_x[4] = {};
};

struct Model {
    Config c;
    std::vector<Layer> L;
    std::vector<Visit> V;   // the visits of a token, in order
    Mat out_w;            // [n_rows][C]
    float *out_norm, *out_res_score, *embd_norm;
    float* lp_query_all = nullptr;   // a looped core's message queries, layer by layer ([n_core][4][C]: Layer::lp_query's)
    const uint8_t* embd;  // host Q8_0 [n_rows][C]
    int embd_type;
};

inline Layer& bind_visit(Model& M, int v) {
    const Visit& Vi = M.V[v];
    Layer& Ly = M.L[Vi.l];
    Ly.kc = Vi.kc, Ly.vc = Vi.vc, Ly.kc8 = Vi.kc8, Ly.vc8 = Vi.vc8, Ly.kv_cap = Vi.kv_cap;
    Ly.h_k = Vi.h_k, Ly.h_v = Vi.h_v, Ly.h_attn = Vi.h_attn, Ly.h_mlp = Vi.h_mlp;
    for (int i = 0; i < 4; ++i) Ly.rec_old[i] = Vi.rec_old[i], Ly.rec_x[i] = Vi.rec_x[i];
    return Ly;
}

extern cublasHandle_t g_blas;
extern cudaStream_t g_s;
extern cudaStream_t g_cs;
extern cudaEvent_t g_ev;
extern bool g_overlap;
extern ik2::cpu::Pool* g_pool;
extern double g_t_cpu, g_t_up;
extern uint64_t g_n_cpu, g_n_wait;
extern bool g_cpu_lazy;

constexpr int kRing = 256;

extern cudaEvent_t g_ring[kRing];
extern uint64_t g_seq_next, g_seq_done;
extern bool g_prof;
extern bool g_mv;
extern unsigned long long* g_pop_device;
extern int g_pop_E_ep;
extern bool g_ep;
extern cudaEvent_t g_ep_x, g_ep_cp[2], g_ep_done;
extern bool g_tp;
extern cudaEvent_t g_tp_h[2];
extern int g_tp_F, g_tp_C;

inline bool dual() { return g_ep || g_tp; }

extern std::vector<uint8_t> g_ep_own;

inline int ep_owner(int l, int e) { return g_ep_own[(size_t) l * g_pop_E_ep + e]; }

extern bool g_flash;
extern int g_n_sm;
extern bool g_group;
extern bool g_group_exp, g_group_attn;
extern FILE* g_trace;

struct Prof {
    double attn = 0, route = 0, upload = 0, experts = 0, rest = 0;
    double t = 0;
    void start() {
        if (g_prof) { CU(cudaStreamSynchronize(g_s)); t = now(); }
    }
    void lap(double& acc) {
        if (!g_prof) return;
        CU(cudaStreamSynchronize(g_s));
        const double n = now();
        acc += n - t;
        t = n;
    }
};

extern Prof g_pf;
extern size_t device_allocated;

template <class T>
T* dalloc(size_t n) {
    T* p = nullptr;
    device_allocated += std::max<size_t>(1, n) * sizeof(T);
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}

extern bool g_dense_q8;

enum class KVPrecision { f16, fp8, fp8_global, fp4_global, fp4 };

extern bool g_kv_managed;
extern bool g_managed_experts;
extern size_t g_kv_room;

struct PrefillTail {
    uint8_t* begin;
    size_t bytes;
};

extern std::vector<PrefillTail> prefill_tails;
extern size_t parked_bytes;
extern bool tails_parked;

template <class T> T* work_alloc(size_t count, size_t hot_bytes = 2ULL << 20) {
    const size_t n = count * sizeof(T);
    if (!g_managed_experts || !(getenv("IK2_PARK_WORK") && atoi(getenv("IK2_PARK_WORK"))) ||
        n <= hot_bytes + (2ULL << 20))
        return dalloc<T>(count);
    T* p = nullptr;
    CU(cudaMallocManaged(&p, n));
    CU(cudaMemPrefetchAsync(p, n, 0, g_s));
    const size_t first = (hot_bytes + 65535) & ~size_t(65535);
    prefill_tails.push_back({(uint8_t*)p + first, n - first});
    parked_bytes += n - first;
    return p;
}

extern bool g_rec;

// ---- the expert cache in VRAM: per-size-class pools, O(1) S3-FIFO with ghost history

// GPU d's half of role r (0 gate, 1 up, 2 down) of an expert: offset in the role's part, bytes (IK2_TP)
inline void tp_part(const Layer& Ly, int d, int r, size_t& off, size_t& bytes) {
    const size_t rows = r < 2 ? (size_t) g_tp_F : (size_t) g_tp_C, rb = Ly.eb[r] / rows, half = rows / 2;
    off = (size_t) d * half * rb, bytes = half * rb;
}
inline size_t tp_bytes(const Layer& Ly, int d) {
    size_t o, b, t = 0;
    for (int r = 0; r < 3; ++r) tp_part(Ly, d, r, o, b), t += b;
    return t;
}
struct ExpertCache {
    static constexpr uint8_t kMaxFreq = 7;
    struct Pool {
        size_t slot_bytes = 0;
        int n_slots = 0, prob_cap = 0, n_layers = 0;
        int n_active = 0;               // slots in use as cache; [n_active, n_slots) lent out (lend)
        uint8_t* data = nullptr;
        std::vector<int> owner;         // slot -> expert idx (l * E + e), -1 if free
        std::vector<uint64_t> used;     // slot -> last tick
        std::vector<uint64_t> sseq;     // slot -> lazy batch seq (0: none)
        std::vector<uint8_t> qid;       // 0=free, 1=prob, 2=main
        std::vector<uint8_t> freq;      // saturating access counter [0..kMaxFreq]
        std::vector<int> prev, next;
        int prob_head = -1, prob_tail = -1, prob_sz = 0;
        int main_head = -1, main_tail = -1, main_sz = 0;
        int free_head = -1;
        uint64_t ghost_seq = 1;
    };

    int n_layers = 0, E = 0;
    std::vector<Pool> pools;
    std::vector<int> layer_pool;       // layer l -> pool index (-1 if on another GPU)
    std::vector<int> loc;              // l * E + e -> slot in pools[layer_pool[l]], or -1
    std::vector<uint64_t> ghost_at;    // l * E + e -> ghost_seq when evicted in decode (0: none)
    int last_handle = -1;
    bool scan = false;   // a long prefill's pass (evict_scan; IK2_SCAN=0: off)
    int tp = -1;         // IK2_TP: this GPU's half of every expert (its index), not whole experts
    uint64_t tick = 0, hits = 0, misses = 0, bytes_up = 0;
    double t_up = 0;

    static void push_back(Pool& P, int q, int s) {
        P.qid[s] = (uint8_t) q;
        P.prev[s] = -1;
        P.next[s] = -1;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (tl < 0) hd = tl = s;
        else P.next[tl] = s, P.prev[s] = tl, tl = s;
        ++sz;
    }
    static int pop_front(Pool& P, int q) {
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        const int s = hd;
        if (s < 0) return -1;
        hd = P.next[s];
        if (hd >= 0) P.prev[hd] = -1;
        else tl = -1;
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
        return s;
    }
    static int evict_from_prob(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        const int n = P.prob_sz;
        for (int i = 0; i < n; ++i) {
            const int s = pop_front(P, 1);
            if (P.used[s] >= t) {
                push_back(P, 1, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    push_back(P, 1, s);
                    continue;
                }
            }
            if (P.freq[s] > 0) {
                P.freq[s] = std::min<uint8_t>(P.freq[s] - 1, 1);
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    static int evict_from_main(Pool& P, uint64_t t, bool& saw_busy_lazy) {
        int pinned_run = 0;
        while (P.main_sz > 0 && pinned_run < P.main_sz) {
            const int s = pop_front(P, 2);
            if (P.used[s] >= t) {
                ++pinned_run;
                push_back(P, 2, s);
                continue;
            }
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) {
                    saw_busy_lazy = true;
                    ++pinned_run;
                    push_back(P, 2, s);
                    continue;
                }
            }
            pinned_run = 0;
            if (P.freq[s] > 0) {
                --P.freq[s];
                push_back(P, 2, s);
                continue;
            }
            return s;
        }
        return -1;
    }
    // A pass that streams nearly every expert of every layer in order (a long prefill), once the cache is full:
    // the victim is the most recently taken probation slot that is free to go (in practice an expert of the layer
    // just computed), not the oldest. Under the usual order such a pass, larger than the cache, evicts the experts of
    // the layers it is about to reach, and the next pass finds none of them (32k after other prompts: 9% hits);
    // like this, the streamed experts take turns in a few slots and the rest of the cache stays for the next pass.
    static int evict_scan(Pool& P, uint64_t t) {
        for (int s = P.prob_tail; s >= 0; s = P.prev[s]) {
            if (P.used[s] >= t) continue;
            if (P.sseq[s] > g_seq_done) {
                lazy_poll();
                if (P.sseq[s] > g_seq_done) continue;
            }
            unlink(P, s);
            return s;
        }
        return -1;
    }
    static int alloc_slot(Pool& P, uint64_t t, bool scan = false) {
        if (P.free_head >= 0) {
            const int s = P.free_head;
            P.free_head = P.next[s];
            P.next[s] = P.prev[s] = -1;
            return s;
        }
        if (scan) {
            const int s = evict_scan(P, t);
            if (s >= 0) return s;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool saw_busy_lazy = false;
            int s = -1;
            if (P.prob_sz > P.prob_cap || P.main_sz == 0) {
                s = evict_from_prob(P, t, saw_busy_lazy);
                if (s < 0) s = evict_from_main(P, t, saw_busy_lazy);
            } else {
                s = evict_from_main(P, t, saw_busy_lazy);
                if (s < 0) {
                    const int prev_main = P.main_sz;
                    s = evict_from_prob(P, t, saw_busy_lazy);
                    if (s < 0 && P.main_sz > prev_main) s = evict_from_main(P, t, saw_busy_lazy);
                }
            }
            if (s >= 0) return s;
            if (attempt == 0 && saw_busy_lazy) {
                CU(cudaStreamSynchronize(g_cs));
                g_seq_done = g_seq_next - 1;
                continue;
            }
            break;
        }
        die("expert cache: no slot free");
    }

    void init(const std::vector<Layer>& L, int dev, int n_layers_in, int E_in, size_t budget) {
        n_layers = n_layers_in;
        E = E_in;
        layer_pool.assign(n_layers, -1);
        loc.assign((size_t) n_layers * E, -1);
        ghost_at.assign((size_t) n_layers * E, 0);
        struct Group {
            size_t sb;
            std::vector<int> layers;
        };
        std::map<size_t, std::vector<int>> by_sb;
        for (int l = 0; l < n_layers; ++l) {
            if (dev >= 0 && L[l].dev != dev) continue;   // (dev -1: every layer, an expert-parallel GPU)
            const size_t eb = tp >= 0 ? tp_bytes(L[l], tp) : L[l].eb[0] + L[l].eb[1] + L[l].eb[2];
            const size_t sb = (eb + 255) / 256 * 256;
            by_sb[sb].push_back(l);
        }
        std::vector<Group> grps;
        for (auto& kv : by_sb) grps.push_back({kv.first, std::move(kv.second)});
        auto min_req_bytes = [&]() {
            size_t s = 0;
            for (const auto& g : grps) s += (size_t) std::min(E, (int) g.layers.size() * E) * g.sb;
            return s;
        };
        while (grps.size() > 1 && (min_req_bytes() > budget || grps.size() > 4)) {
            size_t best_waste = SIZE_MAX;
            size_t best_i = 0;
            for (size_t i = 0; i + 1 < grps.size(); ++i) {
                const size_t w = grps[i].layers.size() * (grps[i + 1].sb - grps[i].sb);
                if (w < best_waste) best_waste = w, best_i = i;
            }
            grps[best_i + 1].layers.insert(grps[best_i + 1].layers.end(), grps[best_i].layers.begin(),
                                           grps[best_i].layers.end());
            grps.erase(grps.begin() + (ptrdiff_t) best_i);
        }
        const size_t hard_min = min_req_bytes();
        if (hard_min > budget)
            die("--cache-gb too small: need at least " + std::to_string((hard_min + (1 << 20) - 1) >> 20) + " MiB");
        size_t soft_min = 0;
        for (const auto& g : grps) soft_min += (size_t) std::min(E + 16, (int) g.layers.size() * E) * g.sb;
        const int min_base = soft_min <= budget ? E + 16 : E;

        pools.resize(grps.size());
        std::vector<int> pmin(grps.size()), pmax(grps.size());
        for (size_t p = 0; p < grps.size(); ++p) {
            pools[p].slot_bytes = grps[p].sb;
            pools[p].n_layers = (int) grps[p].layers.size();
            for (int l : grps[p].layers) layer_pool[l] = (int) p;
            pmax[p] = pools[p].n_layers * E;
            pmin[p] = std::min(min_base, pmax[p]);
        }
        auto used_for = [&](double spl) {
            size_t u = 0;
            for (size_t p = 0; p < pools.size(); ++p) {
                const int ns = std::clamp((int) std::floor(pools[p].n_layers * spl), pmin[p], pmax[p]);
                u += (size_t) ns * pools[p].slot_bytes;
            }
            return u;
        };
        double lo = 0.0, hi = (double) E;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (used_for(mid) <= budget) lo = mid;
            else hi = mid;
        }
        size_t used = 0;
        for (size_t p = 0; p < pools.size(); ++p) {
            pools[p].n_slots = std::clamp((int) std::floor(pools[p].n_layers * lo), pmin[p], pmax[p]);
            used += (size_t) pools[p].n_slots * pools[p].slot_bytes;
        }
        size_t rem = budget - used;
        bool grew = true;
        while (grew) {
            grew = false;
            for (size_t p = 0; p < pools.size(); ++p) {
                if (pools[p].n_slots < pmax[p] && pools[p].slot_bytes <= rem) {
                    ++pools[p].n_slots;
                    rem -= pools[p].slot_bytes;
                    grew = true;
                }
            }
        }
        for (size_t p = 0; p < pools.size(); ++p) {
            Pool& P = pools[p];
            P.prob_cap = std::max(1, P.n_slots / 5);
            P.data = dalloc<uint8_t>((size_t) P.n_slots * P.slot_bytes);
            P.owner.assign(P.n_slots, -1);
            P.used.assign(P.n_slots, 0);
            P.sseq.assign(P.n_slots, 0);
            P.qid.assign(P.n_slots, 0);
            P.freq.assign(P.n_slots, 0);
            P.prev.assign(P.n_slots, -1);
            P.next.assign(P.n_slots, -1);
            for (int s = 0; s < P.n_slots - 1; ++s) P.next[s] = s + 1;
            P.free_head = P.n_slots > 0 ? 0 : -1;
            P.n_active = P.n_slots;
        }
    }
    int total_slots() const {
        int s = 0;
        for (const auto& P : pools) s += P.n_slots;
        return s;
    }
    size_t total_bytes() const {
        size_t b = 0;
        for (const auto& P : pools) b += (size_t) P.n_slots * P.slot_bytes;
        return b;
    }
    uint64_t& slot_sseq(int handle) { return pools[handle >> 20].sseq[handle & 0xfffff]; }
    // the slot of expert e of layer l, its blocks copied in if missing; pinned for this tick (not evicted in it)
    // st: the stream the copy is queued on; *missed (given): whether it was copied in now
    // copy false: the slot is taken but its blocks are the caller's to copy in (copy_in)
    uint8_t* get(const Layer& Ly, int l, int e, uint64_t t, cudaStream_t st = nullptr, bool* missed = nullptr,
                 bool copy = true, bool is_prefill = false) {
        if (!st) st = g_s;
        if (missed) *missed = false;
        const int idx = l * E + e;
        const int p = layer_pool[l];
        Pool& P = pools[p];
        if (loc[idx] >= 0) {
            const int s = loc[idx];
            ++hits;
            P.used[s] = t;
            if (P.freq[s] < kMaxFreq) ++P.freq[s];
            last_handle = (p << 20) | s;
            return P.data + (size_t) s * P.slot_bytes;
        }
        ++misses;
        if (missed) *missed = true;
        const int best = alloc_slot(P, t, scan);
        if (P.owner[best] >= 0) {
            const int old_idx = P.owner[best];
            loc[old_idx] = -1;
            if (!is_prefill) ghost_at[old_idx] = P.ghost_seq;
        }
        if (!is_prefill) ++P.ghost_seq;
        const bool is_ghost = !is_prefill && ghost_at[idx] != 0 && (P.ghost_seq - ghost_at[idx]) <= (uint64_t) P.n_slots;
        P.owner[best] = idx;
        P.used[best] = t;
        P.sseq[best] = 0;
        P.freq[best] = is_ghost ? 1 : 0;
        push_back(P, is_ghost ? 2 : 1, best);
        loc[idx] = best;
        last_handle = (p << 20) | best;
        uint8_t* dst = P.data + (size_t) best * P.slot_bytes;
        if (copy) copy_in(Ly, e, dst, st);
        return dst;
    }
    static void unlink(Pool& P, int s) {
        const int q = P.qid[s];
        if (!q) return;
        int& hd = q == 1 ? P.prob_head : P.main_head;
        int& tl = q == 1 ? P.prob_tail : P.main_tail;
        int& sz = q == 1 ? P.prob_sz : P.main_sz;
        if (P.prev[s] >= 0) P.next[P.prev[s]] = P.next[s];
        else hd = P.next[s];
        if (P.next[s] >= 0) P.prev[P.next[s]] = P.prev[s];
        else tl = P.prev[s];
        P.prev[s] = P.next[s] = -1;
        P.qid[s] = 0;
        --sz;
    }
    static void rebuild_free(Pool& P) {
        P.free_head = -1;
        for (int s = P.n_active - 1; s >= 0; --s)
            if (P.owner[s] < 0 && P.qid[s] == 0) P.next[s] = P.free_head, P.prev[s] = -1, P.free_head = s;
    }
    // Lends the last k cache slots of pool p as one block of scratch (the experts in them are dropped), for a long
    // prefill whose passes stream the experts anyway; give_back() makes them cache slots again. Nothing may be in
    // flight on those slots (the caller synchronizes first).
    uint8_t* lend(int p, int k) {
        Pool& P = pools[p];
        if (k <= 0 || k > P.n_active) return nullptr;
        const int s0 = P.n_active - k;
        for (int s = s0; s < P.n_active; ++s) {
            unlink(P, s);
            if (P.owner[s] >= 0) loc[P.owner[s]] = -1, P.owner[s] = -1;
            P.used[s] = 0, P.sseq[s] = 0, P.freq[s] = 0;
        }
        P.n_active = s0;
        rebuild_free(P);
        return P.data + (size_t) s0 * P.slot_bytes;
    }
    void give_back(int p) {
        Pool& P = pools[p];
        P.n_active = P.n_slots;
        rebuild_free(P);
    }
    bool has(int l, int e) const { return loc[l * E + e] >= 0; }
    void copy_in(const Layer& Ly, int e, uint8_t* dst, cudaStream_t st) {
        if (tp >= 0) {   // (gate's half, up's half, down's half, packed)
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                size_t o, b;
                tp_part(Ly, tp, r, o, b);
                CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r] + o, b, cudaMemcpyHostToDevice, st));
                off += b, bytes_up += b;
            }
            return;
        }
        if (Ly.host_contig) {
            CU(cudaMemcpyAsync(dst, Ly.host[0] + (size_t) e * Ly.slot_bytes, Ly.slot_bytes, cudaMemcpyHostToDevice, st));
            bytes_up += Ly.slot_bytes;
            return;
        }
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            CU(cudaMemcpyAsync(dst + off, Ly.host[r] + (size_t) e * Ly.stride[r], Ly.eb[r], cudaMemcpyHostToDevice, st));
            off += Ly.eb[r];
            bytes_up += Ly.eb[r];
        }
    }
};

// ---- the forward

struct Work {
    static constexpr int kPackRing = 4;
    static constexpr int kMissBatch = 24;
    int Tm, ctx, nk_max, T_sup = 0;
    size_t blk_ld = 0, lg_cap = 0;
    float *h, *res, *r, *a, *tmp, *xq, *xk, *xv, *q, *kk, *vv, *rr, *gate, *sink, *dyn, *kconv, *vconv, *o, *lg;
    float *qkv;         // fused projection staging (aliases gu32)
    float *blocks;      // [n_blk][blk_ld][C]
    uint16_t *x16, *xk16, *xv16, *xq16_sup = nullptr;
    uint16_t *kv_k16 = nullptr, *kv_v16 = nullptr;
    int kv_tmp_cap = 0;
    float* attn_lora_tmp = nullptr;
    int* sparse_pages = nullptr;
    int* pos;
    cudaEvent_t prefill_inputs_copied;
    float* rlog;
    int* route_d;       // contiguous [S] top (int) followed by [S] weights (float)
    // the experts
    uint16_t *ex16, *w16, *h16;
    float *gu32, *y32, *sg, *su, *wt;
    int *gidx, *inv;
    const uint8_t** blob;
    int* tok;
    int* batch_group_row = nullptr;
    bool batch_group_mma = false;
    uint8_t *pack_h[kPackRing] = {}, *pack_d[kPackRing] = {};   // ring-buffered token path tables
    uint16_t* tp_h = nullptr;   // IK2_TP: the whole h, [S][F] FP16 (each GPU fills its half, then the other's)
    uint8_t* tp_pk = nullptr;   // (the decode's tables of this layer on the device: canonical blobs at +1536)
    float *y_h[kPackRing] = {};                                 // ring-buffered CPU lane outputs [64][C]
    int pack_slot = 0;
    float *x_h = nullptr;                                       // CPU lane inputs [8][C], pinned
    float *emb_h = nullptr;                                     // pinned host staging
    int *pos_h = nullptr, *route_h = nullptr;
    int *gidx_h[2] = {}, *inv_h[2] = {};
    float *sg_h[2] = {}, *su_h[2] = {}, *wt_h[2] = {};
    int *argmax_d = nullptr, *argmax_h = nullptr;
    int head_rows = 16;
    std::vector<cudaEvent_t> miss_ev;
    // prefill: the next layer's experts copied in while this one runs (prefetch_layer)
    // the prefetches in flight, a ring by layer: their batches' events, expert -> its batch
    // (-1: not prefetched), the layer each slot is for
    static constexpr int kPfRing = 4;
    std::vector<cudaEvent_t> pf_evs[kPfRing];
    std::vector<int> pf_bs[kPfRing];
    int pf_layers[kPfRing] = {-1, -1, -1, -1};
    std::vector<int> miss_pos;   // expert -> its index among this layer's misses (-1: not one)
    std::vector<uint32_t> use;   // [layer * E + e] how many tokens each expert ran in prefills so far
    std::vector<uint64_t> use_tot;   // [layer] their sum
    float* wtr = nullptr;        // [S] the routing weight of each expert-sorted row
    float* wtr_h[2] = {nullptr, nullptr};
    // prefill attention: the relative bias of up to kRelbRows queries x heads, [rows * H][extent] (IK2_RELB=0: off)
    static constexpr int kRelbRows = 512;
    float* relb = nullptr;
    int relb_ext = 0;
    uint16_t* lg16 = nullptr;   // the prefill attention's probabilities (FP16, lg_cap of them; aliases gu32)
    size_t gu32_bytes = 0;   // (gu32's size: a lent scratch only replaces it when it needs more)
    int T_host = 0;          // tokens the pinned host staging (pos_h, route_h, gidx_h, ..) holds
    std::vector<int> cnt, start, fill, hit_exps, miss_exps;
    std::vector<uint8_t*> slot_ptrs;
    std::vector<uint32_t> seen_stamp, gi_stamp;
    std::vector<char> seen_late;
    std::vector<int> gi_idx;
    uint32_t stamp = 0;
    float* logits;
    // A looped core's passes ([Tm][C] each): the stream the core was first given; the bank of the pass before (the
    // blocks it added after its first, then its final stream); a layer's four messages (q's, k's, v's, the MoE's);
    // its three attention inputs before the messages.
    float *lp_pre = nullptr, *lp_bank = nullptr, *lp_msg = nullptr, *lp_in = nullptr;
    float* lp_msg_all = nullptr;   // a token's messages of a pass, every core layer's ([n_core][4][C]): one launch a pass
    float* lp_g = nullptr;   // the message gates' scratch [3][Tm][2 rank + 2]
    float *lp_a = nullptr, *lp_f = nullptr;   // ... of a chunk's: [3][Tm][rank] and [2][Tm][C]
    float* logits_h = nullptr;   // the resident token path's logits, pinned [n_vocab]
    int *cand_d = nullptr, *cand_h = nullptr, *cand_part = nullptr;   // ... its candidates (k::cand_topk's out), scratch
    // a short prompt's experts by groups, every expert in VRAM (forward): the tables, pinned and on the device -
    // [rg_groups] pointers | n [rg_groups] | tok, row [rg_groups][kGroupMax] | sg, su, wt, id [rg_slots]
    uint8_t *rg_h = nullptr, *rg_d = nullptr;
    int rg_slots = 0, rg_groups = 0;
    size_t rg_bytes = 0;
};

extern std::atomic<std::sig_atomic_t> g_cancel;
extern uint64_t g_cnt_hist[4];
extern uint64_t g_src_hist[4];
extern float* g_taps;
extern int g_taps_ld, g_final_tap;

struct Pictures {
    std::vector<float> rows;
    std::vector<int> row_at;   // a prompt position's row (-1: a token)
    int n = 0;
    void clear() { rows.clear(), row_at.clear(), n = 0; }
    int at(int pos) const { return n && pos < (int) row_at.size() ? row_at[pos] : -1; }
};

extern Pictures g_img;

// a GPU's own: its streams and cuBLAS handle, work buffers and expert cache; use_dev() makes it the current one (the
// globals g_s, g_cs, g_ev, g_blas the kernels and copies go to)
struct DevCtx {
    cudaStream_t s = nullptr, cs = nullptr;
    cudaEvent_t ev = nullptr;
    cublasHandle_t blas = nullptr;
    Work W{};
    ExpertCache EC;
};

extern std::vector<DevCtx> g_dev;
extern int g_cur;

float* dev_f32(const std::vector<float>& v);

#include "ik2_served_experts.cuh"

extern std::vector<float*> g_abl_r;
extern int g_abl_vis;
extern bool g_abl_decode;
extern bool g_abl_in_decode;
extern int g_abl_tokens;

constexpr int kAblRows = 256;

struct AblDecodeScope {   // marks a decode path for its duration
    bool was = g_abl_in_decode;
    AblDecodeScope() { g_abl_in_decode = true; }
    ~AblDecodeScope() { g_abl_in_decode = was; }
};

// (kCand: sampling's candidates - the resident token path finds the largest logits on the device, g_cand; any other
// path gives the logits as kLogits does and leaves g_cand.n at -1)
enum class OutMode { kNone, kArgmax, kLogits, kCand };
constexpr int kCandK = 1000, kCandCap = 4096;   // the largest logits sampling looks at; the candidates a token keeps
struct Cand {
    int n = -1;                    // candidates of the last kCand token (-1: its logits were given instead)
    const int* idx = nullptr;      // [n] token ids, in no order
    const float* val = nullptr;    // [n] their logits
};

extern Cand g_cand;
extern bool g_fast;
extern bool g_fast_want;
extern bool gpu_served;
extern bool g_sparse_decode;
extern bool g_graph;
extern bool g_block;
extern int g_attn_mma;
extern int g_mma_split;

struct FastGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    int attention_bucket = 0;
    bool sparse = false;
};

struct HostExpertPlan {
    ik2::cpu::TransferBuffers buffers;
    std::vector<std::shared_ptr<ik2::cpu::PackedLayer>> packed;
    std::vector<uint8_t> active;
    bool forced = false;
};
extern HostExpertPlan host_plan;
extern bool tier_active;
extern bool stage_cold;

constexpr int kBlockMax = 8;
struct BlockBufs {
    bool ready = false;
    float* emb_h = nullptr;      // pinned [kBlockMax][C]: the window's embedding rows
    int* pos_h = nullptr;        // pinned [kBlockMax]: its positions
    float* logits_h = nullptr;   // pinned [kBlockMax][n_vocab] (kLogits)
    const uint8_t** blob = nullptr;   // group_route's arrays, device
    int *n = nullptr, *tok = nullptr, *row = nullptr, *inv = nullptr;
    float *sg = nullptr, *su = nullptr, *wt = nullptr;
    float* lp_msg = nullptr;     // a pass's messages, [n_core][4][kBlockMax][C]
    k::ConvRec* recs = nullptr;  // every visit's four convs (rollback_block)
    int* zero_slots = nullptr;
    int n_recs = 0;
    int *cand_d = nullptr, *cand_h = nullptr;   // kCand: each row's k::cand_topk out, [kBlockMax][2 + 2 kCandCap]
};

extern BlockBufs g_blk;

struct SequenceRows {
    std::vector<int> slots, buckets;
    int* device_slots = nullptr;
    bool native_fp4 = false;
    int window = 0;   // positive for recorded speculative windows, including a one-row tail
};

struct SampleCfg {   // temp <= 0: greedy
    double temp = 0, top_p = 1;
    int top_k = 0;
    std::mt19937_64* rng = nullptr;
};

using ik2::draft::sample_from;

// When to draft and how wide (IK2_DRAFT_WIDTH the widest; IK2_DRAFT_ADAPT=0: always that wide): a step's expected
// tokens (1 - a^B) / (1 - a) - a the drafts' acceptance, each accepted given the one before it, from the recent
// steps - over its measured time, against a plain token's; a plain run drafts once in a while to look again.
struct SpecCtl {
    int wmax = 5;
    bool adapt = true;
    double c_plain = 7.8e-3, c[kBlockMax + 1] = {};
    double S = 1, F = 1;   // (the drafts accepted, the ones rejected: decayed counts)
    int plain_run = 0;
    int probe = getenv("IK2_DRAFT_PROBE") ? atoi(getenv("IK2_DRAFT_PROBE")) : 8;   // plain tokens before a look
    int probe_w = 4;
    double decay = getenv("IK2_DRAFT_DECAY") ? atof(getenv("IK2_DRAFT_DECAY")) : 0.85;
    uint64_t steps = 0, plain = 0, drafted = 0, accepted = 0;
    uint64_t hist[kBlockMax] = {};
    SpecCtl() {
#ifdef STRATA_USE_HIP
        const double prior[kBlockMax + 1] = {0, 8.4e-3, 13.8e-3, 16.0e-3, 17.8e-3, 19.2e-3, 21.2e-3, 23.2e-3, 25.0e-3};
        c_plain = 8.4e-3;
#else
        const double prior[kBlockMax + 1] = {0, 7.8e-3, 13.3e-3, 15.1e-3, 16.7e-3, 18.2e-3, 19.8e-3, 21.3e-3, 22.8e-3};
#endif
        for (int b = 0; b <= kBlockMax; ++b) c[b] = prior[b];
    }
    int choose(int room) {   // room: the widest a step may be (1: plain)
        const int bmax = std::min(wmax, room);
        if (bmax < 2) return 1;
        if (!adapt) return bmax;
        const double a = std::min(0.98, (S + 0.5) / (S + F + 1.0));
        int best = 1;
        double vb = 1.0 / c_plain;
        for (int b = 2; b <= bmax; ++b) {
            const double e = (1.0 - std::pow(a, b)) / (1.0 - a), v = e / c[b];
            if (v > vb) vb = v, best = b;
        }
        if (best == 1 && plain_run >= probe) best = std::min(probe_w, bmax);   // (a look at the acceptance again)
        return best;
    }
    void took_plain(double t) {
#ifdef STRATA_USE_HIP
        if (t < 0.04 || t < 2.5 * c_plain) c_plain = 0.9 * c_plain + 0.1 * t;
#else
        // Bound capture outliers while still learning costs above the initial GPU prior.
        c_plain = 0.9 * c_plain + 0.1 * std::min(t, std::max(0.04, 2.5 * c_plain));
#endif
        ++plain, ++plain_run;
    }
    void took_spec(int B, int acc, double t) {
#ifdef STRATA_USE_HIP
        if (t < 0.06 || t < 2.5 * c[B]) c[B] = 0.8 * c[B] + 0.2 * t;
#else
        c[B] = 0.8 * c[B] + 0.2 * std::min(t, std::max(0.06, 2.5 * c[B]));
#endif
        S = decay * S + acc, F = decay * F + (acc < B - 1 ? 1 : 0);
        ++steps, drafted += B - 1, accepted += acc, ++hist[acc];
        plain_run = 0;
    }
};

// Layer-major super-chunk prefill (single GPU, no dump, no drafter):
// A long prompt in fewer passes: the prefill scratch for a pass of T tokens (~256 KB a token) is lent by the expert
// cache (ExpertCache::lend), which a long prefill streams through anyway, and given back afterwards - so the passes
// can be as long as the cache allows without the decode's cache being smaller (IK2_BORROW=0: off).
struct BigScratch {
    int pool = -1, T_sup = 0;
    size_t blk_ld = 0;
    float *h = nullptr, *r = nullptr, *tmp = nullptr, *blocks = nullptr, *gu32 = nullptr, *qkv = nullptr,
          *y32 = nullptr, *sg = nullptr, *su = nullptr, *wt = nullptr, *wtr = nullptr, *rlog = nullptr;
    uint16_t *xq16_sup = nullptr, *ex16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *route_d = nullptr, *gidx = nullptr, *inv = nullptr;
    float* lg = nullptr;
    uint16_t* lg16 = nullptr;
    size_t lg_cap = 0;
};

constexpr int kPlacementInterval = 8192;
extern int suffix_window;

struct SuffixStats {
    int forwards = 0;
    int inputs = 0;
    int offered = 0;
    int accepted = 0;
};
using RowSampler = std::function<int(const float*)>;

unsigned long long* device_counts(int layer);
bool host_profile();
void pop_count(int l, const int* top, int S);
void gemm16(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta = 0.0f);
#ifdef STRATA_USE_HIP
bool fused_router(int experts, int picks, int rows);
#endif
bool byte_cache(bool sliding);
size_t kv_row_bytes(size_t elements, bool sliding);
void place_tails(bool host);
void add_attention_lora(const Layer& layer, const float* x, int tokens, float* low, float* out);
void unpack_segments(const float* src, int T, const std::initializer_list<std::pair<float*, int>>& segs, cudaStream_t s);
void mv1(const Mat& m, const float* x, float* y);
void mv1c(const Mat& m, const float* x, float* y, const k::MvConv& c0, const k::MvConv& c1 = k::MvConv{});
bool allow_short_batch_products();
void matmul(const Mat& m, const float* x32, const uint16_t* x16, int T, float* Y, int ldy);
int draft_tap(const Config& c, int l, int vis);
void conv_rec(Layer& Ly, int which, const float* x, const float* hist, int T, int Cc);
void place_pictures(float* h, int T, int pos0, int C, cudaStream_t s);
Work& dev_work(int d);
ExpertCache& dev_cache(int d);
void use_dev(int d);
void abl(float* y, int T, int C, int kind);

Mat mat_rows(const Mat& m, int r0, int n);
void fast_token(Model& M, int tok, int pos, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out, bool generation);
void fast_token_launches(Model& M, Work& W, OutMode out_mode, int attention_bucket, bool sparse = false);
int decode_bucket(int pos);
Visit sequence_visit(Visit v, const Layer& ly, const Config& c, int slot);
void fast_block_launches(Model& M, Work& W, int T, OutMode out_mode, BlockBufs& buf = g_blk,
                         const SequenceRows* rows = nullptr, int attn_bkt = 2);
void fast_block(Model& M, const int* tok, int T, int pos0, OutMode out_mode, std::vector<float>& logits_out,
                std::vector<int>* argmax_out);
bool block_cands(const int* candidates, int t, std::vector<std::pair<float, int>>& cd);
void block_rollback(int n);
std::vector<std::pair<int, double>> sample_dist(std::vector<std::pair<float, int>>& cd, const SampleCfg& sc);
ik2::draft::Draws draft_draws(double temperature, int count, std::mt19937_64& rng);
void spec_step(Model& M, ik2::draft::Drafter* drafter, int anchor, int P, int B, const SampleCfg& sc,
               const std::vector<int>& stops, std::vector<int>& next);
void forward(Model& M, const int* tok, int T, int pos0, bool all_logits, OutMode out_mode, const std::string& dump,
             std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp, bool generation = false);
int big_begin(const Model& M, Work& W, ExpertCache& EC, int n, int ctx_end, BigScratch& bs);
void big_end(Work& W, ExpertCache& EC, BigScratch& bs);
void forward_prefill_sup(Model& M, const int* tok, int T, int pos0, int sub_chunk, OutMode out_mode,
                         std::vector<float>& logits_out, std::vector<int>* argmax_out, double& t_exp);
void rollback(Model& M, int n);
void replace_expert_pools(Model& M, bool managed);
void place_decode(Model& M, int depth);
int verify_suffix(Model& M, const std::vector<int>& hist, int anchor, int remaining, const std::vector<int>& stops,
                  std::vector<int>& next, double& t_exp, SuffixStats& stats, const RowSampler& sample = {});
void embed_row(Model& M, int t, float* raw, float* out);

}  // namespace ik2::engine
