// src/ik2/ik2_main.cu - tintero: Inkling 2 (thinkingmachines/Inkling-2-Small, 594B / ~17B active) on one or two
// GPUs, from an `inkling2` GGUF in llama.cpp PR #32187's layout (docs/INKLING2.md; unsloth/Inkling-2-Small-GGUF's
// files: NVFP4, IQ3_XXS / IQ4_XS and Q*_K experts).
//
//   * The dense weights (attention, router, head) decoded once to FP16 in VRAM (the router in F32); cuBLAS' FP16 GEMMs
//     with F32 sums.  The attention (relative bias, sinks, sliding window, L2 clips) fused: an mma.sync kernel, and a
//     wgmma one on Hopper (sm_90a); the KV cache in FP16 (IK2_KV=fp8: E4M3).
//   * The routed experts as the GGUF holds them, all in pinned host memory (~323 GiB for UD-Q4_K_XL), and a cache of
//     them in VRAM (--cache-gb, at most what the GPU has room for).  A prompt goes through layer by layer in passes of
//     up to 16384 tokens, so each pass streams every expert it misses once, the next layer's prefetched while this one
//     runs; a scan-resistant eviction keeps a pass from flushing the cache.  IK2_WARM fills the cache at load with the
//     experts tokens pick most (a profile written with IK2_POP_OUT).
//   * Decode a token at a time; with IK2_CPU=1 a missing expert is computed on the CPU (AVX-512) when that is sooner
//     than its copy.  Two GPUs (--gpus 2): IK2_EP=1 deals each layer's experts to both (expert parallel - each with its
//     own cache and PCIe link), else the layers are split between them.
//
//   tintero --model FIRST-SHARD.gguf --serve [--max-context 262144] [--cache-gb 60] [--gpus 1] [--threads 8]
//            [--eos a,b] [--lora attn-output-adapter.gguf]
//   tintero --model FIRST-SHARD.gguf --tokens 1,2,3|@FILE [--gen N] [--chunk 2048] [--dump DIR]
// --serve: Maya's engine protocol on stdin / stdout (serve/server.py's StrataEngine: INFO / READY, GEN, PP, T, DONE,
// STOP); --serve-ik2 the older @-line one.  --dump: each layer's output (blk.L.out) and the log-probabilities of every
// prompt position, F32 raw files ([T][C], [T][n_vocab]) - what tools/ik2_compare.py holds against ref/inkling2.py's
// seams.  Settings (IK2_*): docs/SETTINGS.md.
#include "ik2_engine_internal.cuh"

namespace ik2::engine {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "tintero: %s\n", m.c_str());
    std::exit(1);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- the files

struct Shards {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::vector<std::string> paths;
    std::map<std::string, std::pair<int, const strata::TensorInfo*>> by_name;

    explicit Shards(const std::string& first) {
        std::vector<std::string> ps{first};
        const size_t at = first.find("-00001-of-");
        if (at != std::string::npos) {
            const int n = std::atoi(first.c_str() + at + 10);
            for (int i = 2; i <= n; ++i) {
                char num[16];
                std::snprintf(num, sizeof num, "%05d", i);
                std::string p = first;
                p.replace(at + 1, 5, num);
                ps.push_back(p);
            }
        }
        for (const auto& p : ps) {
            files.emplace_back(new strata::GgufFile(p));
            paths.push_back(p);
            for (const auto& t : files.back()->tensors()) by_name[t.name] = {(int) files.size() - 1, &t};
        }
    }
    const strata::TensorInfo* info(const std::string& n) const {
        auto it = by_name.find(n);
        return it == by_name.end() ? nullptr : it->second.second;
    }
    const uint8_t* data(const std::string& n) const {
        auto it = by_name.find(n);
        if (it == by_name.end()) die("no tensor " + n);
        return files[it->second.first]->tensor_data(*it->second.second);
    }
    uint64_t file_offset(const std::string& n) const {
        auto it = by_name.find(n);
        return files[it->second.first]->data_start() + it->second.second->offset;
    }
    int file_of(const std::string& n) const { return by_name.at(n).first; }
    const strata::MetaValue* get(const std::string& key) const { return files[0]->get(key); }
};

size_t tensor_bytes(const strata::TensorInfo& t) {
    // (the blocks decoded here first: strata's block_geometry names NVFP4 but has no geometry for it)
    if (k::block_values((int) t.type)) return (size_t) (t.elements() / k::block_values((int) t.type)) * k::block_bytes((int) t.type);
    int e = 0, b = 0;
    if (!strata::block_geometry(t.type, e, b)) die(std::string("type ") + t.type_name() + " of " + t.name);
    return (size_t) (t.elements() / e) * b;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- the device helpers

cublasHandle_t g_blas;
cudaStream_t g_s;
cudaStream_t g_cs;          // the experts' copies of the token path, beside the compute stream
cudaEvent_t g_ev;
bool g_overlap = true;      // IK2_OVERLAP=0: the copies on the compute stream, before the experts
// The CPU lane (IK2_CPU=1: on): a missing NVFP4 expert computed in host memory by the pool while the others' blocks
// are copied; its own blocks follow on the copy stream without being waited for ("lazy"), so it is in VRAM next time.
ik2::cpu::Pool* g_pool = nullptr;
double g_t_cpu = 0.5e-3, g_t_up = 0.36e-3;   // seconds an expert: the pool's (measured as it runs), the link's
uint64_t g_n_cpu = 0, g_n_wait = 0;
bool g_cpu_lazy = true;   // IK2_CPU_LAZY=0: a CPU-computed expert is not copied in afterwards (the link carries less)
cudaEvent_t g_ring[kRing];                   // a lazy batch's end on the copy stream, by sequence number
uint64_t g_seq_next = 1, g_seq_done = 0;
void lazy_poll() {
    while (g_seq_done + 1 < g_seq_next && cudaEventQuery(g_ring[(g_seq_done + 1) % kRing]) == cudaSuccess) ++g_seq_done;
}

// IK2_PROF=1: the stream waited for at each phase's end, its time added to the phase (slower, but where it goes)
bool g_prof = false;
bool g_mv = true;   // IK2_MV=0: a few tokens' experts through the FP16 GEMMs too
// How often each expert is picked by a token (prefill and decode), [n_layer * E]: IK2_POP_OUT=FILE writes it at
// the end (int32 n_layer, int32 E, then uint64 counts); IK2_WARM=FILE fills the expert cache with the most picked
// experts at load, so the first requests do not start from an empty cache.
std::vector<uint64_t> g_pop;
int g_pop_E = 0;
std::filesystem::path g_profile_out;
// Resident graphs accumulate session deltas without a host read at each visit.
unsigned long long* g_pop_device = nullptr;
unsigned long long* device_counts(int layer) {
    return g_pop_device ? g_pop_device + size_t(layer) * g_pop_E : nullptr;
}
bool host_profile() { return !g_pop.empty() && !g_pop_device; }
void add_picks(size_t index, uint64_t picks) {
    auto& count = g_pop[index];
    while (picks > UINT64_MAX - count) {
        picks -= UINT64_MAX - count;
        count = UINT64_MAX;
        for (auto& value : g_pop) value >>= 1;
    }
    count += picks;
}
void pop_count(int l, const int* top, int S) {
    if (g_pop.empty()) return;
    for (int i = 0; i < S; ++i) add_picks(size_t(l) * g_pop_E + top[i], 1);
}
void pop_dump() {
    if (g_pop_device) {
        std::vector<unsigned long long> delta(g_pop.size());
        CU(cudaMemcpy(delta.data(), g_pop_device, delta.size() * sizeof(delta[0]), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < delta.size(); ++i) add_picks(i, delta[i]);
    }
    ik2::usage::write(g_profile_out, g_pop_E, g_pop);
}
int g_pop_E_ep = 0;
// IK2_EP=1 with --gpus 2: expert parallel. Every layer's attention and dense weights (and the KV cache) on GPU 0;
// expert e of every layer belongs to GPU e & 1, each GPU with its own cache over all layers and its own PCIe link.
// In decode a layer's picks go to their GPUs at once, both links copying the misses side by side.
bool g_ep = false;
cudaEvent_t g_ep_x, g_ep_cp[2], g_ep_done;
// IK2_TP=1 with --gpus 2: the experts tensor-parallel instead. GPU d holds half of every expert: rows [d F/2, ..) of
// gate and up and rows [d C/2, ..) of down, so every miss is half on each link. A layer: each GPU its half of h
// (gate/up/SwiGLU), the halves swapped (P2P), each GPU its half of the output rows (down), concatenated on GPU 0.
// Attention, dense weights and the KV cache on GPU 0 as with IK2_EP.
bool g_tp = false;
cudaEvent_t g_tp_h[2];
int g_tp_F = 0, g_tp_C = 0;
// which GPU an expert belongs to, [n_layer * E]: IK2_EP_SPLIT = GPU 1's share of the picks (0.5; GPU 0 also holds the
// attention, the dense weights and the KV cache). With IK2_WARM's profile each layer's experts are dealt by how often
// they are picked, so the two GPUs' shares come out as asked; without it by a hash of the expert.
std::vector<uint8_t> g_ep_own;
bool g_flash = true;   // IK2_FLASH=0: the prefill's attention (FP16 KV) through the logits in memory and cuBLAS, not fused
int g_n_sm = 0;
bool g_group = true;   // IK2_GROUP=0: 2 .. 8 tokens' experts a slot at a time, their attention a head at a time
bool g_group_exp = true, g_group_attn = true;   // (IK2_GROUP=1: the experts grouped only, 2: the attention only)
FILE* g_trace = nullptr;   // IK2_TRACE=FILE: each layer's picks, records of int32 (layer, T, K, T x K experts)

Prof g_pf;
size_t device_allocated = 0;  // Single-GPU allocations charged against an explicit VRAM budget.
// Captured graphs point at the active cache. Exchange its live rows with another
// slot between requests, leaving every recorded device address valid.
struct SlotStorage {
    struct Region {
        uint8_t* active;
        size_t rows, stride, page;
        size_t bytes(size_t length) const {
            return std::min(rows, (length + page - 1) / page * page) * stride;
        }
    };
    std::vector<Region> regions;
    std::vector<uint8_t*> parked;
    uint8_t* scratch = nullptr;
    static constexpr size_t tile = 4 << 20;

    void add(void* ptr, size_t rows, size_t stride, size_t page = 1) {
        regions.push_back({static_cast<uint8_t*>(ptr), (rows + page - 1) / page * page, stride, page});
    }
    void allocate(int slots) {
        size_t bytes = 0;
        for (const Region& r : regions) bytes += r.rows * r.stride;
        parked.resize(slots);
        for (int i = 1; i < slots; ++i) parked[i] = dalloc<uint8_t>(bytes);
        scratch = dalloc<uint8_t>(tile);
        std::fprintf(stderr, "conversation slots: %d, %.1f MiB per cache, %.1f MiB for exchanges\n",
                     slots, bytes / 1048576., tile / 1048576.);
    }
    void exchange(int from, int to, size_t outgoing, size_t incoming) {
        uint8_t* stored = parked[to];
        auto copy = [](uint8_t* dst, const uint8_t* src, size_t n) {
            CU(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, g_s));
        };
        for (const Region& r : regions) {
            const size_t a = r.bytes(outgoing), b = r.bytes(incoming), both = std::min(a, b);
            for (size_t i = 0; i < both; i += tile) {
                const size_t n = std::min(tile, both - i);
                copy(scratch, r.active + i, n);
                copy(r.active + i, stored + i, n);
                copy(stored + i, scratch, n);
            }
            if (a > b) copy(stored + b, r.active + b, a - b);
            if (b > a) copy(r.active + a, stored + a, b - a);
            // A fresh conversation has zero convolution history. KV rows are written before use.
            if (!incoming && r.rows == 1) CU(cudaMemsetAsync(r.active, 0, r.stride, g_s));
            stored += r.rows * r.stride;
        }
        CU(cudaStreamSynchronize(g_s));
        std::swap(parked[from], parked[to]);
    }
    // Fork the active history in place. The victim's backing allocation now holds the original.
    void preserve(int from, int to, size_t length) {
        uint8_t* dst = parked[to];
        for (const Region& r : regions) {
            const size_t n = r.bytes(length);
            if (n) CU(cudaMemcpyAsync(dst, r.active, n, cudaMemcpyDeviceToDevice, g_s));
            dst += r.rows * r.stride;
        }
        CU(cudaStreamSynchronize(g_s));
        std::swap(parked[from], parked[to]);
    }
};

// Y [T][N] (ldy) = X [T][K] FP16 . W [N][K]^T FP16, F32 sums
void gemm16(const uint16_t* W, int N, int Kd, const uint16_t* X, int T, float* Y, int ldy, float beta) {
#ifdef STRATA_USE_HIP
    static const bool wmma = getenv("IK2_DENSE_WMMA") && atoi(getenv("IK2_DENSE_WMMA"));
    if (wmma && beta == 0.f && T >= 32 && N >= 128 && Kd % 32 == 0) {
        k::rdna3_dense_gemm(W, N, Kd, X, T, Y, ldy, g_s);
        return;
    }
#endif
    const float one = 1.0f;
    CB(cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, Kd, &one, W, CUDA_R_16F, Kd, X, CUDA_R_16F, Kd, &beta, Y,
                    CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

uint8_t* g_stage = nullptr;   // a raw tensor on its way to its FP16 decode
size_t g_stage_bytes = 0;

bool g_dense_q8 = false;
// The KV cache in FP16 (IK2_KV=fp8: E4M3, half the bytes).  Against ref/inkling2.py over 48 tokens FP16 kept every
// top-1 token (KL mean 8.5e-4) and FP8 changed 5 (KL 8.3e-3): Inkling's router is full of near-ties, so FP8's 1.5% in
// layer 0's attention output already moves 7 of 48 tokens' experts.  The full layers' cache is 32 KB a token in FP16.
#ifdef STRATA_USE_HIP
bool fused_router(int experts, int picks, int rows) {
    static const bool enabled = getenv("IK2_FUSED_ROUTER") && atoi(getenv("IK2_FUSED_ROUTER"));
    // One workgroup per row under-utilizes wide RDNA3 GPUs on single-token decode.
    return enabled && experts == 128 && picks == 8 && rows >= 16 && rows <= 64;
}
#endif
KVPrecision g_kv_precision = KVPrecision::f16;
bool packed_cache(bool sliding) {
    return g_kv_precision == KVPrecision::fp4 || (!sliding && g_kv_precision == KVPrecision::fp4_global);
}
bool byte_cache(bool sliding) {
    return g_kv_precision == KVPrecision::fp4 || g_kv_precision == KVPrecision::fp4_global || g_kv_precision == KVPrecision::fp8 ||
           (!sliding && g_kv_precision == KVPrecision::fp8_global);
}
size_t kv_row_bytes(size_t elements, bool sliding) {
    if (packed_cache(sliding))
        return elements / 32 * sizeof(k::Fp4KvBlock);
    return elements * (byte_cache(sliding) ? 1 : sizeof(uint16_t));
}
const char* kv_format_name() {
    switch (g_kv_precision) {
        case KVPrecision::f16: return "f16";
        case KVPrecision::fp8: return "fp8";
        case KVPrecision::fp4: return "fp4";
        case KVPrecision::fp4_global: return "fp4-global";
        case KVPrecision::fp8_global: return "fp8-global";
    }
    std::abort();
}
bool g_kv_managed = false;
bool g_managed_experts = false;
size_t g_kv_room = 0;

std::vector<PrefillTail> prefill_tails;
size_t parked_bytes = 0;
bool tails_parked = false;

void place_tails(bool host) {
    if (prefill_tails.empty() || tails_parked == host) return;
    CU(cudaStreamSynchronize(g_s));
    for (auto tail : prefill_tails)
        CU(cudaMemPrefetchAsync(tail.begin, tail.bytes, host ? cudaCpuDeviceId : 0, g_s));
    CU(cudaStreamSynchronize(g_s));
    tails_parked = host;
    std::fprintf(stderr, "scratch tails: %.3f GiB moved to %s\n", double(parked_bytes) / (1ULL << 30),
                 host ? "host" : "GPU");
}
template <class T> T* kv_alloc(size_t n, bool sliding) {
    if (sliding || !g_kv_managed) return dalloc<T>(n);
    T* ptr = nullptr;
    CU(cudaMallocManaged(&ptr, n * sizeof(T)));
    return ptr;
}
// Q8_0 blocks (2 bytes of FP16 scale, 32 int8 values) taken apart into m.q8 / m.q8d on the device; false if the
// matrix cannot be kept so (another type, or K not a multiple of 128)
bool upload_planar(Mat& m, const std::vector<const uint8_t*>& parts, const std::vector<size_t>& part_values) {
    if (m.type != k::kQ8_0 || m.K % 128) return false;
    size_t n = 0;
    for (size_t v : part_values) n += v;
    std::vector<int8_t> q(n);
    std::vector<uint16_t> d(n / 32);
    size_t b = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        const uint8_t* p = parts[i];
        for (size_t j = 0; j < part_values[i] / 32; ++j, ++b, p += 34) {
            std::memcpy(&d[b], p, 2);
            std::memcpy(&q[b * 32], p + 2, 32);
        }
    }
    m.q8 = dalloc<int8_t>(n);
    m.q8d = dalloc<uint16_t>(n / 32);
    CU(cudaMemcpy(m.q8, q.data(), n, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(m.q8d, d.data(), n / 32 * 2, cudaMemcpyHostToDevice));
    return true;
}

Mat load_mat(const Shards& S, const std::string& n, int N, int Kd) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != (uint64_t) N * Kd) die(n + ": " + std::to_string(t->elements()) + " values");
    if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
        die(n + ": type " + t->type_name() + " not read here");
    Mat m;
    m.type = (int) t->type, m.N = N, m.K = Kd;
    const size_t nb = tensor_bytes(*t);
    if (g_dense_q8) {
        if (upload_planar(m, {S.data(n)}, {(size_t) N * Kd})) return m;
        m.raw = dalloc<uint8_t>(nb);
        CU(cudaMemcpy(m.raw, S.data(n), nb, cudaMemcpyHostToDevice));
        return m;
    }
    if (nb > g_stage_bytes) {
        if (g_stage) CU(cudaFree(g_stage));
        g_stage_bytes = nb;
        CU(cudaMalloc(&g_stage, nb));
    }
    CU(cudaMemcpy(g_stage, S.data(n), nb, cudaMemcpyHostToDevice));
    m.f16 = dalloc<uint16_t>((size_t) N * Kd);
    k::decode_f16(m.type, g_stage, (size_t) N * Kd, m.f16, g_s);
    CU(cudaStreamSynchronize(g_s));
    return m;
}

void add_attention_lora(const Layer& layer, const float* x, int tokens, float* low, float* out) {
    const AttentionLora& lora = layer.attn_lora;
    const int in = layer.wo.K, n = layer.wo.N, rank = lora.rank;
    const float one = 1, zero = 0;
    CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, rank, tokens, in,
                   &one, lora.a, in, x, in, &zero, low, rank));
    CB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, tokens, rank,
                   &lora.scale, lora.b, rank, low, rank, &one, out, n));
}

struct SegPack {
    float* dst[6];
    int off[7];
    int n;
};
// (float4 a thread when every width is a multiple of 4 - the segments' lookup unrolled, so the parameters stay in
// the constant bank instead of being copied to local memory for a dynamic index)
__global__ void unpack_segments4_kernel(const float4* __restrict__ src, int T, SegPack p) {
    const int c4 = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int stride4 = p.off[p.n] >> 2;
    if (c4 >= stride4 || t >= T) return;
    const float4 v = src[(size_t) t * stride4 + c4];
    const int col = c4 << 2;
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            *(float4*) (p.dst[s] + (size_t) t * w + (col - p.off[s])) = v;
        }
    }
}
__global__ void unpack_segments_kernel(const float* __restrict__ src, int T, SegPack p) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int stride = p.off[p.n];
    if (col >= stride || t >= T) return;
    const float v = src[(size_t) t * stride + col];
#pragma unroll
    for (int s = 0; s < 6; ++s) {
        if (s < p.n && col >= p.off[s] && col < p.off[s + 1]) {
            const int w = p.off[s + 1] - p.off[s];
            p.dst[s][(size_t) t * w + (col - p.off[s])] = v;
        }
    }
}
void unpack_segments(const float* src, int T, const std::initializer_list<std::pair<float*, int>>& segs, cudaStream_t s) {
    SegPack p{};
    p.n = (int) segs.size();
    int off = 0, idx = 0;
    bool vec = ((uintptr_t) src & 15) == 0;
    p.off[0] = 0;
    for (const auto& sg : segs) {
        p.dst[idx] = sg.first;
        vec = vec && (sg.second & 3) == 0 && ((uintptr_t) sg.first & 15) == 0;
        off += sg.second;
        p.off[++idx] = off;
    }
    if (vec) {
        const dim3 grid((unsigned) ((off / 4 + 255) / 256), (unsigned) T);
        unpack_segments4_kernel<<<grid, 256, 0, s>>>((const float4*) src, T, p);
    } else {
        const dim3 grid((unsigned) ((off + 255) / 256), (unsigned) T);
        unpack_segments_kernel<<<grid, 256, 0, s>>>(src, T, p);
    }
}

struct MatPart {
    std::string name;
    int N = 0;
    std::vector<float> f32;
};

void ensure_stage(size_t nb) {
    if (nb > g_stage_bytes) {
        if (g_stage) CU(cudaFree(g_stage));
        g_stage_bytes = nb;
        CU(cudaMalloc(&g_stage, nb));
    }
}

Mat load_fused_f16(const Shards& S, const std::vector<MatPart>& parts, int Kd) {
    int N_tot = 0;
    for (const auto& pt : parts) N_tot += pt.N;
    Mat m;
    m.N = N_tot;
    m.K = Kd;
    m.f16 = dalloc<uint16_t>((size_t) N_tot * Kd);
    int row = 0;
    for (const auto& pt : parts) {
        if (pt.f32.empty()) {
            const auto* t = S.info(pt.name);
            if (!t) die("no tensor " + pt.name);
            if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
            if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
                die(pt.name + ": type " + t->type_name() + " not read here");
            const size_t nb = tensor_bytes(*t);
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, S.data(pt.name), nb, cudaMemcpyHostToDevice));
            k::decode_f16((int) t->type, g_stage, (size_t) pt.N * Kd, m.f16 + (size_t) row * Kd, g_s);
        } else {
            const size_t nb = pt.f32.size() * 4;
            ensure_stage(nb);
            CU(cudaMemcpy(g_stage, pt.f32.data(), nb, cudaMemcpyHostToDevice));
            k::to_f16((const float*) g_stage, pt.f32.size(), m.f16 + (size_t) row * Kd, g_s);
        }
        row += pt.N;
    }
    CU(cudaStreamSynchronize(g_s));
    return m;
}

bool try_load_fused_raw(const Shards& S, const std::vector<MatPart>& parts, int Kd, Mat& out) {
    int ty = -1, N_tot = 0;
    size_t nb_tot = 0;
    for (const auto& pt : parts) {
        if (!pt.f32.empty()) return false;
        const auto* t = S.info(pt.name);
        if (!t) die("no tensor " + pt.name);
        if (t->elements() != (uint64_t) pt.N * Kd) die(pt.name + ": " + std::to_string(t->elements()) + " values");
        if (!k::block_values((int) t->type) || Kd % k::block_values((int) t->type))
            die(pt.name + ": type " + t->type_name() + " not read here");
        if (ty < 0) ty = (int) t->type;
        else if (ty != (int) t->type) return false;
        N_tot += pt.N;
        nb_tot += tensor_bytes(*t);
    }
    out = Mat{};
    out.type = ty;
    out.N = N_tot;
    out.K = Kd;
    {
        std::vector<const uint8_t*> ps;
        std::vector<size_t> vs;
        for (const auto& pt : parts) ps.push_back(S.data(pt.name)), vs.push_back((size_t) pt.N * Kd);
        if (upload_planar(out, ps, vs)) return true;
    }
    out.raw = dalloc<uint8_t>(nb_tot);
    size_t off = 0;
    for (const auto& pt : parts) {
        const size_t nb = tensor_bytes(*S.info(pt.name));
        CU(cudaMemcpy(out.raw + off, S.data(pt.name), nb, cudaMemcpyHostToDevice));
        off += nb;
    }
    return true;
}

// one token's product over a dense matrix kept as the GGUF's blocks, or planar
void mv1(const Mat& m, const float* x, float* y) {
    if (m.q8) k::mv_q8p(m.q8, m.q8d, m.N, m.K, x, y, g_s);
    else k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
}
// ... with up to two short convs on its rows: the planar Q8 product has them as its epilogue (a launch less each;
// those rows of y are then not written), any other matrix gets the product and then the convs
void mv1c(const Mat& m, const float* x, float* y, const k::MvConv& c0, const k::MvConv& c1) {
    if (m.q8) {
        k::mv_q8p_conv(m.q8, m.q8d, m.N, m.K, x, y, c0, c1, g_s);
        return;
    }
    k::mv_blocks(m.type, m.raw, m.N, m.K, x, y, g_s);
    for (const k::MvConv* cv : {&c0, &c1})
        if (cv->n) k::conv4_res(y + cv->r0, cv->hist, cv->w, cv->dyn, cv->heads, 1, cv->n, cv->res, cv->y, g_s);
}

bool g_rec = false;
bool allow_short_batch_products() {
    static const bool verify_only = getenv("IK2_VERIFY_ONLY") && atoi(getenv("IK2_VERIFY_ONLY"));
    return !verify_only || g_rec;
}
uint16_t* g_w16 = nullptr;   // a matrix's FP16 decode for the prompt path's GEMMs (the largest of a layer's)
size_t g_w16_n = 0;          // values it holds
// Y [T][N] = X [T][K] . W^T: one token straight over the blocks (x32), more through an FP16 decode and a GEMM (x16).
// A matrix larger than the decode buffer (the head: its few rows of logits) goes a token at a time over its blocks.
void matmul(const Mat& m, const float* x32, const uint16_t* x16, int T, float* Y, int ldy) {
#ifdef STRATA_USE_HIP
    static const bool planar = getenv("IK2_DENSE_WMMA") && atoi(getenv("IK2_DENSE_WMMA"));
    static const bool selective = getenv("IK2_PREFILL_DENSE") && atoi(getenv("IK2_PREFILL_DENSE")) == 2;
    if (planar && m.q8 && T >= 32 && m.N >= 128 && m.K % 32 == 0 && (!m.f16 || (selective && T < 512))) {
        k::rdna3_planar_gemm(m.q8, m.q8d, m.N, m.K, x16, T, Y, ldy, g_s);
        return;
    }
#endif
    if (m.f16 && (!m.q8 || T > 1)) {
        gemm16(m.f16, m.N, m.K, x16, T, Y, ldy);
        return;
    }
    static const bool use_q8_tile = getenv("IK2_Q8_TILE") && atoi(getenv("IK2_Q8_TILE"));
    if (use_q8_tile && allow_short_batch_products() && m.q8 && T >= 2 && T <= 16 && m.K % 32 == 0) {
        k::q8_tile(m.q8, m.q8d, m.N, m.K, x16, T, Y, ldy, g_s);
        return;
    }
    if (m.q8 && T >= 4 && (size_t)m.N * m.K > g_w16_n && g_w16_n >= (size_t)m.K) {
        const int rows = (int)(g_w16_n / m.K);
        for (int r = 0; r < m.N; r += rows) {
            const int n = std::min(rows, m.N - r);
            k::decode_q8p_f16(m.q8 + (size_t)r * m.K, m.q8d + (size_t)r * (m.K / 32), (size_t)n * m.K, g_w16, g_s);
            gemm16(g_w16, n, m.K, x16, T, Y + r, ldy);
        }
        return;
    }
    if (m.K <= 8192 && (T == 1 || (size_t) m.N * m.K > g_w16_n)) {
        for (int t = 0; t < T; ++t) mv1(m, x32 + (size_t) t * m.K, Y + (size_t) t * ldy);
        return;
    }
    if (m.q8) k::decode_q8p_f16(m.q8, m.q8d, (size_t) m.N * m.K, g_w16, g_s);
    else k::decode_f16(m.type, m.raw, (size_t) m.N * m.K, g_w16, g_s);
    gemm16(g_w16, m.N, m.K, x16, T, Y, ldy);
}

std::vector<float> host_f32(const Shards& S, const std::string& n, size_t expect) {
    const auto* t = S.info(n);
    if (!t) die("no tensor " + n);
    if (t->elements() != expect) die(n + ": " + std::to_string(t->elements()) + " values, expected " + std::to_string(expect));
    std::vector<float> v(expect);
    const uint8_t* p = S.data(n);
    if (t->type == 0) {
        std::memcpy(v.data(), p, expect * 4);
    } else if (t->type == 30) {
        for (size_t i = 0; i < expect; ++i) v[i] = bf16_to_f32((uint16_t) (p[2 * i] | (p[2 * i + 1] << 8)));
    } else if (k::block_values((int) t->type)) {
        k::decode_host((int) t->type, p, expect, v.data());
    } else {
        die(n + ": type " + t->type_name());
    }
    return v;
}
float* dev_f32(const std::vector<float>& v) {
    float* p = dalloc<float>(v.size());
    CU(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}
uint16_t* dev_f16(const std::vector<float>& v) {
    const size_t nb = v.size() * 4;
    ensure_stage(nb);
    CU(cudaMemcpy(g_stage, v.data(), nb, cudaMemcpyHostToDevice));
    uint16_t* p = dalloc<uint16_t>(v.size());
    k::to_f16((const float*) g_stage, v.size(), p, g_s);
    CU(cudaStreamSynchronize(g_s));
    return p;
}
// The GGUF can store expanded BF16 in nominal F32 tensors. Only build the
// compact decode copy when every discarded bit is zero.
uint16_t* exact_bf16(const std::vector<float>& values) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t word;
        std::memcpy(&word, &values[i], sizeof(word));
        if (word & 0xffffu) return nullptr;
        bits[i] = word >> 16;
    }
    uint16_t* device = dalloc<uint16_t>(bits.size());
    CU(cudaMemcpy(device, bits.data(), bits.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return device;
}

float* load_f32(const Shards& S, const std::string& n, size_t expect) { return dev_f32(host_f32(S, n, expect)); }
uint16_t* load_f16(const Shards& S, const std::string& n, size_t expect) { return dev_f16(host_f32(S, n, expect)); }

// ---- loading the experts into pinned host memory: pread from the files, several at once

struct Pinned {
    struct Block {
        void* ptr = nullptr;
        size_t bytes = 0;
        bool mapped = false;
        bool registered = false;
    };
    std::vector<Block> blocks;
    ~Pinned() {
        for (const auto& b : blocks) {
            if (b.mapped) {
                if (b.registered) CU(cudaHostUnregister(b.ptr));
#ifndef _WIN32
                ::munmap(b.ptr, b.bytes);
#endif
            } else if (b.ptr) {
                CU(cudaFreeHost(b.ptr));
            }
        }
    }
};

void read_into_strided(const std::string& path, uint64_t off, int n_exp, size_t exp_bytes, size_t dst_stride,
                       uint8_t* dst) {
#ifdef _WIN32
    std::ifstream input(path, std::ios::binary);
    if (!input.seekg(off)) die("cannot seek in " + path);
    for (int e = 0; e < n_exp; ++e)
        if (!input.read(reinterpret_cast<char*>(dst + size_t(e) * dst_stride), exp_bytes))
            die("read failed: " + path);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open " + path);
    const size_t chunk = 64ull << 20;
    for (int e = 0; e < n_exp; ++e) {
        size_t done = 0;
        uint8_t* ep = dst + (size_t) e * dst_stride;
        const uint64_t eoff = off + (uint64_t) e * exp_bytes;
        while (done < exp_bytes) {
            const ssize_t r = ::pread(fd, ep + done, std::min(chunk, exp_bytes - done), (off_t) (eoff + done));
            if (r <= 0) die("read failed: " + path);
            done += (size_t) r;
        }
    }
    posix_fadvise(fd, (off_t) off, (off_t) ((uint64_t) n_exp * exp_bytes), POSIX_FADV_DONTNEED);
    ::close(fd);
#endif
}

// --serve: SIGUSR1 asks the request running to stop (its client went away) at its next token or prompt chunk
static_assert(std::atomic<std::sig_atomic_t>::is_always_lock_free);
std::atomic<std::sig_atomic_t> g_cancel{0};
// IK2_STATS=1: per request, how many tokens the prefill's experts each ran ([0] 1, [1] 2, [2] 3-7, [3] 8+)
uint64_t g_cnt_hist[4] = {0, 0, 0, 0};
// IK2_STATS: a request's experts by where they came from: used and already cached / used and prefetched for this
// pass / used and copied after the routing / prefetched and not used
uint64_t g_src_hist[4] = {0, 0, 0, 0};
#ifndef _WIN32
extern "C" void on_usr1(int) { g_cancel = 1; }
#endif

// Speculative decoding: g_taps [8][g_taps_ld][C] gets the drafter's taps of every position a forward call runs (the
// stream at the start of layers 5, 11, .., 41, the final normed hidden); with g_rec each short conv's history and
// inputs are kept, so that a verify window's rejected part can be taken back (rollback()).
float* g_taps = nullptr;
int g_taps_ld = 0, g_final_tap = 7;   // (the final normed hidden: tap 7, a looped model's 3)
// the drafter's tap a visit's stream (at its start) is, or -1
int draft_tap(const Config& c, int l, int vis) {
    if (c.n_pass == 1) return (l % 6 == 5 && l < c.n_layer - 1) ? l / 6 : -1;
    return vis == 5 ? 0 : vis == 23 ? 1 : vis == 41 ? 2 : -1;
}
void conv_rec(Layer& Ly, int which, const float* x, const float* hist, int T, int Cc) {
    if (!g_rec) return;
    CU(cudaMemcpyAsync(Ly.rec_old[which], hist, (size_t) 3 * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
    CU(cudaMemcpyAsync(Ly.rec_x[which], x, (size_t) T * Cc * 4, cudaMemcpyDeviceToDevice, g_s));
}

// The pictures of the request running (GENI): the image encoder's rows (maya-vision's SVE1 records, [n][C] F32), each
// in place of one image token of the prompt - the model reads them as they are, without the embedding's norm.
Pictures g_img;
// the rows of `path` for the image tokens of ids in order; sig[i]: a hash of position i's row (0: a token), the
// kept state's key beside the ids.  -> an error's text, or empty
std::string load_pictures(const std::string& path, const std::vector<int>& ids, int image_token, int C, Pictures& P,
                          std::vector<uint64_t>& sig) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot read the pictures' file";
    for (;;) {
        int32_t h[5];
        if (std::fread(h, 4, 5, f) != 5) break;
        if (h[0] != 0x31455653 || h[1] <= 0 || h[4] != C) {
            std::fclose(f);
            return "the pictures' file is not the image encoder's for this model";
        }
        const size_t at = P.rows.size(), n = (size_t) h[1] * C;
        P.rows.resize(at + n);
        if (std::fread(P.rows.data() + at, 4, n, f) != n) {
            std::fclose(f);
            return "the pictures' file is cut short";
        }
    }
    std::fclose(f);
    const int n_rows = (int) (P.rows.size() / C);
    P.row_at.assign(ids.size(), -1);
    int k = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != image_token) continue;
        if (k >= n_rows) return "the prompt has more image tokens than its pictures have rows";
        P.row_at[i] = k;
        uint64_t hsh = 1469598103934665603ull;   // FNV-1a over the row's bytes
        const uint8_t* b = (const uint8_t*) (P.rows.data() + (size_t) k * C);
        for (size_t j = 0; j < (size_t) C * 4; ++j) hsh = (hsh ^ b[j]) * 1099511628211ull;
        sig[i] = hsh | 1;
        ++k;
    }
    if (k != n_rows) return "the prompt's image tokens and its pictures' rows do not match";
    P.n = n_rows;
    return "";
}
// the pictures' rows of positions pos0 .. pos0 + T - 1 over their tokens' embeddings in h [T][C]
void place_pictures(float* h, int T, int pos0, int C, cudaStream_t s) {
    if (!g_img.n) return;
    for (int t = 0; t < T;) {
        const int r = g_img.at(pos0 + t);
        if (r < 0) {
            ++t;
            continue;
        }
        int n = 1;
        while (t + n < T && g_img.at(pos0 + t + n) == r + n) ++n;
        CU(cudaMemcpyAsync(h + (size_t) t * C, g_img.rows.data() + (size_t) r * C, (size_t) n * C * 4,
                           cudaMemcpyHostToDevice, s));
        t += n;
    }
}

std::vector<DevCtx> g_dev;
int g_cur = -1;
Work& dev_work(int d) { return g_dev[d].W; }
ExpertCache& dev_cache(int d) { return g_dev[d].EC; }
void use_dev(int d) {
    if (d == g_cur) return;
    if (g_stage) { CU(cudaFree(g_stage)); g_stage = nullptr; g_stage_bytes = 0; }   // (the previous device's)
    CU(cudaSetDevice(d));
    g_cur = d;
    g_s = g_dev[d].s, g_cs = g_dev[d].cs, g_ev = g_dev[d].ev, g_blas = g_dev[d].blas;
}

// IK2_ABLATE=FILE: a direction r (the model's hidden size of raw little-endian floats, made unit length at load) that
// every residual state loses once a write has joined it, y <- y - s r (r . y), s = IK2_ABLATE_SCALE (default 1): the
// normed embedding (pictures included) and each layer visit's attention and MoE results, on every path (prefill, the
// resident token, the block window).  The block bank, the loop's kept stream and its messages are copies of those
// states, so none of them holds r either.  The post convs' histories keep the raw outputs; projecting the sum after
// the conv is what keeps the stream exact.  One GPU only.  Unset: no launch, the paths as before.
std::vector<float*> g_abl_r;   // a device's copy of r; empty: off
float g_abl_s = 1.0f;
// IK2_ABLATE_VISITS=LO:HI: only the writes of visits LO .. HI-1 (the embedding only when LO is 0); default all.
// IK2_ABLATE_MEAN=FILE: 1 + 2 n_visit floats - the embedding's, then each visit's attention and MoE state's target
// component c (the mean over answered harmless prompts): y <- y - s r (r . y - c) instead of to 0.
int g_abl_lo = 0, g_abl_hi = 1 << 30;
std::vector<float> g_abl_c;
int g_abl_vis = -1;   // the visit being run (set by each path's visit loop)
// IK2_ABLATE_PREFILL_ONLY=1: project only while a prompt is read (forward / forward_prefill_sup), not on the tokens
// a request generates (the resident token and block paths)
bool g_abl_decode = true;
bool g_abl_in_decode = false;
// IK2_ABLATE_DECODE_TOKENS=N (the batch server): on generated tokens, project only a request's first N; each decode
// row's strength is read from g_abl_srow (set by the scheduler before every step, so recorded graphs follow it)
int g_abl_tokens = -1;
bool g_abl_keep_norm = false;   // IK2_ABLATE_KEEP_NORM=1: each state keeps its length
// IK2_ABLATE_VISIT_SCALES=LO:HI:S[,LO:HI:S...]: visits LO .. HI-1 use strength S (others IK2_ABLATE_SCALE; the
// embedding too); overrides IK2_ABLATE_VISITS
std::vector<float> g_abl_vs;
float* g_abl_srow = nullptr;     // device [kAblRows]
float* g_abl_srow_h = nullptr;   // pinned mirror
// rows y[0 .. T) ([T][C], on the current GPU's stream); kind 0 the embedding, 1 a visit's attention state, 2 its
// MoE state (the visit: g_abl_vis)
void abl(float* y, int T, int C, int kind) {
    if (g_abl_r.empty() || T <= 0 || (g_abl_in_decode && !g_abl_decode)) return;
    if (kind == 0 ? g_abl_lo > 0 : (g_abl_vis < g_abl_lo || g_abl_vis >= g_abl_hi)) return;
    const int site = kind == 0 ? 0 : 1 + 2 * g_abl_vis + (kind - 1);
    const float c = g_abl_c.empty() ? 0.0f : g_abl_c.at((size_t) site);
    const float s = kind != 0 && !g_abl_vs.empty() ? g_abl_vs.at((size_t) g_abl_vis) : g_abl_s;
    if (s == 0.0f) return;
    k::project_out(y, g_abl_r[g_cur], s, c, T, C, g_s,
                   g_abl_in_decode && g_abl_tokens >= 0 && T <= kAblRows ? g_abl_srow : nullptr, g_abl_keep_norm);
}

Cand g_cand;

#include "ik2_batch_serve.cuh"

}  // namespace ik2::engine

int main(int argc, char** argv) {
    using namespace ik2::engine;

    std::string model, toks, dump, draft_path, lora_path;
    int gen = 16, chunk = 2048, ctx = 262144, threads = 8, n_gpu = 1;
    int cache_slots = getenv("IK2_CACHE_SLOTS") ? std::atoi(getenv("IK2_CACHE_SLOTS")) : 1;
    bool serve = false, serve_maya = true;   // (--serve: Maya's engine protocol; --serve-ik2: the @-line one)
    std::vector<int> eos_ids;                 // --eos a,b: the tokens a request ends at (Maya's protocol)
    size_t cache_limit = SIZE_MAX, vram_limit = SIZE_MAX;
    auto free_for_budget = [&](size_t actual_free) {
        if (vram_limit == SIZE_MAX) return actual_free;
        const size_t charged = std::min(device_allocated, vram_limit);
        return std::min(actual_free, vram_limit - charged);
    };
    auto gib = [](const std::string& value) {
        char* end = nullptr;
        const double bytes = std::strtod(value.c_str(), &end) * 1073741824.0;
        if (*end || !(bytes >= 1 && bytes < double(SIZE_MAX))) die("invalid GiB budget: " + value);
        return size_t(bytes);
    };
    int image_token = 200054;   // --image-token: Inkling 2's <|unused_200054|>, the template's placeholder of a picture
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value of " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") {   // (--tokens @FILE: the ids from a file - a long prompt passes a shell's limit)
            toks = next();
            if (!toks.empty() && toks[0] == '@') {
                std::ifstream in(toks.substr(1));
                toks.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                while (!toks.empty() && std::isspace((unsigned char) toks.back())) toks.pop_back();
            }
        }
        else if (a == "--gen") gen = std::atoi(next().c_str());
        else if (a == "--chunk") chunk = std::atoi(next().c_str());
        else if (a == "--ctx" || a == "--max-context") ctx = std::atoi(next().c_str());   // (--max-context: serve/server.py's name)
        else if (a == "--cache-slots") cache_slots = std::atoi(next().c_str());
        else if (a == "--cache-gb") cache_limit = gib(next());
        else if (a == "--vram-gb") vram_limit = gib(next());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--gpus") n_gpu = std::atoi(next().c_str());
        else if (a == "--draft") draft_path = next();
        else if (a == "--lora") lora_path = next();
        else if (a == "--serve") serve = true;
        else if (a == "--serve-ik2") serve = true, serve_maya = false;
        else if (a == "--eos") {
            const std::string v = next();
            for (const char* p = v.c_str(); *p;) {
                eos_ids.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        else if (a == "--image-token") image_token = std::atoi(next().c_str());
        else if (a == "--dump") dump = next();
        else die("unknown argument " + a);
    }
    if (vram_limit != SIZE_MAX && n_gpu != 1) die("--vram-gb requires one GPU");
    g_prof = getenv("IK2_PROF") && std::atoi(getenv("IK2_PROF"));
#ifdef STRATA_USE_HIP
    for (const char* flag : {"IK2_GROUP_MMA", "IK2_WGMMA"}) {
        if (getenv(flag) && std::atoi(getenv(flag))) die(std::string(flag) + " requires CUDA tensor cores");
    }
    if (serve && !draft_path.empty() && getenv("IK2_BLOCK") && !atoi(getenv("IK2_BLOCK")))
        die("--serve with --draft requires block verification");
    if (n_gpu != 1) die("HIP supports one RDNA 3 or 3.5 device");
    g_flash = getenv("IK2_FLASH") && std::atoi(getenv("IK2_FLASH"));
    cudaDeviceProp hip_props{};
    CU(cudaGetDeviceProperties(&hip_props, 0));
    std::fprintf(stderr, "ik2 HIP %s: %s prefill attention\n", hip_props.gcnArchName, g_flash ? "RDNA3 WMMA" : "hipBLAS");
#else
    g_flash = !(getenv("IK2_FLASH") && !std::atoi(getenv("IK2_FLASH")));
#endif
    g_mv = !getenv("IK2_MV") || std::atoi(getenv("IK2_MV"));
    g_group = !getenv("IK2_GROUP") || std::atoi(getenv("IK2_GROUP"));
    if (getenv("IK2_GROUP")) {
        const int v = std::atoi(getenv("IK2_GROUP"));
        g_group_exp = v == 1 || v >= 3, g_group_attn = v >= 2;
    }
    g_dense_q8 = getenv("IK2_DENSE") && std::string(getenv("IK2_DENSE")) == "q8";
    if (getenv("IK2_KV")) {
        const std::string kv = getenv("IK2_KV");
        if (kv == "f16" || kv == "fp16") g_kv_precision = KVPrecision::f16;
        else if (kv == "fp8" || kv == "e4m3") g_kv_precision = KVPrecision::fp8;
        else if (kv == "fp8-global") g_kv_precision = KVPrecision::fp8_global;
        else if (kv == "fp4") {
#ifdef STRATA_USE_HIP
            g_kv_precision = KVPrecision::fp4;
#else
            die("IK2_KV=fp4 requires HIP; use fp4-global for CUDA");
#endif
        }
        else if (kv == "fp4-global") g_kv_precision = KVPrecision::fp4_global;
        else die("unknown IK2_KV=" + kv + " (expected f16, fp8, fp8-global, fp4 or fp4-global)");
    }
    if (getenv("IK2_TRACE") && *getenv("IK2_TRACE")) {
        g_trace = std::fopen(getenv("IK2_TRACE"), "wb");
        if (!g_trace) die(std::string("cannot write ") + getenv("IK2_TRACE"));
    }
    if (model.empty() || (toks.empty() && !serve))
        die("usage: tintero --model FIRST.gguf (--tokens 1,2,3 [--gen N] | --serve) [--chunk 2048] [--ctx N] ...");
    const int batch_width = getenv("IK2_BATCH") ? std::atoi(getenv("IK2_BATCH")) : 1;
    if (const char* v = getenv("IK2_SPARSE_DECODE")) {
        if (std::string(v) != "0" && std::string(v) != "1") die("IK2_SPARSE_DECODE expects 0 or 1");
        g_sparse_decode = std::string(v) == "1";
    }
    if (g_sparse_decode) {
        if (batch_width != 1 || !draft_path.empty() || (getenv("IK2_SUFFIX") && atoi(getenv("IK2_SUFFIX"))))
            die("IK2_SPARSE_DECODE requires single-stream decode without speculation");
    }
    if (batch_width < 1 || batch_width > 32) die("IK2_BATCH must be between 1 and 32");
    const bool batch_draft = batch_width > 1 && !draft_path.empty();
    if (batch_draft && batch_width > 8) die("batched --draft supports up to eight slots");
    const int batch_rows = batch_width * (batch_draft ? kBlockMax : 1);
    if (batch_width > 1) {
        if (!serve || !serve_maya || (cache_slots != 1 && cache_slots != batch_width))
            die("IK2_BATCH requires --serve without a separate cache-slot count");
        cache_slots = batch_width;
    }
    if (chunk < 1 || ctx < 1) die("--chunk and --ctx must be positive");
    if (cache_slots < 1 || cache_slots > ctx) die("--cache-slots must be between 1 and the total context size");
    if (cache_slots > 1 && (!serve || n_gpu != 1)) die("multiple cache slots require a single-GPU server");
    ctx /= cache_slots;
    std::vector<int> prompt;
    for (const char* p = toks.c_str(); *p;) {
        prompt.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    const double t_start = now();
    Shards S(model);
    Model M;
    Config& c = M.c;
    const auto* arch = S.get("general.architecture");
    const bool physical_layers = arch && arch->s == "inkling2_loop";
    const std::string a = physical_layers ? "inkling2_loop." : "inkling2.";
    // a key under PR #32187's name, else under llama.cpp master's; a value master writes a visit (an array of equal
    // values) read as one
    auto key = [&](const char* k1, const char* k2 = nullptr) -> double {
        const auto* v = S.get(a + k1);
        if (!v && k2) v = S.get(a + k2);
        if (!v) die(std::string("no key ") + a + k1);
        return v->items.empty() ? v->num() : v->items[0].num();
    };
    const int block_count = (int) key("block_count");
    c.C = (int) key("embedding_length");
    c.H = (int) key("attention.head_count");
    c.KV = (int) key("attention.head_count_kv");
    c.D = (int) key("attention.key_length");
    c.E = (int) key("expert_count");
    c.K = (int) key("expert_used_count");
    c.F = (int) key("expert_feed_forward_length");
    c.window = (int) key("attention.sliding_window");
    c.d_rel = (int) key("d_rel", "attention.relative_length");
    c.extent = (int) key("rel_extent", "attention.relative_extent");
    c.extent_swa = S.get(a + "rel_extent_swa") ? (int) key("rel_extent_swa") : c.window;   // (master: the window)
    c.block = (int) key("attn_res.block_size");
    c.n_vocab = (int) key("unpadded_vocab_size", "vocab_size_unpadded");
    c.eps = (float) key("attention.layer_norm_rms_epsilon");
    c.route_scale = (float) key("expert_weights_scale");
    c.qk_clip = (float) key("qk_clip_threshold", "attention.qk_clip");
    c.rel_clip = (float) key("rel_clip_threshold", "attention.relative_clip");
    c.alpha = (float) key("log_scaling_alpha", "attention.log_scale_alpha");
    c.n_floor = (float) key("log_scaling_n_floor", "attention.log_scale_floor");
    c.gate_cap = (float) key("mlp_gate_softcap", "activation.situ_beta");
    c.up_cap = (float) key("mlp_up_softcap", "activation.situ_linear_beta");
    c.logit_cap = (float) key("final_logit_softcapping");
    // inkling2 counts visits; inkling2_loop counts the physical layers once.
    if (physical_layers) c.n_pass = (int) key("loop.count");
    else if (S.get(a + "num_loops")) c.n_pass = (int) key("num_loops");
    if (c.n_pass < 1) die("loop count must be positive");
    if (c.n_pass > 1) {
        c.core0 = (int) key("loop.core_start", "loop.start"), c.core1 = (int) key("loop.core_end", "loop.end"), c.rank = (int) key("loop.gate_rank");
        if (c.core0 <= 0 || c.core1 <= c.core0 || c.core0 % c.block || (c.core1 - c.core0) % c.block || c.rank <= 0)
            die("the looped core must start after the first block and span whole blocks");
    }
    const int n_core = c.core1 - c.core0;
    const int extra_visits = (c.n_pass - 1) * n_core;
    c.n_layer = physical_layers ? block_count : block_count - extra_visits;
    c.n_visit = c.n_layer + extra_visits;
    if (c.core1 > c.n_layer || (physical_layers && S.get(a + "loop.varying") && key("loop.varying")))
        die("unsupported loop configuration");
    {   // the pattern is a visit's; a layer's is its first visit's (the same in every pass)
        const auto* sp = S.get(a + "attention.sliding_window_pattern");
        if (!sp || (int) sp->items.size() != block_count) die("sliding_window_pattern");
        for (int l = 0; l < c.n_layer; ++l)
            c.sliding.push_back(sp->items[physical_layers || l < c.core1 ? l : l + extra_visits].u != 0);
    }
    // (a row of an expert is whole blocks of its type: checked per tensor below; 64 covers NVFP4 and Q8_0)
    if (c.D != 128 || c.H % c.KV || c.C % 64 || c.F % 64) die("unexpected shapes");
    const int n_sliding = (int) std::count(c.sliding.begin(), c.sliding.end(), true);
    std::printf("tintero: %d layers (%d sliding), %d experts (%d a token), width %d, vocab %d; %zu shards\n",
                c.n_layer, n_sliding, c.E, c.K, c.C, c.n_vocab, S.files.size());
    if (c.n_pass > 1) {
        std::printf("looped: layers %d .. %d run %d times - %d layer visits a token; message gates of rank %d\n", c.core0,
                    c.core1 - 1, c.n_pass, c.n_visit, c.rank);
        // if (n_gpu != 1) die("a looped model: one GPU so far (--gpus 1)");
        if (!draft_path.empty() && (c.C != 2304 || c.n_visit != 48)) die("--draft: a looped model other than Inkling-2-Mini");
    }
    const auto default_profile = ik2::usage::profile_path(model);
    const char* warm = getenv("IK2_WARM");
    const char* output = getenv("IK2_POP_OUT");
    g_profile_out = output ? std::filesystem::path(output) : default_profile;
    g_pop_E = c.E;
    auto counts = ik2::usage::read(warm ? std::filesystem::path(warm) : default_profile, c.n_layer, c.E);
    g_tp_F = c.F, g_tp_C = c.C;
    if (n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"))) {
        const double share = getenv("IK2_EP_SPLIT") ? std::atof(getenv("IK2_EP_SPLIT")) : 0.5;
        g_pop_E_ep = c.E;
        g_ep_own.assign((size_t) c.n_layer * c.E, 0);
        const auto& cnt = counts;
        for (int l = 0; l < c.n_layer; ++l) {
            if (cnt.empty()) {
                for (int e = 0; e < c.E; ++e)
                    g_ep_own[(size_t) l * c.E + e] = ((uint32_t) (e * 2654435761u + l * 40503u) >> 8) % 10000 < share * 10000;
                continue;
            }
            std::vector<int> ord(c.E);
            for (int e = 0; e < c.E; ++e) ord[e] = e;
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[(size_t) l * c.E + a] > cnt[(size_t) l * c.E + b]; });
            double got[2] = {0, 0};
            const double want[2] = {1.0 - share, share};
            for (int e : ord) {   // (the GPU furthest below its share takes the next most picked)
                const double v = (double) cnt[(size_t) l * c.E + e] + 1.0;
                const int d = got[0] / want[0] <= got[1] / want[1] ? 0 : 1;
                g_ep_own[(size_t) l * c.E + e] = (uint8_t) d;
                got[d] += v;
            }
        }
    }

    {
        int nd = 0;
        CU(cudaGetDeviceCount(&nd));
        if (n_gpu < 1 || n_gpu > nd) die("--gpus " + std::to_string(n_gpu) + ": " + std::to_string(nd) + " visible");
    }
    g_dev.resize(n_gpu);
    for (int d = 0; d < n_gpu; ++d) {
        CU(cudaSetDevice(d));
        CU(cudaStreamCreate(&g_dev[d].s));
        CU(cudaStreamCreate(&g_dev[d].cs));
        CU(cudaEventCreateWithFlags(&g_dev[d].ev, cudaEventDisableTiming));
        CB(cublasCreate(&g_dev[d].blas));
        CB(cublasSetStream(g_dev[d].blas, g_dev[d].s));
        CB(cublasSetMathMode(g_dev[d].blas, CUBLAS_DEFAULT_MATH));
    }
    g_ep = n_gpu == 2 && getenv("IK2_EP") && std::atoi(getenv("IK2_EP"));
    g_tp = n_gpu == 2 && !g_ep && getenv("IK2_TP") && std::atoi(getenv("IK2_TP"));
    if (const char* p = getenv("IK2_ABLATE"); p && *p) {
        if (n_gpu != 1) die("IK2_ABLATE: one GPU only");
        std::vector<float> r((size_t) c.C);
        FILE* f = std::fopen(p, "rb");
        if (!f) die(std::string("IK2_ABLATE: cannot read ") + p);
        const size_t got = std::fread(r.data(), sizeof(float), r.size() + 1, f);
        std::fclose(f);
        if (got != r.size()) die("IK2_ABLATE: " + std::string(p) + " is not " + std::to_string(c.C) + " floats");
        double n2 = 0.0;
        for (float x : r) n2 += (double) x * x;
        if (!(n2 > 0.0) || !std::isfinite(n2)) die("IK2_ABLATE: a zero or non-finite direction");
        for (float& x : r) x = (float) (x / std::sqrt(n2));
        if (getenv("IK2_ABLATE_SCALE")) g_abl_s = (float) std::atof(getenv("IK2_ABLATE_SCALE"));
        if (!(g_abl_s >= 0.0f) || !std::isfinite(g_abl_s)) die("IK2_ABLATE_SCALE: 0 or a positive number");
        g_abl_r.resize(1);
        CU(cudaSetDevice(0));
        device_allocated += r.size() * sizeof(float);
        CU(cudaMalloc(&g_abl_r[0], r.size() * sizeof(float)));
        CU(cudaMemcpy(g_abl_r[0], r.data(), r.size() * sizeof(float), cudaMemcpyHostToDevice));
        g_abl_decode = !(getenv("IK2_ABLATE_PREFILL_ONLY") && std::atoi(getenv("IK2_ABLATE_PREFILL_ONLY")));
        if (const char* n = getenv("IK2_ABLATE_DECODE_TOKENS"); n && *n) g_abl_tokens = std::max(0, std::atoi(n));
        g_abl_keep_norm = getenv("IK2_ABLATE_KEEP_NORM") && std::atoi(getenv("IK2_ABLATE_KEEP_NORM"));
        if (const char* v = getenv("IK2_ABLATE_VISIT_SCALES"); v && *v) {
            g_abl_vs.assign((size_t) c.n_visit, 0.0f);
            std::string spec = v;
            for (size_t at = 0; at < spec.size();) {
                const size_t end = std::min(spec.find(',', at), spec.size());
                int lo = 0, hi = 0;
                float sv = 0.0f;
                if (std::sscanf(spec.substr(at, end - at).c_str(), "%d:%d:%f", &lo, &hi, &sv) != 3 || lo < 0 ||
                    hi > c.n_visit || lo >= hi)
                    die("IK2_ABLATE_VISIT_SCALES: LO:HI:S[,LO:HI:S...]");
                for (int i = lo; i < hi; ++i) g_abl_vs[(size_t) i] = sv;
                at = end + 1;
            }
            g_abl_lo = 0, g_abl_hi = c.n_visit;   // (the scales say which visits)
        }
        if (const char* v = getenv("IK2_ABLATE_VISITS"); v && *v) {
            if (std::sscanf(v, "%d:%d", &g_abl_lo, &g_abl_hi) != 2 || g_abl_lo < 0 || g_abl_hi <= g_abl_lo)
                die("IK2_ABLATE_VISITS: LO:HI");
        }
        if (const char* m = getenv("IK2_ABLATE_MEAN"); m && *m) {
            g_abl_c.resize(1 + 2 * (size_t) c.n_visit);
            FILE* fm = std::fopen(m, "rb");
            if (!fm) die(std::string("IK2_ABLATE_MEAN: cannot read ") + m);
            const size_t gm = std::fread(g_abl_c.data(), sizeof(float), g_abl_c.size() + 1, fm);
            std::fclose(fm);
            if (gm != g_abl_c.size()) die("IK2_ABLATE_MEAN: not 1 + 2 n_visit floats");
        }
        device_allocated += kAblRows * sizeof(float);
        CU(cudaMalloc(&g_abl_srow, kAblRows * sizeof(float)));
        CU(cudaMallocHost(&g_abl_srow_h, kAblRows * sizeof(float)));
        for (int i = 0; i < kAblRows; ++i) g_abl_srow_h[i] = 1.0f;   // (until a scheduler says otherwise: all)
        CU(cudaMemcpy(g_abl_srow, g_abl_srow_h, kAblRows * sizeof(float), cudaMemcpyHostToDevice));
        std::fprintf(stderr, "ablate: %s, scale %g, visits %d:%d%s%s, decode tokens %d\n", p, (double) g_abl_s, g_abl_lo,
                     g_abl_hi, g_abl_c.empty() ? "" : ", mean targets", g_abl_decode ? "" : ", prompt only", g_abl_tokens);
    }
    if (dual()) {
        int p01 = 0, p10 = 0;
        CU(cudaDeviceCanAccessPeer(&p01, 0, 1));
        CU(cudaDeviceCanAccessPeer(&p10, 1, 0));
        if (p01 && p10) {
            CU(cudaSetDevice(0));
            CU(cudaDeviceEnablePeerAccess(1, 0));
            CU(cudaSetDevice(1));
            CU(cudaDeviceEnablePeerAccess(0, 0));
        }
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_ep_x, cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_cp[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_ep_cp[1], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&g_ep_done, cudaEventDisableTiming));
        CU(cudaSetDevice(0));
        CU(cudaEventCreateWithFlags(&g_tp_h[0], cudaEventDisableTiming));
        CU(cudaSetDevice(1));
        CU(cudaEventCreateWithFlags(&g_tp_h[1], cudaEventDisableTiming));
        std::printf("%s over 2 GPUs (peer access %s)\n", g_tp ? "experts tensor-parallel" : "expert parallel",
                    p01 && p10 ? "on" : "off: copies through the host");
    }
    use_dev(0);
    g_overlap = !getenv("IK2_OVERLAP") || std::atoi(getenv("IK2_OVERLAP"));
    for (auto& e : g_ring) CU(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    // (off by default: on an RTX PRO 6000 node's Xeon 6737P an NVFP4 expert takes 0.73 ms on 7 threads against 0.33 ms
    //  over PCIe, and the lane measured 16.4 against 16.8 tok/s - IK2_CPU=1 to try it)
    if (getenv("IK2_CPU") && std::atoi(getenv("IK2_CPU")) && ik2::cpu::available() && threads > 1 && n_gpu == 1)
        g_pool = new ik2::cpu::Pool(threads - 1);
    if (getenv("IK2_UP_MS")) g_t_up = std::atof(getenv("IK2_UP_MS")) * 1e-3;
    g_cpu_lazy = !getenv("IK2_CPU_LAZY") || std::atoi(getenv("IK2_CPU_LAZY"));
    if (n_gpu > 1 && getenv("IK2_DENSE")) die("IK2_DENSE with --gpus > 1");

    // Do the experts all fit in VRAM beside the dense weights and the K / V cache (Inkling-2-Mini's 14.7 GiB on a 24 GB
    // card)?  Then they all stay there: the dense weights are kept as the GGUF's blocks unless IK2_DENSE says otherwise
    // (half an FP16 copy's bytes, a token's products straight over them) and tokens take the resident path
    // (IK2_FAST=0: off; IK2_GRAPH=0: its launches one by one instead of a CUDA graph a token).
    const bool resident_only = getenv("IK2_RESIDENT_ONLY") && atoi(getenv("IK2_RESIDENT_ONLY"));
    gpu_served = getenv("IK2_GPU_SERVED") && atoi(getenv("IK2_GPU_SERVED"));
    if (gpu_served && (resident_only || n_gpu != 1 || g_pool))
        die("IK2_GPU_SERVED requires one GPU, pinned experts and a partial cache");
#ifdef STRATA_USE_HIP
    if (gpu_served) die("IK2_GPU_SERVED requires CUDA");
#endif
    if (resident_only && n_gpu != 1) die("IK2_RESIDENT_ONLY requires one GPU");
    g_graph = !(getenv("IK2_GRAPH") && !std::atoi(getenv("IK2_GRAPH")));
    g_block = !(getenv("IK2_BLOCK") && !std::atoi(getenv("IK2_BLOCK")));
#ifdef STRATA_USE_HIP
    g_attn_mma = getenv("IK2_ATTN_MMA") ? std::atoi(getenv("IK2_ATTN_MMA")) : 0;
    if (g_attn_mma < 0 || g_attn_mma > 3) die("IK2_ATTN_MMA expects 0, 1, 2 or 3");
    g_mma_split = k::attn_decode_mma_blocks(c.KV);
#else
    g_attn_mma = c.D == 128 && c.H / c.KV <= 8 ? (getenv("IK2_ATTN_MMA") ? std::atoi(getenv("IK2_ATTN_MMA")) : 1) : 0;
    g_mma_split = k::attn_decode_mma_blocks(c.KV);
#endif
    if (n_gpu == 1 && getenv("IK2_KV_MANAGED") && atoi(getenv("IK2_KV_MANAGED"))) {
        int concurrent = 0;
        CU(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0));
        if (!concurrent) die("IK2_KV_MANAGED requires concurrent managed memory");
        g_kv_managed = true;
        std::printf("global KV: on-demand managed pages, sliding KV in VRAM\n");
    }
    if (n_gpu == 1 && !g_pool) {
        size_t exp_b = 0, dense_b = 0, kv_b = 0;
        for (const auto& kv : S.by_name) {
            const auto& t = *kv.second.second;
            if (!k::block_values((int) t.type)) continue;
            if (t.name.find("_exps.weight") != std::string::npos) exp_b += tensor_bytes(t);
            else if (t.name != "token_embd.weight") dense_b += tensor_bytes(t);
        }
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t cap = c.sliding[l] ? (size_t) std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : (size_t) ctx;
            if (!g_kv_managed || c.sliding[l])
                kv_b += (l >= c.core0 && l < c.core1 ? c.n_pass : 1) * cap * kv_row_bytes(c.KV * c.D, c.sliding[l]) * 2;
        }
        kv_b *= (size_t)cache_slots;
        size_t fr0 = 0, tot0 = 0;
        CU(cudaMemGetInfo(&fr0, &tot0));
        fr0 = std::min(fr0, vram_limit);
        // With resident-only loading the actual post-workspace budget decides how many experts fit.
        const size_t reserve = resident_only ? size_t{256} << 20 : size_t{3} << 29;
        const bool fits = exp_b + dense_b + kv_b + reserve <= fr0 && exp_b <= cache_limit;
        if ((fits || gpu_served) && !getenv("IK2_DENSE")) g_dense_q8 = true;
        g_fast_want = (fits || gpu_served) && g_dense_q8 && !(getenv("IK2_FAST") && !std::atoi(getenv("IK2_FAST")));
        if (gpu_served && !g_fast_want) die("IK2_GPU_SERVED requires planar dense weights and fast decode");
        if (resident_only && !g_fast_want) die("resident-only loading: the model, KV cache and requested expert budget must fit on the GPU with planar dense weights");
        stage_cold = getenv("IK2_STAGE_COLD") && atoi(getenv("IK2_STAGE_COLD"));
        g_managed_experts = !gpu_served && g_fast_want && g_kv_managed && getenv("IK2_TIER") && atoi(getenv("IK2_TIER"));
        std::printf("experts %.1f GiB, dense weights %.1f GiB as blocks, K / V cache %.1f GiB, %.1f GiB free: %s\n",
                    exp_b / 1073741824.0, dense_b / 1073741824.0, kv_b / 1073741824.0, fr0 / 1073741824.0,
                    fits ? "every expert stays in VRAM" : "the experts in host memory, a cache of them in VRAM");
    }

    // the dense weights
    const double t_dense = now();
    const size_t C = c.C, HD = (size_t) c.H * c.D, KD = (size_t) c.KV * c.D;
    M.L.resize(c.n_layer);
    // two layouts: llama.cpp PR #32187's (ref/inkling2.py's: the short convs folded, gate and up apart) and unsloth's
    // "GGUF_shard_rewrite" one that llama.cpp's master reads (names of its own, the convs' residual and scale outside
    // their weights, gate and up fused - gate's rows first - with one scale): the second's folded here as it loads
    const bool rw = S.info("blk.0.ffn_gate_up_exps.weight") != nullptr;
    auto nm = [&](const std::string& b, const char* old_name) -> std::string {
        static const std::map<std::string, std::string> m = {
            {"attn_r.weight", "attn_rel.weight"}, {"attn_r_clip.weight", "attn_rel_clip.weight"},
            {"attn_rel_proj.weight", "attn_rel_bias.weight"}, {"attn_res_score.weight", "attn_res_query.weight"},
            {"ffn_res_score.weight", "ffn_res_query.weight"}, {"output_res_score.weight", "output_res_query.weight"},
            {"attn_q_in_norm.weight", "attn_q_inp_norm.weight"}, {"attn_k_in_norm.weight", "attn_k_inp_norm.weight"},
            {"attn_v_in_norm.weight", "attn_v_inp_norm.weight"}, {"shortconv_k.weight", "attn_k_conv.weight"},
            {"shortconv_v.weight", "attn_v_conv.weight"}, {"shortconv_attn.weight", "attn_post_conv.weight"},
            {"shortconv_mlp.weight", "ffn_post_conv.weight"}, {"shortconv_k_dyn.weight", "attn_k_conv_proj.weight"},
            {"shortconv_v_dyn.weight", "attn_v_conv_proj.weight"}};
        auto it = m.find(old_name);
        return b + (rw && it != m.end() ? it->second : std::string(old_name));
    };
    // a conv's weights [C'][4] with the residual (tap 3: the current token) and the output scale folded in
    auto conv_w = [&](const std::string& b, const char* n, size_t Cc, bool dynamic) -> float* {
        auto w = host_f32(S, nm(b, n), Cc * 4);
        if (rw) {
            const float sc = 1.0f / std::sqrt((float) ((dynamic ? 2 : 1) * 4 + 1));
            for (size_t i = 0; i < Cc; ++i) {
                for (int j = 0; j < 4; ++j) w[i * 4 + j] *= sc;
                w[i * 4 + 3] += sc;
            }
        }
        return dev_f32(w);
    };
    auto dyn_host = [&](const std::string& b, const char* n) -> std::vector<float> {
        auto w = host_f32(S, nm(b, n), (size_t) c.KV * 4 * C);
        if (rw)
            for (auto& x : w) x *= 1.0f / std::sqrt(9.0f);
        return w;
    };
    std::unique_ptr<strata::GgufFile> lora_file;
    std::vector<ik2::LoraPair> lora_pairs;
    int max_lora_rank = 0;
    if (!lora_path.empty()) {
        try {
            lora_file = std::make_unique<strata::GgufFile>(lora_path);
            lora_pairs = ik2::attention_lora_pairs(*lora_file, arch ? arch->s : "", c.n_layer, HD, C);
        } catch (const std::exception& e) {
            die("--lora " + lora_path + ": " + e.what());
        }
    }
    std::printf("layout: %s\n", rw ? "GGUF_shard_rewrite tensor layout" : "llama.cpp PR #32187");
    struct ExpJob {
        int l;
        std::string name;   // a tensor read whole into pinned memory
        std::vector<int> roles;
    };
    std::vector<ExpJob> exp_jobs;
    size_t host_total = 0;
    M.embd_norm = load_f32(S, "token_embd_norm.weight", C);   // (on GPU 0, where the embedding rows go)
    std::vector<float> lp_q, lp_kv;   // a looped core's message queries, [2 n_core][C] each
    std::vector<float> lp_all;
    if (c.n_pass > 1 && !physical_layers) {
        lp_q = host_f32(S, "loop_res_query.weight", (size_t) 2 * n_core * C);
        lp_kv = host_f32(S, "loop_kv_res_query.weight", (size_t) 2 * n_core * C);
    }
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& Ly = M.L[l];
        const std::string b = "blk." + std::to_string(l) + ".";
        Ly.dev = dual() ? 0 : l * n_gpu / c.n_layer;
        use_dev(Ly.dev);
        Ly.sliding = c.sliding[l];
        auto sink_v = host_f32(S, b + "attn_sink_proj.weight", (size_t) c.H * C);
        if (Ly.sliding) {
            std::vector<MatPart> qkv_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {b + "attn_k.weight", (int) KD, {}},
                {b + "attn_v.weight", (int) KD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            if (!g_dense_q8) {
                qkv_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqkvrg = load_fused_f16(S, qkv_parts, (int) C);
            } else {
                if (!try_load_fused_raw(S, qkv_parts, (int) C, Ly.wqkvrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                    Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.sink16 = dev_f16(sink_v);
            }
        } else {
            std::vector<MatPart> qrg_parts = {
                {b + "attn_q.weight", (int) HD, {}},
                {nm(b, "attn_r.weight"), c.H * c.d_rel, {}},
                {b + "attn_gate.weight", (int) HD, {}},
            };
            auto kd_v = dyn_host(b, "shortconv_k_dyn.weight");
            auto vd_v = dyn_host(b, "shortconv_v_dyn.weight");
            if (!g_dense_q8) {
                qrg_parts.push_back({"", c.H, std::move(sink_v)});
                Ly.wqrg = load_fused_f16(S, qrg_parts, (int) C);
                Ly.wk_dyn = load_fused_f16(S, {{b + "attn_k.weight", (int) KD, {}}, {"", c.KV * 4, std::move(kd_v)}},
                                           (int) C);
                Ly.wv_dyn = load_fused_f16(S, {{b + "attn_v.weight", (int) KD, {}}, {"", c.KV * 4, std::move(vd_v)}},
                                           (int) C);
            } else {
                if (!try_load_fused_raw(S, qrg_parts, (int) C, Ly.wqrg)) {
                    Ly.wq = load_mat(S, b + "attn_q.weight", (int) HD, (int) C);
                    Ly.wr = load_mat(S, nm(b, "attn_r.weight"), c.H * c.d_rel, (int) C);
                    Ly.wgate = load_mat(S, b + "attn_gate.weight", (int) HD, (int) C);
                }
                Ly.wk = load_mat(S, b + "attn_k.weight", (int) KD, (int) C);
                Ly.wv = load_mat(S, b + "attn_v.weight", (int) KD, (int) C);
                Ly.sink16 = dev_f16(sink_v);
                Ly.sc_k_dyn16 = dev_f16(kd_v);
                Ly.sc_v_dyn16 = dev_f16(vd_v);
            }
        }
        Ly.wo = load_mat(S, b + "attn_output.weight", (int) C, (int) HD);
        if (lora_file && lora_pairs[l].rank) {
            const auto& pair = lora_pairs[l];
            auto upload = [&](const strata::TensorInfo& t) {
                float* dst = dalloc<float>(t.elements());
                CU(cudaMemcpy(dst, lora_file->tensor_data(t), t.elements() * sizeof(float), cudaMemcpyHostToDevice));
                return dst;
            };
            Ly.attn_lora = {upload(*pair.a), upload(*pair.b), pair.rank, pair.scale};
            max_lora_rank = std::max(max_lora_rank, pair.rank);
        }
        Ly.router16 = load_f16(S, b + "ffn_gate_inp.weight", (size_t) c.E * C);
        Ly.probs_b = load_f32(S, b + "exp_probs_b.bias", c.E);
        Ly.ffn_norm = load_f32(S, b + "ffn_norm.weight", C);
        if (Ly.sliding) {
            Ly.attn_norm = load_f32(S, b + "attn_norm.weight", C);
        } else {
            Ly.q_in_norm = load_f32(S, nm(b, "attn_q_in_norm.weight"), C);
            Ly.k_in_norm = load_f32(S, nm(b, "attn_k_in_norm.weight"), C);
            Ly.v_in_norm = load_f32(S, nm(b, "attn_v_in_norm.weight"), C);
        }
        Ly.q_clip = load_f32(S, b + "attn_q_clip.weight", HD);
        Ly.r_clip = load_f32(S, nm(b, "attn_r_clip.weight"), (size_t) c.H * c.d_rel);
        {   // [extent][d_rel]: PR #32187's attn_rel_proj is [d_rel][extent], the rewrite's attn_rel_bias already this
            const int ext = Ly.sliding ? c.extent_swa : c.extent;
            const auto v = host_f32(S, nm(b, "attn_rel_proj.weight"), (size_t) ext * c.d_rel);
            std::vector<float> t(v.size());
            if (rw) {
                t = v;
            } else {
                for (int r = 0; r < c.d_rel; ++r)
                    for (int e = 0; e < ext; ++e) t[(size_t) e * c.d_rel + r] = v[(size_t) r * ext + e];
            }
            Ly.rel_proj = dev_f32(t);
        }
        if (l > 0) {
            const std::string rn = nm(b, "attn_res_score.weight");
            const auto* ti = S.info(rn);
            if (!ti) die(rn);
            Ly.n_res_q = (int) (ti->elements() / C);
            Ly.res_score = load_f32(S, rn, (size_t) Ly.n_res_q * C);
        }
        Ly.ffn_res_score = load_f32(S, nm(b, "ffn_res_score.weight"), C);
        Ly.sc_k = conv_w(b, "shortconv_k.weight", KD, !Ly.sliding);
        Ly.sc_v = conv_w(b, "shortconv_v.weight", KD, !Ly.sliding);
        Ly.sc_attn = conv_w(b, "shortconv_attn.weight", C, false);
        Ly.sc_mlp = conv_w(b, "shortconv_mlp.weight", C, false);
        // the experts: (role tensor, its roles); a fused gate_up is gate's F rows then up's, an expert's 2 F rows
        std::vector<std::pair<std::string, std::vector<int>>> parts;
        if (rw) parts = {{"gate_up", {0, 1}}, {"down", {2}}};
        else parts = {{"gate", {0}}, {"up", {1}}, {"down", {2}}};
        for (const auto& pr : parts) {
            const std::string n = b + "ffn_" + pr.first + "_exps.weight";
            const auto* ti = S.info(n);
            if (!ti) die("no " + n);
            const int nr = (int) pr.second.size();
            if (ti->elements() != (uint64_t) c.E * c.F * C * nr) die(n + ": shape");
            if (!k::block_values((int) ti->type)) die(n + ": type " + ti->type_name() + " not decoded here");
            if (C % k::block_values((int) ti->type) || c.F % k::block_values((int) ti->type))
                die(n + ": a row is not whole " + ti->type_name() + " blocks");
            const size_t per = tensor_bytes(*ti) / c.E;
            std::vector<float> sc;
            const std::string sn = b + "ffn_" + pr.first + "_exps.scale";
            if (S.info(sn)) sc = host_f32(S, sn, c.E);
            else if (ti->type == k::kNVFP4) die(n + ": NVFP4 without its .scale");
            for (int i = 0; i < nr; ++i) {
                const int r = pr.second[i];
                Ly.ty[r] = (int) ti->type;
                Ly.eb[r] = per / nr;
                Ly.stride[r] = per;
                Ly.scale[r] = sc;
            }
            host_total += tensor_bytes(*ti);
            exp_jobs.push_back({l, n, pr.second});
        }
        if (c.n_pass > 1 && l >= c.core0 && l < c.core1) {   // the core: its later passes' message gates and queries
            static const char* const legacy[4] = {"attn", "k", "v", "mlp"};
            static const char* const per_layer[4] = {"q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"};
            const auto& mod = physical_layers ? per_layer : legacy;
            for (int m = 0; m < 4; ++m) {
                const std::string p = b + "loop_" + mod[m] + "_";
                const bool compact = g_fast_want && getenv("IK2_GATE_BF16") && atoi(getenv("IK2_GATE_BF16"));
                const auto down = host_f32(S, p + "down.weight", (size_t)c.rank * 2 * C);
                Ly.lp_down[m] = dev_f32(down);
                if (compact) Ly.lp_bf[m][0] = exact_bf16(down);
                for (int fg = 0; fg < 2; ++fg) {   // (the file's [C][rank] by rank, [rank][C]: k::loop_gate's layout)
                    const auto w = host_f32(S, p + (fg ? "gain.weight" : "forget.weight"), C * c.rank);
                    std::vector<float> t(w.size());
                    for (size_t ch = 0; ch < C; ++ch)
                        for (int r = 0; r < c.rank; ++r) t[(size_t) r * C + ch] = w[ch * c.rank + r];
                    (fg ? Ly.lp_gain[m] : Ly.lp_forget[m]) = dev_f32(t);
                    if (compact) Ly.lp_bf[m][fg + 1] = exact_bf16(t);
                }
            }
            // loop_res_query: rows 2 i (q's) and 2 i + 1 (the MoE's) of core layer i; loop_kv_res_query: k's and v's
            const size_t i = (size_t) (l - c.core0);
            std::vector<float> q(4 * C);
            if (physical_layers) {
                q = host_f32(S, b + "loop_msg_query.weight", 4 * C);
            } else {
                std::memcpy(q.data(), lp_q.data() + 2 * i * C, C * 4);
                std::memcpy(q.data() + C, lp_kv.data() + 2 * i * C, C * 4);
                std::memcpy(q.data() + 2 * C, lp_kv.data() + (2 * i + 1) * C, C * 4);
                std::memcpy(q.data() + 3 * C, lp_q.data() + (2 * i + 1) * C, C * 4);
            }
            Ly.lp_query = dev_f32(q);
            lp_all.insert(lp_all.end(), q.begin(), q.end());
            if (l == c.core1 - 1) M.lp_query_all = dev_f32(lp_all);
        }
    }
    // Each visit owns its KV. Sliding layers use ring buffers in both CLI and server mode.
    const bool paged_kv = getenv("IK2_PAGED_KV") && atoi(getenv("IK2_PAGED_KV"));
    if (g_sparse_decode && (paged_kv || g_kv_precision != KVPrecision::fp8 || c.D != 128 || g_attn_mma != 3))
        die("IK2_SPARSE_DECODE requires token-major IK2_KV=fp8 and IK2_ATTN_MMA=3 with 128-wide heads");
    if (packed_cache(false) && c.D != 128)
        die("FP4 KV requires 128-wide heads");
#ifdef STRATA_USE_HIP
    if (paged_kv && (!byte_cache(false) || c.D != 128)) die("IK2_PAGED_KV requires quantized global KV with 128-wide heads");
#else
    if (paged_kv) die("IK2_PAGED_KV requires the HIP backend");
#endif
    M.V.resize(c.n_visit);
    for (int v = 0; v < c.n_visit; ++v) {
        Visit& Vi = M.V[v];
        if (c.n_pass == 1 || v < c.core0) Vi.l = v;
        else if (v < c.core0 + c.n_pass * n_core) Vi.l = c.core0 + (v - c.core0) % n_core, Vi.pass = (v - c.core0) / n_core;
        else Vi.l = v - (c.n_pass - 1) * n_core;
        Layer& Ly = M.L[Vi.l];
        Ly.fp4_kv = packed_cache(Ly.sliding);
        Ly.paged_kv = paged_kv && !Ly.sliding;
        use_dev(Ly.dev);
        Vi.kv_cap = Ly.sliding ? std::min(ctx, c.window + std::max(chunk, ik2::draft::kBlock)) : ctx;
        if (byte_cache(Ly.sliding)) {
            const size_t slots = Ly.paged_kv ? ((size_t)Vi.kv_cap + 15) / 16 * 16 : Vi.kv_cap;
            Vi.kc8 = kv_alloc<uint8_t>(batch_width * slots * kv_row_bytes(KD, Ly.sliding), Ly.sliding);
            Vi.vc8 = kv_alloc<uint8_t>(batch_width * slots * kv_row_bytes(KD, Ly.sliding), Ly.sliding);
        } else {
            Vi.kc = kv_alloc<uint16_t>((size_t)batch_width * Vi.kv_cap * KD, Ly.sliding);
            Vi.vc = kv_alloc<uint16_t>((size_t)batch_width * Vi.kv_cap * KD, Ly.sliding);
        }
        Vi.h_k = dalloc<float>(batch_width * 3 * KD);
        Vi.h_v = dalloc<float>(batch_width * 3 * KD);
        Vi.h_attn = dalloc<float>(batch_width * 3 * C);
        Vi.h_mlp = dalloc<float>(batch_width * 3 * C);
    }
    c.n_rows = (int) (S.info("output.weight")->elements() / C);
    if (lora_file) {
        std::fprintf(stderr, "loaded LoRA %s: %zu attention output projections, max rank %d\n",
                     lora_path.c_str(), lora_file->tensors().size() / 2, max_lora_rank);
        lora_file.reset();
    }
    M.out_w = load_mat(S, "output.weight", c.n_rows, (int) C);
    if (g_dense_q8) {   // (a layer's largest; the head's logits go a token at a time over its blocks)
        g_w16_n = (2 * HD + 2 * KD + (size_t) c.H * c.d_rel) * C;
        g_w16 = dalloc<uint16_t>(g_w16_n);
    }
    M.out_norm = load_f32(S, "output_norm.weight", C);
    M.out_res_score = load_f32(S, rw ? "output_res_query.weight" : "output_res_score.weight", C);
    M.embd = S.data("token_embd.weight");
    M.embd_type = (int) S.info("token_embd.weight")->type;
    if (!k::block_values(M.embd_type)) die("token_embd's type");
    if (g_stage) { CU(cudaFree(g_stage)); g_stage = nullptr; g_stage_bytes = 0; }
    {
        size_t kv_bytes = 0;
        for (const auto& Vi : M.V) kv_bytes += (size_t) Vi.kv_cap * kv_row_bytes(KD, M.L[Vi.l].sliding) * 2;
        std::printf("kv cache: %s%s (%.2f GiB across %d layer%s)\n", kv_format_name(),
                    paged_kv ? ", 16-token global pages" : "",
                    kv_bytes / 1073741824.0, c.n_visit, c.n_pass > 1 ? " visits" : "s");
        std::fflush(stdout);
    }
    size_t fr = 0, tot = 0;
    for (int d = 0; d < n_gpu; ++d) {
        use_dev(d);
        CU(cudaMemGetInfo(&fr, &tot));
        std::printf("dense weights in VRAM: %.1f s; GPU %d: %.1f GiB of %.1f free\n", now() - t_dense, d,
                    fr / 1073741824.0, tot / 1073741824.0);
        std::fflush(stdout);
    }

    // the experts into pinned host memory: the allocations, then the reads, several files at once.  IK2_HOST=mmap:
    // read where the files are mapped instead (pageable: slower copies, no 323 GiB of host memory - for several
    // GPUs whose caches hold most of the experts)
    const double t_host = now();
    Pinned pin;
    double t_alloc = 0;
    // (the resident path keeps every expert in VRAM, so no pinned host copy is needed: the experts go from where the
    //  files are mapped straight to their slots, and stay in the page cache for the next load - IK2_HOST=pinned: as
    //  for the other models)
    const bool host_mmap = getenv("IK2_HOST") ? std::string(getenv("IK2_HOST")) == "mmap" : g_fast_want && !gpu_served;
    if (gpu_served && host_mmap) die("IK2_GPU_SERVED requires IK2_HOST=pinned");
    if (host_mmap) {
        if (g_pool) die("IK2_HOST=mmap with the CPU lane");
        for (const auto& J : exp_jobs) {
            Layer& Ly = M.L[J.l];
            const uint8_t* p = S.data(J.name);
            for (size_t i = 0; i < J.roles.size(); ++i) Ly.host[J.roles[i]] = p + i * Ly.eb[J.roles[i]];
#ifndef _WIN32
            {   // (read ahead: a cold file otherwise comes in a page fault at a time)
                const size_t pg = 4096, a0 = (size_t) p & ~(pg - 1);
                ::madvise((void*) a0, (size_t) p - a0 + tensor_bytes(*S.info(J.name)), MADV_WILLNEED);
            }
#endif
        }
        std::printf("experts: read where the files are mapped (IK2_HOST=mmap)\n");
        std::fflush(stdout);
    } else {
        // Pinned host memory: anonymous mappings read into and then registered with CUDA, a layer as soon as its
        // reads are done (323 GiB in ~60 s from VAST); IK2_ALLOC=hostalloc: cudaHostAlloc instead, which pins at
        // ~3 GiB/s (the driver serializes it; 110-190 s)
#ifndef _WIN32
        bool use_mmap_reg = !(getenv("IK2_ALLOC") && std::string(getenv("IK2_ALLOC")) == "hostalloc");
        if (use_mmap_reg) {
            const size_t probe_bytes = 2ULL << 20;
            void* mp = ::mmap(nullptr, probe_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mp == MAP_FAILED) {
                use_mmap_reg = false;
            } else {
                *((volatile uint8_t*) mp) = 0;
                if (cudaHostRegister(mp, probe_bytes, cudaHostRegisterPortable) != cudaSuccess) {
                    (void)cudaGetLastError();
                    use_mmap_reg = false;
                } else {
                    CU(cudaHostUnregister(mp));
                }
                ::munmap(mp, probe_bytes);
            }
        }
        pin.blocks.resize(c.n_layer);
        if (use_mmap_reg) {
            for (int l = 0; l < c.n_layer; ++l) {
                const Layer& Ly = M.L[l];
                const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
                const size_t layer_bytes = (size_t) c.E * slot_bytes;
                void* p = ::mmap(nullptr, layer_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) {
                    for (int k = 0; k < l; ++k) ::munmap(pin.blocks[k].ptr, pin.blocks[k].bytes);
                    pin.blocks.assign(c.n_layer, {});
                    use_mmap_reg = false;
                    break;
                }
                pin.blocks[l] = {p, layer_bytes, true, false};
            }
        }
#else
        bool use_mmap_reg = false;
        pin.blocks.resize(c.n_layer);
#endif
        // (pinned: the allocations and the reads overlap - a layer's reads start as soon as its buffer exists;
        // IK2_ALLOC_THREADS threads allocate, default 1)
        std::vector<std::atomic<int>> ready(c.n_layer);
        for (auto& r : ready) r.store(use_mmap_reg ? 1 : 0);
        for (int l = 0; l < c.n_layer; ++l) {
            Layer& Ly = M.L[l];
            const size_t slot_bytes = Ly.eb[0] + Ly.eb[1] + Ly.eb[2];
            Ly.host_contig = true;
            Ly.slot_bytes = slot_bytes;
            Ly.stride[0] = Ly.stride[1] = Ly.stride[2] = slot_bytes;
        }
        std::vector<std::thread> alloc_th;
        std::atomic<int> next_alloc{0};
        double t_alloc_end = 0;
        std::atomic<int> alloc_left{use_mmap_reg ? 0 : c.n_layer};
        if (!use_mmap_reg) {
            const int na = std::max(1, getenv("IK2_ALLOC_THREADS") ? std::atoi(getenv("IK2_ALLOC_THREADS")) : 1);
            for (int t = 0; t < na; ++t)
                alloc_th.emplace_back([&]() {
                    for (;;) {
                        const int l = next_alloc++;
                        if (l >= c.n_layer) return;
                        const size_t layer_bytes = (size_t) c.E * M.L[l].slot_bytes;
                        void* p = nullptr;
                        CU(cudaHostAlloc(&p, layer_bytes, cudaHostAllocPortable | (gpu_served ? cudaHostAllocMapped : 0)));
                        pin.blocks[l] = {p, layer_bytes, false, false};
                        ready[l].store(1, std::memory_order_release);
                        if (--alloc_left == 0) t_alloc_end = now();
                    }
                });
        }
        auto wait_layer = [&](int l) {
            while (!ready[l].load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            Layer& Ly = M.L[l];
            uint8_t* buf = (uint8_t*) pin.blocks[l].ptr;
            Ly.host[0] = buf;
            Ly.host[1] = buf + Ly.eb[0];
            Ly.host[2] = buf + Ly.eb[0] + Ly.eb[1];
        };
        std::vector<std::atomic<int>> jobs_left(c.n_layer);
        for (auto& v : jobs_left) v.store(0);
        for (const auto& J : exp_jobs) ++jobs_left[J.l];
        std::mutex reg_m;
        std::condition_variable reg_cv;
        std::deque<int> reg_q;
        int reg_fail = 0;
        std::thread reg_th;
        if (use_mmap_reg)
            reg_th = std::thread([&]() {   // registers each layer once its reads are done, while the others are read
                for (int done = 0; done < c.n_layer; ++done) {
                    int l;
                    {
                        std::unique_lock<std::mutex> lk(reg_m);
                        reg_cv.wait(lk, [&] { return !reg_q.empty(); });
                        l = reg_q.front();
                        reg_q.pop_front();
                    }
                    if (cudaHostRegister(pin.blocks[l].ptr, pin.blocks[l].bytes, cudaHostRegisterPortable | (gpu_served ? cudaHostRegisterMapped : 0)) == cudaSuccess)
                        pin.blocks[l].registered = true;
                    else
                        (void)cudaGetLastError(), ++reg_fail;
                }
            });
        auto layer_read = [&](int l) {
            if (!use_mmap_reg) return;
            {
                std::lock_guard<std::mutex> lk(reg_m);
                reg_q.push_back(l);
            }
            reg_cv.notify_one();
        };
        for (int l = 0; l < c.n_layer; ++l)
            if (jobs_left[l].load() == 0) layer_read(l);
        std::vector<std::thread> th;
        std::atomic<int> next{0};
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&]() {
                for (;;) {
                    const int j = next++;
                    if (j >= (int) exp_jobs.size()) return;
                    const auto& J = exp_jobs[j];
                    wait_layer(J.l);
                    const Layer& Ly = M.L[J.l];
                    const size_t role_off = J.roles[0] == 0 ? 0 : (J.roles[0] == 1 ? Ly.eb[0] : Ly.eb[0] + Ly.eb[1]);
                    const size_t exp_bytes = tensor_bytes(*S.info(J.name)) / c.E;
                    uint8_t* dst = (uint8_t*) pin.blocks[J.l].ptr + role_off;
                    read_into_strided(S.paths[S.file_of(J.name)], S.file_offset(J.name), c.E, exp_bytes,
                                      Ly.slot_bytes, dst);
                    if (--jobs_left[J.l] == 0) layer_read(J.l);
                }
            });
        for (auto& t : alloc_th) t.join();
        for (auto& t : th) t.join();
        for (int l = 0; l < c.n_layer; ++l) wait_layer(l);   // (host[] of layers no job read)
        t_alloc = (use_mmap_reg ? 0.0 : t_alloc_end - t_host);
        const double t_read = now() - t_host - t_alloc;   // (the reads left after the last allocation)
        double t_pin = 0;
        if (use_mmap_reg) {
            const double tp0 = now();
            reg_th.join();   // (the layers read last are still registering)
            t_pin = now() - tp0;
            if (gpu_served && reg_fail) die("could not map expert memory");
            if (reg_fail > 0)
                std::fprintf(stderr, "tintero: warning: cudaHostRegister failed on %d layers (running pageable)\n",
                             reg_fail);
        }
        if (use_mmap_reg) {
            std::printf("experts in pinned (mmap+register) memory: %.1f GiB; allocating %.1f s, reading %.1f s "
                        "(%.2f GiB/s), pinning %.1f s\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read, t_pin);
        } else {
            std::printf("experts in pinned memory: %.1f GiB; allocating %.1f s, reading %.1f s (%.2f GiB/s)\n",
                        host_total / 1073741824.0, t_alloc, t_read, host_total / 1073741824.0 / t_read);
        }
        std::fflush(stdout);
    }

    // the drafter (before the expert cache takes what VRAM is left)
    ik2::draft::Drafter* drafter = nullptr;
    uint16_t* draft_head = M.out_w.f16;
    float* draft_mask = nullptr;
    if (!draft_path.empty()) {
        if (n_gpu != 1) die("--draft: one GPU only so far");
        g_final_tap = c.n_pass == 1 ? 7 : 3;
        const double t0 = now();
        try {
            size_t free_before = 0, total = 0;
            if (vram_limit != SIZE_MAX) CU(cudaMemGetInfo(&free_before, &total));
            drafter = ik2::draft::load(draft_path, c.C, ctx, g_blas, g_s, batch_width);
            if (vram_limit != SIZE_MAX) {
                size_t free_after = 0;
                CU(cudaMemGetInfo(&free_after, &total));
                device_allocated += free_before - std::min(free_before, free_after);
            }
            draft_mask = ik2::draft::dflash2_mask(drafter);
#ifdef STRATA_USE_HIP
            if (batch_draft && !draft_mask) die("batched HIP drafting requires DFlash2");
#endif
            if (draft_mask && (g_final_tap != 3 || c.n_vocab != 200058))
                die("DFlash2 requires the Inkling-2-Mini target");
            if (!draft_head && M.out_w.q8 && !draft_mask) {
                const size_t n = (size_t) c.n_rows * c.C;
                draft_head = dalloc<uint16_t>(n);
                k::decode_q8p_f16(M.out_w.q8, M.out_w.q8d, n, draft_head, g_s);
            }
            if (!draft_head && !M.out_w.q8) die("--draft: the output head is neither FP16 nor planar Q8_0");
            ik2::draft::fold(drafter, draft_head, c.n_rows);
        } catch (const std::exception& e) {
            die(e.what());
        }
        if (M.out_w.q8 && (draft_mask || !M.out_w.f16)) {
            ik2::draft::set_head_q8(drafter, M.out_w.q8, M.out_w.q8d);
            CU(cudaStreamSynchronize(g_s));
            if (draft_head && draft_head != M.out_w.f16) {
                CU(cudaFree(draft_head));
                device_allocated -= size_t(c.n_rows) * c.C * sizeof(uint16_t);
            }
            draft_head = nullptr;
        }
        g_taps_ld = std::max({chunk, ik2::draft::kBlock, batch_rows});
        g_taps = dalloc<float>((size_t) (g_final_tap + 1) * g_taps_ld * c.C);
        if (batch_width == 1) for (auto& Vi : M.V) {
            const size_t cc[4] = {KD, KD, C, C};
            for (int i = 0; i < 4; ++i) {
                Vi.rec_old[i] = dalloc<float>(3 * cc[i]);
                Vi.rec_x[i] = dalloc<float>((size_t) ik2::draft::kBlock * cc[i]);
            }
        }
        std::printf("drafter: %s, %.2f GiB of VRAM, %.1f s\n", draft_path.c_str(),
                    ik2::draft::bytes(drafter) / 1073741824.0, now() - t0);
    }

#ifdef STRATA_USE_HIP
    constexpr int max_suffix_inputs = 512;
#else
    constexpr int max_suffix_inputs = 128;
#endif
    if (getenv("IK2_SUFFIX") && n_gpu == 1 && g_fast_want && !drafter)
        suffix_window = std::min({std::max(0, atoi(getenv("IK2_SUFFIX"))), chunk, max_suffix_inputs});
    if (suffix_window >= 8) {
        for (auto& v : M.V) {
            const size_t cc[] = {KD, KD, C, C};
            for (int j = 0; j < 4; ++j) {
                v.rec_old[j] = dalloc<float>(3 * cc[j]);
                v.rec_x[j] = dalloc<float>((size_t)suffix_window * cc[j]);
            }
        }
        std::printf("suffix verification: up to %d inputs per target forward\n", suffix_window);
    } else
        suffix_window = 0;

    SlotStorage slot_storage;
    if (batch_width == 1 && cache_slots > 1) {
        for (const Visit& v : M.V) {
            const size_t page = M.L[v.l].paged_kv ? 16 : 1;
            const size_t row = kv_row_bytes(KD, M.L[v.l].sliding);
            slot_storage.add(v.kc8 ? (void*)v.kc8 : (void*)v.kc, v.kv_cap, row, page);
            slot_storage.add(v.vc8 ? (void*)v.vc8 : (void*)v.vc, v.kv_cap, row, page);
            for (auto* h : {v.h_k, v.h_v}) slot_storage.add(h, 1, 3 * KD * sizeof(float));
            for (auto* h : {v.h_attn, v.h_mlp}) slot_storage.add(h, 1, 3 * C * sizeof(float));
        }
        if (drafter)
            for (const auto& span : ik2::draft::cache_spans(drafter))
                slot_storage.add(span.ptr, span.rows, span.row_bytes);
        slot_storage.allocate(cache_slots);
    }
    // the work buffers and the expert cache, a set a GPU
    for (int d = 0; d < n_gpu; ++d) {
    use_dev(d);
    CU(cudaMemGetInfo(&fr, &tot));
    fr = free_for_budget(fr);
    Work& W = g_dev[d].W;
    W.Tm = std::max(batch_rows, drafter ? std::max(chunk, kBlockMax) : chunk);
    W.ctx = ctx;
    W.nk_max = std::min(ctx, ctx);
    // Tokens a layer-major prefill pass (every expert not in VRAM streams in once a pass): 16384 when there is room
    // (a pass's scratch is ~280 KB a token: 4.6 GB, taken from the expert cache), else 4096.
    // (a looped model's prompts go through forward(): forward_prefill_sup has no loop yet)
    // ... nor the resident path's models: a pass borrows cache slots, which would evict experts
    W.T_sup = ((n_gpu == 1 || dual()) && draft_path.empty() && c.n_pass == 1 && !g_fast_want && fr >= (20ULL << 30)) ? (fr >= (60ULL << 30) ? 16384 : 4096) : 0;
    // IK2_SUP=N: tokens a layer-major prefill pass (each pass streams every expert not in the cache once)
    if (W.T_sup > 0 && getenv("IK2_SUP")) W.T_sup = std::max(W.Tm, std::atoi(getenv("IK2_SUP")));
    const size_t Tm = W.Tm, S8 = Tm * c.K;
    const size_t T_buf = std::max(Tm, (size_t) W.T_sup), S_buf = T_buf * c.K;
    W.blk_ld = T_buf;
    if (max_lora_rank) W.attn_lora_tmp = work_alloc<float>(T_buf * max_lora_rank);
    for (float** p : {&W.h, &W.r, &W.tmp})
        *p = work_alloc<float>(T_buf * C);
    for (float** p : {&W.res, &W.a, &W.xq, &W.xk, &W.xv})
        *p = work_alloc<float>(Tm * C);
    W.q = work_alloc<float>(Tm * HD), W.gate = work_alloc<float>(Tm * HD), W.o = work_alloc<float>(Tm * HD);
    W.kk = work_alloc<float>(Tm * KD), W.vv = work_alloc<float>(Tm * KD), W.kconv = work_alloc<float>(Tm * KD);
    W.vconv = work_alloc<float>(Tm * KD);
    W.rr = work_alloc<float>(Tm * c.H * c.d_rel), W.sink = work_alloc<float>(Tm * c.H);
    W.dyn =
        work_alloc<float>(2 * Tm * c.KV * 4); // (a full layer's dynamic taps: k's, and on the resident path v's beside)
    W.lg_cap = std::max((size_t) 8 * c.H * (size_t) ctx + (size_t) 8 * c.H * 128 * 130, (size_t) 64 << 20);
    const bool flash_only = g_flash && c.D == 128 && (!getenv("IK2_RELB") || atoi(getenv("IK2_RELB")));
    if (flash_only && g_fast_want) W.lg_cap = std::max((size_t)64 << 20, (size_t)8 * c.H * 128 * 130);
#ifdef STRATA_USE_HIP
    if (flash_only && g_fast_want) {
        const size_t flash = std::min<size_t>(Tm, Work::kRelbRows) * c.H * 130 * 16;
        const size_t short_attention = size_t{8} * c.H * 130 * 128;
        const size_t expert = (S8 * std::max<size_t>(C, c.F) * sizeof(uint16_t) + sizeof(float) - 1) / sizeof(float);
        W.lg_cap = std::max({flash, short_attention, expert});
    }
#endif
    W.lg = work_alloc<float>(W.lg_cap, 72ULL << 20);
    W.blocks = work_alloc<float>((size_t)(c.n_layer / c.block + 1) * T_buf * C);
    W.x16 = work_alloc<uint16_t>(Tm * std::max(C, HD)), W.xk16 = work_alloc<uint16_t>(Tm * C),
    W.xv16 = work_alloc<uint16_t>(Tm * C);
    if (W.T_sup > 0 || dual()) W.xq16_sup = work_alloc<uint16_t>(T_buf * C);
    if (!getenv("IK2_RELB") || std::atoi(getenv("IK2_RELB"))) {
        W.relb_ext = std::max(c.extent, c.extent_swa);
        const size_t rows = g_fast_want ? std::min<size_t>(Tm, Work::kRelbRows) : Work::kRelbRows;
        W.relb = work_alloc<float>(rows * c.H * W.relb_ext);
    }
    if (byte_cache(false) && !(flash_only && g_fast_want)) {
        W.kv_tmp_cap = std::min(ctx, 32768);
        W.kv_k16 = work_alloc<uint16_t>((size_t)W.kv_tmp_cap * KD);
        W.kv_v16 = work_alloc<uint16_t>((size_t)W.kv_tmp_cap * KD);
    }
    if (g_sparse_decode) W.sparse_pages = work_alloc<int>(c.KV * k::kSparsePages);
    W.pos = work_alloc<int>(T_buf);
    W.rlog = work_alloc<float>(T_buf * c.E), W.route_d = work_alloc<int>(S_buf * 2);
    W.ex16 = work_alloc<uint16_t>(T_buf * C),
    W.w16 = work_alloc<uint16_t>((size_t)2 * c.F * C); // (one expert's tokens)
    W.h16 = work_alloc<uint16_t>(std::max({size_t(64), T_buf, size_t(batch_rows * c.K)}) * c.F);
    W.gu32_bytes = 4 * std::max({Tm * C, // two FP16 limbs per input element
                  S8 * 2 * (size_t) c.F,
                  Tm * std::max<size_t>(2 * HD + 2 * KD + (size_t) c.H * c.d_rel + c.H,
                                        (2 * HD + (size_t) c.H * c.d_rel + c.H) + 2 * (KD + (size_t) c.KV * 4)),
                  T_buf * 2 * (size_t) c.F, (W.lg_cap + 1) / 2});
    W.gu32 = work_alloc<float>(W.gu32_bytes / 4);
    W.qkv = W.gu32;
    W.lg16 = (uint16_t*) W.gu32;
    W.y32 = work_alloc<float>(std::max({size_t(64), T_buf, size_t(batch_rows * c.K)}) * C);
    if (g_tp)
        W.tp_h = work_alloc<uint16_t>(std::max<size_t>(64, S_buf) *
                                      c.F); // (one expert's tokens; the token path's <= 64 rows)
    W.wtr = work_alloc<float>(S_buf);
    W.sg = work_alloc<float>(S_buf), W.su = work_alloc<float>(S_buf), W.wt = work_alloc<float>(S_buf);
    W.gidx = work_alloc<int>(S_buf), W.inv = work_alloc<int>(S_buf);
    size_t token_entries = S8;
#ifndef STRATA_USE_HIP
    const char* group_moe = getenv("IK2_BATCH_GROUP_MOE");
    if (batch_draft || (batch_width > 1 && group_moe && std::atoi(group_moe))) {
        W.batch_group_mma = group_moe && std::atoi(group_moe) == 2;
        const size_t entries = size_t(std::min(batch_rows * c.K, k::kGroupRouteSlots)) * k::kGroupMax;
        token_entries = std::max(token_entries, entries);
        W.batch_group_row = work_alloc<int>(entries);
    }
#endif
    W.blob = work_alloc<const uint8_t*>(S8), W.tok = work_alloc<int>(token_entries);
    for (int i = 0; i < Work::kPackRing; ++i) {
        CU(cudaHostAlloc(&W.pack_h[i], 8192, cudaHostAllocPortable));
        W.pack_d[i] = work_alloc<uint8_t>(8192);
        if (g_pool) CU(cudaHostAlloc(&W.y_h[i], (size_t) 64 * C * 4, cudaHostAllocPortable));
    }
    W.T_host = (int) std::max<size_t>(T_buf, W.T_sup > 0 ? (size_t) ctx : 0);   // (a lent scratch can take up to ctx)
    const size_t S_host = (size_t) W.T_host * c.K;
    for (int i = 0; i < 2; ++i) {
        CU(cudaHostAlloc(&W.gidx_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.inv_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.sg_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.su_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wt_h[i], S_host * 4, cudaHostAllocPortable));
        CU(cudaHostAlloc(&W.wtr_h[i], S_host * 4, cudaHostAllocPortable));
    }
    CU(cudaHostAlloc(&W.x_h, (size_t) 8 * C * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.emb_h, Tm * C * 4, cudaHostAllocPortable));
    if (draft_mask) {
        embed_row(M, ik2::draft::kDFlash2MaskToken, W.tmp, draft_mask);
        CU(cudaStreamSynchronize(g_s));
    }
    CU(cudaEventCreateWithFlags(&W.prefill_inputs_copied, cudaEventDisableTiming));
    CU(cudaHostAlloc(&W.pos_h, (size_t) W.T_host * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.route_h, S_host * 8, cudaHostAllocPortable));
    W.argmax_d = work_alloc<int>(Tm);
    CU(cudaHostAlloc(&W.argmax_h, Tm * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.logits_h, (size_t) c.n_rows * 4, cudaHostAllocPortable));
    CU(cudaHostAlloc(&W.cand_h, (size_t) (2 + 2 * kCandCap) * 4, cudaHostAllocPortable));
    {   // (IK2_RG_SLOTS: the slots - tokens x picks - up to which a prompt's experts go by groups; 0: never)
        // (96 tokens by default: past ~128 the FP16 GEMMs are the faster way)
        const int want = getenv("IK2_RG_SLOTS") ? std::atoi(getenv("IK2_RG_SLOTS")) : 96 * c.K;
        W.rg_slots = std::max(0, std::min(want, (int) std::max<size_t>(64, T_buf)));
        W.rg_groups = c.E + W.rg_slots / k::kGroupMax;
        W.rg_bytes = (size_t) W.rg_groups * (8 + 4 + 8 * k::kGroupMax) + (size_t) W.rg_slots * 16;
        CU(cudaHostAlloc(&W.rg_h, W.rg_bytes, cudaHostAllocPortable));
        W.rg_d = work_alloc<uint8_t>(W.rg_bytes);
    }
    W.cand_d = work_alloc<int>(2 + 2 * kCandCap);
    W.cand_part = work_alloc<int>(k::cand_scratch(c.n_vocab));
    const int n_ev = (c.E + Work::kMissBatch - 1) / Work::kMissBatch;
    W.miss_ev.resize(n_ev);
    for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.miss_ev[i], cudaEventDisableTiming));
    for (int r = 0; r < Work::kPfRing; ++r) {
        W.pf_evs[r].resize(n_ev);
        for (int i = 0; i < n_ev; ++i) CU(cudaEventCreateWithFlags(&W.pf_evs[r][i], cudaEventDisableTiming));
    }
    W.cnt.assign(c.E + 1, 0);
    W.start.assign(c.E + 1, 0);
    W.fill.assign(c.E + 1, 0);
    W.hit_exps.reserve(c.E);
    W.miss_exps.reserve(c.E);
    W.slot_ptrs.assign(c.E, nullptr);
    W.seen_stamp.assign(c.E, 0);
    W.gi_stamp.assign(c.E, 0);
    W.seen_late.assign(c.E, 0);
    W.gi_idx.assign(c.E, 0);
#ifdef STRATA_USE_HIP
    if (suffix_window >= 32) W.head_rows = std::min(64, suffix_window / 16 * 16);
#endif
    W.head_rows = std::max(W.head_rows, batch_rows);
    W.logits = dalloc<float>((size_t) W.head_rows * c.n_rows);
    if (c.n_pass > 1) {
        W.lp_pre = work_alloc<float>(Tm * C);
        W.lp_bank = work_alloc<float>((size_t)(n_core / c.block) * Tm *
                                      C); // (the pass's blocks after its first, its final stream)
        W.lp_msg = work_alloc<float>(4 * Tm * C);
        W.lp_msg_all = dalloc<float>((size_t) n_core * 4 * C * batch_rows);
        W.lp_in = work_alloc<float>(3 * Tm * C);
        W.lp_g = work_alloc<float>(3 * Tm * (2 * (size_t)c.rank + 2));
        W.lp_a = work_alloc<float>(3 * Tm * (size_t)c.rank);
        W.lp_f = work_alloc<float>(2 * Tm * C);
    }
    for (auto& Vi : M.V) {   // the conv histories start at zero
        if (M.L[Vi.l].dev != d) continue;
        CU(cudaMemset(Vi.h_k, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_v, 0, 3 * KD * 4));
        CU(cudaMemset(Vi.h_attn, 0, 3 * C * 4));
        CU(cudaMemset(Vi.h_mlp, 0, 3 * C * 4));
    }
    ExpertCache& EC = g_dev[d].EC;
    CU(cudaMemGetInfo(&fr, &tot));
    fr = free_for_budget(fr);
    // (what is left free: 2 GiB beside a cache that streams experts in and out; a quarter of one when every expert
    //  stays in VRAM - nothing more is allocated but a token's graph)
    const size_t room = g_fast_want ? (size_t) 256 << 20 : (size_t) 2 << 30;
    const size_t budget = std::min(cache_limit, fr > room ? fr - room : (size_t) 0);
    EC.tp = g_tp ? d : -1;
    size_t prefill_budget = budget;
    if (gpu_served) {
        size_t largest = 0;
        for (const Layer& ly : M.L) largest = std::max(largest, (ly.slot_bytes + 255) / 256 * 256);
        prefill_budget = std::min(budget, largest * c.E);
    }
    EC.init(M.L, dual() ? -1 : d, c.n_layer, c.E, prefill_budget);
    if (resident_only && EC.total_slots() != c.n_layer * c.E)
        die("resident-only loading: insufficient VRAM after workspace allocation; reduce --ctx or --chunk");
    std::printf("expert cache, GPU %d: %d slots across %zu pools (%.1f GiB); load %.1f s in all\n", d, EC.total_slots(),
                EC.pools.size(), EC.total_bytes() / 1073741824.0, now() - t_start);
    for (size_t p = 0; p < EC.pools.size(); ++p)
        std::printf("  pool %zu: %d slots of %.1f MiB (%.1f GiB, %d layers)\n", p, EC.pools[p].n_slots,
                    EC.pools[p].slot_bytes / 1048576.0,
                    (double) EC.pools[p].n_slots * EC.pools[p].slot_bytes / 1073741824.0, EC.pools[p].n_layers);
    if ((!counts.empty() && (!g_fast_want || warm)) || g_ep) {
        // Expert-parallel prefetch starts with uniform popularity on a cold run.
        std::vector<uint64_t> uniform;
        if (counts.empty()) uniform.assign(size_t(c.n_layer) * c.E, 1);
        const auto& cnt = counts.empty() ? uniform : counts;
        const double tw = now();
        const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
        int n = 0;
        for (size_t p = 0; p < EC.pools.size(); ++p) {
            std::vector<int> idx;
            for (int l = 0; l < c.n_layer; ++l)
                if (EC.layer_pool[l] == (int) p)
                    for (int e = 0; e < c.E; ++e)
                        if (g_ep ? ep_owner(l, e) == d : cnt[l * c.E + e] > 0) idx.push_back(l * c.E + e);
            // the most picked first; the chosen ones then go in least picked first, so that the evictions that
            // start from the oldest take the least picked
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
            const int k = std::min((int) idx.size(), EC.pools[p].n_active);
            for (int i = k - 1; i >= 0; --i) {
                const int l = idx[i] / c.E, e = idx[i] % c.E;
                EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true);
                if (!counts.empty())
                    EC.pools[p].freq[EC.loc[idx[i]]] = std::min<uint64_t>(2, cnt[idx[i]]);
                ++n;
            }
        }
        CU(cudaStreamSynchronize(g_cs));
        ++EC.tick;   // (taken in the tick the first prefetch uses, they would all count as in use: none evictable)
        // the profile also seeds the prefetch's picture of which experts a pass uses (W.use, as 100k picks a
        // layer): without it the first long-enough request prefetches all 18k experts
        W.use.assign((size_t) c.n_layer * c.E, 0), W.use_tot.assign(c.n_layer, 0);
        for (int l = 0; l < c.n_layer; ++l) {
            long double sum = 0;
            for (int e = 0; e < c.E; ++e) sum += cnt[(size_t) l * c.E + e];
            if (sum <= 0) continue;
            for (int e = 0; e < c.E; ++e) {
                W.use[(size_t) l * c.E + e] = (uint64_t) (100000.0L * cnt[(size_t) l * c.E + e] / sum + 0.5L);
                W.use_tot[l] += W.use[(size_t) l * c.E + e];
            }
        }
        std::printf("warm: %d experts (%.1f GiB) copied in %.1f s\n", n, (EC.bytes_up - b0) / 1073741824.0,
                    now() - tw);
        EC.misses = m0, EC.bytes_up = b0;
    }
    if (g_fast_want && d == 0) {   // the resident path: every expert in now, and where each is on the device
        auto blocks = [](const Mat& m) { return m.raw || m.q8; };   // (kept as the GGUF's blocks, whole or planar)
        bool all = !EC.pools.empty(), raw = blocks(M.out_w);
        for (const auto& P : EC.pools) all = all && P.n_active >= P.n_layers * c.E;
        for (const auto& Ly : M.L)
            raw = raw && blocks(Ly.wo) && (Ly.sliding ? blocks(Ly.wqkvrg) : (blocks(Ly.wqrg) && blocks(Ly.wk) && blocks(Ly.wv)));
        if (gpu_served) {
            if (!raw) die("GPU-served expert decode requires dense block weights");
            served.init(M, budget - EC.total_bytes());
            g_fast = true;
        } else if (all && raw) {
            const double tw = now();
            const uint64_t m0 = EC.misses, b0 = EC.bytes_up;
            int n = 0;
#ifndef STRATA_USE_HIP
            bool use_fast_load = false;
            constexpr size_t kStageBytes = 64ULL << 20;
            uint8_t* stage[2] = {};
            cudaEvent_t stage_done[2] = {};
            bool stage_used[2] = {false, false};
            int stage_idx = 0;
            const int pack_threads = std::max(1, std::min(threads > 0 ? threads : 8, 12));
            const char* fast_load = getenv("IK2_FAST_LOAD");
            if (host_mmap && EC.tp < 0 && (!fast_load || atoi(fast_load) != 0)) {
                if (cudaHostAlloc(&stage[0], kStageBytes, cudaHostAllocDefault) == cudaSuccess &&
                    cudaHostAlloc(&stage[1], kStageBytes, cudaHostAllocDefault) == cudaSuccess) {
                    CU(cudaEventCreateWithFlags(&stage_done[0], cudaEventDisableTiming));
                    CU(cudaEventCreateWithFlags(&stage_done[1], cudaEventDisableTiming));
                    use_fast_load = true;
                } else {
                    (void)cudaGetLastError();
                    if (stage[0]) cudaFreeHost(stage[0]);
                    stage[0] = stage[1] = nullptr;
                }
            }
            for (int l = 0; l < c.n_layer; ++l) {
                const Layer& ly = M.L[l];
                const size_t pitch = EC.pools[EC.layer_pool[l]].slot_bytes;
                const bool bulk = use_fast_load && pitch <= kStageBytes;
                // Warm profiles can leave holes. Only combine consecutive missing
                // experts that the allocator placed in consecutive physical slots.
                uint8_t* first = nullptr;
                int start = 0, count = 0;
                auto flush = [&] {
                    if (!count) return;
                    const size_t active_bytes = ly.eb[0] + ly.eb[1] + ly.eb[2];
                    const int chunk_e = std::max(1, (int)(kStageBytes / pitch));
                    for (int off_e = 0; off_e < count; off_e += chunk_e) {
                        const int cnt = std::min(chunk_e, count - off_e);
                        const int b = stage_idx++ & 1;
                        if (stage_used[b]) CU(cudaEventSynchronize(stage_done[b]));
                        uint8_t* sbuf = stage[b];
                        const int nt = std::min(pack_threads, cnt);
                        auto pack_range = [&](int lo, int hi) {
                            for (int k = lo; k < hi; ++k) {
                                uint8_t* dst_e = sbuf + (size_t)k * pitch;
                                size_t off = 0;
                                for (int r = 0; r < 3; ++r) {
                                    std::memcpy(dst_e + off,
                                                ly.host[r] + (size_t)(start + off_e + k) * ly.stride[r],
                                                ly.eb[r]);
                                    off += ly.eb[r];
                                }
                            }
                        };
                        if (nt <= 1) {
                            pack_range(0, cnt);
                        } else {
                            std::vector<std::thread> workers;
                            workers.reserve(nt - 1);
                            for (int t = 0; t < nt; ++t) {
                                const int lo = cnt * t / nt, hi = cnt * (t + 1) / nt;
                                if (t + 1 < nt) workers.emplace_back(pack_range, lo, hi);
                                else pack_range(lo, hi);
                            }
                            for (auto& w : workers) w.join();
                        }
                        CU(cudaMemcpyAsync(first + (size_t)off_e * pitch, sbuf, (size_t)cnt * pitch,
                                           cudaMemcpyHostToDevice, g_cs));
                        CU(cudaEventRecord(stage_done[b], g_cs));
                        stage_used[b] = true;
                        EC.bytes_up += (size_t)cnt * active_bytes;
                    }
                    count = 0;
                };
                for (int e = 0; e < c.E; ++e) {
                    if (EC.has(l, e)) continue;
                    uint8_t* dst = EC.get(ly, l, e, EC.tick, g_cs, nullptr, !bulk, true);
                    ++n;
                    if (!bulk) continue;
                    if (count && (e != start + count || dst != first + count * pitch)) flush();
                    if (!count) first = dst, start = e;
                    ++count;
                }
                flush();
            }
            CU(cudaStreamSynchronize(g_cs));
            if (use_fast_load) {
                CU(cudaEventDestroy(stage_done[0]));
                CU(cudaEventDestroy(stage_done[1]));
                CU(cudaFreeHost(stage[0]));
                CU(cudaFreeHost(stage[1]));
            }
#else
            for (int l = 0; l < c.n_layer; ++l)
                for (int e = 0; e < c.E; ++e)
                    if (!EC.has(l, e)) EC.get(M.L[l], l, e, EC.tick, g_cs, nullptr, true, true), ++n;
            CU(cudaStreamSynchronize(g_cs));
#endif
            ++EC.tick;
            for (int l = 0; l < c.n_layer; ++l) {
                Layer& Ly = M.L[l];
                const auto& P = EC.pools[EC.layer_pool[l]];
                std::vector<const uint8_t*> tab(c.E);
                for (int e = 0; e < c.E; ++e) tab[e] = P.data + (size_t) EC.loc[(size_t) l * c.E + e] * P.slot_bytes;
                if (stage_cold)
                    Ly.tier_cold = dalloc<uint8_t>(c.E);
                Ly.tab_blob = dalloc<const uint8_t*>(c.E);
                CU(cudaMemcpy(Ly.tab_blob, tab.data(), (size_t) c.E * sizeof(void*), cudaMemcpyHostToDevice));
                Ly.tab_h = tab;
                for (int r = 0; r < 3; ++r)
                    if (!Ly.scale[r].empty()) Ly.tab_sc[r] = dev_f32(Ly.scale[r]);
            }
            EC.misses = m0, EC.bytes_up = b0;
            g_fast = true;
            std::printf("resident: every expert in VRAM (%d copied in %.1f s); tokens take the resident path%s\n", n,
                        now() - tw, g_graph ? ", one graph replay per decode step" : "");
        } else {
            if (resident_only) die("resident-only loading: the dense matrix layout cannot use the resident decoder");
            std::printf("resident path off: %s\n", all ? "a dense matrix is not kept as blocks" : "the cache does not hold every expert");
        }
    }
    std::fflush(stdout);
    }
    if (!g_profile_out.empty()) {
        g_pop = std::move(counts);
        if (g_pop.empty()) g_pop.resize(size_t(c.n_layer) * c.E);
        if (g_fast && !output) {
            g_pop_device = dalloc<unsigned long long>(g_pop.size());
            CU(cudaMemset(g_pop_device, 0, g_pop.size() * sizeof(*g_pop_device)));
        }
    }
    if (g_sparse_decode) {
        if (!g_fast || g_prof || g_rec || g_trace || host_profile() || !dump.empty())
            die("IK2_SPARSE_DECODE requires the resident decoder without tracing, profiling or dumps");
        std::fprintf(stderr, "sparse decode enabled: 224 x 64 global keys from position 32768; dense prefill\n");
    }
#ifdef STRATA_USE_HIP
    if ((drafter || suffix_window) && g_block) {
        const bool planar = M.out_w.q8 && std::all_of(M.L.begin(), M.L.end(), [](const Layer& ly) {
            return ly.wo.q8 && (ly.sliding ? ly.wqkvrg.q8 != nullptr : ly.wqrg.q8 && ly.wk.q8 && ly.wv.q8);
        });
        if (!planar) {
            g_block = false;
            std::fprintf(stderr, "HIP block verification requires planar Q8 projections and output head; using ordinary verification\n");
        }
    }
    if (serve && drafter && !(g_fast && g_block))
        die("HIP drafter serving requires all experts resident and planar Q8 projections and head");
#endif
    // Keep space for allocations made on demand, including the image encoder.
    const size_t reserve =
        (size_t)(getenv("IK2_HEADROOM_MIB") ? std::max(0, atoi(getenv("IK2_HEADROOM_MIB"))) : 1536) << 20;
#ifdef STRATA_USE_HIP
    if (g_fast && getenv("IK2_PREFILL_DENSE") && atoi(getenv("IK2_PREFILL_DENSE"))) {
        std::vector<Mat*> projections;
        size_t elements = 0;
        const bool selective = atoi(getenv("IK2_PREFILL_DENSE")) == 2;
        for (auto& layer : M.L) {
            for (Mat* m : {&layer.wqkvrg, &layer.wqrg, &layer.wk, &layer.wv, &layer.wo}) {
                const bool fused = m == &layer.wqkvrg || m == &layer.wqrg;
                if (m->q8 && !m->f16 && (!selective || fused)) {
                    projections.push_back(m);
                    elements += (size_t)m->N * m->K;
                }
            }
        }
        size_t free_bytes, total_bytes;
        CU(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (elements && elements * sizeof(uint16_t) + reserve <= free_bytes) {
            auto* dst = dalloc<uint16_t>(elements);
            for (auto* m : projections) {
                const size_t n = (size_t)m->N * m->K;
                m->f16 = dst;
                k::decode_q8p_f16(m->q8, m->q8d, n, dst, g_s);
                dst += n;
            }
            CU(cudaStreamSynchronize(g_s));
            std::fprintf(stderr, "prefill dense projections: %zu FP16 matrices, %.0f MiB\n",
                         projections.size(), elements / 524288.0);
        } else {
            std::fprintf(stderr, "prefill dense projections: using per-call dequantization (%.0f MiB available after headroom)\n",
                         free_bytes > reserve ? (free_bytes - reserve) / 1048576.0 : 0.0);
        }
    }
#endif
    if (g_managed_experts) {
        CU(cudaStreamSynchronize(g_s));
        size_t remaining = 0, total = 0;
        CU(cudaMemGetInfo(&remaining, &total));
        g_kv_room = remaining > reserve ? remaining - reserve : 0;
    }
    // the caches' counters, summed over the GPUs
    auto ec_sum = [&](auto f) {
        double v = 0;
        for (auto& D : g_dev) v += (double) f(D.EC);
        return v;
    };

    if (serve) {
        if (eos_ids.empty()) {
            if (const auto* e = S.get("tokenizer.ggml.eos_token_id")) eos_ids.push_back((int) e->u);
            eos_ids.push_back(199999);   // (Inkling 2's <|endoftext|>)
        }
        std::vector<uint8_t> embd_owned;
        if (g_fast && host_mmap && !g_managed_experts) {
            const size_t embd_bytes = (size_t) c.n_vocab * (c.C / k::block_values(M.embd_type)) * k::block_bytes(M.embd_type);
            embd_owned.assign(M.embd, M.embd + embd_bytes);
            M.embd = embd_owned.data();
            for (auto& Ly : M.L) Ly.host[0] = Ly.host[1] = Ly.host[2] = nullptr;
            S.by_name.clear();
            S.files.clear();
        }
        if (batch_width > 1) {
            serve_sequences(M, batch_width, ctx, chunk, image_token, eos_ids, drafter, draft_head);
            pop_dump();
            return 0;
        }
        // --serve: requests on stdin, a line each - "GEN max_new temperature top_p seed stop,stop ids,ids,..." - and
        // the answer on stdout as "@TOK id" lines then "@END reason prompt_tokens cached_tokens prompt_s gen_s".  The
        // tokens whose state the model holds are kept: a request that starts with them runs only what follows.
        double t_exp = 0;
        // Snapshots of every short conv's history (plus sliding-layer KV windows in host RAM) at the end of a
        // request's prompt (before its generation): an agent's next turn usually agrees with the last one up to
        // there and no further (the model's own reply comes back re-rendered, its thinking dropped), so the state
        // goes back to that point instead of the start. Full-layer KV needs nothing: positions rewrite cleanly.
        struct Snap {
            int pos = -1;
            std::vector<float*> h;   // [layer * 4 + conv]
            std::vector<std::vector<uint8_t>> kv_k, kv_v;
        };
        struct Conversation {
            std::vector<int> hist;
            std::vector<uint64_t> hist_sig;
            std::vector<float> last;
            std::vector<Snap> snaps;
            int snap_next = 0;
            uint64_t used = 0;

            int checkpoint(size_t shared, size_t length) const {
                int best = -1;
                for (size_t i = 0; i < snaps.size(); ++i) {
                    const auto pos = snaps[i].pos;
                    if (pos >= 0 && (size_t)pos <= shared && (size_t)pos < length &&
                        (best < 0 || pos > snaps[best].pos)) best = (int)i;
                }
                return best;
            }
        };
        std::vector<Conversation> conversations(cache_slots);
        int active_slot = 0;
        uint64_t request_order = 0;
        for (auto& convo : conversations) convo.snaps.resize(std::max(1, 8 / cache_slots));
        for (auto& convo : conversations) for (auto& sn : convo.snaps) {
            sn.kv_k.resize(M.V.size());
            sn.kv_v.resize(M.V.size());
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                auto& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                const size_t cc[4] = {KD, KD, C, C};
                for (int i = 0; i < 4; ++i) sn.h.push_back(dalloc<float>(3 * cc[i]));
            }
        }
        auto snap_copy = [&](Snap& sn, bool save) {
            const size_t cc[4] = {KD, KD, C, C};
            const cudaMemcpyKind kv_dir = save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            const size_t row_b = kv_row_bytes(KD, true);
            for (size_t l = 0; l < M.V.size(); ++l) {   // (l: a visit)
                Layer& Ly = bind_visit(M, (int) l);
                use_dev(Ly.dev);
                float* hs[4] = {Ly.h_k, Ly.h_v, Ly.h_attn, Ly.h_mlp};
                for (int i = 0; i < 4; ++i) {
                    float* sv = sn.h[l * 4 + i];
                    CU(cudaMemcpyAsync(save ? sv : hs[i], save ? hs[i] : sv, 3 * cc[i] * 4, cudaMemcpyDeviceToDevice, g_s));
                }
                if (sn.pos > 0 && Ly.sliding && Ly.kv_cap < ctx) {
                    const int w = std::min(sn.pos, c.window);
                    const int k_start = sn.pos - w;
                    const int s0 = k_start % Ly.kv_cap;
                    const int n1 = std::min(w, Ly.kv_cap - s0);
                    const int n2 = w - n1;
                    uint8_t* dk = Ly.kc8 ? Ly.kc8 : (uint8_t*) Ly.kc;
                    uint8_t* dv = Ly.vc8 ? Ly.vc8 : (uint8_t*) Ly.vc;
                    const size_t bytes = (size_t)w * row_b;
                    if (save && sn.kv_k[l].size() < bytes) {
                        sn.kv_k[l].resize(bytes);
                        sn.kv_v[l].resize(bytes);
                    }
                    uint8_t* hk = sn.kv_k[l].data();
                    uint8_t* hv = sn.kv_v[l].data();
                    CU(cudaMemcpyAsync(save ? hk : dk + (size_t) s0 * row_b, save ? dk + (size_t) s0 * row_b : hk,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    CU(cudaMemcpyAsync(save ? hv : dv + (size_t) s0 * row_b, save ? dv + (size_t) s0 * row_b : hv,
                                       (size_t) n1 * row_b, kv_dir, g_s));
                    if (n2 > 0) {
                        CU(cudaMemcpyAsync(save ? hk + (size_t) n1 * row_b : dk,
                                           save ? dk : hk + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                        CU(cudaMemcpyAsync(save ? hv + (size_t) n1 * row_b : dv,
                                           save ? dv : hv + (size_t) n1 * row_b, (size_t) n2 * row_b, kv_dir, g_s));
                    }
                }
            }
            for (auto& D : g_dev) CU(cudaStreamSynchronize(D.s));
        };
        auto reset = [&](Conversation& slot) {
            for (auto& sn : slot.snaps) sn.pos = -1;
            for (int vis = 0; vis < (int) M.V.size(); ++vis) {
                Layer& Ly = bind_visit(M, vis);
                use_dev(Ly.dev);
                CU(cudaMemset(Ly.h_k, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_v, 0, 3 * KD * 4));
                CU(cudaMemset(Ly.h_attn, 0, 3 * C * 4));
                CU(cudaMemset(Ly.h_mlp, 0, 3 * C * 4));
            }
            slot.hist.clear();
            slot.hist_sig.clear();
            slot.last.clear();
            slot.used = 0;
        };
        auto parse_ints = [](const std::string& t) {
            std::vector<int> v;
            for (const char* p = t.c_str(); *p;) {
                if (*p == '-' && p[1] == 0) break;
                v.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
            return v;
        };
        if (resident_only) {
            size_t free_bytes, total_bytes;
            CU(cudaMemGetInfo(&free_bytes, &total_bytes));
            std::printf("resident server: %.0f MiB of VRAM free before graph capture\n", free_bytes / 1048576.0);
        }
#ifndef _WIN32
        std::signal(SIGUSR1, on_usr1);
#endif
        // Maya's protocol (serve/server.py's StrataEngine): "GEN <max_new> [key=value ...] <ids>" in; "PP ..." per
        // prompt pass, "T <id>" per token and "DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel>
        // 0 0 <reused>" out; "STOP" mid-request ends it at the next token (or between the prompt's passes), "QUIT"
        // ends the engine.  The tokens a request ends at: --eos, else the GGUF's eos and <|endoftext|>.
        if (serve_maya) {
            std::printf("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d cache_slots=%d\n", kv_format_name(), ctx, n_gpu, cache_slots);
            std::printf("READY %d stop\n", ctx);
        } else {
            std::printf("@READY ctx %d\n", ctx);
        }
        std::fflush(stdout);
        // stdin on a thread: a STOP while a request runs is seen at once (the request loop looks at g_cancel)
        // (static: the reader thread is detached and may outlive this scope at QUIT)
        static std::mutex in_m;
        static std::condition_variable in_cv;
        static std::deque<std::string> in_q;
        static bool in_eof = false;
        static std::atomic<bool> busy{false};
        std::thread([]() {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    if (busy) g_cancel = 1;
                    continue;
                }
                std::lock_guard<std::mutex> lk(in_m);
                in_q.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_m);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& l) {
            std::unique_lock<std::mutex> lk(in_m);
            in_cv.wait(lk, [&] { return !in_q.empty() || in_eof; });
            if (in_q.empty()) return false;
            l = std::move(in_q.front());
            in_q.pop_front();
            return true;
        };
        std::string line;
        std::mt19937_64 rng(1234);
        // speculative decoding (--draft, the resident path): the controller decides each step; IK2_DRAFT_WIDTH the
        // widest window, IK2_DRAFT_ADAPT=0 always that wide
        SpecCtl sctl;
        const bool spec_on = drafter && g_fast && g_block;
        sctl.wmax = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, kBlockMax) : 5;
        sctl.adapt = !(getenv("IK2_DRAFT_ADAPT") && !atoi(getenv("IK2_DRAFT_ADAPT")));
        while (next_line(line)) {
            std::istringstream in(line);
            std::string cmd, stops_s, ids_s, emb_path;
            int max_new = 0, top_k = 0;
            double temp = 0, top_p = 1;
            long long seed = -1;
            in >> cmd;
            if (cmd == "QUIT") {
                break;
            }
            // GENI: GEN with the request's pictures - "GENI <max_new> [key=value ...] <pictures file> <ids>", the file
            // the image encoder's rows (SVE1 records back to back), one row an image token of the ids in order
            const bool geni = cmd == "GENI";
            if (cmd != "GEN" && !geni) {   // (Maya's PAUSE / WRAP and the like: not offered in READY, so not sent)
                if (!serve_maya) {
                    std::printf("@END error unknown_command 0 0 0 0\n");
                    std::fflush(stdout);
                }
                continue;
            }
            std::vector<int> stops;
            if (serve_maya) {
                stops = eos_ids;
                in >> max_new;
                std::string w;
                while (in >> w) {
                    const size_t eq = w.find('=');
                    if (eq == std::string::npos) {
                        if (geni && emb_path.empty() && ids_s.empty() && w.find_first_not_of("0123456789,-") != std::string::npos)
                            emb_path = w;   // (the pictures' file comes before the ids)
                        else
                            ids_s = w;
                        continue;
                    }
                    const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                    if (k == "temperature") temp = std::atof(v.c_str());
                    else if (k == "top_p") top_p = std::atof(v.c_str());
                    else if (k == "top_k") top_k = std::atoi(v.c_str());
                    else if (k == "seed") seed = std::atoll(v.c_str());
                    else if (k == "stop_token") stops.push_back(std::atoi(v.c_str()));
                }
            } else {
                in >> max_new >> temp >> top_p >> seed >> stops_s >> ids_s;
                stops = parse_ints(stops_s);
            }
            const std::vector<int> ids = parse_ints(ids_s);
            if (ids.empty() || (int) ids.size() + max_new > ctx) {
                if (serve_maya)
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n",
                                ids.empty() ? "an empty prompt" :
                                    ("the prompt and max_tokens pass the context (" + std::to_string(ctx) + ")").c_str(),
                                ids.size());
                else
                    std::printf("@END error %s %zu 0 0 0\n", ids.empty() ? "empty_prompt" : "context_length", ids.size());
                std::fflush(stdout);
                continue;
            }
            g_img.clear();
            std::vector<uint64_t> sig(ids.size(), 0);
            if (!emb_path.empty()) {
                const std::string err = load_pictures(emb_path, ids, image_token, c.C, g_img, sig);
                if (!err.empty()) {
                    g_img.clear();
                    std::printf("ERR %s\nDONE 0 %zu 0 0 cancel 0 0 0\n", err.c_str(), ids.size());
                    std::fflush(stdout);
                    continue;
                }
            }
            g_cancel = 0;
            busy = true;
            if (seed >= 0) rng.seed((uint64_t) seed);
            size_t L = 0, best_reusable = 0;
            int selected = active_slot, restore = -1;
            for (int i = 0; i < cache_slots; ++i) {
                const auto& candidate = conversations[i];
                size_t shared = 0;
                while (shared < candidate.hist.size() && shared < ids.size() && candidate.hist[shared] == ids[shared] &&
                       (shared < candidate.hist_sig.size() ? candidate.hist_sig[shared] : 0) == sig[shared])
                    ++shared;
                const bool can_append = shared == candidate.hist.size() &&
                                        (shared < ids.size() || !candidate.last.empty());
                const int checkpoint = can_append ? -1 : candidate.checkpoint(shared, ids.size());
                const size_t reusable = can_append ? shared : checkpoint < 0 ? 0 : candidate.snaps[checkpoint].pos;
                const bool preferred = reusable ? candidate.used > conversations[selected].used
                                                : candidate.used < conversations[selected].used;
                if (i == 0 || reusable > best_reusable || (reusable == best_reusable && preferred)) {
                    restore = checkpoint;
                    selected = i;
                    L = shared;
                    best_reusable = reusable;
                }
            }
            if (selected != active_slot) {
                slot_storage.exchange(active_slot, selected, conversations[active_slot].hist.size(),
                                      conversations[selected].hist.size());
                active_slot = selected;
            }
            if (restore >= 0 && cache_slots > 1) {
                // Keep both descendants when there is space. With every slot occupied,
                // rewind the matching chat in place so its retry cannot evict a different chat.
                auto empty = std::find_if(conversations.begin(), conversations.end(),
                                          [](const Conversation& c) { return c.hist.empty(); });
                if (empty != conversations.end()) {
                    const int target = (int)(empty - conversations.begin());
                    auto& source = conversations[selected];
                    auto& checkpoint = source.snaps[restore];
                    slot_storage.preserve(selected, target, source.hist.size());
                    snap_copy(checkpoint, false);
                    L = checkpoint.pos;
                    empty->hist.assign(source.hist.begin(), source.hist.begin() + L);
                    empty->hist_sig.assign(source.hist_sig.begin(), source.hist_sig.begin() + std::min(L, source.hist_sig.size()));
                    empty->last.clear();
                    for (auto& sn : empty->snaps) sn.pos = -1;
                    active_slot = target;
                }
            }
            auto& slot = conversations[active_slot];
            auto& hist = slot.hist;
            auto& hist_sig = slot.hist_sig;
            auto& last = slot.last;
            auto& snaps = slot.snaps;
            auto& snap_next = slot.snap_next;
            if (L < hist.size() || L == ids.size()) {
                // back to the latest snapshot at or before L that leaves a token to run (one past it is dropped:
                // its position's state follows from a prefix the request does not share)
                if (L == hist.size() && L == ids.size() && !last.empty()) {
                    // (the same tokens again: the last logits answer)
                } else if (restore >= 0) {
                    const int sp = snaps[restore].pos;
                    snap_copy(snaps[restore], false);
                    for (auto& sn : snaps)
                        if (sn.pos > sp) sn.pos = -1;
                    hist.resize(sp);
                    if (hist_sig.size() > (size_t) sp) hist_sig.resize(sp);
                    last.clear();
                    L = sp;
                } else {
                    reset(slot), L = 0;
                }
            }
            const int cached = (int) L;
            const double t0 = now();
            const uint64_t miss0 = dev_cache(0).misses, bytes0 = dev_cache(0).bytes_up;
            for (auto& v : g_cnt_hist) v = 0;
            for (auto& v : g_src_hist) v = 0;
            bool cancelled = false;
            if (tier_active && ids.size() - L >= (size_t)2 * chunk) replace_expert_pools(M, false);
            if (L < ids.size()) {
                const int T_sup = g_dev[0].W.T_sup;
                const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
                if (T_sup > 0 && (n_gpu == 1 || dual()) && (int) (ids.size() - L) > 64) {
                    BigScratch bs;
                    const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) (ids.size() - L), (int) ids.size(), bs);
                    for (size_t p0 = L; p0 < ids.size(); p0 += T_pass) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(T_pass, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward_prefill_sup(M, ids.data() + p0, n, (int) p0, sub_chunk,
                                            last_c ? OutMode::kLogits : OutMode::kNone, lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (serve_maya) {   // (read, total, ms, tok/s, reused)
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), (int) L);
                            std::fflush(stdout);
                        }
                    }
                    big_end(g_dev[0].W, dev_cache(0), bs);
                } else {
                    for (size_t p0 = L; p0 < ids.size(); p0 += chunk) {
                        if (g_cancel) {   // (a prompt half run: nothing of it is kept)
                            cancelled = true;
                            break;
                        }
                        const int n = (int) std::min<size_t>(chunk, ids.size() - p0);
                        const bool last_c = p0 + n == ids.size();
                        std::vector<float> lo;
                        forward(M, ids.data() + p0, n, (int) p0, false, last_c ? OutMode::kLogits : OutMode::kNone,
                                std::string(), lo, nullptr, t_exp);
                        if (last_c) last = std::move(lo);
                        if (drafter) {   // (the drafter's context for these positions, from their taps)
                            ik2::draft::prefill_context(drafter, g_taps, g_taps_ld, n, (int) p0, ids.size());
                            if (last_c) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
                        }
                        if (serve_maya && ids.size() - L > (size_t)chunk) {
                            const double ms = (now() - t0) * 1e3;
                            std::printf("PP %zu %zu %.0f %.1f %d\n", p0 + n, ids.size(), ms,
                                        (p0 + n - L) / std::max(1e-3, ms / 1e3), cached);
                            std::fflush(stdout);
                        }
                    }
                }
            }
            if (cancelled) {
                reset(slot);
                if (serve_maya) std::printf("DONE 0 %zu %.1f 0 cancel 0 0 %d\n", ids.size(), (now() - t0) * 1e3, cached);
                else std::printf("@END cancelled %zu %d %.3f %.3f\n", ids.size(), cached, now() - t0, 0.0);
                std::fflush(stdout);
                busy = false;
                continue;
            }
            if (L < ids.size()) {
                hist = ids;
                hist_sig = sig;
                // Branching invalidates newer snapshots. Reuse an empty slot before evicting a shared prefix.
                for (size_t i = 0; i < snaps.size(); ++i) {
                    const int slot = (snap_next + (int)i) % (int)snaps.size();
                    if (snaps[slot].pos < 0) {
                        snap_next = slot;
                        break;
                    }
                }
                Snap& sn = snaps[snap_next];
                snap_next = (snap_next + 1) % (int) snaps.size();
                sn.pos = (int) hist.size();
                snap_copy(sn, true);
            }
            place_decode(M, (int)ids.size());
            slot.used = ++request_order;
            const double t1 = now();
            if (getenv("IK2_STATS"))
                std::printf("prefill stats: %zu tokens run, %llu expert copies (%.1f GiB) in %.2f s; experts that ran 1 / 2 / 3-7 / 8+ "
                            "tokens: %llu / %llu / %llu / %llu\n",
                            ids.size() - (size_t) cached, (unsigned long long) (dev_cache(0).misses - miss0),
                            (dev_cache(0).bytes_up - bytes0) / 1073741824.0, t1 - t0,
                            (unsigned long long) g_cnt_hist[0], (unsigned long long) g_cnt_hist[1],
                            (unsigned long long) g_cnt_hist[2], (unsigned long long) g_cnt_hist[3]);
            if (getenv("IK2_STATS"))
                std::printf("prefill sources: used and cached %llu, used and prefetched %llu, used and copied after the "
                            "routing %llu, prefetched and unused %llu\n",
                            (unsigned long long) g_src_hist[0], (unsigned long long) g_src_hist[1],
                            (unsigned long long) g_src_hist[2], (unsigned long long) g_src_hist[3]);
            // softmax(logits / T) over the top_p mass of the 1000 largest of the (logit, token) pairs given (the
            // whole vocabulary, or the candidates the resident token path found on the device: every logit of the
            // highest bins that hold 1000 - the same 1000 either way; ties: the lower token first)
            auto pick_top = [&](std::vector<std::pair<float, int>>& cd) -> int {
                const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kCandK) : kCandK);
                std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                                  [](const std::pair<float, int>& a2, const std::pair<float, int>& b2) {
                                      return a2.first > b2.first || (a2.first == b2.first && a2.second < b2.second);
                                  });
                std::vector<double> pr(kk);
                double z = 0;
                for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
                double acc = 0;
                int keep = kk;
                for (int j = 0; j < kk; ++j) {
                    acc += pr[j] / z;
                    if (acc >= top_p) {
                        keep = j + 1;
                        break;
                    }
                }
                double zk = 0;
                for (int j = 0; j < keep; ++j) zk += pr[j];
                double u = std::uniform_real_distribution<double>(0, zk)(rng);
                for (int j = 0; j < keep; ++j) {
                    u -= pr[j];
                    if (u <= 0) return cd[j].second;
                }
                return cd[keep - 1].second;
            };
            auto pick = [&](const float* r) -> int {
                if (temp <= 0) return (int) (std::max_element(r, r + c.n_vocab) - r);
                std::vector<std::pair<float, int>> cd(c.n_vocab);
                for (int v = 0; v < c.n_vocab; ++v) cd[v] = {r[v], v};
                return pick_top(cd);
            };
            auto pick_cand = [&]() -> int {
                std::vector<std::pair<float, int>> cd(g_cand.n);
                for (int i = 0; i < g_cand.n; ++i) cd[i] = {g_cand.val[i], g_cand.idx[i]};
                return pick_top(cd);
            };
            // IK2_CAND=0: sampling takes the whole vocabulary's logits from the device every token
            const bool use_cand = !(getenv("IK2_CAND") && !std::atoi(getenv("IK2_CAND")));
            bool cand = false;   // the last token's logits came as candidates (g_cand), not in `last`
            const char* reason = "length";
            int n_out = 0;
            int greedy_tok = temp <= 0 && !last.empty() ? pick(last.data()) : -1;
            int pending_sample = -1;
            const RowSampler suffix_sample = temp > 0 ? RowSampler(pick) : RowSampler{};
            int suffix_at = 0;
            SuffixStats suffix_stats;
            size_t next_placement = hist.size() + kPlacementInterval;
            for (int i = 0; i < max_new; ++i) {
                if (g_managed_experts && hist.size() >= next_placement) {
                    place_decode(M, (int)hist.size());
                    next_placement = hist.size() + kPlacementInterval;
                }
                if (g_cancel) {
                    reason = "cancelled";
                    break;
                }
                const int tok = temp <= 0             ? greedy_tok
                                : pending_sample >= 0 ? pending_sample
                                : cand                ? pick_cand()
                                                      : pick(last.data());
                pending_sample = -1;
                ++n_out;
                std::printf(serve_maya ? "T %d\n" : "@TOK %d\n", tok);
                std::fflush(stdout);
                if (std::find(stops.begin(), stops.end(), tok) != stops.end()) {
                    reason = "stop";
                    break;
                }
                if (i + 1 == max_new) break;
                if (suffix_window && i >= suffix_at) {
                    std::vector<int> next;
                    const int n =
                        verify_suffix(M, hist, tok, max_new - i - 1, stops, next, t_exp, suffix_stats, suffix_sample);
                    if (n) {
                        hist.push_back(tok);
                        for (int j = 0; j < n - 1; ++j) {
                            std::printf(serve_maya ? "T %d\n" : "@TOK %d\n", next[j]);
                            hist.push_back(next[j]);
                        }
                        std::fflush(stdout);
                        n_out += n - 1;
                        i += n - 1;
                        if (temp <= 0)
                            greedy_tok = next.back();
                        else
                            pending_sample = next.back();
                        last.clear();
                        if (g_cancel) {
                            reason = "cancelled";
                            break;
                        }
                        if (n < 8) suffix_at = i + 16;
                        continue;
                    }
                }
                const int B = spec_on ? sctl.choose(std::min(max_new - i - 1, ctx - (int) hist.size())) : 1;
                if (B >= 2) {
                    SampleCfg sc;
                    sc.temp = temp, sc.top_p = top_p, sc.top_k = top_k, sc.rng = &rng;
                    std::vector<int> next;
                    const double ts = now();
                    spec_step(M, drafter, tok, (int) hist.size(), B, sc, stops, next);
                    CU(cudaStreamSynchronize(g_s));
                    sctl.took_spec(B, (int) next.size() - 1, now() - ts);
                    hist.push_back(tok);
                    for (size_t j = 0; j + 1 < next.size(); ++j) {
                        std::printf(serve_maya ? "T %d\n" : "@TOK %d\n", next[j]);
                        hist.push_back(next[j]);
                    }
                    std::fflush(stdout);
                    n_out += (int) next.size() - 1;
                    i += (int) next.size() - 1;
                    if (temp <= 0) greedy_tok = next.back();
                    else pending_sample = next.back();
                    last.clear();
                    cand = false;
                    if (g_cancel) {
                        reason = "cancelled";
                        break;
                    }
                    continue;
                }
                const double tp0 = now();
                if (temp <= 0) {
                    std::vector<float> lo;
                    std::vector<int> ag;
                    forward(M, &tok, 1, (int) hist.size(), false, OutMode::kArgmax, std::string(), lo, &ag, t_exp, true);
                    greedy_tok = ag[0];
                    last.clear();
                } else {
                    std::vector<float> lo;
                    forward(M, &tok, 1, (int) hist.size(), false, use_cand ? OutMode::kCand : OutMode::kLogits,
                            std::string(), lo, nullptr, t_exp, true);
                    cand = use_cand && g_cand.n >= 0;
                    last = std::move(lo);
                }
                if (spec_on) {   // (the drafter's context for the token: its taps, row 0)
                    ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, 1, (int) hist.size());
                    ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, 0);
                    CU(cudaStreamSynchronize(g_s));
                    sctl.took_plain(now() - tp0);
                }
                hist.push_back(tok);
            }
            if (suffix_stats.forwards)
                std::fprintf(stderr, "suffix: %d target forwards verified %d inputs, accepted %d/%d proposals\n",
                             suffix_stats.forwards, suffix_stats.inputs, suffix_stats.accepted, suffix_stats.offered);
            if (spec_on && getenv("IK2_STATS"))
                std::fprintf(stderr, "speculative: %llu windows, %llu of %llu drafts accepted (%.2f tok/win), %llu plain; win %.1f ms (2..6: %.1f %.1f %.1f %.1f %.1f), plain %.1f ms\n",
                             (unsigned long long) sctl.steps, (unsigned long long) sctl.accepted,
                             (unsigned long long) sctl.drafted,
                             sctl.steps ? 1.0 + (double) sctl.accepted / sctl.steps : 0.0,
                             (unsigned long long) sctl.plain,
                             sctl.c[std::min(sctl.wmax, kBlockMax)] * 1e3,
                             sctl.c[2] * 1e3, sctl.c[3] * 1e3, sctl.c[4] * 1e3, sctl.c[5] * 1e3, sctl.c[6] * 1e3,
                             sctl.c_plain * 1e3);
            if (serve_maya)
                std::printf("DONE %d %zu %.1f %.1f %s %d %d %d\n", n_out, ids.size(), (t1 - t0) * 1e3,
                            (now() - t1) * 1e3, std::strcmp(reason, "cancelled") == 0 ? "cancel" : reason,
                            suffix_stats.accepted, suffix_stats.offered, cached);
            else
                std::printf("@END %s %zu %d %.3f %.3f\n", reason, ids.size(), cached, t1 - t0, now() - t1);
            std::fflush(stdout);
            busy = false;
        }
        pop_dump();
        return 0;
    }
    if ((int) prompt.size() + gen > ctx) die("prompt + gen past --ctx");
    // the prompt
    std::vector<float> logits;
    double t_exp = 0;
    const double tp = now();
    const bool all = !dump.empty();
    const char* prompt_output = getenv("IK2_PREFILL_LOGITS_OUT");
    std::ofstream prompt_logits;
    if (prompt_output) {
        prompt_logits.open(prompt_output, std::ios::binary);
        if (!prompt_logits) die(std::string("cannot write prompt logits to ") + prompt_output);
    }
    const bool all_rows = all || prompt_output;
    int first_best = -1;
    const int T_sup = g_dev[0].W.T_sup;
    const int sub_chunk = std::min(g_dev[0].W.Tm, 512);
    if (T_sup > 0 && (n_gpu == 1 || dual()) && !drafter && !all_rows && (int) prompt.size() > 64) {
        BigScratch bs;
        const int T_pass = big_begin(M, g_dev[0].W, dev_cache(0), (int) prompt.size(), (int) prompt.size(), bs);
        if (T_pass != T_sup) std::printf("prefill passes of %d tokens (scratch lent by the expert cache)\n", T_pass);
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += T_pass) {
            const int n = std::min(T_pass, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = (last && gen > 0) ? OutMode::kArgmax : OutMode::kNone;
            std::vector<float> lo;
            std::vector<int> ag;
            forward_prefill_sup(M, prompt.data() + p0, n, p0, sub_chunk, om, lo, &ag, t_exp);
            if (last && !ag.empty()) first_best = ag[0];
        }
        big_end(g_dev[0].W, dev_cache(0), bs);
    } else {
        for (int p0 = 0; p0 < (int) prompt.size(); p0 += chunk) {
            const int n = std::min(chunk, (int) prompt.size() - p0);
            const bool last = p0 + n == (int) prompt.size();
            const OutMode om = all_rows ? OutMode::kLogits : (!last ? OutMode::kNone : ((gen > 0 || drafter) ? OutMode::kArgmax : OutMode::kNone));
            std::vector<float> lo;
            std::vector<int> ag;
            forward(M, prompt.data() + p0, n, p0, all_rows, om, dump, lo, &ag, t_exp);
            if (prompt_output) {
                if (lo.size() != (size_t)n * c.n_vocab) die("unexpected number of prompt logits");
                prompt_logits.write(reinterpret_cast<const char*>(lo.data()), lo.size() * sizeof(float));
                if (!prompt_logits) die("writing prompt logits failed");
                if (last) first_best = (int)(std::max_element(lo.end() - c.n_vocab, lo.end()) - (lo.end() - c.n_vocab));
            }
            if (all) logits.insert(logits.end(), lo.begin(), lo.end());
            else if (last && !ag.empty()) first_best = ag[0];
            if (drafter) {
                ik2::draft::prefill_context(drafter, g_taps, g_taps_ld, n, p0, prompt.size());
                if (last) ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n - 1);
            }
        }
    }
    if (prompt_output) {
        prompt_logits.close();
        if (!prompt_logits) die("closing prompt logits failed");
    }
    const double prompt_s = now() - tp;
    std::printf("prompt: %zu tokens in %.2f s (%.1f tok/s); experts %.2f s of it, uploads %.2f s; cache %llu hits, "
                "%llu misses, %.1f GiB copied\n",
                prompt.size(), prompt_s, prompt.size() / prompt_s, t_exp, ec_sum([](ExpertCache& e) { return e.t_up; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.hits; }),
                (unsigned long long) ec_sum([](ExpertCache& e) { return e.misses; }),
                ec_sum([](ExpertCache& e) { return e.bytes_up; }) / 1073741824.0);
    std::fflush(stdout);
    if (all) {
        // log-probabilities of every prompt position
        std::vector<float> lp(logits.size());
        for (size_t t = 0; t < prompt.size(); ++t) {
            const float* r = logits.data() + t * c.n_vocab;
            double m = -1e30, s = 0;
            for (int v = 0; v < c.n_vocab; ++v) m = std::max(m, (double) r[v]);
            for (int v = 0; v < c.n_vocab; ++v) s += std::exp(r[v] - m);
            const double ls = m + std::log(s);
            for (int v = 0; v < c.n_vocab; ++v) lp[t * c.n_vocab + v] = (float) (r[v] - ls);
        }
        FILE* f = std::fopen((dump + "/logprobs.f32").c_str(), "wb");
        std::fwrite(lp.data(), 4, lp.size(), f);
        std::fclose(f);
        const float* r_last = logits.data() + (prompt.size() - 1) * c.n_vocab;
        first_best = (int) (std::max_element(r_last, r_last + c.n_vocab) - r_last);
        logits.clear();
    }
    if (gen) place_decode(M, (int)prompt.size());
    // greedy generation
    g_pf = Prof{};
    std::vector<int> out;
    const double tg = now();
    const uint64_t h0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.hits; }),
                   m0 = (uint64_t) ec_sum([](ExpertCache& e) { return e.misses; });
    int n_steps = 0;
    const int verify_check = getenv("IK2_VERIFY_CHECK") ? std::atoi(getenv("IK2_VERIFY_CHECK")) : 0;
    const int block_check = getenv("IK2_BLOCK_CHECK") ? std::atoi(getenv("IK2_BLOCK_CHECK")) : 0;
    double t_draft = 0, t_verify = 0;
    std::vector<int> acc_hist(ik2::draft::kBlock, 0);
    // IK2_FORCE=FILE (comma-separated ids): the tokens fed are these instead of the model's own picks, and how often
    // its pick agrees with the next one is counted - two builds then run the same tokens (a fair speed comparison)
    // and the agreement measures their numerics on real text
    std::vector<int> force;
    int agree = 0;
    if (getenv("IK2_FORCE")) {
        FILE* ff = std::fopen(getenv("IK2_FORCE"), "r");
        if (!ff) die("IK2_FORCE: cannot read the file");
        int v;
        while (std::fscanf(ff, "%d,", &v) == 1) force.push_back(v);
        std::fclose(ff);
    }
    // IK2_LOGITS_OUT=FILE: every generated token's logits [n_vocab] F32 appended there (with IK2_FORCE: two builds'
    // logits over the same tokens, to compare)
    FILE* lo = getenv("IK2_LOGITS_OUT") ? std::fopen(getenv("IK2_LOGITS_OUT"), "wb") : nullptr;
    const bool use_suffix = suffix_window && !lo && force.empty() && !drafter;
    std::vector<int> suffix_hist;
    if (use_suffix) suffix_hist = prompt;
    int suffix_at = 0;
    SuffixStats suffix_stats;
    int next_placement = kPlacementInterval;
    for (int i = 0, best = first_best; i < gen && !drafter; ++i) {
        if (g_managed_experts && i >= next_placement) {
            place_decode(M, (int)prompt.size() + i);
            next_placement = i + kPlacementInterval;
        }
        const int tok = i < (int) force.size() ? force[i] : best;
        if (i < (int) force.size()) agree += best == force[i];
        out.push_back(tok);
        std::printf("%d%s", tok, i + 1 < gen ? "," : "\n");
        std::fflush(stdout);
        if (i + 1 == gen) break;
        if (use_suffix) {
            std::vector<int> next;
            const int n =
                i < suffix_at ? 0 : verify_suffix(M, suffix_hist, tok, gen - i - 1, {}, next, t_exp, suffix_stats);
            suffix_hist.push_back(tok);
            if (n) {
                for (int j = 0; j < n - 1; ++j) {
                    out.push_back(next[j]);
                    suffix_hist.push_back(next[j]);
                    std::printf("%d,", next[j]);
                }
                i += n - 1;
                best = next.back();
                if (n < 8) suffix_at = i + 16;
                continue;
            }
        }
        std::vector<int> ag;
        if (lo) logits.clear();
        forward(M, &tok, 1, (int) prompt.size() + i, false, lo ? OutMode::kLogits : OutMode::kArgmax, std::string(),
                logits, &ag, t_exp, true);
        if (lo) {
            std::fwrite(logits.data(), sizeof(float), (size_t) c.n_vocab, lo);
            best = (int) (std::max_element(logits.begin(), logits.begin() + c.n_vocab) - logits.begin());
        } else {
            best = ag[0];
        }
    }
    if (lo) std::fclose(lo);
    if (!force.empty())
        std::printf("forced: %d tokens fed, the model's pick agreed with the next one %d times (%.2f%%)\n",
                    (int) std::min<size_t>(force.size(), out.size()), agree,
                    100.0 * agree / std::max<size_t>(1, std::min<size_t>(force.size(), out.size())));

    if (suffix_stats.forwards)
        std::fprintf(stderr, "suffix: %d target forwards verified %d inputs, accepted %d/%d proposals\n",
                     suffix_stats.forwards, suffix_stats.inputs, suffix_stats.accepted, suffix_stats.offered);
    SpecCtl ctl;
    if (drafter && g_fast && g_block) {
        // (the resident path: the controller picks plain tokens or a window; greedy, as the CLI's generation is)
        ctl.wmax = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, kBlockMax) : 5;
        ctl.adapt = !(getenv("IK2_DRAFT_ADAPT") && !atoi(getenv("IK2_DRAFT_ADAPT")));
        int anchor = first_best, P = (int) prompt.size();
        out.push_back(anchor);
        const SampleCfg sc;
        std::vector<int> next;
        while ((int) out.size() < gen && P < ctx) {
            const int room = std::min(gen - (int) out.size(), ctx - P);
            const int B = ctl.choose(room);
            const double ts = now();
            if (B < 2) {
                std::vector<float> lo;
                std::vector<int> ag;
                forward(M, &anchor, 1, P, false, OutMode::kArgmax, std::string(), lo, &ag, t_exp);
                ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, 1, P);
                ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, 0);
                CU(cudaStreamSynchronize(g_s));
                ctl.took_plain(now() - ts);
                anchor = ag[0], P += 1;
                out.push_back(anchor);
                continue;
            }
            spec_step(M, drafter, anchor, P, B, sc, std::vector<int>{}, next);
            CU(cudaStreamSynchronize(g_s));
            ctl.took_spec(B, (int) next.size() - 1, now() - ts);
            for (int j = 0; j < (int) next.size() && (int) out.size() < gen; ++j) out.push_back(next[j]);
            P += (int) next.size();
            anchor = next.back();
        }
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    } else if (drafter) {
        // draft 7 after the anchor, verify the 8 in one call, keep the agreeing prefix and the target's next token
        float *raw = dalloc<float>(c.C), *anchor_h = dalloc<float>(c.C);
        auto argmax_row = [&](const float* r) { return (int) (std::max_element(r, r + c.n_vocab) - r); };
        int anchor = first_best, P = (int) prompt.size();
        out.push_back(anchor);
        // IK2_DRAFT_WIDTH: the tokens a step verifies, the anchor and width - 1 drafts (2 .. 8)
        const int width = getenv("IK2_DRAFT_WIDTH") ? std::clamp(atoi(getenv("IK2_DRAFT_WIDTH")), 2, ik2::draft::kBlock)
                                                    : ik2::draft::kBlock;
        const bool margin_check = getenv("IK2_MARGIN") != nullptr;
        while ((int) out.size() < gen && P < ctx) {
            const int B = std::min(width, ctx - P);
            const double t0 = now();
            embed_row(M, anchor, raw, anchor_h);
            int dr[ik2::draft::kBlock - 1];
            if (B > 1) ik2::draft::propose(drafter, anchor, anchor_h, P, draft_head, c.n_rows, c.n_vocab, dr, B - 1);
            const double t1 = now();
            int tk[ik2::draft::kBlock];
            tk[0] = anchor;
            for (int j = 1; j < B; ++j) tk[j] = dr[j - 1];
            g_rec = true;
            logits.clear();
            std::vector<int> ag;
            const bool need_logits = n_steps < verify_check || margin_check;
            const bool use_block = g_fast && g_block && !need_logits;
            if (use_block && n_steps < block_check) {
                // (IK2_BLOCK_CHECK=n: the first n windows through both paths, their logits compared; forward()'s
                //  then taken)
                std::vector<float> lb;
                fast_block(M, tk, B, P, OutMode::kLogits, lb, nullptr);
                block_rollback(0);
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
                double mx = 0, kl = 0;
                int dis = 0;
                for (int j = 0; j < B; ++j) {
                    const float* a = logits.data() + (size_t) j * c.n_vocab;
                    const float* b = lb.data() + (size_t) j * c.n_vocab;
                    double ma = -1e30, mb = -1e30, sa = 0, sb = 0;
                    for (int v = 0; v < c.n_vocab; ++v) ma = std::max(ma, (double) a[v]), mb = std::max(mb, (double) b[v]);
                    for (int v = 0; v < c.n_vocab; ++v) sa += std::exp(a[v] - ma), sb += std::exp(b[v] - mb);
                    for (int v = 0; v < c.n_vocab; ++v) {
                        mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                        const double pa = std::exp(a[v] - ma) / sa;
                        if (pa > 0) kl += pa * ((a[v] - ma - std::log(sa)) - (b[v] - mb - std::log(sb)));
                    }
                    dis += argmax_row(a) != argmax_row(b);
                }
                std::printf("block check, step %d (%d tokens at %d): max |dlogit| %.4f, KL %.2e a row, argmax differs in %d\n",
                            n_steps, B, P, mx, kl / B, dis);
                ag.clear();
                for (int j = 0; j < B; ++j) ag.push_back(argmax_row(logits.data() + (size_t) j * c.n_vocab));
            } else if (use_block)
                fast_block(M, tk, B, P, OutMode::kArgmax, logits, &ag);
            else
                forward(M, tk, B, P, true, need_logits ? OutMode::kLogits : OutMode::kArgmax, std::string(), logits,
                        &ag, t_exp);
            if (n_steps < verify_check) {
                // the same window again through each other path (the convs' history taken back first): the
                // logits' largest difference and the argmax disagreements against the grouped run
                const bool ge = g_group_exp, ga = g_group_attn;
                const int modes[3][2] = {{0, 0}, {1, 0}, {0, 1}};
                const char* names[3] = {"neither grouped", "experts grouped only", "attention grouped only"};
                for (int m = 0; m < 3; ++m) {
                    rollback(M, 0);
                    g_group_exp = modes[m][0], g_group_attn = modes[m][1];
                    std::vector<float> lb;
                    forward(M, tk, B, P, true, OutMode::kLogits, std::string(), lb, nullptr, t_exp);
                    double mx = 0;
                    int dis = 0;
                    for (int j = 0; j < B; ++j) {
                        const float* a = logits.data() + (size_t) j * c.n_vocab;
                        const float* b = lb.data() + (size_t) j * c.n_vocab;
                        for (int v = 0; v < c.n_vocab; ++v) mx = std::max(mx, (double) std::fabs(a[v] - b[v]));
                        dis += argmax_row(a) != argmax_row(b);
                    }
                    std::printf("verify check, step %d, grouped vs %s: max |dlogit| %.4f, argmax differs in %d of %d rows\n",
                                n_steps, names[m], mx, dis, B);
                }
                g_group_exp = ge, g_group_attn = ga;
                rollback(M, 0);
                logits.clear();
                forward(M, tk, B, P, true, OutMode::kLogits, std::string(), logits, nullptr, t_exp);
            }
            g_rec = false;
            int g[ik2::draft::kBlock];
            if (need_logits) {
                for (int j = 0; j < B; ++j) g[j] = argmax_row(logits.data() + (size_t) j * c.n_vocab);
            } else {
                for (int j = 0; j < B; ++j) g[j] = ag[j];
            }
            if (margin_check) {   // the target's top-2 margin at each verified position (near ties flip)
                for (int j = 0; j < B; ++j) {
                    const float* r = logits.data() + (size_t) j * c.n_vocab;
                    float b1 = -1e30f, b2 = -1e30f;
                    int i1 = -1, i2 = -1;
                    for (int v = 0; v < c.n_vocab; ++v) {
                        if (r[v] > b1) b2 = b1, i2 = i1, b1 = r[v], i1 = v;
                        else if (r[v] > b2) b2 = r[v], i2 = v;
                    }
                    if (b1 - b2 < 0.05f)
                        std::printf("near tie at position %d: %d %.4f, %d %.4f\n", P + j, i1, b1, i2, b2);
                }
            }
            int n = 0;
            while (n < B - 1 && dr[n] == g[n]) ++n;
            if (n + 1 < B) {
                if (use_block && n_steps >= block_check) block_rollback(n + 1);
                else rollback(M, n + 1);
            }
            ik2::draft::write_context(drafter, g_taps, g_taps_ld, 0, n + 1, P);
            ik2::draft::set_f_rnn(drafter, g_taps, g_taps_ld, n);
            for (int j = 0; j < n && (int) out.size() < gen; ++j) out.push_back(dr[j]);
            if ((int) out.size() < gen) out.push_back(g[n]);
            anchor = g[n];
            P += n + 1;
            ++n_steps;
            ++acc_hist[n];
            t_draft += t1 - t0;
            t_verify += now() - t1;
        }
        for (size_t j = 0; j < out.size(); ++j) std::printf("%d%s", out[j], j + 1 < out.size() ? "," : "\n");
    }
    const double gen_s = now() - tg;
    if (drafter && (ctl.steps || ctl.plain)) {
        std::printf("speculative: %llu windows (%llu drafts, %llu accepted: %.2f tokens a window; accepted drafts:",
                    (unsigned long long) ctl.steps, (unsigned long long) ctl.drafted, (unsigned long long) ctl.accepted,
                    ctl.steps ? 1.0 + (double) ctl.accepted / ctl.steps : 0.0);
        for (int j = 0; j < kBlockMax; ++j) std::printf(" %llu", (unsigned long long) ctl.hist[j]);
        std::printf("), %llu plain tokens; a window %.1f ms (widths 2 .. 6:", (unsigned long long) ctl.plain,
                    ctl.c[std::min(ctl.wmax, kBlockMax)] * 1e3);
        for (int b = 2; b <= 6; ++b) std::printf(" %.1f", ctl.c[b] * 1e3);
        std::printf("), a plain token %.1f ms\n", ctl.c_plain * 1e3);
    }
    if (drafter && n_steps) {
        std::printf("speculative: %d steps, %.2f tokens a step (accepted drafts: ", n_steps,
                    (double) (out.size() - 1) / n_steps);
        for (int j = 0; j < ik2::draft::kBlock; ++j) std::printf("%d%s", acc_hist[j], j + 1 < ik2::draft::kBlock ? " " : "");
        std::printf("); drafting %.1f ms a step, verifying %.1f ms\n", t_draft * 1e3 / n_steps, t_verify * 1e3 / n_steps);
    }
    if (g_pool)
        std::printf("CPU lane: %d threads, %llu experts computed on the CPU (%.2f ms each at the end), %llu waited for\n",
                    g_pool->size(), (unsigned long long) g_n_cpu, g_t_cpu * 1e3, (unsigned long long) g_n_wait);
    if (g_prof)
        std::printf("profile (all calls): attention %.2f s, router %.2f s, uploads %.2f s, experts %.2f s, rest %.2f s\n",
                    g_pf.attn, g_pf.route, g_pf.upload, g_pf.experts, g_pf.rest);
    if (out.size() > 1)
        std::printf("generation: %d tokens in %.2f s (%.2f tok/s); cache %llu hits, %llu misses in it\n",
                    (int) out.size() - 1, gen_s, (out.size() - 1) / gen_s, (unsigned long long) (ec_sum([](ExpertCache& e) { return e.hits; }) - h0),
                    (unsigned long long) (ec_sum([](ExpertCache& e) { return e.misses; }) - m0));
    if (gpu_served)
        std::printf("GPU-served expert misses=%llu copies=%llu reserve_refills=%llu loop_reuses=%llu\n",
                    (unsigned long long)served.misses, (unsigned long long)served.copies, (unsigned long long)served.refills,
                    (unsigned long long)served.loop_reuses);
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fflush(stdout);
    if (g_trace) std::fclose(g_trace);
    pop_dump();
    return 0;
}
