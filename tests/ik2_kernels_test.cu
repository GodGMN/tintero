// tests/ik2_kernels_test.cu - tintero's kernels (src/ik2/ik2_kernels.cu) on random data, no model, each against a
// double reference written from ref/inkling2.py's ops:
//   * rms, the scaled L2 clip (with and without the full layers' log scaling), the short conv (static and dynamic taps,
//     the history carried over calls of 1, 2, 3 and many tokens = one call over all of them), the attention residual
//     (1 and 3 queries, 1 to 9 states);
//   * a head's attention weights: the relative bias out to the extent, the window, the sink, queries and keys at
//     offsets (a chunk of a prompt against the cache);
//   * the router (the picks as a set and the weights), the softcapped SwiGLU with the experts' scales, the combine;
//   * the decodes (NVFP4, Q8_0, Q5_K, Q6_K): the device's F32 values bit for bit the host's, FP16 = F32 rounded (NVFP4:
//     exact), NVFP4 against the format's definition.  (Q8_0 / Q5_K / Q6_K against ggml's values: not here yet - the
//     real model's seams cover them.)
#include "../src/ik2/ik2_iq_tables.h"
#include "../src/ik2/ik2_kernels.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = ik2::k;

namespace {

int g_fail = 0;
std::mt19937 rng(20261116);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

void release(std::initializer_list<void*> buffers) {
    for (void* ptr : buffers) CU(cudaFree(ptr));
}

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }
std::vector<float> fvec(size_t n, float a = 1.0f) {
    std::vector<float> v(n);
    for (auto& x : v) x = frand(a);
    return v;
}

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
T* dev_n(size_t n) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    CU(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaDeviceSynchronize());
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// |got - want| <= atol + rtol |want| everywhere
void expect(const char* what, const std::vector<float>& got, const std::vector<double>& want, double rtol,
            double atol = 1e-6) {
    double worst = 0.0;
    size_t bad = 0, at = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]), lim = atol + rtol * std::fabs(want[i]);
        if (!(err <= lim) || !std::isfinite(got[i])) {
            if (!bad) at = i;
            ++bad;
        }
        worst = std::max(worst, err / (atol + rtol * std::fabs(want[i])));
    }
    if (bad) {
        ++g_fail;
        std::printf("FAIL %-34s %zu of %zu off (first at %zu: got %g want %g)\n", what, bad, want.size(), at,
                    (double) got[at], want[at]);
    } else {
        std::printf("ok   %-34s %zu values, worst %.2f of the tolerance\n", what, want.size(), worst);
    }
}
void expect_true(const char* what, bool ok) {
    if (!ok) ++g_fail;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

#ifndef STRATA_USE_HIP
#include "../src/ik2/ik2_served_kernels.cuh"

void test_served_copy() {
    constexpr size_t bytes = 4112; // One vector beyond a full 256-thread block.
    uint8_t *pinned, *mapped;
    CU(cudaHostAlloc(&pinned, 2 * bytes, cudaHostAllocMapped));
    CU(cudaHostGetDevicePointer(&mapped, pinned, 0));
    for (size_t i = 0; i < 2 * bytes; ++i) pinned[i] = uint8_t(i % 251 + 1);
    auto cache = dev_n<uint8_t>(3 * bytes);
    auto table = dev<const uint8_t*>({cache, mapped + bytes});
    auto reserve = dev<const uint8_t*>({cache + bytes, cache + 2 * bytes});
    auto picks = dev<int>({0, 1});
    auto selected = dev<const uint8_t*>({cache, mapped + bytes});
    auto records = dev_n<int>(2);
    copy_served_experts<<<dim3(1, 2), 256>>>(picks, table, mapped, reserve, bytes, records);
    publish_served_experts<<<1, 2>>>(records, 2, table, reserve, selected);
    expect_true("publish only the missed expert", host(records, 2) == std::vector<int>{0, ~1} &&
                host(table, 2) == std::vector<const uint8_t*>{cache, cache + 2 * bytes} &&
                host(selected, 2) == host(table, 2));
    auto weights = host(cache, 3 * bytes);
    expect_true("copy the complete mapped blob", std::equal(weights.begin() + 2 * bytes, weights.end(), pinned + bytes));
    expect_true("preserve resident and unused slots", std::all_of(weights.begin(), weights.begin() + 2 * bytes,
                                                                  [](uint8_t x) { return x == 0; }));
    copy_served_experts<<<dim3(1, 2), 256>>>(picks, table, mapped, reserve, bytes, records);
    expect_true("later visit reuses published slot", host(records, 2) == std::vector<int>{0, 1});
    release({cache, table, reserve, picks, selected, records});
    CU(cudaFreeHost(pinned));
}
#endif

float h2f_host(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) return (sign ? -1.0f : 1.0f) * std::ldexp((float) m, -24);
    if (e == 31) return m ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + (float) m / 1024.0f, (int) e - 15);
}
uint16_t f2h_host(float f) { return (uint16_t) __half_as_ushort(__float2half_rn(f)); }

// ---------------------------------------------------------------------------------------------------------------

void test_rms() {
    const int rows = 5, D = 4608;
    auto x = fvec((size_t) rows * D, 3.0f), w = fvec(D);
    float *dx = dev(x), *dw = dev(w), *dy = dev_n<float>((size_t) rows * D);
    for (int with_w = 0; with_w < 2; ++with_w) {
        const float eps = with_w ? 1e-6f : 1e-5f;
        k::rms(dx, with_w ? dw : nullptr, rows, D, eps, dy, 0);
        std::vector<double> want((size_t) rows * D);
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int d = 0; d < D; ++d) ss += (double) x[(size_t) r * D + d] * x[(size_t) r * D + d];
            const double rs = 1.0 / std::sqrt(ss / D + eps);
            for (int d = 0; d < D; ++d) want[(size_t) r * D + d] = x[(size_t) r * D + d] * rs * (with_w ? w[d] : 1.0);
        }
        expect(with_w ? "rms (weight)" : "rms (no weight)", host(dy, want.size()), want, 2e-5);
    }
    k::rms(dx, nullptr, rows * 36, 128, 1e-6f, dy, 0);   // v's rows: [T][Hkv][128]
    std::vector<double> want((size_t) rows * D);
    for (int r = 0; r < rows * 36; ++r) {
        double ss = 0;
        for (int d = 0; d < 128; ++d) ss += (double) x[(size_t) r * 128 + d] * x[(size_t) r * 128 + d];
        for (int d = 0; d < 128; ++d) want[(size_t) r * 128 + d] = x[(size_t) r * 128 + d] / std::sqrt(ss / 128 + 1e-6);
    }
    expect("rms (rows of 128)", host(dy, want.size()), want, 2e-5);
    release({dx, dw, dy});
}

// the refusal ablation's projection: against a double reference, the component left along r, rows independent of T
void test_project_out() {
    const int T = 9, D = 2304;
    auto x = fvec((size_t) T * D, 4.0f), r = fvec(D);
    double n2 = 0;
    for (float v : r) n2 += (double) v * v;
    for (float& v : r) v = (float) (v / std::sqrt(n2));
    float* dr = dev(r);
    const float cases[][2] = {{1.0f, 0.0f}, {1.3f, 0.0f}, {0.8f, 0.0f}, {1.0f, 2.5f}};
    for (const auto& sc : cases) {
        float* dy = dev(x);
        k::project_out(dy, dr, sc[0], sc[1], T, D, 0);
        std::vector<double> want((size_t) T * D);
        for (int t = 0; t < T; ++t) {
            double d = 0;
            for (int e = 0; e < D; ++e) d += (double) r[e] * x[(size_t) t * D + e];
            for (int e = 0; e < D; ++e) want[(size_t) t * D + e] = x[(size_t) t * D + e] - sc[0] * (d - sc[1]) * r[e];
        }
        const auto got = host(dy, want.size());
        char name[64];
        std::snprintf(name, sizeof name, "project_out (s %.1f, c %.1f)", sc[0], sc[1]);
        expect(name, got, want, 2e-5, 1e-4);
        if (sc[0] == 1.0f) {   // what is left along r: c
            double worst = 0;
            for (int t = 0; t < T; ++t) {
                double d = 0;
                for (int e = 0; e < D; ++e) d += (double) r[e] * got[(size_t) t * D + e];
                worst = std::max(worst, std::fabs(d - sc[1]));
            }
            expect_true("project_out leaves c along r", worst <= 1e-3);
        }
        float* d1 = dev(std::vector<float>(x.begin() + (size_t) 4 * D, x.begin() + (size_t) 5 * D));   // row 4 alone
        k::project_out(d1, dr, sc[0], sc[1], 1, D, 0);
        const auto one = host(d1, D);
        expect_true("project_out row bits do not depend on T",
                    std::memcmp(one.data(), got.data() + (size_t) 4 * D, (size_t) D * sizeof(float)) == 0);
        release({dy, d1});
    }
    {   // per-row strengths: row t takes srow[t] * s (0 leaves it as it was)
        std::vector<float> srow(T);
        for (int t = 0; t < T; ++t) srow[t] = t % 3 == 0 ? 0.0f : 1.0f + 0.1f * t;
        float *dy = dev(x), *ds = dev(srow);
        k::project_out(dy, dr, 1.0f, 0.0f, T, D, 0, ds);
        std::vector<double> want((size_t) T * D);
        for (int t = 0; t < T; ++t) {
            double d = 0;
            for (int e = 0; e < D; ++e) d += (double) r[e] * x[(size_t) t * D + e];
            for (int e = 0; e < D; ++e) want[(size_t) t * D + e] = x[(size_t) t * D + e] - srow[t] * d * r[e];
        }
        expect("project_out (per-row strengths)", host(dy, want.size()), want, 2e-5, 1e-4);
        release({dy, ds});
    }
    {   // norm-preserving: the projection, then each row back to its length
        float* dy = dev(x);
        k::project_out(dy, dr, 1.0f, 0.0f, T, D, 0, nullptr, true);
        std::vector<double> want((size_t) T * D);
        for (int t = 0; t < T; ++t) {
            double d = 0, n0 = 0, n1 = 0;
            for (int e = 0; e < D; ++e) d += (double) r[e] * x[(size_t) t * D + e], n0 += (double) x[(size_t) t * D + e] * x[(size_t) t * D + e];
            for (int e = 0; e < D; ++e) {
                want[(size_t) t * D + e] = x[(size_t) t * D + e] - d * r[e];
                n1 += want[(size_t) t * D + e] * want[(size_t) t * D + e];
            }
            for (int e = 0; e < D; ++e) want[(size_t) t * D + e] *= std::sqrt(n0 / n1);
        }
        expect("project_out (norm kept)", host(dy, want.size()), want, 2e-5, 1e-4);
        release({dy});
    }
    k::project_out(nullptr, dr, 1.0f, 0.0f, 0, D, 0);   // T 0: nothing
    release({dr});
}

void test_l2clip() {
    const int rows = 7, th_cases = 2;
    for (int c = 0; c < 3; ++c) {
        const int G = c == 2 ? 48 : c == 1 ? 8 : 48, D = c == 2 ? 16 : 128;
        const bool scaled = c != 1, tau = c == 0;
        auto x = fvec((size_t) rows * G * D, 4.0f), sc = fvec((size_t) G * D, 2.0f);
        for (size_t i = 0; i < (size_t) G * D; ++i) x[i] *= 0.01f;   // row 0: under the threshold
        std::vector<int> pos(rows);
        for (int r = 0; r < rows; ++r) pos[r] = r == 0 ? 5 : 100000 * r + 77777;   // some past the floor
        float *dx = dev(x), *dsc = dev(sc), *dy = dev_n<float>(x.size());
        int* dpos = dev(pos);
        const float th = c == 2 ? 8.0f : 22.627416610717773f, alpha = 0.1f, n_floor = 128000.0f;
        k::l2clip(dx, scaled ? dsc : nullptr, rows, G, D, th, tau ? dpos : nullptr, alpha, n_floor, dy, 0);
        std::vector<double> want(x.size());
        int clipped = 0;
        for (int r = 0; r < rows; ++r)
            for (int g = 0; g < G; ++g) {
                const size_t o = ((size_t) r * G + g) * D;
                double ss = 0;
                for (int d = 0; d < D; ++d) {
                    const double v = (double) x[o + d] * (scaled ? sc[(size_t) g * D + d] : 1.0);
                    ss += v * v;
                }
                double f = std::min(1.0, th / (std::sqrt(ss) + 1e-6));
                clipped += f < 1.0;
                if (tau) f *= 1.0 + alpha * std::log(std::max(1.0, (pos[r] + 1.0) / n_floor));
                for (int d = 0; d < D; ++d) want[o + d] = (double) x[o + d] * (scaled ? sc[(size_t) g * D + d] : 1.0) * f;
            }
        char name[64];
        std::snprintf(name, sizeof name, "l2clip G%d D%d%s%s (%d clipped)", G, D, scaled ? " scaled" : "",
                      tau ? " tau" : "", clipped);
        expect(name, host(dy, want.size()), want, 2e-5);
        release({dx, dsc, dy, dpos});
    }
    (void) th_cases;
}

// the reference conv over the whole sequence (ref/inkling2.py's conv)
std::vector<double> conv_ref(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>* dyn,
                             int heads, int T, int C) {
    std::vector<double> y((size_t) T * C);
    auto xp = [&](int r, int c) -> double { return r < 3 ? 0.0 : x[(size_t) (r - 3) * C + c]; };
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            double a = 0;
            for (int j = 0; j < 4; ++j) a += (double) w[(size_t) c * 4 + j] * xp(t + j, c);
            if (dyn)
                for (int lag = 0; lag < 4; ++lag)
                    a += (double) (*dyn)[(size_t) t * heads * 4 + (size_t) (c / (C / heads)) * 4 + lag] * xp(t + 3 - lag, c);
            y[(size_t) t * C + c] = a;
        }
    return y;
}

void test_conv() {
    const int T = 41, C = 1024, heads = 8;
    auto x = fvec((size_t) T * C), w = fvec((size_t) C * 4), dyn = fvec((size_t) T * heads * 4);
    float *dx = dev(x), *dw = dev(w), *dd = dev(dyn), *dy = dev_n<float>((size_t) T * C);
    for (int with_dyn = 0; with_dyn < 2; ++with_dyn) {
        const auto want = conv_ref(x, w, with_dyn ? &dyn : nullptr, heads, T, C);
        // one call
        float* hist = dev_n<float>((size_t) 3 * C);
        k::conv4(dx, hist, dw, with_dyn ? dd : nullptr, heads, T, C, dy, 0);
        expect(with_dyn ? "conv4 dynamic, one call" : "conv4 static, one call", host(dy, want.size()), want, 2e-5);
        // the same tokens in calls of 1, 2, 3, 5, the rest: the history carries
        CU(cudaMemset(hist, 0, (size_t) 3 * C * sizeof(float)));
        CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
        const int parts[] = {1, 2, 3, 1, 5, 1, 1, T};
        int t0 = 0;
        for (int p : parts) {
            const int n = std::min(p, T - t0);
            if (n <= 0) break;
            k::conv4(dx + (size_t) t0 * C, hist, dw, with_dyn ? dd + (size_t) t0 * heads * 4 : nullptr, heads, n, C,
                     dy + (size_t) t0 * C, 0);
            t0 += n;
        }
        expect(with_dyn ? "conv4 dynamic, in pieces" : "conv4 static, in pieces", host(dy, want.size()), want, 2e-5);
        for (int tb : {16, 17}) {
            CU(cudaMemset(hist, 0, (size_t) 3 * C * sizeof(float)));
            CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
            k::conv4(dx, hist, dw, with_dyn ? dd : nullptr, heads, tb, C, dy, 0);
            k::conv4(dx + (size_t) tb * C, hist, dw, with_dyn ? dd + (size_t) tb * heads * 4 : nullptr, heads, T - tb,
                     C, dy + (size_t) tb * C, 0);
            char name[64];
            std::snprintf(name, sizeof name, "conv4 %s, split at %d", with_dyn ? "dyn" : "static", tb);
            expect(name, host(dy, want.size()), want, 2e-5);
        }
        release({hist});
    }
    release({dx, dw, dd, dy});
}

void test_attn_res(int T, int C) {
    for (int n : {1, 2, 9})
        for (int nq : {1, 3}) {
            std::vector<std::vector<float>> st(n);
            k::ResStates rs{};
            std::vector<float*> ds(n);
            for (int i = 0; i < n; ++i) st[i] = fvec((size_t) T * C, 1.0f + i), ds[i] = dev(st[i]), rs.st[i] = ds[i];
            auto q = fvec((size_t) nq * C, 3.0f);
            float *dq = dev(q), *dout = dev_n<float>((size_t) nq * T * C);
            k::attn_res(rs, n, dq, nq, T, C, 1e-6f, dout, 0);
            std::vector<double> want((size_t) nq * T * C);
            for (int t = 0; t < T; ++t) {
                std::vector<double> r(n);
                for (int i = 0; i < n; ++i) {
                    double ss = 0;
                    for (int d = 0; d < C; ++d) ss += (double) st[i][(size_t) t * C + d] * st[i][(size_t) t * C + d];
                    r[i] = 1.0 / std::sqrt(ss / C + 1e-6);
                }
                for (int qi = 0; qi < nq; ++qi) {
                    std::vector<double> sc(n);
                    double m = -1e300, sum = 0;
                    for (int i = 0; i < n; ++i) {
                        double dot = 0;
                        for (int d = 0; d < C; ++d) dot += (double) q[(size_t) qi * C + d] * st[i][(size_t) t * C + d] * r[i];
                        sc[i] = dot / std::sqrt((double) C);
                        m = std::max(m, sc[i]);
                    }
                    for (int i = 0; i < n; ++i) sum += (sc[i] = std::exp(sc[i] - m));
                    for (int d = 0; d < C; ++d) {
                        double a = 0;
                        for (int i = 0; i < n; ++i) a += sc[i] / sum * st[i][(size_t) t * C + d] * r[i];
                        want[((size_t) qi * T + t) * C + d] = a;
                    }
                }
            }
            char name[64];
            std::snprintf(name, sizeof name, "attn_res C=%d %d states, %d queries", C, n, nq);
            expect(name, host(dout, want.size()), want, 1e-4, 1e-6);
            for (auto* p : ds) release({p});
            release({dq, dout});
        }
}

void test_attn_probs() {
    struct Case {
        int Tq, nk, qpos0, kpos0, extent, window;
        const char* name;
    };
    const int d_rel = 16, H = 48;
    const Case cases[] = {
        {37, 37, 0, 0, 24, 1 << 30, "attn_probs full, from 0"},
        {20, 150, 130, 0, 64, 1 << 30, "attn_probs full, a chunk"},
        {33, 90, 200, 143, 32, 32, "attn_probs sliding window 32"},
        {1, 300, 299, 0, 1024, 1 << 30, "attn_probs one token"},
        {5, 2100, 3000, 905, 1024, 1024, "attn_probs sliding 1024"},
    };
    for (const Case& c : cases) {
        auto lg = fvec((size_t) c.Tq * c.nk, 40.0f);
        auto r = fvec((size_t) c.Tq * H * d_rel, 2.0f);        // [T][H][d_rel]; head 5's rows are used
        auto proj = fvec((size_t) c.extent * d_rel, 1.0f);
        auto sink = fvec((size_t) c.Tq * H, 3.0f);
        const int head = 5;
        const float scale = 1.0f / std::sqrt(128.0f);
        float *dl = dev(lg), *dr = dev(r), *dp = dev(proj), *dsk = dev(sink);
        k::attn_probs(dl, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dr + head * d_rel, H * d_rel, d_rel, dp, c.extent,
                      c.window, dsk + head, H, 0);
        std::vector<double> want((size_t) c.Tq * c.nk);
        for (int i = 0; i < c.Tq; ++i) {
            std::vector<double> l(c.nk + 1);
            double m = -1e300;
            for (int j = 0; j < c.nk; ++j) {
                const long dist = (long) (c.qpos0 + i) - (c.kpos0 + j);
                if (dist < 0 || dist >= c.window) {
                    l[j] = -INFINITY;
                    continue;
                }
                double v = (double) lg[(size_t) i * c.nk + j] * scale;
                if (dist < c.extent)
                    for (int e = 0; e < d_rel; ++e)
                        v += (double) r[((size_t) i * H + head) * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = v;
                m = std::max(m, v);
            }
            l[c.nk] = sink[(size_t) i * H + head];
            m = std::max(m, l[c.nk]);
            double sum = 0;
            for (int j = 0; j <= c.nk; ++j) sum += std::exp(l[j] - m);
            for (int j = 0; j < c.nk; ++j) want[(size_t) i * c.nk + j] = std::exp(l[j] - m) / sum;
        }
        expect(c.name, host(dl, want.size()), want, 2e-4, 1e-9);
        release({dl, dr, dp, dsk});
    }

    // Batched attn_probs_f16: [H][Tq][nk] F32 logits + [Tq][H][d_rel] r + [Tq][H] sink -> [H][Tq][nk] FP16
    for (const Case& c : cases) {
        auto lg = fvec((size_t) H * c.Tq * c.nk, 40.0f);
        auto r = fvec((size_t) c.Tq * H * d_rel, 2.0f);
        auto proj = fvec((size_t) c.extent * d_rel, 1.0f);
        auto sink = fvec((size_t) c.Tq * H, 3.0f);
        const float scale = 1.0f / std::sqrt(128.0f);
        float *dl = dev(lg), *dr = dev(r), *dp = dev(proj), *dsk = dev(sink);
        uint16_t* dout = dev_n<uint16_t>((size_t) H * c.Tq * c.nk);
        k::attn_probs_f16(dl, dout, H, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dr, d_rel, dp, c.extent, c.window, dsk, 0);
        const auto raw = host(dout, (size_t) H * c.Tq * c.nk);
        std::vector<float> got(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) got[i] = h2f_host(raw[i]);
        std::vector<double> want(raw.size(), 0.0);
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < c.Tq; ++i) {
                std::vector<double> l(c.nk + 1);
                double m = -1e300;
                for (int j = 0; j < c.nk; ++j) {
                    const long dist = (long) (c.qpos0 + i) - (c.kpos0 + j);
                    if (dist < 0 || dist >= c.window) {
                        l[j] = -INFINITY;
                        continue;
                    }
                    double v = (double) lg[((size_t) h * c.Tq + i) * c.nk + j] * scale;
                    if (dist < c.extent)
                        for (int e = 0; e < d_rel; ++e)
                            v += (double) r[((size_t) i * H + h) * d_rel + e] * proj[(size_t) dist * d_rel + e];
                    l[j] = v;
                    m = std::max(m, v);
                }
                l[c.nk] = sink[(size_t) i * H + h];
                m = std::max(m, l[c.nk]);
                double sum = 0;
                for (int j = 0; j <= c.nk; ++j) sum += std::exp(l[j] - m);
                for (int j = 0; j < c.nk; ++j) want[((size_t) h * c.Tq + i) * c.nk + j] = std::exp(l[j] - m) / sum;
            }
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "%s (f16)", c.name);
        expect(nm, got, want, 2e-3, 1e-4);
        // the same with the bias as a table [Tq * H][extent] (the prefill's path)
        std::vector<float> bt((size_t) c.Tq * H * c.extent);
        for (int i = 0; i < c.Tq; ++i)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < c.extent; ++d) {
                    double b = 0;
                    for (int e = 0; e < d_rel; ++e)
                        b += (double) r[((size_t) i * H + h) * d_rel + e] * proj[(size_t) d * d_rel + e];
                    bt[((size_t) i * H + h) * c.extent + d] = (float) b;
                }
        float* dbt = dev(bt);
        k::attn_probs_bias_f16(dl, dout, H, c.Tq, c.nk, c.qpos0, c.kpos0, scale, dbt, c.extent, c.window, dsk, 0);
        const auto raw2 = host(dout, (size_t) H * c.Tq * c.nk);
        for (size_t i = 0; i < raw2.size(); ++i) got[i] = h2f_host(raw2[i]);
        std::snprintf(nm, sizeof nm, "%s (f16, bias table)", c.name);
        expect(nm, got, want, 2e-3, 1e-4);
        release({dl, dr, dp, dsk, dout, dbt});
    }
}

void test_route(int E) {
    const int T = 300, K = 8;
    auto lg = fvec((size_t) T * E, 6.0f), bias = fvec(E, 0.2f);
    // Several waves share the cutoff score. The lower expert IDs must win the tie.
    for (int e = 0; e < E; e += 4) {
        lg[e] = 8.0f;
        bias[e] = 0.4f;
    }
    float *dl = dev(lg), *db = dev(bias), *dw = dev_n<float>((size_t) T * K);
    int* dt = dev_n<int>((size_t) T * K);
    k::route(dl, db, T, E, K, 8.0f, dt, dw, 0);
    const auto top = host(dt, (size_t) T * K);
    const auto w = host(dw, (size_t) T * K);
    bool picks_ok = true;
    std::vector<double> want_w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<std::pair<double, int>> v(E);
        for (int e = 0; e < E; ++e) v[e] = {-(1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + e])) + bias[e]), e};
        std::sort(v.begin(), v.end());
        std::vector<int> pick(K);
        for (int i = 0; i < K; ++i) pick[i] = v[i].second;
        std::sort(pick.begin(), pick.end());
        double sum = 0;
        for (int i = 0; i < K; ++i) sum += 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]]));
        for (int i = 0; i < K; ++i) {
            picks_ok &= top[(size_t) t * K + i] == pick[i];
            want_w[(size_t) t * K + i] = 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]])) / (sum + 1e-20) * 8.0;
        }
    }
    expect_true("route: the picks", picks_ok);
    if (picks_ok) expect("route: the weights", w, want_w, 2e-5);
    // route_token: a token's route and route_gather in one launch, the same values to the bit (ties among them:
    // the logits of the first rows repeated)
    {
        std::vector<const uint8_t*> tab(E);
        for (int e = 0; e < E; ++e) tab[e] = (const uint8_t*) (uintptr_t) (0x1000 + 64 * e);
        auto sgt = fvec(E, 2.0f), sut = fvec(E, 2.0f), sdt = fvec(E, 2.0f);
        const uint8_t** dtab = dev(tab);
        float *dsg = dev(sgt), *dsu = dev(sut), *dsd = dev(sdt);
        bool same = true;
        std::vector<unsigned long long> usage(E);
        auto* device_usage = dev(usage);
        for (int t = 0; t < 40; ++t) {
            std::vector<float> row(lg.begin() + (size_t) t * E, lg.begin() + (size_t) (t + 1) * E);
            if (t >= 30)   // ties: a handful of equal logits and biases among the largest
                for (int e = 0; e < E; e += 7) row[e] = 5.5f;
            std::vector<float> b2 = bias;
            if (t >= 30)
                for (int e = 0; e < E; e += 7) b2[e] = 0.1f;
            float *dr = dev(row), *db2 = dev(b2);
            int *t1 = dev_n<int>(K), *t2 = dev_n<int>(K), *k1 = dev_n<int>(K), *k2 = dev_n<int>(K), *i1 = dev_n<int>(K),
                *i2 = dev_n<int>(K);
            float *w1 = dev_n<float>(K), *w2 = dev_n<float>(K), *g1 = dev_n<float>(K), *g2 = dev_n<float>(K),
                  *u1 = dev_n<float>(K), *u2 = dev_n<float>(K), *x1 = dev_n<float>(K), *x2 = dev_n<float>(K);
            const uint8_t **b1 = dev_n<const uint8_t*>(K), **bb2 = dev_n<const uint8_t*>(K);
            k::route(dr, db2, 1, E, K, 8.0f, t1, w1, 0);
            k::route_gather(t1, w1, dtab, dsg, dsu, dsd, K, K, b1, k1, i1, g1, u1, x1, 0);
            k::route_token(dr, db2, E, K, 8.0f, dtab, dsg, dsu, dsd, t2, w2, bb2, k2, i2, g2, u2, x2, 0, nullptr, 0, nullptr, false, nullptr, device_usage);
            for (int e : host(t1, K)) ++usage[e];
            same &= host(t1, K) == host(t2, K) && host(k1, K) == host(k2, K) && host(i1, K) == host(i2, K);
            same &= host(b1, K) == host(bb2, K);
            const auto a1 = host(w1, K), a2 = host(w2, K), c1 = host(g1, K), c2 = host(g2, K), d1 = host(u1, K),
                       d2 = host(u2, K), e1 = host(x1, K), e2 = host(x2, K);
            same &= !std::memcmp(a1.data(), a2.data(), K * 4) && !std::memcmp(c1.data(), c2.data(), K * 4) &&
                    !std::memcmp(d1.data(), d2.data(), K * 4) && !std::memcmp(e1.data(), e2.data(), K * 4);
            for (int* q : {t1, t2, k1, k2, i1, i2}) release({q});
            for (float* q : {w1, w2, g1, g2, u1, u2, x1, x2, dr, db2}) release({q});
            release({b1, bb2});
        }
        expect_true("route_token: route and route_gather's values to the bit (40 tokens, ties among them)", same);
        expect_true("route_token: per-expert usage", usage == host(device_usage, E));
        k::count_routes(dt, T * K, device_usage, 0);
        for (int e : top) ++usage[e];
        expect_true("batched per-expert usage", usage == host(device_usage, E));
        release({device_usage, dtab, dsg, dsu, dsd});
    }
    release({dl, db, dw, dt});
}

#ifdef STRATA_USE_HIP
void test_project_route() {
    constexpr int E = 128, K = 8, C = 256;
    auto wf = fvec(E*C);
    for (float& value : wf) value = std::round(value * 8) / 8;
    std::vector<uint16_t> wh(wf.size());
    for (size_t i = 0; i < wf.size(); ++i) wh[i] = f2h_host(wf[i]);
    auto x = fvec(64*C);
    for (float& value : x) value = std::round(value * 8) / 8;
    std::fill_n(x.begin(), C, 0.f); // all logits and biases tie for the first token
    auto* w = dev(wh);
    auto *dx = dev(x), *bias = dev(std::vector<float>(E)), *logs = dev_n<float>(64*E);
    auto *a = dev_n<int>(64*K), *b = dev_n<int>(64*K);
    auto *aw = dev_n<float>(64*K), *bw = dev_n<float>(64*K);
    for (int rows : {1, 8, 32, 64}) {
        for (int i = 0; i < rows; ++i) k::mv_f16(w, E, C, dx+i*C, logs+i*E, 0);
        k::route_rows(logs, bias, rows, E, K, 1.f, a, aw, 0);
        k::rdna3_project_route(w, dx, C, rows, bias, 1.f, b, bw, 0);
        expect_true("fused router picks and weights", host(a,rows*K) == host(b,rows*K) &&
                    host(aw,rows*K) == host(bw,rows*K));
    }
    release({w,dx,bias,logs,a,b,aw,bw});
}
#endif

void test_moe_pieces() {
    const int n = 37, F = 2304, C = 4608, T = 11, K = 8;
    // swiglu_cap over FP16 gate | up rows
    std::vector<uint16_t> gu((size_t) n * 2 * F);
    std::vector<float> guf(gu.size());
    {
        auto v = fvec(gu.size(), 12.0f);
        float* dv = dev(v);
        std::vector<int> idx(n * 2 * F / C);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
        int* di = dev(idx);
        uint16_t* dg = dev_n<uint16_t>(gu.size());
        k::gather_f16(dv, di, (int) idx.size(), C, dg, 0);   // (also gather_f16 with the identity)
        gu = host(dg, gu.size());
        bool ok = true;
        for (size_t i = 0; i < gu.size(); ++i) {
            guf[i] = h2f_host(gu[i]);
            ok &= std::fabs(guf[i] - v[i]) <= 6e-4f * std::fabs(v[i]) + 1e-7f;
        }
        expect_true("gather_f16: FP16 of the rows", ok);
        release({dv, di, dg});
    }
    auto sg = fvec(n, 2.0f), su = fvec(n, 3.0f);
    uint16_t *dgu = dev(gu), *dh = dev_n<uint16_t>((size_t) n * F);
    float *dsg = dev(sg), *dsu = dev(su);
    k::swiglu_cap(dgu, n, F, dsg, dsu, 4.0f, 25.0f, dh, 0);
    const auto h = host(dh, (size_t) n * F);
    std::vector<float> hf(h.size());
    std::vector<double> want(h.size());
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < F; ++c) {
            const double g = (double) guf[(size_t) r * 2 * F + c] * sg[r], u = (double) guf[(size_t) r * 2 * F + F + c] * su[r];
            want[(size_t) r * F + c] = 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g)) * 25.0 * std::tanh(u / 25.0);
            hf[(size_t) r * F + c] = h2f_host(h[(size_t) r * F + c]);
        }
    expect("swiglu_cap (FP16 out)", hf, want, 6e-4, 1e-6);
    release({dgu, dh, dsg, dsu});

    // the combine: token t's K slots are rows inv[t][i] of y
    const int rows = T * K;
    std::vector<uint16_t> y((size_t) rows * C);
    std::vector<float> yf(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = (uint16_t) (rng() & 0x7BFF) | (uint16_t) ((rng() & 1) << 15);   // any finite FP16
        if (((y[i] >> 10) & 31) > 20) y[i] &= ~(uint16_t) (8 << 10);          // (keep the sums small)
        yf[i] = h2f_host(y[i]);
    }
    std::vector<int> inv(rows);
    for (int i = 0; i < rows; ++i) inv[i] = i;
    std::shuffle(inv.begin(), inv.end(), rng);
    auto wt = fvec(rows, 2.0f), base = fvec((size_t) T * C, 5.0f);
    uint16_t* dy = dev(y);
    int* dinv = dev(inv);
    float *dwt = dev(wt), *dout = dev(base);
    for (int add = 1; add >= 0; --add) {
        k::moe_combine(dy, dinv, dwt, T, K, C, dout, add, 0);
        std::vector<double> w2((size_t) T * C);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) {
                double a = add ? base[(size_t) t * C + c] : 0.0;
                for (int i = 0; i < K; ++i) a += (double) wt[(size_t) t * K + i] * yf[(size_t) inv[(size_t) t * K + i] * C + c];
                w2[(size_t) t * C + c] = a;
            }
        expect(add ? "moe_combine (add)" : "moe_combine (overwrite)", host(dout, w2.size()), w2, 2e-5, 2e-4);
    }
    release({dy, dinv, dwt, dout});

    auto lgt = fvec(50000, 90.0f);
    float* dlg = dev(lgt);
    k::softcap(dlg, lgt.size(), 30.0f, 0);
    std::vector<double> w3(lgt.size());
    for (size_t i = 0; i < lgt.size(); ++i) w3[i] = 30.0 * std::tanh((double) lgt[i] / 30.0);
    expect("softcap", host(dlg, lgt.size()), w3, 2e-5);
    auto o = fvec(4096, 3.0f), g = fvec(4096, 8.0f);
    float *dob = dev(o), *dgb = dev(g);
    k::gate_mul(dob, dgb, o.size(), 0);
    std::vector<double> w4(o.size());
    for (size_t i = 0; i < o.size(); ++i) w4[i] = (double) o[i] / (1.0 + std::exp(-(double) g[i]));
    expect("gate_mul", host(dob, o.size()), w4, 2e-5);
    release({dlg, dob, dgb});
}

// NVFP4 by the format's definition (ggml's dequantize_row_nvfp4): 64 values in 36 bytes - four UE4M3 scales, then for
// each 16 values 8 bytes, value j < 8 the low nibble of byte j and value j + 8 its high one
double nvfp4_def(const uint8_t* blocks, size_t v) {
    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const uint8_t* b = blocks + (v / 64) * 36;
    const int i = (int) (v % 64), sb = i / 16, j = i % 16;
    const uint8_t x = b[sb];
    double d = 0.0;
    if (x != 0 && x != 0x7F) {
        const int e = (x >> 3) & 0xF, m = x & 7;
        d = (e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7)) * 0.5;
    }
    const uint8_t byte = b[4 + sb * 8 + j % 8];
    return kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * d;
}

void test_decode() {
    const int types[] = {k::kNVFP4, k::kQ8_0, k::kQ5_K, k::kQ6_K, k::kIQ3_XXS, k::kIQ4_XS, k::kIQ4_NL};
    const char* names[] = {"NVFP4", "Q8_0", "Q5_K", "Q6_K", "IQ3_XXS", "IQ4_XS", "IQ4_NL"};
    for (int ti = 0; ti < (int) (sizeof types / sizeof types[0]); ++ti) {
        const int type = types[ti], bv = k::block_values(type);
        const size_t bb = k::block_bytes(type), nblk = 777, n = nblk * bv;
        std::vector<uint8_t> raw(nblk * bb);
        for (auto& b : raw) b = (uint8_t) rng();
        // the F16 scales: finite and moderate (random bits would give inf / nan scales)
        for (size_t b = 0; b < nblk; ++b) {
            auto fix = [&](size_t off) {
                uint8_t* p = raw.data() + b * bb + off;
                uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
                h = (uint16_t) ((h & 0x83FF) | ((uint16_t) (8 + (h >> 10) % 8) << 10));
                p[0] = (uint8_t) h, p[1] = (uint8_t) (h >> 8);
            };
            if (type == k::kQ8_0 || type == k::kIQ4_NL) fix(0);
            if (type == k::kQ5_K) fix(0), fix(2);
            if (type == k::kQ6_K) fix(208);
        if (type == k::kIQ3_XXS || type == k::kIQ4_XS) fix(0);
        }
        std::vector<float> hostv(n);
        k::decode_host(type, raw.data(), n, hostv.data());
        uint8_t* dr = dev(raw);
        float* d32 = dev_n<float>(n);
        uint16_t* d16 = dev_n<uint16_t>(n);
        k::decode_f32(type, dr, n, d32, 0);
        k::decode_f16(type, dr, n, d16, 0);
        const auto g32 = host(d32, n);
        const auto g16 = host(d16, n);
        bool same = std::memcmp(g32.data(), hostv.data(), n * sizeof(float)) == 0;
        char name[80];
        std::snprintf(name, sizeof name, "decode %s: device F32 = host, bit for bit", names[ti]);
        expect_true(name, same);
        bool f16_ok = true, exact = true;
        for (size_t i = 0; i < n; ++i) {
            const float back = h2f_host(g16[i]);
            f16_ok &= std::fabs(back - hostv[i]) <= 4.9e-4f * std::fabs(hostv[i]) + 6e-8f || !std::isfinite(back);
            exact &= back == hostv[i];
        }
        std::snprintf(name, sizeof name, "decode %s: FP16 = the value rounded%s", names[ti],
                      type == k::kNVFP4 ? " (exact)" : "");
        expect_true(name, f16_ok && (type != k::kNVFP4 || exact));
        if (type == k::kNVFP4) {
            bool def = true;
            for (size_t i = 0; i < n; ++i) def &= (double) hostv[i] == nvfp4_def(raw.data(), i);
            expect_true("decode NVFP4: the format's definition", def);
        }
        release({dr, d32, d16});
    }
}

// random blocks of `type` whose F16 scales are finite and moderate
std::vector<uint8_t> rand_blocks(int type, size_t n_values) {
    const size_t bb = k::block_bytes(type), nb = n_values / k::block_values(type);
    std::vector<uint8_t> raw(nb * bb);
    for (auto& b : raw) b = (uint8_t) rng();
    for (size_t b = 0; b < nb; ++b) {
        auto fix = [&](size_t off) {
            uint8_t* p = raw.data() + b * bb + off;
            uint16_t h = (uint16_t) (p[0] | (p[1] << 8));
            h = (uint16_t) ((h & 0x83FF) | ((uint16_t) (8 + (h >> 10) % 8) << 10));
            p[0] = (uint8_t) h, p[1] = (uint8_t) (h >> 8);
        };
        if (type == k::kQ8_0 || type == k::kIQ4_NL) fix(0);
        if (type == k::kQ5_K) fix(0), fix(2);
        if (type == k::kQ6_K) fix(208);
        if (type == k::kIQ3_XXS || type == k::kIQ4_XS) fix(0);
    }
    return raw;
}

// ggml's dequantize_row_iq3_xxs / dequantize_row_iq4_xs, transcribed (ksigns_iq2xs[i] = i | parity(i) << 7)
static const uint32_t g_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
static const int8_t g_iq4nl[16] = {IK2_IQ4NL_VALUES};
void ggml_iq3_xxs(const uint8_t* x, float* y, size_t k) {
    for (size_t i = 0; i < k / 256; ++i) {
        const uint8_t* blk = x + i * 98;
        const float d = h2f_host((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, 4);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint32_t s7 = (aux32 >> 7 * l) & 127;
                uint32_t par = s7 ^ (s7 >> 4);   // (its parity, folded: portable)
                par ^= par >> 2;
                par ^= par >> 1;
                const uint8_t signs = (uint8_t) (s7 | (par & 1) << 7);
                const uint8_t* grid1 = (const uint8_t*) (g_iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*) (g_iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & (1 << j) ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & (1 << (j + 4)) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}
void ggml_iq4_xs(const uint8_t* x, float* y, size_t k) {
    for (size_t i = 0; i < k / 256; ++i) {
        const uint8_t* blk = x + i * 136;
        const float d = h2f_host((uint16_t) (blk[0] | (blk[1] << 8)));
        const uint16_t scales_h = (uint16_t) (blk[2] | (blk[3] << 8));
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8;
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * g_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * g_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}
void test_iq_ref() {
    for (int type : {k::kIQ3_XXS, k::kIQ4_XS, k::kIQ4_NL}) {
        const size_t n = 256 * 333;
        auto raw = rand_blocks(type, n);
        std::vector<float> want(n), got(n);
        if (type == k::kIQ4_NL) {
            for (size_t row = 0; row < n / 32; ++row) {
                const auto* block = raw.data() + 18 * row;
                const float scale = h2f_host(block[0] | (block[1] << 8));
                for (int i = 0; i < 16; ++i) {
                    want[32 * row + i] = scale * g_iq4nl[block[2 + i] & 15];
                    want[32 * row + i + 16] = scale * g_iq4nl[block[2 + i] >> 4];
                }
            }
        } else {
            (type == k::kIQ3_XXS ? ggml_iq3_xxs : ggml_iq4_xs)(raw.data(), want.data(), n);
        }
        k::decode_host(type, raw.data(), n, got.data());
        expect(type == k::kIQ3_XXS ? "decode_host IQ3_XXS = ggml" : type == k::kIQ4_XS ? "decode_host IQ4_XS = ggml" : "decode_host IQ4_NL = ggml", got,
               std::vector<double>(want.begin(), want.end()), 0.0, 0.0);
    }
}

void test_experts_mv() {
    const int S = 5;
    struct Case {
        int tg, tu, td;
        const char* name;
        int C = 512, F = 256;
    };
    const Case cases[] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4, "NVFP4"},
                          {k::kQ5_K, k::kQ5_K, k::kQ6_K, "Q5_K / Q6_K"},
                          {k::kQ8_0, k::kNVFP4, k::kQ6_K, "mixed"},
                          {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4, "IQ3_XXS / NVFP4"},
                          {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K, "IQ4_XS / Q6_K"},
                          {k::kIQ4_NL, k::kIQ4_NL, k::kIQ4_NL, "IQ4_NL"},
                          {k::kIQ4_NL, k::kIQ4_NL, k::kIQ4_NL, "IQ4_NL gate K=2304", 2304, 32},
                          {k::kIQ4_NL, k::kIQ4_NL, k::kIQ4_NL, "IQ4_NL down K=1152", 32, 1152},
                          {k::kQ8_0, k::kQ8_0, k::kQ8_0, "Q8_0 gate K=2304", 2304, 32},
                          {k::kQ8_0, k::kQ8_0, k::kQ8_0, "Q8_0 down K=1152", 32, 1152}};
    for (const Case& cs : cases) {
        const int C = cs.C, F = cs.F;
        const int E = 3;
        auto g = rand_blocks(cs.tg, (size_t) E * F * C), u = rand_blocks(cs.tu, (size_t) E * F * C),
             d = rand_blocks(cs.td, (size_t) E * C * F);
        const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E;
        // each expert's blob: gate | up | down, 256-byte aligned slots
        const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
        std::vector<uint8_t> pool(slot * E);
        for (int e = 0; e < E; ++e) {
            std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
            std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
        }
        uint8_t* dpool = dev(pool);
        const int pick[S] = {2, 0, 2, 1, 0}, tok[S] = {0, 1, 1, 2, 0};
        std::vector<const uint8_t*> blob(S);
        for (int i = 0; i < S; ++i) blob[i] = dpool + pick[i] * slot;
        const uint8_t** dblob = nullptr;
        CU(cudaMalloc(&dblob, S * sizeof(void*)));
        CU(cudaMemcpy(dblob, blob.data(), S * sizeof(void*), cudaMemcpyHostToDevice));
        auto x = fvec((size_t) 3 * C, 2.0f);
        float *dx = dev(x), *dgu = dev_n<float>((size_t) S * 2 * F), *dy = dev_n<float>((size_t) S * C);
        int* dtok = dev(std::vector<int>(tok, tok + S));
        k::experts_gate_up(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
        std::vector<uint16_t> h((size_t) S * F);
        std::vector<float> hf(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
            hf[i] = h2f_host(h[i]);
        }
        uint16_t* dh = dev(h);
        k::experts_down(dblob, S, cs.td, eb0 + eb1, dh, C, F, dy, 0);
        std::vector<double> want_gu((size_t) S * 2 * F), want_y((size_t) S * C);
        std::vector<float> row(C > F ? C : F);
        for (int i = 0; i < S; ++i) {
            const int e = pick[i];
            for (int r = 0; r < 2 * F; ++r) {
                const bool up = r >= F;
                const int rr = up ? r - F : r;
                const int ty = up ? cs.tu : cs.tg;
                const uint8_t* w = (up ? u.data() + e * eb1 : g.data() + e * eb0) + (size_t) rr * (eb0 / F);
                if (up) w = u.data() + e * eb1 + (size_t) rr * (eb1 / F);
                k::decode_host(ty, w, C, row.data());
                double a = 0;
                for (int c = 0; c < C; ++c) a += (double) row[c] * x[(size_t) tok[i] * C + c];
                want_gu[(size_t) i * 2 * F + r] = a;
            }
            for (int r = 0; r < C; ++r) {
                k::decode_host(cs.td, d.data() + e * eb2 + (size_t) r * (eb2 / C), F, row.data());
                double a = 0;
                for (int c = 0; c < F; ++c) a += (double) row[c] * hf[(size_t) i * F + c];
                want_y[(size_t) i * C + r] = a;
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "experts_gate_up %s", cs.name);
        expect(name, host(dgu, want_gu.size()), want_gu, 2e-4, 2e-3);
        std::snprintf(name, sizeof name, "experts_down %s", cs.name);
        expect(name, host(dy, want_y.size()), want_y, 2e-4, 2e-3);
        if (cs.tg == cs.tu && (cs.tg == k::kIQ4_NL || cs.tg == k::kQ8_0)) {
            auto sg = fvec(S), su = fvec(S);
            auto *dg = dev(sg), *du = dev(su);
            auto* hidden = dev_n<uint16_t>((size_t)S * F);
            k::experts_gate_up_swiglu(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, dg, du, 12.f, 19.f, hidden, 0);
            const auto halves = host(hidden, (size_t)S * F);
            std::vector<float> got(halves.size());
            std::vector<double> expected(got.size());
            for (int i = 0; i < S; ++i) {
                for (int j = 0; j < F; ++j) {
                    const double g = want_gu[(size_t)i * 2 * F + j] * sg[i];
                    const double u = want_gu[(size_t)i * 2 * F + F + j] * su[i];
                    expected[(size_t)i * F + j] = 12. * std::tanh(g / 12.) / (1. + std::exp(-g)) *
                                                   19. * std::tanh(u / 19.);
                    got[(size_t)i * F + j] = h2f_host(halves[(size_t)i * F + j]);
                }
            }
            expect(cs.tg == k::kIQ4_NL ? "IQ4_NL fused SwiGLU vs FP64" : "Q8_0 fused SwiGLU vs FP64", got, expected, 1e-3, 1e-4);
            release({hidden, dg, du});
        }
        // S = 1 path (<4, 2> launch config)
        k::experts_gate_up(dblob, dtok, 1, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
        k::experts_down(dblob, 1, cs.td, eb0 + eb1, dh, C, F, dy, 0);
        std::snprintf(name, sizeof name, "experts_gate_up S=1 %s", cs.name);
        expect(name, host(dgu, 2 * F), std::vector<double>(want_gu.begin(), want_gu.begin() + 2 * F), 2e-4, 2e-3);
        std::snprintf(name, sizeof name, "experts_down S=1 %s", cs.name);
        expect(name, host(dy, C), std::vector<double>(want_y.begin(), want_y.begin() + C), 2e-4, 2e-3);
        release({dpool, dblob, dx, dgu, dy, dtok, dh});
    }
}

#ifdef STRATA_USE_HIP
void test_rdna3_equal_attention_scores() {
    constexpr int D = 128, nk = 65;
    std::vector<uint8_t> keys(nk * D), values(nk * D);
    for (int d = 0; d < D; ++d) {
        keys[d] = (13 * d + 23) % 126;
        keys[(nk - 1) * D + d] = (13 * ((7 * d) % D) + 23) % 126;
        values[d] = 0x78;
        values[(nk - 1) * D + d] = 0xf8;
    }
    // A constant query gives both permutations the same dot product. Their opposite values cancel exactly.
    auto* q = dev(std::vector<float>(D, 1.234567f));
    auto* k0 = dev(keys);
    auto* v = dev(values);
    auto* zero = dev(std::vector<float>(16, 0.f));
    auto* pos = dev(std::vector<int>{nk - 1});
    auto* scratch = dev_n<float>(2 * 130);
    auto* y = dev_n<float>(D);
    const float scale = 1.f / std::sqrt(128.f) / 16.f;
    const std::vector<double> expected(D, 0.);
    k::attn_decode_fp8(q, zero, zero, k0, v, 0, 1, 1, D, 0, nk, nk - 1, scale, 1.f / 32.f,
                       zero, 16, 0, nk, scratch, y, 0);
    expect("equal E4M3 scores in separate splits", host(y, D), expected, 0., 5e-6);
    k::attn_decode_fp8_at(q, zero, zero, k0, v, 0, 1, 1, pos, 2, scale, 1.f / 32.f,
                          zero, 16, 0, nk, scratch, y, 0);
    expect("equal resident E4M3 scores in separate splits", host(y, D), expected, 0., 5e-6);
    release({q, k0, v, zero, pos, scratch, y});
}

void test_rdna3_attention_output_projection() {
    constexpr int rows = 37, width = 3072;
    std::mt19937 gen(193);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> x(width);
    std::vector<int8_t> q(rows * width);
    std::vector<uint16_t> d(rows * (width / 32));
    for (float& v : x) v = random(gen);
    for (int8_t& v : q) v = (int)(gen() % 256) - 128;
    for (uint16_t& v : d) v = f2h_host(random(gen) / 32);
    auto* dx = dev(x);
    auto* dq = dev(q);
    auto* dd = dev(d);
    auto* dy = dev_n<float>(rows);
    k::mv_q8p(dq, dd, rows, width, dx, dy, 0);
    std::vector<double> expected(rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c)
            expected[r] += (double)h2f_host(d[(r * width + c) / 32]) * q[r * width + c] * x[c];
    expect("RDNA3 3072-wide Q8 projection vs FP64", host(dy, rows), expected, 2e-5, 2e-5);
    release({dx, dq, dd, dy});
}

void test_rdna3_loop_gate() {
    constexpr int C = 2304, R = 64, jobs = 3;
    std::mt19937 gen(714);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<uint16_t> weights(4 * C * R);
    std::vector<float> w(weights.size()), x(jobs * C), m(jobs * C);
    for (size_t i = 0; i < w.size(); ++i) {
        float v = random(gen) / 32;
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        weights[i] = bits >> 16;
        bits &= 0xffff0000u;
        std::memcpy(&w[i], &bits, sizeof(bits));
    }
    for (float& v : x) v = random(gen);
    for (float& v : m) v = random(gen);
    auto *dx = dev(x), *dm = dev(m), *scratch = dev_n<float>(jobs * (2 * R + 2));
    auto* dw = dev(weights);
    k::LoopGate gates[jobs]{};
    std::vector<double> expected(jobs * C);
    for (int j = 0; j < jobs; ++j) {
        const int off = j * C;
        gates[j] = {dx + off, dm + off, nullptr, nullptr, nullptr, dx + off, {dw, dw + 2 * R * C, dw + 3 * R * C}};
        double sx = 0., sm = 0.;
        for (int c = 0; c < C; ++c) { sx += (double)x[off+c] * x[off+c]; sm += (double)m[off+c] * m[off+c]; }
        double rank[R];
        for (int r = 0; r < R; ++r) {
            double ax = 0., am = 0.;
            for (int c = 0; c < C; ++c) {
                ax += (double)w[2*r*C + c] * x[off+c];
                am += (double)w[(2*r+1)*C + c] * m[off+c];
            }
            rank[r] = ax / std::sqrt(sx/C + 1e-6) + am / std::sqrt(sm/C + 1e-6);
        }
        for (int c = 0; c < C; ++c) {
            double forget = 0., gain = 0.;
            for (int r = 0; r < R; ++r) {
                forget += w[(2*R+r)*C + c] * rank[r];
                gain += w[(3*R+r)*C + c] * rank[r];
            }
            expected[off+c] = (1. - std::tanh(forget)) * x[off+c] + std::tanh(gain) * m[off+c];
        }
    }
    k::loop_gates(gates, jobs, 1, C, R, 1e-6f, scratch, 0);
    expect("three in-place BF16 loop gates vs FP64", host(dx, expected.size()), expected, 2e-5, 2e-6);
    release({dx, dm, scratch, dw});
}

void test_iq4nl_gate_cancellation() {
    constexpr int C = 512, F = 2;
    constexpr size_t row_bytes = C / 32 * 18, gate_bytes = F * row_bytes;
    std::vector<uint8_t> weights(2 * gate_bytes, 0x88); // Both nibbles decode to +1.
    for (size_t b = 0; b < weights.size(); b += 18) {
        weights[b] = 0;
        weights[b + 1] = 0x3c; // Unit FP16 block scale.
    }
    for (int r = 0; r < F; ++r) weights[gate_bytes + r * row_bytes + 2] = 0x89;
    std::vector<float> x(C);
    x[0] = 4096.f;
    x[4] = -4096.f;
    x[64] = 1.f / 8192;
    auto* pool = dev(weights);
    auto* blobs = dev(std::vector<const uint8_t*>{pool});
    auto* tokens = dev(std::vector<int>{0});
    auto* input = dev(x);
    auto* hidden = dev_n<uint16_t>(F);
    k::experts_gate_up_swiglu(blobs, tokens, 1, k::kIQ4_NL, k::kIQ4_NL, gate_bytes, input, C, F,
                              nullptr, nullptr, 12.f, 19.f, hidden, 0);
    const double gate = x[64], up = 12. * x[0] + x[64];
    const double expected = 12. * std::tanh(gate / 12.) / (1. + std::exp(-gate)) * 19. * std::tanh(up / 19.);
    const auto result = host(hidden, F);
    std::vector<float> activations;
    for (uint16_t bits : result) activations.push_back(h2f_host(bits));
    expect("IQ4_NL cancelling gate vs FP64", activations, std::vector<double>(F, expected), 5e-4, 1e-7);
    release({hidden, input, tokens, blobs, pool});
}
#endif

// the grouped kernels against the per-slot ones: 8 tokens, variable group sizes (1, 2, 3, 5, 8) over all quant types
void test_experts_grouped() {
    const int C = 512, F = 256, E = 5, T = 8, K = 8, S = T * K;
    const int types[][3] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4},
                            {k::kQ5_K, k::kQ5_K, k::kQ6_K},
                            {k::kQ8_0, k::kQ8_0, k::kQ8_0},
                            {k::kQ8_0, k::kNVFP4, k::kQ6_K},
                            {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4},
                            {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K}};
    const char* tnames[] = {"NVFP4", "Q5_K/Q6_K", "Q8_0", "Q8_0/NVFP4/Q6_K", "IQ3_XXS/NVFP4", "IQ4_XS/Q6_K"};
    for (int ti = 0; ti < 6; ++ti) {
        const auto& ty = types[ti];
        auto g = rand_blocks(ty[0], (size_t) E * F * C), u = rand_blocks(ty[1], (size_t) E * F * C),
             d = rand_blocks(ty[2], (size_t) E * C * F);
        const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E, slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
        std::vector<uint8_t> pool(slot * E);
        for (int e = 0; e < E; ++e) {
            std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
            std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
        }
        uint8_t* dpool = dev(pool);
        // Assign experts so group sizes hit 1, 2, 3, 5, 8 across the 5 experts (distinct within each token)
        const int want_ng[E] = {1, 2, 3, 5, 8};
        std::vector<const uint8_t*> blob(S);
        std::vector<int> tok(S), gn, gtok, grow;
        std::vector<const uint8_t*> gb;
        std::vector<int> pick(S, -1);
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < want_ng[e]; ++t) pick[t * K + e] = e;
        std::vector<int> idx;   // the slots in use
        for (int i = 0; i < S; ++i)
            if (pick[i] >= 0) idx.push_back(i);
        const int n = (int) idx.size();
        std::vector<int> gi(E, -1);
        for (int r = 0; r < n; ++r) {
            const int i = idx[r], e = pick[i];
            blob[r] = dpool + e * slot;
            tok[r] = i / K;
            if (gi[e] < 0) {
                gi[e] = (int) gb.size();
                gb.push_back(dpool + e * slot);
                gn.push_back(0);
                gtok.resize(gtok.size() + k::kGroupMax);
                grow.resize(grow.size() + k::kGroupMax);
            }
            const int gg = gi[e];
            gtok[gg * k::kGroupMax + gn[gg]] = i / K;
            grow[gg * k::kGroupMax + gn[gg]] = r;
            ++gn[gg];
        }
        const uint8_t** dblob = nullptr;
        const uint8_t** dgb = nullptr;
        CU(cudaMalloc(&dblob, n * sizeof(void*)));
        CU(cudaMalloc(&dgb, gb.size() * sizeof(void*)));
        CU(cudaMemcpy(dblob, blob.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dgb, gb.data(), gb.size() * sizeof(void*), cudaMemcpyHostToDevice));
        int *dtok = dev(tok), *dgn = dev(gn), *dgt = dev(gtok), *dgr = dev(grow);
        auto x = fvec((size_t) T * C, 2.0f);
        std::vector<uint16_t> h((size_t) n * F);
        for (auto& v : h) v = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
        float *dx = dev(x), *gu1 = dev_n<float>((size_t) n * 2 * F), *gu2 = dev_n<float>((size_t) n * 2 * F);
        float *y1 = dev_n<float>((size_t) n * C), *y2 = dev_n<float>((size_t) n * C);
        uint16_t* dh = dev(h);
        k::experts_gate_up(dblob, dtok, n, ty[0], ty[1], eb0, dx, C, F, gu1, 0);
        k::experts_gate_up_g(dgb, dgn, dgt, dgr, (int) gb.size(), ty[0], ty[1], eb0, dx, C, F, gu2, 0);
        k::experts_down(dblob, n, ty[2], eb0 + eb1, dh, C, F, y1, 0);
        k::experts_down_g(dgb, dgn, dgr, (int) gb.size(), ty[2], eb0 + eb1, dh, C, F, y2, 0);
        const auto a1 = host(gu1, (size_t) n * 2 * F), b1 = host(y1, (size_t) n * C);
        std::vector<double> wa(a1.begin(), a1.end()), wb(b1.begin(), b1.end());
        char name[80];
        std::snprintf(name, sizeof name, "experts_gate_up_g (%zu groups) %s", gb.size(), tnames[ti]);
        // (two F32 orders of the same sums: random NVFP4 blocks make terms of 1e3 - 1e4 that cancel to a few units,
        //  so the absolute part of the tolerance follows the terms, not the result)
        expect(name, host(gu2, wa.size()), wa, 1e-4, 2e-2);
        std::snprintf(name, sizeof name, "experts_down_g %s", tnames[ti]);
        // (the same two orders over F terms: 1 of 9728 sums went 2e-3 off at a value of 6.5, IQ4_XS / Q6_K on H200)
        expect(name, host(y2, wb.size()), wb, 1e-4, 5e-3);
        for (void* q : {(void*) dpool, (void*) dblob, (void*) dgb, (void*) dtok, (void*) dgn, (void*) dgt, (void*) dgr,
                        (void*) dx, (void*) gu1, (void*) gu2, (void*) y1, (void*) y2, (void*) dh})
            release({q});
    }
}

// several queries against attn_decode a query at a time
void test_attn_multi() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, nq = 8;
    auto kc = fvec((size_t) ctx * KV * D), vc = fvec((size_t) ctx * KV * D);
    auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f), sink = fvec((size_t) nq * H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    float *dk = dev(kc), *dv = dev(vc), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
          *one = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);
    struct Case { int pos0, window; const char* name; };
    const Case cases[] = {{3000, 1 << 30, "attn_multi full, 8 queries at 3000"}, {2500, 1024, "attn_multi sliding 1024"},
                          {0, 1 << 30, "attn_multi from position 0"}, {100, 1 << 30, "attn_multi at 100 (one run)"}};
    for (const Case& cs : cases) {
        const int k0 = cs.window < (1 << 30) ? std::max(0, cs.pos0 - cs.window + 1) : 0, nk = cs.pos0 + nq - k0;
        k::attn_multi(dq, dr, ds, dk, dv, nq, H, KV, k0, nk, cs.pos0, scale, dp, d_rel, 1024, cs.window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = cs.pos0 + qi, kq = cs.window < (1 << 30) ? std::max(0, qp - cs.window + 1) : 0;
            k::attn_decode(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk, dv, H, KV, D,
                           kq, qp + 1 - kq, qp, scale, dp, d_rel, 1024, cs.window, scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect(cs.name, got, want, 1e-4, 1e-6);
    }
    for (float* p : {dk, dv, dq, dr, ds, dp, scratch, out, one}) release({p});
}

void test_mv_blocks() {
    struct Case {
        int type, N, K;
        const char* name;
    };
    const Case cases[] = {
        {k::kQ8_0, 384, 4608, "Q8_0 N=384"},
        {k::kQ8_0, 1031, 6144, "Q8_0 N=1031"},
        {k::kQ8_0, 2307, 4608, "Q8_0 N=2307"},
        {k::kQ8_0, 33, 96, "Q8_0 odd nb"},
        {k::kNVFP4, 1031, 6144, "NVFP4 N=1031"},
        {k::kQ6_K, 1031, 6144, "Q6_K N=1031"},
        {k::kQ5_K, 513, 4608, "Q5_K N=513"},
        {k::kIQ3_XXS, 1031, 4608, "IQ3_XXS N=1031"},
        {k::kIQ4_XS, 513, 4608, "IQ4_XS N=513"},
    };
    for (const Case& cs : cases) {
        const int N = cs.N, K = cs.K;
        auto raw = rand_blocks(cs.type, (size_t) N * K);
        auto x = fvec(K, 2.0f);
        uint8_t* dw = dev(raw);
        float *dx = dev(x), *dy = dev_n<float>(N);
        k::mv_blocks(cs.type, dw, N, K, dx, dy, 0);
        std::vector<double> want(N);
        std::vector<float> row(K);
        const size_t rb = raw.size() / N;
        for (int r = 0; r < N; ++r) {
            k::decode_host(cs.type, raw.data() + (size_t) r * rb, K, row.data());
            double a = 0;
            for (int c = 0; c < K; ++c) a += (double) row[c] * x[c];
            want[r] = a;
        }
        char name[64];
        std::snprintf(name, sizeof name, "mv_blocks %s", cs.name);
        // Cancellation over K=4608..6144 random blocks leaves ~1e-2 F32 rounding residual (same as test_experts_grouped).
        expect(name, host(dy, N), want, 2e-4, K > 2048 ? 3e-2 : 2e-3);
        release({dw, dx, dy});
    }
}

void test_attn_decode() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3000;
    struct Case {
        int k0, nk, qpos, extent, window;
        const char* name;
    };
    const Case cases[] = {{0, 1, 0, 1024, 1 << 30, "attn_decode first token"},
                          {0, 2500, 2499, 1024, 1 << 30, "attn_decode full, 2500 keys"},
                          {1476, 1024, 2499, 1024, 1024, "attn_decode sliding 1024"},
                          {0, 200, 199, 1024, 1 << 30, "attn_decode full, 200 keys (one run)"},
                          {0, 2999, 2998, 64, 1 << 30, "attn_decode full, 2999 keys, extent 64"}};
    auto kc = fvec((size_t) ctx * KV * D), vc = fvec((size_t) ctx * KV * D);
    auto q = fvec((size_t) H * D, 3.0f), r = fvec((size_t) H * d_rel, 2.0f), sink = fvec(H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    float *dk = dev(kc), *dv = dev(vc), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) H * ctx + (size_t) H * 32 * 130), *dout = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);
    for (const Case& cs : cases) {
        k::attn_decode(dq, dr, ds, dk, dv, H, KV, D, cs.k0, cs.nk, cs.qpos, scale, dp, d_rel, cs.extent, cs.window,
                       scratch, dout, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(cs.nk);
            double m = sink[h];
            for (int j = 0; j < cs.nk; ++j) {
                const int dist = cs.qpos - (cs.k0 + j);
                if (dist < 0 || dist >= cs.window) {
                    l[j] = -INFINITY;
                    continue;
                }
                double a = 0;
                for (int d = 0; d < D; ++d) a += (double) q[h * D + d] * kc[((size_t) (cs.k0 + j) * KV + g) * D + d];
                a *= scale;
                if (dist < cs.extent)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < cs.nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < cs.nk; ++j)
                    o += std::exp(l[j] - m) / sum * vc[((size_t) (cs.k0 + j) * KV + g) * D + d];
                want[(size_t) h * D + d] = o;
            }
        }
        expect(cs.name, host(dout, want.size()), want, 2e-4, 1e-6);
    }
    for (float* p : {dk, dv, dq, dr, ds, dp, scratch, dout}) release({p});
}

void test_argmax() {
    // argmax_sum: multi-block (200058), single-block (777), null b/c, unaligned, and tie-breaking (lower index)
    {
        const int n = 200058;
        auto a = fvec(n, 5.0f), b = fvec(n, 5.0f), c = fvec(n, 5.0f);
        // Set two exact ties across blocks (500 and 150000) above the random range
        a[500] = 20.0f, b[500] = 10.0f, c[500] = 5.0f;
        a[150000] = 20.0f, b[150000] = 10.0f, c[150000] = 5.0f;
        float *da = dev(a), *db = dev(b), *dc = dev(c);
        int* dout = dev_n<int>(1);
        k::argmax_sum(da, db, dc, n, dout, 0);
        expect_true("argmax_sum (a+b+c, tie)", host(dout, 1)[0] == 500);
        a[199999] = 50.0f;
        CU(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        k::argmax_sum(da, nullptr, nullptr, n, dout, 0);
        expect_true("argmax_sum (a only, multi)", host(dout, 1)[0] == 199999);
        k::argmax_sum(da + 1, db + 1, nullptr, n - 1, dout, 0);
        int want_unaligned = 0;
        float best_u = -1e30f;
        for (int i = 0; i < n - 1; ++i) {
            const float v = a[1 + i] + b[1 + i];
            if (v > best_u) best_u = v, want_unaligned = i;
        }
        expect_true("argmax_sum (unaligned)", host(dout, 1)[0] == want_unaligned);
        k::argmax_sum(da, db, dc, 777, dout, 0);
        expect_true("argmax_sum (single block)", host(dout, 1)[0] == 500);
        release({da, db, dc, dout});
    }
    // argmax_rows: 8 rows x 200058 cols with ld = 201024, plus single-block case
    {
        const int rows = 8, cols = 200058, ld = 201024;
        auto x = fvec((size_t) rows * ld, 5.0f);
        x[3 * (size_t) ld + 1234] = 30.0f;
        x[3 * (size_t) ld + 180000] = 30.0f;
        float* dx = dev(x);
        int* dout = dev_n<int>(rows);
        k::argmax_rows(dx, ld, rows, cols, dout, 0);
        const auto got = host(dout, rows);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            int w = 0;
            for (int c = 1; c < cols; ++c)
                if (x[(size_t) r * ld + c] > x[(size_t) r * ld + w]) w = c;
            ok &= got[r] == w;
        }
        expect_true("argmax_rows (8 x 200058, tie)", ok);
        k::argmax_rows(dx, ld, 1, 500, dout, 0);
        int w1 = 0;
        for (int c = 1; c < 500; ++c)
            if (x[c] > x[w1]) w1 = c;
        expect_true("argmax_rows (1 x 500)", host(dout, 1)[0] == w1);
        release({dx, dout});
    }
    // argmax_gemv_f16: W [R][ld] FP16, x [R] F32, base [n] F32, bias_f16 [n] FP16
    {
        const int R = 256, n = 200058, ld = 201024;
        std::vector<uint16_t> wh((size_t) R * ld), bh(n);
        for (auto& v : wh) v = f2h_host(frand(0.5f));
        for (auto& v : bh) v = f2h_host(frand(1.0f));
        auto x = fvec(R, 0.5f), base = fvec(n, 2.0f);
        uint16_t *dw = dev(wh), *dbias = dev(bh);
        float *dx = dev(x), *dbase = dev(base);
        int* dout = dev_n<int>(1);
        k::argmax_gemv_f16(dw, ld, dx, dbase, dbias, R, n, dout, 0);
        auto ref_gemv = [&](int off_w, int w_ld, int nn, bool use_base, bool use_bias) {
            int best_i = 0;
            float best_v = -INFINITY;
            for (int v = 0; v < nn; ++v) {
                float acc = 0.0f;
                for (int r = 0; r < R; ++r) acc = std::fma(h2f_host(wh[(size_t) r * w_ld + off_w + v]), x[r], acc);
                if (use_base) acc += base[v];
                if (use_bias) acc += h2f_host(bh[v]);
                if (acc > best_v) best_v = acc, best_i = v;
            }
            return best_i;
        };
        expect_true("argmax_gemv_f16 (200058)", host(dout, 1)[0] == ref_gemv(0, ld, n, true, true));
        k::argmax_gemv_f16(dw, ld, dx, nullptr, nullptr, R, 200057, dout, 0);
        expect_true("argmax_gemv_f16 (no base/bias, tail)", host(dout, 1)[0] == ref_gemv(0, ld, 200057, false, false));
        k::argmax_gemv_f16(dw + 1, ld, dx, dbase, dbias, R, 777, dout, 0);
        expect_true("argmax_gemv_f16 (unaligned, single)", host(dout, 1)[0] == ref_gemv(1, ld, 777, true, true));
        release({dw, dbias, dx, dbase, dout});
    }
}

void test_attn_f16() {
    const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, kv_cap = 1056, window = 1024, nq = 8;
    auto kc_f = fvec((size_t) ctx * KV * D), vc_f = fvec((size_t) ctx * KV * D);
    std::vector<uint16_t> kc_lin(kc_f.size()), vc_lin(vc_f.size());
    for (size_t i = 0; i < kc_f.size(); ++i) {
        kc_lin[i] = f2h_host(kc_f[i]);
        vc_lin[i] = f2h_host(vc_f[i]);
    }
    // Ring-buffer copy holding positions [1476 .. 2507] wrapped mod kv_cap
    std::vector<uint16_t> kc_ring((size_t) kv_cap * KV * D, 0), vc_ring((size_t) kv_cap * KV * D, 0);
    for (int pos = 1476; pos < 2510; ++pos) {
        const int slot = pos % kv_cap;
        std::memcpy(kc_ring.data() + (size_t) slot * KV * D, kc_lin.data() + (size_t) pos * KV * D,
                    (size_t) KV * D * sizeof(uint16_t));
        std::memcpy(vc_ring.data() + (size_t) slot * KV * D, vc_lin.data() + (size_t) pos * KV * D,
                    (size_t) KV * D * sizeof(uint16_t));
    }
    auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f), sink = fvec((size_t) nq * H, 3.0f),
         proj = fvec((size_t) 1024 * d_rel);
    uint16_t *dk_lin = dev(kc_lin), *dv_lin = dev(vc_lin), *dk_ring = dev(kc_ring), *dv_ring = dev(vc_ring);
    float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
    float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
          *one = dev_n<float>((size_t) H * D);
    const float scale = 1.0f / std::sqrt(128.0f);

    // 1. attn_decode_f16 linear (kv_cap=0, full) and ring-buffer (kv_cap=1056, sliding) vs double reference
    for (int ring = 0; ring < 2; ++ring) {
        const int k0 = ring ? 1476 : 0, nk = ring ? 1024 : 2500, qpos = 2499, w = ring ? window : (1 << 30);
        k::attn_decode_f16(dq, dr, ds, ring ? dk_ring : dk_lin, ring ? dv_ring : dv_lin, ring ? kv_cap : 0, H, KV, D,
                           k0, nk, qpos, scale, dp, d_rel, 1024, w, scratch, one, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(nk);
            double m = sink[h];
            for (int j = 0; j < nk; ++j) {
                const int dist = qpos - (k0 + j);
                if (dist < 0 || dist >= w) {
                    l[j] = -INFINITY;
                    continue;
                }
                double a = 0;
                for (int d = 0; d < D; ++d)
                    a += (double) q[h * D + d] * h2f_host(kc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                a *= scale;
                if (dist < 1024)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < nk; ++j)
                    o += std::exp(l[j] - m) / sum * h2f_host(vc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                want[(size_t) h * D + d] = o;
            }
        }
        expect(ring ? "attn_decode_f16 ring 1056" : "attn_decode_f16 linear 2500", host(one, want.size()), want, 2e-4, 1e-6);
    }

    // 2. attn_multi_f16 ring-buffer (pos0 = 2500, k0 = 1477, nk = 1031) vs attn_decode_f16
    {
        const int pos0 = 2500, k0 = pos0 - window + 1, nk = pos0 + nq - k0;
        k::attn_multi_f16(dq, dr, ds, dk_ring, dv_ring, kv_cap, nq, H, KV, k0, nk, pos0, scale, dp, d_rel, 1024,
                          window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = pos0 + qi, kq = qp - window + 1;
            k::attn_decode_f16(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk_ring,
                               dv_ring, kv_cap, H, KV, D, kq, qp + 1 - kq, qp, scale, dp, d_rel, 1024, window,
                               scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect("attn_multi_f16 ring 1056", got, want, 1e-4, 1e-6);
    }

    for (uint16_t* p : {dk_lin, dv_lin, dk_ring, dv_ring}) release({p});
    for (float* p : {dq, dr, ds, dp, scratch, out, one}) release({p});
}

// The resident token path's kernels: the planar Q8 product (with a short conv on some rows, with rows that take
// another input) and the FP16 one against a double reference; the launches that stand for several - rms_multi,
// attn_res_jobs, moe_combine_conv, the sinks in the clips' launch - against the ones they replace, to the bit; a
// looped layer's message gate, a token's kernels and a chunk's, against a double reference.
void test_token_path() {
    const int C = 2304;
    {   // mv_q8p and mv_q8p_conv
        const int N = 300, K = C, nb = K / 32;
        std::vector<int8_t> q((size_t) N * K);
        for (auto& v : q) v = (int8_t) ((int) (rng() % 255) - 127);
        std::vector<uint16_t> d((size_t) N * nb);
        for (auto& v : d) v = f2h_host(0.001f + 0.01f * std::fabs(frand()));
        auto x = fvec(K, 2.0f), xa = fvec(K, 2.0f), xb = fvec(K, 2.0f);
        int8_t* dq = dev(q);
        uint16_t* dd = dev(d);
        float *dx = dev(x), *dxa = dev(xa), *dxb = dev(xb), *dy = dev_n<float>(N);
        auto dot = [&](int row, const std::vector<float>& in) {
            double a = 0;
            for (int b = 0; b < nb; ++b) {
                double sb = 0;
                for (int j = 0; j < 32; ++j) sb += (double) q[(size_t) row * K + b * 32 + j] * in[b * 32 + j];
                a += (double) h2f_host(d[(size_t) row * nb + b]) * sb;
            }
            return a;
        };
        k::mv_q8p(dq, dd, N, K, dx, dy, 0);
        std::vector<double> want(N);
        for (int r = 0; r < N; ++r) want[r] = dot(r, x);
        expect("mv_q8p", host(dy, N), want, 2e-4, 1e-3);
        // rows 40..99 through a conv with dynamic taps and a residual, rows 200..259 through a plain one; rows
        // 100..149 take xa, 150..199 xb
        const int n0 = 60, n1 = 60, heads = 4;
        auto h0 = fvec(3 * n0), w0 = fvec(4 * n0), dyn = fvec(heads * 4), res = fvec(n0), h1 = fvec(3 * n1), w1 = fvec(4 * n1);
        float *dh0 = dev(h0), *dw0 = dev(w0), *ddyn = dev(dyn), *dres = dev(res), *dh1 = dev(h1), *dw1 = dev(w1);
        float *dy0 = dev_n<float>(n0), *dy1 = dev_n<float>(n1);
        CU(cudaMemset(dy, 0, N * sizeof(float)));
        k::mv_q8p_conv(dq, dd, N, K, dx, dy, k::MvConv{40, n0, dh0, dw0, ddyn, heads, dres, dy0},
                       k::MvConv{200, n1, dh1, dw1, nullptr, 0, nullptr, dy1}, 0, k::MvAlt{dxa, 100, 50},
                       k::MvAlt{dxb, 150, 50});
        const auto gy = host(dy, N), gy0 = host(dy0, n0), gy1 = host(dy1, n1), gh0 = host(dh0, 3 * n0);
        std::vector<float> got;
        std::vector<double> ref;
        for (int r = 0; r < N; ++r) {
            const double v = dot(r, r >= 100 && r < 150 ? xa : r >= 150 && r < 200 ? xb : x);
            if (r >= 40 && r < 40 + n0) {
                const int c = r - 40;
                const float* dc = dyn.data() + (size_t) (c / (n0 / heads)) * 4;
                double a = (double) w0[c * 4] * h0[c] + (double) w0[c * 4 + 1] * h0[n0 + c] +
                           (double) w0[c * 4 + 2] * h0[2 * n0 + c] + (double) w0[c * 4 + 3] * v;
                a += dc[0] * v + (double) dc[1] * h0[2 * n0 + c] + (double) dc[2] * h0[n0 + c] + (double) dc[3] * h0[c];
                got.push_back(gy0[c]), ref.push_back(a + res[c]);
                got.push_back(gh0[2 * n0 + c]), ref.push_back(v);         // (the history takes the row)
                got.push_back(gh0[c]), ref.push_back(h0[n0 + c]);
            } else if (r >= 200 && r < 200 + n1) {
                const int c = r - 200;
                got.push_back(gy1[c]);
                ref.push_back((double) w1[c * 4] * h1[c] + (double) w1[c * 4 + 1] * h1[n1 + c] +
                              (double) w1[c * 4 + 2] * h1[2 * n1 + c] + (double) w1[c * 4 + 3] * v);
            } else {
                got.push_back(gy[r]), ref.push_back(v);
            }
        }
        expect("mv_q8p_conv: convs and other inputs", got, ref, 2e-4, 2e-3);
        // mv_f16
        const int Nf = 40;
        std::vector<uint16_t> wf((size_t) Nf * K);
        for (auto& v : wf) v = f2h_host(frand());
        uint16_t* dwf = dev(wf);
        float* dyf = dev_n<float>(Nf);
        k::mv_f16(dwf, Nf, K, dx, dyf, 0);
        std::vector<double> wantf(Nf);
        for (int r = 0; r < Nf; ++r)
            for (int i = 0; i < K; ++i) wantf[r] += (double) h2f_host(wf[(size_t) r * K + i]) * x[i];
        expect("mv_f16", host(dyf, Nf), wantf, 2e-4, 1e-3);
#ifdef STRATA_USE_HIP
        // Side projections include two different inputs and a row count not divisible by four.
        if (K == 2304) {
            k::MvF16Jobs side{};
            float* first = dev_n<float>(7), *second = dev_n<float>(12), *third = dev_n<float>(21);
            side.job[0] = {dwf, dx, first, 7};
            side.job[1] = {dwf + 7 * K, dxa, second, 12};
            side.job[2] = {dwf + 19 * K, dxb, third, 21};
            k::mv_q8p_conv(dq, dd, N, K, dx, dy, {}, {}, 0, {}, {}, side);
            const std::vector<float> inputs[] = {x, xa, xb};
            int base = 0;
            for (int j = 0; j < 3; ++j) {
                std::vector<double> reference(side.job[j].rows);
                for (int r = 0; r < side.job[j].rows; ++r)
                    for (int i = 0; i < K; ++i)
                        reference[r] += (double)h2f_host(wf[(size_t)(base + r) * K + i]) * inputs[j][i];
                expect("attention side projection", host(side.job[j].output, reference.size()), reference, 2e-4, 1e-3);
                base += side.job[j].rows;
            }
            std::vector<double> main_rows(N);
            for (int r = 0; r < N; ++r) main_rows[r] = dot(r, x);
            expect("attention main projection after side rows", host(dy, N), main_rows, 2e-4, 2e-3);
            release({first, second, third});
        }
#endif
        // the sinks in the clips' launch: mv_f16's values to the bit
        {
            const int H = 24, KV = 8, D = 128, d_rel = 16;
            auto qv = fvec((size_t) H * D), kv = fvec((size_t) KV * D), vv = fvec((size_t) KV * D), rv = fvec((size_t) H * d_rel);
            float *dqv = dev(qv), *dkv = dev(kv), *dvv = dev(vv), *drv = dev(rv), *dsk = dev_n<float>(H);
            uint16_t *dkc = dev_n<uint16_t>((size_t) 4 * KV * D), *dvc = dev_n<uint16_t>((size_t) 4 * KV * D);
            std::vector<int> pos = {2};
            int* dpos = dev(pos);
            k::prep_qkv_r_f16_at(dqv, nullptr, dkv, dkc, dvv, dvc, drv, nullptr, H, KV, D, d_rel, 22.6f, 8.0f, 1e-6f, dpos,
                                 false, 0.1f, 128000.0f, 0, nullptr, 0, dwf, dx, K, dsk);
            const auto a = host(dsk, H), b = host(dyf, H);
            expect_true("prep_qkv_r_f16_at: the sinks are mv_f16's to the bit", !std::memcmp(a.data(), b.data(), H * 4));
            for (float* p : {dqv, dkv, dvv, drv, dsk}) release({p});
            release({dkc, dvc, dpos});
        }
        release({dq, dd, dwf});
        for (float* p : {dx, dxa, dxb, dy, dh0, dw0, ddyn, dres, dh1, dw1, dy0, dy1, dyf}) release({p});
    }
    {   // rms_multi against rms; moe_combine_conv against moe_combine_f32 + conv4_res; attn_res_jobs against attn_res
        auto x3 = fvec((size_t) 3 * C, 3.0f), w3 = fvec((size_t) 3 * C);
        float *dx3 = dev(x3), *dw3 = dev(w3), *ya = dev_n<float>((size_t) 3 * C), *yb = dev_n<float>((size_t) 3 * C);
        for (int j = 0; j < 3; ++j) k::rms(dx3 + (size_t) j * C, dw3 + (size_t) j * C, 1, C, 1e-6f, ya + (size_t) j * C, 0);
        k::rms_multi(k::RmsJobs{{dx3, dx3 + C, dx3 + 2 * C}, {dw3, dw3 + C, dw3 + 2 * C}, {yb, yb + C, yb + 2 * C}}, 3, C,
                     1e-6f, 0);
        const auto ra = host(ya, (size_t) 3 * C), rb = host(yb, (size_t) 3 * C);
        expect_true("rms_multi: rms's values to the bit", !std::memcmp(ra.data(), rb.data(), ra.size() * 4));

        const int K = 8;
        auto y = fvec((size_t) K * C), wt = fvec(K), hist = fvec((size_t) 3 * C), tw = fvec((size_t) 4 * C), res = fvec(C);
        std::vector<int> inv = {3, 1, 0, 2, 7, 5, 6, 4};
        float *dyk = dev(y), *dwt = dev(wt), *dha = dev(hist), *dhb = dev(hist), *dtw = dev(tw), *dres = dev(res);
        float *tmp = dev_n<float>(C), *oa = dev_n<float>(C), *ob = dev_n<float>(C);
        int* dinv = dev(inv);
        k::moe_combine_f32(dyk, dinv, dwt, 1, K, C, tmp, false, 0);
        k::conv4_res(tmp, dha, dtw, nullptr, 0, 1, C, dres, oa, 0);
        k::moe_combine_conv(dyk, dinv, dwt, K, C, dhb, dtw, dres, ob, 0);
        const auto a1 = host(oa, C), b1 = host(ob, C), a2 = host(dha, (size_t) 3 * C), b2 = host(dhb, (size_t) 3 * C);
        expect_true("moe_combine_conv: moe_combine_f32 and conv4_res's values to the bit",
                    !std::memcmp(a1.data(), b1.data(), C * 4) && !std::memcmp(a2.data(), b2.data(), a2.size() * 4));

        const int n = 5;
        auto st = fvec((size_t) n * C, 2.0f), qq = fvec((size_t) 4 * C);
        float *dst = dev(st), *dqq = dev(qq), *ra4 = dev_n<float>((size_t) 4 * C), *rb4 = dev_n<float>((size_t) 4 * C);
        k::ResStates states{};
        for (int i = 0; i < n; ++i) states.st[i] = dst + (size_t) i * C;
        k::attn_res(states, n, dqq, 2, 1, C, 1e-6f, ra4, 0);
        k::attn_res(states, n, dqq + (size_t) 2 * C, 2, 1, C, 1e-6f, ra4 + (size_t) 2 * C, 0);
        k::attn_res_jobs(states, n, dqq, 2, 2, C, 1e-6f, rb4, 0);
        const auto a4 = host(ra4, (size_t) 4 * C), b4 = host(rb4, (size_t) 4 * C);
        expect_true("attn_res_jobs: two attn_res's values to the bit", !std::memcmp(a4.data(), b4.data(), a4.size() * 4));
        for (float* p : {dx3, dw3, ya, yb, dyk, dwt, dha, dhb, dtw, dres, tmp, oa, ob, dst, dqq, ra4, rb4}) release({p});
        release({dinv});
    }
    {   // a looped layer's gate: y = (1 - tanh(F g)) x + tanh(G g) m, g = D [rms(x); rms(m)]
        const int T = 3, R = 64;
        const float eps = 1e-6f;
        auto x = fvec((size_t) T * C, 2.0f), m = fvec((size_t) T * C, 2.0f), D = fvec((size_t) R * 2 * C, 0.05f),
             F = fvec((size_t) R * C, 0.3f), G = fvec((size_t) R * C, 0.3f);
        std::vector<double> want((size_t) T * C), a1((size_t) T * R), a2((size_t) T * R), fw((size_t) T * C), gw((size_t) T * C);
        for (int t = 0; t < T; ++t) {
            double sx = 0, sm = 0;
            for (int c = 0; c < C; ++c) sx += (double) x[(size_t) t * C + c] * x[(size_t) t * C + c], sm += (double) m[(size_t) t * C + c] * m[(size_t) t * C + c];
            const double rx = 1.0 / std::sqrt(sx / C + eps), rm = 1.0 / std::sqrt(sm / C + eps);
            std::vector<double> g(R);
            for (int r = 0; r < R; ++r) {
                double p1 = 0, p2 = 0;
                for (int c = 0; c < C; ++c) {
                    p1 += (double) D[(size_t) r * 2 * C + c] * x[(size_t) t * C + c];
                    p2 += (double) D[(size_t) r * 2 * C + C + c] * m[(size_t) t * C + c];
                }
                a1[(size_t) t * R + r] = p1, a2[(size_t) t * R + r] = p2;
                g[r] = rx * p1 + rm * p2;
            }
            for (int c = 0; c < C; ++c) {
                double f = 0, gn = 0;
                for (int r = 0; r < R; ++r) f += (double) F[(size_t) r * C + c] * g[r], gn += (double) G[(size_t) r * C + c] * g[r];
                fw[(size_t) t * C + c] = f, gw[(size_t) t * C + c] = gn;
                want[(size_t) t * C + c] = (1.0 - std::tanh(f)) * x[(size_t) t * C + c] + std::tanh(gn) * m[(size_t) t * C + c];
            }
        }
        float *dx = dev(x), *dm = dev(m), *dD = dev(D), *dF = dev(F), *dG = dev(G), *dy = dev_n<float>((size_t) T * C);
        float* dsc = dev_n<float>((size_t) 3 * T * (2 * R + 2));
        const k::LoopGate gt{dx, dm, dD, dF, dG, dy};
        k::loop_gates(&gt, 1, T, C, R, eps, dsc, 0);
        expect("loop_gates (a token's kernels)", host(dy, want.size()), want, 2e-4, 2e-4);
        // a chunk's: the products given (here the reference's), the two kernels around them
        std::vector<float> a1f(a1.begin(), a1.end()), a2f(a2.begin(), a2.end());
        float *da1 = dev(a1f), *da2 = dev(a2f), *dg = dev_n<float>((size_t) T * R);
        k::loop_gate_mix(dx, dm, da1, da2, T, C, R, eps, dg, 0);
        const auto gg = host(dg, (size_t) T * R);
        std::vector<float> ff((size_t) T * C), gf((size_t) T * C);   // f = g F, gn = g G from the kernel's g
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) {
                double f = 0, gn = 0;
                for (int r = 0; r < R; ++r) f += (double) F[(size_t) r * C + c] * gg[(size_t) t * R + r], gn += (double) G[(size_t) r * C + c] * gg[(size_t) t * R + r];
                ff[(size_t) t * C + c] = (float) f, gf[(size_t) t * C + c] = (float) gn;
            }
        float *df = dev(ff), *dgn = dev(gf);
        CU(cudaMemset(dy, 0, (size_t) T * C * sizeof(float)));
        k::loop_gate_blend(dx, dm, df, dgn, (size_t) T * C, dy, 0);
        expect("loop_gate_mix, loop_gate_blend (a chunk's)", host(dy, want.size()), want, 2e-4, 2e-4);
        for (float* p : {dx, dm, dD, dF, dG, dy, dsc, da1, da2, dg, df, dgn}) release({p});
    }
}

// cand_topk: every one of the K largest values is among the gathered, each gathered value is its token's, and the
// count says when the buffer was too small (a flat row)
void test_cand_topk() {
    const int n = 200058, K = 1000, cap = 4096;
    int* dscr = dev_n<int>(k::cand_scratch(n));
    int* dout = dev_n<int>(2 + 2 * cap);
    for (int kind = 0; kind < 3; ++kind) {
        std::vector<float> x(n);
        std::normal_distribution<float> nd(0.0f, kind == 0 ? 3.0f : 0.5f);
        for (auto& v : x) v = kind == 2 ? 1.25f : std::max(-29.9f, std::min(29.9f, nd(rng)));
        if (kind == 1)   // (ties at the cut and above it)
            for (int i = 0; i < n; i += 97) x[i] = 1.5f;
        float* dx = dev(x);
        k::cand_topk(dx, n, -30.0f, 30.0f, K, cap, dscr, dout, 0);
        const auto out = host(dout, 2 + 2 * cap);
        const int cnt = out[0];
        if (kind == 2) {
            expect_true("cand_topk: a flat row reports more values than the buffer holds", cnt == n);
            release({dx});
            continue;
        }
        bool ok = cnt >= K && cnt <= cap;
        std::vector<char> got(n, 0);
        for (int i = 0; i < std::min(cnt, cap) && ok; ++i) {
            const int id = out[2 + i];
            float v;
            std::memcpy(&v, &out[2 + cap + i], 4);
            ok = id >= 0 && id < n && !got[id] && v == x[id];
            if (ok) got[id] = 1;
        }
        if (ok) {   // nothing left out is larger than the K-th largest value
            std::vector<float> sorted = x;
            std::nth_element(sorted.begin(), sorted.begin() + (K - 1), sorted.end(), std::greater<float>());
            const float kth = sorted[K - 1];
            for (int i = 0; i < n && ok; ++i) ok = got[i] || x[i] < kth;
        }
        char label[96];
        std::snprintf(label, sizeof label, "cand_topk: the %d largest of %d are among its %d (%s)", K, n, cnt,
                      kind ? "ties" : "spread");
        expect_true(label, ok);
        release({dx});
    }
    release({dscr, dout});
}

float e4m3fn_to_f32_host(uint8_t q);
// attn_decode_mma_{f16,fp8} (one token's attention, a warp straight from the cache) against a double reference:
// 1 to 8 heads a K / V head, linear and ring caches, a window, the bias within and past its extent, the sink, a
// first token, a context shorter than a step, splits with and without keys; the FP8 cache with every finite byte
// (the subnormals among them).
#ifdef STRATA_USE_HIP
constexpr int attention_formats[] = {0, 1, 2, 3, 4}; // Half, E4M3 rows/pages, E2M1 rows/pages.
#else
constexpr int attention_formats[] = {0, 1, 3};
#endif

k::Fp8KvLayout test_fp8_layout(int format, int ring) {
    if (format == 3) return k::Fp8KvLayout::global_fp4();
#ifdef STRATA_USE_HIP
    if (format == 5) return k::Fp8KvLayout::fp4_ring(ring);
    if (format == 2 || format == 4) return k::Fp8KvLayout::global_pages(format == 4);
#endif
    return ring;
}

constexpr float kv_fp4_levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};

// Fill a packed cache independently of the writer. Return the exact decoded values for the CPU oracle.
template<class T> void fp4_fixture(std::vector<uint8_t>& bytes, std::vector<T>& decoded) {
    bytes.resize(decoded.size() / 32 * 18);
    for (size_t i = 0; i < decoded.size(); i += 32) {
        uint8_t* block = bytes.data() + i / 32 * 18;
        const uint16_t scale = f2h_host((1 + rng() % 15) / 8.f);
        std::memcpy(block, &scale, 2);
        for (int j = 0; j < 32; j += 2) {
            const unsigned pair = rng() & 255;
            block[2 + j/2] = pair;
            for (int n = 0; n < 2; ++n) {
                const int code = pair >> (n*4);
                const float val = kv_fp4_levels[code & 7] * (code & 8 ? -1 : 1) * h2f_host(scale);
                if constexpr (sizeof(T) == 2) decoded[i+j+n] = f2h_host(val);
                else decoded[i+j+n] = val;
            }
        }
    }
}

size_t page_offset(int token, int head, int d, int heads, bool value) {
    const size_t page = ((size_t)(token / 16) * heads + head) * 2048;
    return page + (value ? d * 16 + token % 16 : (token % 16) * 128 + d);
}

std::vector<uint8_t> page_cache(const std::vector<uint8_t>& linear, int heads, bool value) {
    const int positions = linear.size() / (128 * heads);
    std::vector<uint8_t> pages((size_t)((positions + 15) / 16) * heads * 2048, 0x7f);
    for (int t = 0; t < positions; ++t)
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < 128; ++d)
                pages[page_offset(t, h, d, heads, value)] = linear[((size_t)t * heads + h) * 128 + d];
    return pages;
}

// Independent serialization of [page][head][token][four E2M1 blocks].
std::vector<uint8_t> page_fp4(const std::vector<uint8_t>& rows, int heads) {
    constexpr size_t bytes = 4 * sizeof(k::Fp4KvBlock);
    const size_t tokens = rows.size() / (heads * bytes);
    std::vector<uint8_t> pages(((tokens + 15) / 16) * 16 * heads * bytes, 0xff);
    for (size_t t = 0; t < tokens; ++t)
        for (int h = 0; h < heads; ++h)
            std::copy_n(rows.data() + (t * heads + h) * bytes, bytes,
                        pages.data() + ((t / 16 * heads + h) * 16 + t % 16) * bytes);
    return pages;
}

// Two windows in non-adjacent slots, each rewound to a different accepted prefix.
void test_window_rollback() {
    constexpr int slots = 4, B = 5, T = 2 * B, C = 14, heads = 2, stride = 8, ld = C + 3;
    auto input = fvec(T * ld), weights = fvec(4 * C), initial = fvec(slots * 3 * C);
    auto residual = fvec(T * C), dynamic = fvec(T * heads * 4);
    auto *x = dev(input), *w = dev(weights), *hist = dev(initial), *res = dev(residual), *dyn = dev(dynamic);
    auto *old = dev_n<float>(slots * 3 * C), *record = dev_n<float>(slots * stride * C), *out = dev_n<float>(T * C);
    std::vector<int> ids(T, 1);
    std::fill_n(ids.begin(), B, 3);
    auto* slots_d = dev(ids);
    k::conv_windows(slots_d, T, B, stride, x, ld, {0, C, hist, w, dyn, heads, res, out}, old, record, 0);
    std::vector<double> state(initial.begin(), initial.end()), kept(state), expected(T * C);
    std::vector<k::ConvRec> recs;
    for (int t = 0; t < T; ++t) {
        const int slot = ids[t], offset = slot * 3 * C;
        for (int c = 0; c < C; ++c) {
            double* h = state.data() + offset + c;
            const double taps[] = {h[0], h[C], h[2 * C], input[t * ld + c]};
            double y = residual[t * C + c];
            for (int j = 0; j < 4; ++j)
                y += taps[j] * weights[4 * c + j] + taps[3 - j] * dynamic[(t * heads + c / (C / heads)) * 4 + j];
            expected[t * C + c] = y;
            h[0] = taps[1]; h[C] = taps[2]; h[2 * C] = taps[3];
        }
        if (t % B + 1 == (slot == 3 ? 1 : 4)) {
            std::copy_n(state.begin() + offset, 3 * C, kept.begin() + offset);
            recs.push_back({old + offset, record + slot * stride * C, hist + offset, C});
        }
    }
    expect("multi-slot convolution", host(out, T * C), expected, 2e-5, 2e-5);
    expect("multi-slot histories", host(hist, state.size()), state, 0, 0);
    auto* records = dev(recs);
    k::rollback_block(records, 1, 1, 0);
    k::rollback_block(records + 1, 1, 4, 0);
    expect("different accepted prefixes and untouched slots", host(hist, kept.size()), kept, 0, 0);
    for (auto* ptr : {x, w, hist, res, dyn, old, record, out}) CU(cudaFree(ptr));
    CU(cudaFree(records)); CU(cudaFree(slots_d));
}

void test_sequence_kernels() {
    constexpr int T = 5, N = 7, K = 256, ld = N + 3;
    std::vector<int> slots = {5, 0, 3, 2, 1};
    auto x = fvec(T * K), weights = fvec(N * 4), history = fvec(6 * 3 * N);
    std::vector<int8_t> q(N * K);
    std::vector<uint16_t> scale(N * K / 32);
    for (size_t i = 0; i < q.size(); ++i) q[i] = int(i % 101) - 50;
    for (size_t i = 0; i < scale.size(); ++i) scale[i] = f2h_host((1 + i % 7) / 64.f);
    auto* dx = dev(x);
    auto* dq = dev(q);
    auto* ds = dev(scale);
    auto* dy = dev_n<float>(T * ld);
    auto* dw = dev(weights);
    auto* dh = dev(history);
    auto* ids = dev(slots);
    auto* out = dev_n<float>(T * N);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CU(cudaDeviceSynchronize());
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    k::mv_q8p_sequences(dq, ds, N, K, dx, T, dy, ld, stream);
    k::conv_sequences(ids, T, dy, ld, {0, N, dh, dw, nullptr, 0, nullptr, out}, stream);
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<double> projected(T * ld), convolved(T * N), hist(history.begin(), history.end());
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            double& v = projected[t * ld + n];
            for (int i = 0; i < K; ++i)
                v += double(x[t * K + i]) * q[n * K + i] * h2f_host(scale[(n * K + i) / 32]);
        }
    }
    for (int replay = 0; replay < 2; ++replay) {
        CU(cudaMemcpy(ids, slots.data(), T * sizeof(int), cudaMemcpyHostToDevice));
        CU(cudaGraphLaunch(exec, stream));
        CU(cudaStreamSynchronize(stream));
        for (int t = 0; t < T; ++t) {
            for (int n = 0; n < N; ++n) {
                const double v = projected[t * ld + n];
                const int h = slots[t] * 3 * N + n;
                double y = v * weights[n * 4 + 3];
                for (int j = 0; j < 3; ++j) y += hist[h + j * N] * weights[n * 4 + j];
                convolved[t * N + n] = y;
                hist[h] = hist[h + N];
                hist[h + N] = hist[h + 2 * N];
                hist[h + 2 * N] = v;
            }
        }
        expect("sequence convolution", host(out, T * N), convolved, 2e-5, 2e-5);
        expect("sequence slot histories", host(dh, hist.size()), hist, 2e-5, 2e-5);
        std::reverse(slots.begin(), slots.end());
    }
    expect("sequence projection and padding", host(dy, T * ld), projected, 2e-5, 2e-5);
    CU(cudaGraphExecDestroy(exec)); CU(cudaGraphDestroy(graph)); CU(cudaStreamDestroy(stream));
    CU(cudaFree(dx)); CU(cudaFree(dq)); CU(cudaFree(ds)); CU(cudaFree(dy)); CU(cudaFree(dw));
    CU(cudaFree(dh)); CU(cudaFree(ids)); CU(cudaFree(out));
}

void test_sparse_decode() {
    // Different landmark keys win for each KV head. The last page has one live token.
    constexpr int H = 4, KV = 2, D = 128, n = 32769, page_size = 64;
    constexpr int budget = k::kSparsePages, total = (n + page_size - 1) / page_size;
    std::vector<uint8_t> keys(size_t(n) * KV * D), values(keys.size());
    std::vector<uint16_t> q(H * D);
    for (int h = 0; h < H; ++h) q[h * D + h % 2] = f2h_host(1.f);
    for (int t = 0; t < n; ++t) {
        for (int kv = 0; kv < KV; ++kv) {
            const int page = t / page_size;
            const size_t row = (size_t(t) * KV + kv) * D;
            if (page % 3 == kv && t % page_size == (kv ? 48 : 16)) keys[row + kv] = 0x48; // 4
            values[row] = (page % 3 == kv || t == n - 1) ? 0x40 : 0xb8; // 2 or -1
        }
    }
    auto* dq = dev(q);
    auto* dk = dev(keys);
    auto* dv = dev(values);
    auto* pos = dev(std::vector<int>{n - 1});
    auto* rel = dev(std::vector<float>(H, 1.f));
    auto* projection = dev(std::vector<float>{4.f, 2.f});
    auto* sink = dev(std::vector<float>(H, 8.f));
    auto* pages = dev_n<int>(KV * budget);
    auto* out = dev_n<float>(H * D);
    auto* scratch = dev_n<float>(k::attn_decode_mma_scratch(H, 64));
    k::sparse_decode(dq, rel, sink, dk, dv, H, KV, pos, 1.f, 1.f, projection, 1, 2,
                            pages, scratch, out, nullptr, nullptr, nullptr);
    const auto actual = host(pages, KV * budget);
    std::vector<int> selected(KV * budget);
    std::vector<double> expected(H * D);
    for (int kv = 0; kv < KV; ++kv) {
        double z[2] = {std::exp(8.), std::exp(8.)}, sum[2] = {}; // sink
        for (int slot = 0; slot < budget; ++slot) {
            int page = slot;
            if (slot >= budget - 24) page = total - budget + slot;
            else if (slot >= 2) {
                const int begin = 2 + (slot - 2) * (total - 26) / (budget - 26);
                const int end = 2 + (slot - 1) * (total - 26) / (budget - 26);
                page = begin;
                for (int candidate = begin; candidate < end; ++candidate)
                    if (candidate % 3 == kv) { page = candidate; break; }
            }
            selected[kv * budget + slot] = page;
            for (int t = page * page_size; t < std::min(n, (page + 1) * page_size); ++t) {
                const size_t row = (size_t(t) * KV + kv) * D;
                for (int h = 0; h < 2; ++h) {
                    const double bias = t == n - 1 ? 4. : t == n - 2 ? 2. : 0.;
                    const double w = std::exp((keys[row + h] ? 4. : 0.) + bias);
                    z[h] += w;
                    sum[h] += w * (values[row] == 0x40 ? 2. : -1.);
                }
            }
        }
        for (int h = 0; h < 2; ++h) expected[(kv * 2 + h) * D] = sum[h] / z[h];
    }
    expect_true("sparse page ids by KV head", actual == selected);
    expect("sparse decode vs selected-key FP64", host(out, H * D), expected, 2e-3, 5e-5);
    CU(cudaFree(dq)); CU(cudaFree(dk)); CU(cudaFree(dv)); CU(cudaFree(pos));
    CU(cudaFree(rel)); CU(cudaFree(projection)); CU(cudaFree(sink));
    CU(cudaFree(pages)); CU(cudaFree(out)); CU(cudaFree(scratch));
}

// Each captured launch owns one sequence slot and reads its position on replay.
void test_fp4_sequence_replay() {
    constexpr int slots = 2, cap = 19, D = 128, splits = 3;
    std::vector<k::Fp4KvBlock> values(slots * cap * D / 32);
    for (int slot = 0; slot < slots; ++slot)
        for (int t = 0; t < cap; ++t)
            for (int b = 0; b < D / 32; ++b) {
                auto& block = values[(slot * cap + t) * (D / 32) + b];
                block.scale = f2h_host(t + 1);
                std::memset(block.pairs, slot ? 0xaa : 0x22, sizeof(block.pairs));
            }
    auto* cache = dev(values);
    auto* q = dev(std::vector<uint16_t>(D, 0));
    auto* zero = dev(std::vector<float>(16, 0));
    auto* positions = dev_n<int>(slots);
    auto* out = dev_n<float>(slots * D);
    auto* scratch = dev_n<float>(k::attn_decode_mma_scratch(1, splits));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CU(cudaDeviceSynchronize());
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int slot = 0; slot < slots; ++slot) {
        const auto* row = reinterpret_cast<const uint8_t*>(cache + slot * cap * (D / 32));
        k::attn_decode_mma_fp8(q, zero, zero, row, row, k::Fp8KvLayout::global_fp4(), 1, 1,
                               positions + slot, 0, splits, 1.f, 1.f, zero, 16, 1, cap,
                               scratch, out + slot * D, nullptr, nullptr, stream);
    }
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    for (int replay = 0; replay < 2; ++replay) {
        const int pos[slots] = {replay ? 18 : 0, replay ? 4 : 17};
        CU(cudaMemcpy(positions, pos, sizeof(pos), cudaMemcpyHostToDevice));
        CU(cudaGraphLaunch(exec, stream));
        CU(cudaStreamSynchronize(stream));
        std::vector<double> want(slots * D);
        // All scores are zero. The extra sink contributes one to the denominator.
        for (int slot = 0; slot < slots; ++slot)
            std::fill_n(want.begin() + slot * D, D, (slot ? -.5 : .5) * (pos[slot] + 1));
        expect("FP4 batch slots at replayed positions", host(out, want.size()), want, 1e-5, 1e-5);
    }
    CU(cudaGraphExecDestroy(exec)); CU(cudaGraphDestroy(graph)); CU(cudaStreamDestroy(stream));
    CU(cudaFree(cache)); CU(cudaFree(q)); CU(cudaFree(zero)); CU(cudaFree(positions)); CU(cudaFree(out)); CU(cudaFree(scratch));
}

void test_deep_attention() {
    constexpr int n = 392957, D = 128, H = 3, T = 2, splits = 512, padded = ((n + 15) / 16) * 16;
    const int marks[] = {n / 20, n / 2, 19 * n / 20, n - 1};
    std::vector<uint8_t> keys(size_t(padded) * D), values(keys.size());
    for (int i = 0; i < 4; ++i) {
        keys[size_t(marks[i]) * D] = 0x58 + i; // 16, 18, 20, 22 in E4M3.
        for (int d = 0; d < D; ++d) values[size_t(marks[i]) * D + d] = d & (1 << i) ? 0x38 : 0xb8;
    }
    std::fill(keys.begin() + size_t(n) * D, keys.end(), 0x7e);
    std::fill(values.begin() + size_t(n) * D, values.end(), 0x7e);
    std::vector<uint16_t> query(T * H * D);
    std::vector<double> expected(query.size());
    for (int t = 0; t < T; ++t) {
        const int seen = n - 1 + t, active = t == 0 ? 3 : 4;
        for (int h = 0; h < H; ++h) {
            const double q = 1. - h / 8.;
            query[(t * H + h) * D] = f2h_host(q);
            double normalizer = seen - active + 1.;  // Unmarked keys and the sink all score zero.
            for (int i = 0; i < active; ++i) normalizer += std::exp(q * (16 + 2 * i));
            for (int d = 0; d < D; ++d)
                for (int i = 0; i < active; ++i)
                    expected[(t * H + h) * D + d] += (d & (1 << i) ? 1. : -1.) * std::exp(q * (16 + 2 * i)) / normalizer;
        }
    }
    auto* dq = dev(query);
    auto* zero = dev(std::vector<float>(H * 16, 0.f));
    auto* pos = dev(std::vector<int>{n - 1});
    auto* out = dev_n<float>(query.size());
    const size_t scratch_n = std::max(k::attn_decode_mma_scratch(H, splits), size_t(T * H * 130 * 16));
    auto* scratch = dev_n<float>(scratch_n);
    std::vector<k::Fp4KvBlock> packed_k(size_t(padded) * D / 32), packed_v(packed_k.size());
    for (int i = 0; i < 4; ++i) {
        const size_t head = size_t(marks[i]) * D / 32;
        packed_k[head].scale = f2h_host((16.f + 2 * i) / 4.f);
        packed_k[head].pairs[0] = 6; // code 6 is +4; other channels are zero.
        for (int d = 0; d < D; d += 2) {
            auto& block = packed_v[head + d / 32];
            block.scale = f2h_host(1.f);
            const int lo = d & (1 << i) ? 2 : 10;
            const int hi = (d + 1) & (1 << i) ? 2 : 10;
            block.pairs[(d % 32) / 2] = lo | (hi << 4);
        }
    }
    for (size_t block = size_t(n) * D / 32; block < packed_k.size(); ++block) {
        packed_k[block].scale = f2h_host(64.f);
        std::memset(packed_k[block].pairs, 0x77, sizeof(packed_k[block].pairs));
        packed_v[block] = packed_k[block];
    }
    const char* prefill_names[] = {"393K linear causal prefill", "393K paged causal prefill", "393K FP4 causal prefill", "393K paged FP4 prefill"};
    const char* decode_names[] = {"393K linear decode", "393K paged decode", "393K FP4 decode", "393K paged FP4 decode"};
    for (int format : attention_formats) {
        if (!format) continue;
        const auto layout = test_fp8_layout(format, 0);
        auto* dk = format >= 3 ? reinterpret_cast<uint8_t*>(dev(packed_k)) :
                   dev(format == 2 ? page_cache(keys, 1, false) : keys);
        auto* dv = format >= 3 ? reinterpret_cast<uint8_t*>(dev(packed_v)) :
                   dev(format == 2 ? page_cache(values, 1, true) : values);
        k::attn_flash_fp8(dq, H * D, dk, dv, layout, H, 1, D, T, n - T, 1.f, 1.f, zero, 1, n,
                          zero, out, H * D, scratch, scratch_n, 16, nullptr);
        expect(prefill_names[format - 1], host(out, query.size()), expected, 2e-3, 5e-5);
        k::attn_decode_mma_fp8(dq + H * D, zero, zero, dk, dv, layout, H, 1, pos, 0, splits, 1.f, 1.f,
                               zero, 16, 1, n, scratch, out, nullptr, nullptr, nullptr);
        expect(decode_names[format - 1], host(out, H * D),
               std::vector<double>(expected.begin() + H * D, expected.end()), 2e-3, 5e-5);
        CU(cudaFree(dk)); CU(cudaFree(dv));
    }
    CU(cudaFree(dq)); CU(cudaFree(zero)); CU(cudaFree(pos)); CU(cudaFree(out)); CU(cudaFree(scratch));
}

void test_attn_decode_mma() {
    const int D = 128, d_rel = 16;
    struct Case {
        int H, KV, qpos, window, extent, kv_cap, n_split;
        const char* name;
    };
    const Case cases[] = {
        {12, 4, 0, 1 << 30, 64, 0, 1, "first token"},
        {12, 4, 9, 1 << 30, 64, 0, 3, "10 keys, 3 blocks"},
        {24, 8, 2499, 1 << 30, 1024, 0, 5, "2500 keys, 3 heads a K/V head"},
#ifdef STRATA_USE_HIP
        {12, 4, 9, 1 << 30, 64, 0, 129, "10 keys, many empty splits"},
        {6, 2, 1067, 512, 128, 640, 257, "ring with 257 splits"},
#endif
        {8, 1, 1000, 1 << 30, 200, 0, 2, "1001 keys, 8 heads a K/V head"},
        {4, 4, 333, 1 << 30, 64, 0, 7, "334 keys, 1 head a K/V head"},
        {12, 2, 2000, 256, 256, 400, 2, "ring 400, window 256, 6 heads"},
        {10, 2, 5000, 1024, 1024, 1056, 4, "ring 1056, window 1024, 5 heads"},
    };
    for (int fp8 : attention_formats)
        for (const Case& c : cases) {
            if (fp8 >= 2 && c.kv_cap) continue;
            const int H = c.H, KV = c.KV, n_pos = c.qpos + 1, rows = c.kv_cap > 0 ? c.kv_cap : n_pos;
            const float k_inv = fp8 ? 1.0f / 16.0f : 1.0f, v_inv = fp8 ? 1.0f / 32.0f : 1.0f;
            // the cache's values as the kernel reads them: kf, vf [n_pos][KV * D] (before the scales)
            std::vector<float> kf((size_t) n_pos * KV * D), vf((size_t) n_pos * KV * D);
            std::vector<uint16_t> k16, v16;
            std::vector<uint8_t> k8, v8;
            if (fp8) k8.assign((size_t) rows * KV * D, 0x7e), v8.assign((size_t) rows * KV * D, 0x7e);
            else k16.assign((size_t) rows * KV * D, 0x7e00), v16.assign((size_t) rows * KV * D, 0x7e00);   // (NaN: unused)
            for (int pos = 0; pos < n_pos; ++pos) {
                const size_t pr = (size_t) (c.kv_cap > 0 ? pos % c.kv_cap : pos) * KV * D;
                for (int x = 0; x < KV * D; ++x) {
                    const size_t i = (size_t) pos * KV * D + x;
                    if (fp8) {
                        uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
                        if ((bk & 0x7F) >= 0x60) bk &= 0xDFu;   // (magnitudes under 32, as a clipped K times 16)
                        if ((bv & 0x7F) >= 0x60) bv &= 0xDFu;
                        k8[pr + x] = bk, v8[pr + x] = bv;
                        kf[i] = e4m3fn_to_f32_host(bk), vf[i] = e4m3fn_to_f32_host(bv);
                    } else {
                        k16[pr + x] = f2h_host(frand(1.0f)), v16[pr + x] = f2h_host(frand(1.0f));
                        kf[i] = h2f_host(k16[pr + x]), vf[i] = h2f_host(v16[pr + x]);
                    }
                }
            }
            if (fp8 >= 3) {
                fp4_fixture(k8, kf); fp4_fixture(v8, vf);
                if (fp8 == 4) { k8 = page_fp4(k8, KV); v8 = page_fp4(v8, KV); }
            }
            auto qf = fvec((size_t) H * D, fp8 ? 2.0f : 4.0f);
            std::vector<uint16_t> q16(qf.size());
            for (size_t i = 0; i < qf.size(); ++i) q16[i] = f2h_host(qf[i]);
            auto r = fvec((size_t) H * d_rel, 2.0f), proj = fvec((size_t) c.extent * d_rel), sink = fvec((size_t) H, 3.0f);
            auto gate = fvec((size_t) H * D, 3.0f);   // (the FP8 runs go through the gate and take the FP16 copy)
            const float scale = 1.0f / std::sqrt((float) D);
            uint16_t* dq = dev(q16);
            float *dr = dev(r), *dp = dev(proj), *ds = dev(sink), *dout = dev_n<float>((size_t) H * D), *dg = dev(gate);
            uint16_t* dout16 = dev_n<uint16_t>((size_t) H * D);
            float* dscr = dev_n<float>(k::attn_decode_mma_scratch(H, c.n_split));
            std::vector<int> pos_h = {c.qpos};
            int* dpos = dev(pos_h);
            for (size_t i = 0; i < qf.size(); ++i) qf[i] = h2f_host(q16[i]);
            float *dq32 = dev(qf), *gated = dev_n<float>(qf.size());
            float* split_part = dev_n<float>((size_t)H * c.n_split * 130);
            std::vector<float> scalar, multi;
            if (fp8) {
                uint8_t *dk = dev(fp8 == 2 ? page_cache(k8, KV, false) : k8), *dv = dev(fp8 == 2 ? page_cache(v8, KV, true) : v8);
                k::attn_decode_mma_fp8(dq, dr, ds, dk, dv, test_fp8_layout(fp8, c.kv_cap), H, KV, dpos, -1, c.n_split, scale * k_inv, v_inv, dp,
                                       d_rel, c.extent, c.window, dscr, dout, dg, dout16, 0);
                k::attn_decode_fp8_at(dq32, dr, ds, dk, dv, test_fp8_layout(fp8, c.kv_cap), H, KV, dpos, c.n_split, scale * k_inv,
                                      v_inv, dp, d_rel, c.extent, c.window, split_part, gated, 0, dg);
                if (fp8 >= 2) {
                    float* tmp = dev_n<float>((size_t)H * 128 * 130);
                    float* result = dev_n<float>((size_t)H * D);
                    k::attn_decode_fp8(dq32, dr, ds, dk, dv, test_fp8_layout(fp8, 0), H, KV, D, 0, n_pos, c.qpos,
                                       scale * k_inv, v_inv, dp, d_rel, c.extent, c.window, tmp, result, 0);
                    scalar = host(result, (size_t)H * D);
                    k::attn_multi_fp8(dq32, dr, ds, dk, dv, test_fp8_layout(fp8, 0), 1, H, KV, 0, n_pos, c.qpos,
                                      scale * k_inv, v_inv, dp, d_rel, c.extent, c.window, tmp, result, 0);
                    multi = host(result, (size_t)H * D);
                    release({tmp, result});
                }
                CU(cudaDeviceSynchronize());
                release({dk, dv});
            } else {
                uint16_t *dk = dev(k16), *dv = dev(v16);
                k::attn_decode_mma_f16(dq, dr, ds, dk, dv, c.kv_cap, H, KV, nullptr, c.qpos, c.n_split, scale, dp, d_rel,
                                       c.extent, c.window, dscr, dout, nullptr, nullptr, 0);
                k::attn_decode_f16_at(dq32, dr, ds, dk, dv, c.kv_cap, H, KV, dpos, c.n_split, scale, dp,
                                      d_rel, c.extent, c.window, split_part, gated, 0);
                CU(cudaDeviceSynchronize());
                release({dk, dv});
            }
            const int lo = std::max(0, c.qpos - (c.window < (1 << 30) ? c.window : c.qpos + 1) + 1);
            std::vector<double> want((size_t) H * D);
            double mag = 0;   // the largest sum of |terms| of an output (the tolerance below)
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(c.qpos + 1 - lo);
                double m = sink[h];
                for (int kp = lo; kp <= c.qpos; ++kp) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) h2f_host(q16[(size_t) h * D + d]) * kf[((size_t) kp * KV + g) * D + d];
                    a *= (double) (scale * k_inv);
                    if (c.qpos - kp < c.extent)
                        for (int e = 0; e < d_rel; ++e)
                            a += (double) r[h * d_rel + e] * proj[(size_t) (c.qpos - kp) * d_rel + e];
                    l[kp - lo] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[h] - m);
                for (double a : l) sum += std::exp(a - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0, oa = 0;
                    for (int kp = lo; kp <= c.qpos; ++kp) {
                        const double t = std::exp(l[kp - lo] - m) / sum * vf[((size_t) kp * KV + g) * D + d];
                        o += t, oa += std::fabs(t);
                    }
                    const double gs = fp8 ? 1.0 / (1.0 + std::exp(-(double) gate[(size_t) h * D + d])) : 1.0;
                    want[(size_t) h * D + d] = o * (double) v_inv * gs;
                    mag = std::max(mag, oa * (double) v_inv);
                }
            }
            char label[96];
            std::snprintf(label, sizeof label, "mma %s %s", fp8 == 4 ? "paged fp4" : fp8 == 3 ? "block fp4" : fp8 == 2 ? "paged fp8" : fp8 ? "fp8" : "f16", c.name);
            // (P is FP16 in the P V product: 2^-11 of a term)
            expect("split attention and gate", host(gated, want.size()), want, 1e-3, 1e-5);
            if (fp8 >= 2) {
                auto ungated = want;
                for (size_t i = 0; i < ungated.size(); ++i) ungated[i] *= 1.0 + std::exp(-(double)gate[i]);
                expect("paged scalar decode", scalar, ungated, 1e-3, 1e-5);
                expect("paged multi-query decode", multi, ungated, 1e-3, 1e-5);
            }
            release({dq32, gated, split_part});
            const auto got = host(dout, want.size());
            expect(label, got, want, 1e-2, std::max(1e-6, 1e-3 * mag));
            if (fp8) {   // the FP16 copy: the output's rounding
                const auto h16 = host(dout16, want.size());
                bool same = true;
                for (size_t i = 0; i < got.size(); ++i) same &= h16[i] == f2h_host(got[i]);
                if (!same) expect_true("mma fp8: the FP16 copy is the output rounded", false);
            }
            release({dq, dpos, dout16});
            for (float* p : {dr, dp, ds, dout, dscr, dg}) release({p});
        }
}

// attn_flash_f16 (the fused prefill attention) against a double reference: linear and ring caches, a window, the
// relative bias table, the sink, a chunk from 0 and one at an offset, 1 to 7 key splits (some of them empty).
void test_attn_flash() {
    const int H = 12, KV = 2, D = 128;
    struct Case {
        int Tq, qpos0, window, extent, kv_cap, n_split;
        const char* name;
        bool block = false;
        bool fp4_only = false;
    };
    std::vector<Case> cases = {
        {65, 113, 71, 9, 0, 1, "fp4 sliding mask across query tiles", false, true},
        {2, 8191, 1 << 30, 11, 0, 3, "fp4 8193 keys, final single-key tile", false, true},
        {130, 0, 1 << 30, 64, 0, 1, "attn_flash from 0"},
        {100, 300, 1 << 30, 64, 0, 3, "attn_flash at 300, 3 splits"},
        {77, 1500, 1 << 30, 200, 0, 7, "attn_flash at 1500, 7 splits"},
        {70, 2000, 256, 256, 400, 2, "attn_flash ring 400, window 256"},
        {9, 40, 1 << 30, 64, 0, 7, "attn_flash 9 queries, empty splits"},
        {3, 0, 1 << 30, 23, 0, 7, "verify attention from 0, empty splits", true},
        {1, 33, 1 << 30, 23, 0, 3, "verify one column tile", true},
        {2, 95, 1 << 30, 23, 0, 3, "verify two column tiles", true},
#ifdef STRATA_USE_HIP
        {8, 127, 64, 35, 0, 3, "verify attention with an unaligned window", true},
        {8, 127, 64, 35, 80, 3, "verify attention ring, three column tiles", true},
#endif
    };
#ifdef STRATA_USE_HIP
    for (int t = 4; t <= 7; ++t)
        cases.push_back({t, 31, 23, 9, 0, 3, "FP4 verify across page boundary", true, true});
#endif
    std::vector<int> formats(std::begin(attention_formats), std::end(attention_formats));
#ifdef STRATA_USE_HIP
    formats.push_back(5);
#endif
    for (int fp8 : formats) for (const Case& c : cases) {
        if (c.fp4_only && fp8 < 3) continue;
        if (fp8 == 5 ? !c.kv_cap : (fp8 >= 2 && c.kv_cap)) continue;
        const int n_pos = c.qpos0 + c.Tq;   // positions written so far
        const int rows = c.kv_cap > 0 ? c.kv_cap : n_pos;
        std::vector<uint8_t> k8((size_t) rows * KV * D), v8(k8.size());
        std::vector<uint16_t> kc((size_t) rows * KV * D, 0x7e00), vc((size_t) rows * KV * D, 0x7e00);   // (NaN: unused)
        for (int pos = 0; pos < n_pos; ++pos) {
            const int pr = c.kv_cap > 0 ? pos % c.kv_cap : pos;
            for (int x = 0; x < KV * D; ++x) {
                const size_t idx = (size_t) pr * KV * D + x;
                if (fp8) {
                    k8[idx] = (rng() & 0x80u) | (rng() % 0x60u);
                    v8[idx] = (rng() & 0x80u) | (rng() % 0x60u);
                    kc[idx] = f2h_host(e4m3fn_to_f32_host(k8[idx]));
                    vc[idx] = f2h_host(e4m3fn_to_f32_host(v8[idx]));
                } else {
                    kc[idx] = f2h_host(frand(1.0f));
                    vc[idx] = f2h_host(frand(1.0f));
                }
            }
        }
        if (fp8 >= 3) {
            fp4_fixture(k8, kc); fp4_fixture(v8, vc);
            if (fp8 == 4) { k8 = page_fp4(k8, KV); v8 = page_fp4(v8, KV); }
        }
        const int q_ld = H * D + (c.block ? 0 : 64);
        auto qf = fvec((size_t) c.Tq * q_ld, 4.0f);
        std::vector<uint16_t> q16(qf.size());
        for (size_t i = 0; i < qf.size(); ++i) q16[i] = f2h_host(qf[i]);
        auto bias = fvec((size_t) c.Tq * H * c.extent, 2.0f), sink = fvec((size_t) c.Tq * H, 3.0f);
        constexpr int dr = 5, r_ld = H * dr + 9, gate_ld = H * D + 13;
        std::vector<float> r, proj, gate;
        if (c.block) {
            r = fvec(c.Tq * r_ld);
            proj = fvec(c.extent * dr);
            gate = fvec(c.Tq * gate_ld);
            for (int t = 0; t < c.Tq; ++t) for (int h = 0; h < H; ++h) for (int d = 0; d < c.extent; ++d) {
                double sum = 0;
                for (int k = 0; k < dr; ++k) sum += (double)r[t * r_ld + h * dr + k] * proj[d * dr + k];
                bias[(t * H + h) * c.extent + d] = (float)sum;
            }
        }
        const float scale = (fp8 ? 1.0f / 16.0f : 1.0f) / std::sqrt((float) D);
        const float v_scale = fp8 && !c.fp4_only ? 1.0f / 32.0f : 1.0f;
        const int o_ld = H * D + (c.block ? 0 : 32);
        uint16_t* dq = dev(q16);
        float *db = dev(bias), *ds = dev(sink), *dout = dev_n<float>((size_t) c.Tq * o_ld);
        const size_t scr = c.block ? k::attn_block_scratch(H, c.Tq, c.n_split) : (size_t) 8 * c.Tq * H * (D + 2);
        float* dscr = dev_n<float>(scr);
        if (c.block) {
            int* pos = dev(std::vector<int>{c.qpos0});
            float *rel = dev(r), *pr = dev(proj), *g = dev(gate);
            uint16_t* out16 = dev_n<uint16_t>((size_t)c.Tq * o_ld);
            if (fp8) {
                auto *dk = dev(fp8 == 2 ? page_cache(k8, KV, false) : k8), *dv = dev(fp8 == 2 ? page_cache(v8, KV, true) : v8);
                k::attn_block_fp8(dq, rel, r_ld, ds, dk, dv, test_fp8_layout(fp8, c.kv_cap), H, KV, pos, c.Tq, c.n_split,
                                  scale, v_scale, pr, dr, c.extent, c.window, dscr, dout, g, gate_ld, out16, 0);
                CU(cudaDeviceSynchronize());
                release({dk, dv});
            } else {
                auto *dk = dev(kc), *dv = dev(vc);
                k::attn_block_f16(dq, rel, r_ld, ds, dk, dv, c.kv_cap, H, KV, pos, c.Tq, c.n_split,
                                  scale, pr, dr, c.extent, c.window, dscr, dout, g, gate_ld, out16, 0);
                CU(cudaDeviceSynchronize());
                release({dk, dv});
            }
            auto fp = host(dout, (size_t)c.Tq * o_ld);
            auto half = host(out16, fp.size());
            bool rounded = true;
            for (size_t k = 0; k < fp.size(); ++k) rounded &= half[k] == f2h_host(fp[k]);
            expect_true("verify attention returns rounded FP16 output", rounded);
            release({pos, out16, rel, pr, g});
        } else if (fp8) {
            uint8_t *dk = dev(fp8 == 2 ? page_cache(k8, KV, false) : k8), *dv = dev(fp8 == 2 ? page_cache(v8, KV, true) : v8);
            k::attn_flash_fp8(dq, q_ld, dk, dv, test_fp8_layout(fp8, c.kv_cap), H, KV, D, c.Tq, c.qpos0, scale, v_scale, db,
                              c.extent, c.window, ds, dout, o_ld, dscr, scr, c.n_split, 0);
            CU(cudaDeviceSynchronize());
            release({dk, dv});
        } else {
            uint16_t *dk = dev(kc), *dv = dev(vc);
            k::attn_flash_f16(dq, q_ld, dk, dv, c.kv_cap, H, KV, D, c.Tq, c.qpos0, scale, db, c.extent, c.window,
                              ds, dout, o_ld, dscr, scr, c.n_split, 0);
            CU(cudaDeviceSynchronize());
            release({dk, dv});
        }
        const auto out = host(dout, (size_t) c.Tq * o_ld);
        std::vector<float> got;
        std::vector<double> want;
        for (int i = 0; i < c.Tq; ++i) {
            const int qp = c.qpos0 + i, lo = std::max(0, qp - c.window + 1);
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(qp + 1 - lo);
                double m = sink[(size_t) i * H + h];
                for (int kp = lo; kp <= qp; ++kp) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) h2f_host(q16[(size_t) i * q_ld + h * D + d]) *
                             h2f_host(kc[((size_t) (c.kv_cap > 0 ? kp % c.kv_cap : kp) * KV + g) * D + d]);
                    a *= scale;
                    if (qp - kp < c.extent) a += bias[((size_t) i * H + h) * c.extent + (qp - kp)];
                    l[kp - lo] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[(size_t) i * H + h] - m);
                for (double a : l) sum += std::exp(a - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0;
                    for (int kp = lo; kp <= qp; ++kp)
                        o += std::exp(l[kp - lo] - m) / sum * h2f_host(vc[((size_t) (c.kv_cap > 0 ? kp % c.kv_cap : kp) * KV + g) * D + d]);
                    if (c.block) o /= 1.0 + std::exp(-(double)gate[(size_t)i * gate_ld + h * D + d]);
                    want.push_back(o * v_scale);
                    got.push_back(out[(size_t) i * o_ld + h * D + d]);
                }
            }
        }
        char name[96];
        std::snprintf(name, sizeof name, "%s %s", c.name, fp8 == 5 ? "ring fp4" : fp8 == 4 ? "paged fp4" : fp8 == 3 ? "block fp4" : fp8 == 2 ? "paged fp8" : fp8 ? "fp8" : "f16");
        expect(name, got, want, 1e-2, 2e-3);
        release({dq});
        for (float* p : {db, ds, dout, dscr}) release({p});
    }
}

void test_round2_fused() {
    // 1. rms_f16: (with/without w) x (y16+y32, y16-only, y32-only) + unaligned/non-mult4 tail
    {
        struct Case {
            int rows, D, off;
            bool with_w, w16, w32;
            const char* name;
        };
        const Case cases[] = {
            {5, 4608, 0, true, true, true, "rms_f16 w y16+y32"},
            {5, 4608, 0, false, true, false, "rms_f16 no-w y16-only"},
            {5, 128, 0, true, false, true, "rms_f16 w y32-only"},
            {4, 130, 1, true, true, true, "rms_f16 unaligned tail"},
        };
        for (const Case& cs : cases) {
            auto x = fvec((size_t) cs.rows * cs.D + cs.off, 3.0f), w = fvec(cs.D + cs.off);
            float *dx = dev(x), *dw = dev(w), *dy32 = dev_n<float>((size_t) cs.rows * cs.D + cs.off);
            uint16_t* dy16 = dev_n<uint16_t>((size_t) cs.rows * cs.D + cs.off);
            const float eps = 1e-6f;
            k::rms_f16(dx + cs.off, cs.with_w ? dw + cs.off : nullptr, cs.rows, cs.D, eps,
                       cs.w16 ? dy16 + cs.off : nullptr, cs.w32 ? dy32 + cs.off : nullptr, 0);
            std::vector<double> want((size_t) cs.rows * cs.D);
            for (int r = 0; r < cs.rows; ++r) {
                double ss = 0;
                for (int d = 0; d < cs.D; ++d) {
                    const double v = x[cs.off + (size_t) r * cs.D + d];
                    ss += v * v;
                }
                const double rs = 1.0 / std::sqrt(ss / cs.D + eps);
                for (int d = 0; d < cs.D; ++d)
                    want[(size_t) r * cs.D + d] = x[cs.off + (size_t) r * cs.D + d] * rs * (cs.with_w ? w[cs.off + d] : 1.0);
            }
            if (cs.w32) {
                const auto got32 = host(dy32 + cs.off, want.size());
                char nm[64];
                std::snprintf(nm, sizeof nm, "%s (f32)", cs.name);
                expect(nm, got32, want, 2e-5);
            }
            if (cs.w16) {
                const auto raw16 = host(dy16 + cs.off, want.size());
                std::vector<float> got16(want.size());
                for (size_t i = 0; i < want.size(); ++i) got16[i] = h2f_host(raw16[i]);
                char nm[64];
                std::snprintf(nm, sizeof nm, "%s (f16)", cs.name);
                expect(nm, got16, want, 2e-3, 1e-4);
            }
            release({dx, dw, dy32, dy16});
        }
    }

    // 2. gate_mul_f16: f16-only vs f16+f32, non-mult4 sizes, n < 4, and unaligned o16
    {
        struct Case {
            size_t n;
            int off16;
            bool write_f32;
            const char* name;
        };
        const Case cases[] = {
            {4096, 0, false, "gate_mul_f16 f16-only"},
            {4099, 0, true, "gate_mul_f16 tail f16+f32"},
            {3, 0, true, "gate_mul_f16 n=3"},
            {1031, 1, false, "gate_mul_f16 unaligned o16"},
        };
        for (const Case& cs : cases) {
            auto o = fvec(cs.n, 3.0f), g = fvec(cs.n, 4.0f);
            float *do_ = dev(o), *dg = dev(g);
            uint16_t* do16 = dev_n<uint16_t>(cs.n + cs.off16);
            k::gate_mul_f16(do_, dg, cs.n, do16 + cs.off16, cs.write_f32, 0);
            std::vector<double> want(cs.n), orig(o.begin(), o.end());
            for (size_t i = 0; i < cs.n; ++i) want[i] = (double) o[i] * (1.0 / (1.0 + std::exp(-(double) g[i])));
            const auto raw16 = host(do16 + cs.off16, cs.n);
            std::vector<float> got16(cs.n);
            for (size_t i = 0; i < cs.n; ++i) got16[i] = h2f_host(raw16[i]);
            expect(cs.name, got16, want, 2e-3, 1e-4);
            const auto got32 = host(do_, cs.n);
            expect(cs.write_f32 ? "gate_mul_f16 f32 updated" : "gate_mul_f16 f32 untouched", got32,
                   cs.write_f32 ? want : orig, 1e-5);
            release({do_, dg, do16});
        }
    }

    // 3. prep_qkv_r_f16: sliding (pos=nullptr, kv_cap wraparound) and full (pos!=nullptr, kv_cap=0)
    {
        const int T = 5, H = 48, KV = 8, D = 128, d_rel = 16, kv_cap = 1056;
        const float qk_clip = 22.627416610717773f, rel_clip = 8.0f, eps = 1e-6f, alpha = 0.1f, n_floor = 128000.0f;
        for (int full = 0; full < 2; ++full) {
            const int cap = full ? 0 : kv_cap, pos0 = full ? 100 : 1054;
            const int kc_rows = cap > 0 ? cap : pos0 + T;
            auto q = fvec((size_t) T * H * D, 4.0f), qc = fvec((size_t) H * D, 2.0f),
                 k_in = fvec((size_t) T * KV * D, 4.0f), v_in = fvec((size_t) T * KV * D, 3.0f),
                 r = fvec((size_t) T * H * d_rel, 4.0f), rc = fvec((size_t) H * d_rel, 2.0f);
            for (size_t i = 0; i < (size_t) H * D; ++i) q[i] *= 0.01f;
            std::vector<int> pos = {5, 130000, 200000, 127999, 300000};
            float *dq = dev(q), *dqc = dev(qc), *dk = dev(k_in), *dv = dev(v_in), *dr = dev(r), *drc = dev(rc);
            uint16_t *dkc = dev_n<uint16_t>((size_t) kc_rows * KV * D),
                     *dvc = dev_n<uint16_t>((size_t) kc_rows * KV * D);
            int* dpos = dev(pos);
            k::prep_qkv_r_f16(dq, dqc, dk, dkc, dv, dvc, dr, drc, T, H, KV, D, d_rel, qk_clip, rel_clip, eps,
                              full ? dpos : nullptr, alpha, n_floor, pos0, cap, 0);
            std::vector<double> wq(q.size()), wr(r.size()), wk((size_t) T * KV * D), wv((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const double tau =
                    full ? (1.0 + alpha * std::log(std::max(1.0, ((double) pos[t] + 1.0) / n_floor))) : 1.0;
                for (int h = 0; h < H; ++h) {
                    double ss = 0;
                    for (int d = 0; d < D; ++d) {
                        const double v = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d];
                        ss += v * v;
                    }
                    const double f = std::min(1.0, qk_clip / (std::sqrt(ss) + 1e-6)) * tau;
                    for (int d = 0; d < D; ++d)
                        wq[((size_t) t * H + h) * D + d] = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d] * f;

                    double sr = 0;
                    for (int d = 0; d < d_rel; ++d) {
                        const double v = (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d];
                        sr += v * v;
                    }
                    const double fr = std::min(1.0, rel_clip / (std::sqrt(sr) + 1e-6)) * tau;
                    for (int d = 0; d < d_rel; ++d)
                        wr[((size_t) t * H + h) * d_rel + d] =
                            (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d] * fr;
                }
                for (int kv = 0; kv < KV; ++kv) {
                    double sk = 0, sv = 0;
                    for (int d = 0; d < D; ++d) {
                        const double kv_v = k_in[((size_t) t * KV + kv) * D + d];
                        const double vv_v = v_in[((size_t) t * KV + kv) * D + d];
                        sk += kv_v * kv_v;
                        sv += vv_v * vv_v;
                    }
                    const double fk = std::min(1.0, qk_clip / (std::sqrt(sk) + 1e-6));
                    const double rs = 1.0 / std::sqrt(sv / D + eps);
                    for (int d = 0; d < D; ++d) {
                        wk[((size_t) t * KV + kv) * D + d] = k_in[((size_t) t * KV + kv) * D + d] * fk;
                        wv[((size_t) t * KV + kv) * D + d] = v_in[((size_t) t * KV + kv) * D + d] * rs;
                    }
                }
            }
            const auto kc_all = host(dkc, (size_t) kc_rows * KV * D);
            const auto vc_all = host(dvc, (size_t) kc_rows * KV * D);
            std::vector<float> got_k((size_t) T * KV * D), got_v((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const int slot = cap > 0 ? (pos0 + t) % cap : (pos0 + t);
                for (size_t j = 0; j < (size_t) KV * D; ++j) {
                    got_k[(size_t) t * KV * D + j] = h2f_host(kc_all[(size_t) slot * KV * D + j]);
                    got_v[(size_t) t * KV * D + j] = h2f_host(vc_all[(size_t) slot * KV * D + j]);
                }
            }
            expect(full ? "prep_qkv_r_f16 full q" : "prep_qkv_r_f16 ring q", host(dq, wq.size()), wq, 2e-5);
            expect(full ? "prep_qkv_r_f16 full r" : "prep_qkv_r_f16 ring r", host(dr, wr.size()), wr, 2e-5);
            expect(full ? "prep_qkv_r_f16 full k" : "prep_qkv_r_f16 ring k", got_k, wk, 2e-3, 1e-4);
            expect(full ? "prep_qkv_r_f16 full v" : "prep_qkv_r_f16 ring v", got_v, wv, 2e-3, 1e-4);
            for (float* p : {dq, dqc, dk, dv, dr, drc}) release({p});
            release({dkc, dvc, dpos});
        }
    }

    // 4. experts_gate_up_swiglu vs experts_gate_up + swiglu_cap_f32
    {
        const int C = 512, F = 256, E = 3;
        struct Case {
            int tg, tu, off;
            bool with_scale;
            const char* name;
        };
        const Case cases[] = {
            {k::kNVFP4, k::kNVFP4, 0, true, "NVFP4 scaled"},
            {k::kQ5_K, k::kQ5_K, 0, true, "Q5_K scaled"},
            {k::kQ6_K, k::kQ6_K, 0, false, "Q6_K unscaled"},
            {k::kQ8_0, k::kQ8_0, 0, true, "Q8_0 aligned"},
            {k::kQ8_0, k::kQ8_0, 2, true, "Q8_0 unaligned"},
            {k::kQ8_0, k::kNVFP4, 0, true, "Q8_0/NVFP4 mixed"},
            {k::kIQ3_XXS, k::kIQ3_XXS, 0, true, "IQ3_XXS scaled"},
            {k::kIQ4_XS, k::kIQ4_XS, 0, false, "IQ4_XS unscaled"},
        };
        for (const Case& cs : cases) {
            auto g = rand_blocks(cs.tg, (size_t) E * F * C), u = rand_blocks(cs.tu, (size_t) E * F * C);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, slot = (eb0 + eb1 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E + cs.off, 0);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + cs.off + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + cs.off + e * slot + eb0, u.data() + e * eb1, eb1);
            }
            uint8_t* dpool = dev(pool);
            const int S_max = 5, pick[S_max] = {2, 0, 2, 1, 0}, tok[S_max] = {0, 1, 1, 2, 0};
            std::vector<const uint8_t*> blob(S_max);
            for (int i = 0; i < S_max; ++i) blob[i] = dpool + cs.off + pick[i] * slot;
            const uint8_t** dblob = nullptr;
            CU(cudaMalloc(&dblob, S_max * sizeof(void*)));
            CU(cudaMemcpy(dblob, blob.data(), S_max * sizeof(void*), cudaMemcpyHostToDevice));
            auto x = fvec((size_t) 3 * C, 1.5f), sg = fvec(S_max, 0.5f), su = fvec(S_max, 0.5f);
            std::vector<float> ones(S_max, 1.0f);
            for (int i = 0; i < S_max; ++i) {
                sg[i] = 0.7f + 0.1f * i;
                su[i] = 0.8f + 0.05f * i;
            }
            float *dx = dev(x), *dgu = dev_n<float>((size_t) S_max * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dones = dev(ones);
            int* dtok = dev(std::vector<int>(tok, tok + S_max));
            uint16_t *dh_ref = dev_n<uint16_t>((size_t) S_max * F), *dh_got = dev_n<uint16_t>((size_t) S_max * F);
            const float gcap = 20.0f, ucap = 20.0f;
            for (int S : {1, 5}) {
                k::experts_gate_up(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, dgu, 0);
                k::swiglu_cap_f32(dgu, S, F, cs.with_scale ? dsg : dones, cs.with_scale ? dsu : dones, gcap, ucap,
                                  dh_ref, 0);
                k::experts_gate_up_swiglu(dblob, dtok, S, cs.tg, cs.tu, eb0, dx, C, F, cs.with_scale ? dsg : nullptr,
                                          cs.with_scale ? dsu : nullptr, gcap, ucap, dh_got, 0);
                const auto r16 = host(dh_ref, (size_t) S * F), g16 = host(dh_got, (size_t) S * F);
                std::vector<double> want((size_t) S * F);
                std::vector<float> got((size_t) S * F);
                for (size_t i = 0; i < want.size(); ++i) {
                    want[i] = h2f_host(r16[i]);
                    got[i] = h2f_host(g16[i]);
                }
                char nm[64];
                std::snprintf(nm, sizeof nm, "experts_gu_swiglu S=%d %s", S, cs.name);
                expect(nm, got, want, 3e-3, 1e-2);
            }
            release({dpool, dblob, dx, dgu, dsg, dsu, dones, dtok, dh_ref, dh_got});
        }
    }

    // 5. experts_gate_up_swiglu_g vs experts_gate_up_g + swiglu_cap_f32
    {
        const int C = 512, F = 256, E = 5, T = 8, K = 8, S = T * K;
        const int types[][2] = {{k::kNVFP4, k::kNVFP4},
                                {k::kQ5_K, k::kQ5_K},
                                {k::kQ6_K, k::kQ6_K},
                                {k::kQ8_0, k::kQ8_0},
                                {k::kQ8_0, k::kNVFP4},
                                {k::kIQ3_XXS, k::kIQ3_XXS},
                                {k::kIQ4_XS, k::kIQ4_XS}};
        const char* tnames[] = {"NVFP4", "Q5_K", "Q6_K", "Q8_0", "Q8_0/NVFP4", "IQ3_XXS", "IQ4_XS"};
        const int want_ng[E] = {1, 2, 3, 5, 8};
        for (int ti = 0; ti < 7; ++ti) {
            const int tg = types[ti][0], tu = types[ti][1];
            auto g = rand_blocks(tg, (size_t) E * F * C), u = rand_blocks(tu, (size_t) E * F * C);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, slot = (eb0 + eb1 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            }
            uint8_t* dpool = dev(pool);
            std::vector<int> pick(S, -1), gn, gtok, grow, idx;
            std::vector<const uint8_t*> gb;
            for (int e = 0; e < E; ++e)
                for (int t = 0; t < want_ng[e]; ++t) pick[t * K + e] = e;
            for (int i = 0; i < S; ++i)
                if (pick[i] >= 0) idx.push_back(i);
            const int n = (int) idx.size();
            std::vector<int> gi(E, -1);
            for (int r = 0; r < n; ++r) {
                const int i = idx[r], e = pick[i];
                if (gi[e] < 0) {
                    gi[e] = (int) gb.size();
                    gb.push_back(dpool + e * slot);
                    gn.push_back(0);
                    gtok.resize(gtok.size() + k::kGroupMax);
                    grow.resize(grow.size() + k::kGroupMax);
                }
                const int gg = gi[e];
                gtok[gg * k::kGroupMax + gn[gg]] = i / K;
                grow[gg * k::kGroupMax + gn[gg]] = r;
                ++gn[gg];
            }
            const uint8_t** dgb = nullptr;
            CU(cudaMalloc(&dgb, gb.size() * sizeof(void*)));
            CU(cudaMemcpy(dgb, gb.data(), gb.size() * sizeof(void*), cudaMemcpyHostToDevice));
            int *dgn = dev(gn), *dgt = dev(gtok), *dgr = dev(grow);
            auto x = fvec((size_t) T * C, 1.5f), sg = fvec(n), su = fvec(n);
            std::vector<float> ones(n, 1.0f);
            for (int i = 0; i < n; ++i) {
                sg[i] = 0.6f + 0.03f * i;
                su[i] = 0.9f + 0.02f * i;
            }
            float *dx = dev(x), *dgu = dev_n<float>((size_t) n * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dones = dev(ones);
            uint16_t *dh_ref = dev_n<uint16_t>((size_t) n * F), *dh_got = dev_n<uint16_t>((size_t) n * F);
            const float gcap = 20.0f, ucap = 20.0f;
            for (int with_scale = 0; with_scale < 2; ++with_scale) {
                k::experts_gate_up_g(dgb, dgn, dgt, dgr, (int) gb.size(), tg, tu, eb0, dx, C, F, dgu, 0);
                k::swiglu_cap_f32(dgu, n, F, with_scale ? dsg : dones, with_scale ? dsu : dones, gcap, ucap, dh_ref,
                                  0);
                k::experts_gate_up_swiglu_g(dgb, dgn, dgt, dgr, (int) gb.size(), tg, tu, eb0, dx, C, F,
                                            with_scale ? dsg : nullptr, with_scale ? dsu : nullptr, gcap, ucap, dh_got,
                                            0);
                const auto r16 = host(dh_ref, (size_t) n * F), g16 = host(dh_got, (size_t) n * F);
                std::vector<double> want((size_t) n * F);
                std::vector<float> got((size_t) n * F);
                for (size_t i = 0; i < want.size(); ++i) {
                    want[i] = h2f_host(r16[i]);
                    got[i] = h2f_host(g16[i]);
                }
                char nm[64];
                std::snprintf(nm, sizeof nm, "experts_gu_swiglu_g %s%s", tnames[ti], with_scale ? "" : " (no scale)");
                expect(nm, got, want, 3e-3, 1e-2);
            }
            for (void* p : {(void*) dpool, (void*) dgb, (void*) dgn, (void*) dgt, (void*) dgr, (void*) dx, (void*) dgu,
                            (void*) dsg, (void*) dsu, (void*) dones, (void*) dh_ref, (void*) dh_got})
                release({p});
        }
    }
}

float e4m3fn_to_f32_host(uint8_t q) {
    const uint32_t mag = q & 0x7Fu;
    const float sgn = (q & 0x80u) ? -1.0f : 1.0f;
    if (mag == 0x7Fu) return NAN;
    const uint32_t e = (mag >> 3) & 15u, m = mag & 7u;
    if (e == 0) return sgn * std::ldexp((float) m, -9);
    return sgn * std::ldexp(1.0f + (float) m / 8.0f, (int) e - 7);
}

template<class Launch>
float time_attention_graph(Launch launch, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t start, end;
    CU(cudaEventCreate(&start));
    CU(cudaEventCreate(&end));
    for (int i = 0; i < 5; ++i) CU(cudaGraphLaunch(exec, stream));
    CU(cudaEventRecord(start, stream));
    constexpr int runs = 32;
    for (int i = 0; i < runs; ++i) CU(cudaGraphLaunch(exec, stream));
    CU(cudaEventRecord(end, stream));
    CU(cudaEventSynchronize(end));
    float ms;
    CU(cudaEventElapsedTime(&ms, start, end));
    CU(cudaEventDestroy(start));
    CU(cudaEventDestroy(end));
    CU(cudaGraphExecDestroy(exec));
    CU(cudaGraphDestroy(graph));
    return ms * 1000 / runs;
}

void test_sequence_attention(int depth, int rows_per_slot, bool time_graphs, bool ring = false, bool paged = false, bool fp4_ring = false) {
    constexpr int H = 24, KV = 8, D = 128, dr = 16, extent = 1024;
    const int window = ring ? 64 : 1 << 30;
    constexpr int KD = KV * D, HD = H * D, ld = 2 * HD + H * dr, slots = 3;
    const int used = depth + rows_per_slot + 8;
    const int T = 2 * rows_per_slot, cap = ring ? 80 : paged ? (used + 15) / 16 * 16 : used, splits = depth < 1024 ? 16 : 32;
    const auto layout = test_fp8_layout(fp4_ring ? 5 : ring ? 1 : paged ? 4 : 3, ring ? cap : 0);
    const size_t stride = layout.fp4 ? (size_t)cap * KD / 32 * sizeof(k::Fp4KvBlock) : (size_t)cap * KD;
    std::vector<int> slot_ids, positions;
    for (int t = 0; t < T; ++t) {
        slot_ids.push_back(t < rows_per_slot ? 2 : 0);
        positions.push_back((t < rows_per_slot ? depth : depth - 19) + t % rows_per_slot);
    }
    auto* ds = dev(slot_ids);
    auto* dp = dev(positions);
    std::vector<float> decoded((size_t)slots * cap * KD);
    std::vector<uint8_t> packed_k, packed_v;
    if (!layout.fp4) {
        packed_k.assign(slots * stride, 0x30);
        packed_v.assign(slots * stride, 0x38);
    } else {
        fp4_fixture(packed_k, decoded);
        fp4_fixture(packed_v, decoded);
    }
    auto *kc = dev(packed_k), *vc = dev(packed_v);
    auto *old_k = dev(packed_k), *old_v = dev(packed_v);
    auto qrg = fvec(T * ld);
    auto *q = dev(qrg), *old_q = dev(qrg);
    auto *keys = dev(fvec(T * KD, 3)), *values = dev(fvec(T * KD, 3));
    auto *sink = dev(fvec(T * H)), *rel = dev(fvec(extent * dr));
    auto *out = dev_n<float>(T * HD), *old_out = dev_n<float>(T * HD);
    auto* old_half = dev_n<uint16_t>(T * HD);
    auto* scratch = dev_n<float>((size_t)T * k::attn_decode_mma_scratch(H, splits));
#ifdef STRATA_USE_HIP
    uint16_t* q16 = nullptr;
#else
    auto* q16 = dev_n<uint16_t>(T * HD);
#endif
    cudaStream_t stream;
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CU(cudaDeviceSynchronize());
    auto per_row = [&] {
        for (int t = 0; t < T; ++t) {
            auto *krow = old_k + slot_ids[t] * stride, *vrow = old_v + slot_ids[t] * stride;
            auto *qr = old_q + t * ld, *r = qr + HD;
            k::prep_qkv_r_fp8_at(qr, nullptr, keys + t * KD, krow, values + t * KD, vrow, r, nullptr,
                                 H, KV, D, dr, 22, 8, 1e-6f, dp + t, !ring, 0.2f, 64, layout,
                                 16, 32, q16, stream);
#ifdef STRATA_USE_HIP
            k::attn_decode_fp8_at(qr, r, sink + t * H, krow, vrow, layout, H, KV, dp + t, splits,
                                  (1.f / std::sqrt((float)D)) / 16, 1.f / 32, rel, dr, extent, window,
                                  scratch, old_out + t * HD, stream, r + H * dr, old_half + t * HD);
#else
            k::attn_decode_mma_fp8(q16, r, sink + t * H, krow, vrow, layout, H, KV, dp + t, 0, splits,
                                   (1.f / std::sqrt((float)D)) / 16, 1.f / 32, rel, dr, extent, window,
                                   scratch, old_out + t * HD, r + H * dr, old_half + t * HD, stream);
#endif
        }
    };
    auto batched = [&] {
        k::sequence_attention_quantized(q, q + HD, q + HD + H * dr, ld, nullptr, nullptr, keys, values,
                                        kc, vc, layout, stride, ds, dp, T, H, KV, dr, 22, 8, 1e-6f,
                                        0.2f, 64, 16, 32, rel, extent, window, splits, sink, scratch, out, stream, q16);
    };
    per_row();
    batched();
    CU(cudaStreamSynchronize(stream));
    expect_true("batch keys including untouched slots", host(kc, slots * stride) == host(old_k, slots * stride));
    expect_true("batch values including untouched slots", host(vc, slots * stride) == host(old_v, slots * stride));
    const auto actual = host(out, T * HD), expected = host(old_out, T * HD);
    expect_true("batch attention equals per-row attention",
                std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v); }) &&
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
    if (rows_per_slot > 1) {
        // Resume after different accepted prefixes and overwrite rejected KV with new rows.
        auto replacement = fvec(T * KD, 3);
        CU(cudaMemcpyAsync(values, replacement.data(), replacement.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        for (int t = 0; t < T; ++t) positions[t] += slot_ids[t] == 2 ? 1 : rows_per_slot - 1;
        CU(cudaMemcpyAsync(dp, positions.data(), T * sizeof(int), cudaMemcpyHostToDevice, stream));
        CU(cudaMemcpyAsync(old_q, qrg.data(), qrg.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CU(cudaMemcpyAsync(q, qrg.data(), qrg.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        per_row();
        batched();
        CU(cudaStreamSynchronize(stream));
        expect_true("causal attention after per-slot rollback", host(out, T * HD) == host(old_out, T * HD));
    }
    if (time_graphs) {
        for (int pass = 0; pass < 2; ++pass) {
            CU(cudaMemcpyAsync(old_q, qrg.data(), qrg.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CU(cudaMemcpyAsync(q, qrg.data(), qrg.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CU(cudaStreamSynchronize(stream));
            float old_us, new_us;
            if (pass == 0) old_us = time_attention_graph(per_row, stream);
            new_us = time_attention_graph(batched, stream);
            if (pass == 1) old_us = time_attention_graph(per_row, stream);
            std::printf("FP4 attention rows=%d depth=%d pass=%d per_row_us=%.2f batched_us=%.2f\n",
                        T, depth, pass + 1, old_us, new_us);
        }
    }
    CU(cudaStreamDestroy(stream));
    for (auto* ptr : {q, old_q, keys, values, sink, rel, out, old_out, scratch}) CU(cudaFree(ptr));
    for (auto* ptr : {kc, vc, old_k, old_v}) CU(cudaFree(ptr));
    CU(cudaFree(old_half));
    if (q16) CU(cudaFree(q16));
    CU(cudaFree(ds));
    CU(cudaFree(dp));
}

static_assert(k::kv_page_head(1048575, 63, 64) * (4 * sizeof(k::Fp4KvBlock)) ==
              1048576ull * 64 * 72 - 72, "FP4 page byte offsets must exceed 32 bits");

void test_fp4_kv_write(bool paged = false) {
    constexpr int D = 128, T = 3, KV = 2, KD = D * KV;
    const int P = paged ? 262143 : 17;
    const int capacity = paged ? ((P + T + 15) / 16) * 16 : P + T;
    const auto layout = test_fp8_layout(paged ? 4 : 3, 0);
    auto key = fvec(T * KD, 9.f), value = fvec(T * KD, 7.f);
    std::fill_n(key.begin(), 32, 0.f);
    std::fill_n(value.begin()+KD, 32, 0.f);
    // An unclipped head with an exact scale exercises every E2M1 midpoint.
    std::fill_n(key.begin() + D, D, 0.f);
    const float midpoints[] = {.25f, -.75f, 1.25f, -1.75f, 2.5f, -3.5f, 5.f, 6.f};
    std::copy_n(midpoints, 8, key.begin() + D);
    auto* dk = dev(key); auto* dv = dev(value);
    auto* q = dev(fvec(T * D)); auto* r = dev(fvec(T * 16));
    auto* kc = dev_n<uint8_t>(size_t(capacity) * KD / 32 * sizeof(k::Fp4KvBlock));
    auto* vc = dev_n<uint8_t>(size_t(capacity) * KD / 32 * sizeof(k::Fp4KvBlock));
    k::prep_qkv_r_fp8(q, nullptr, dk, kc, dv, vc, r, nullptr, T, 1, KV, D, 16,
                      22.f, 8.f, 1e-6f, nullptr, 0.f, 1.f, P, layout, 16.f, 32.f, nullptr);
    auto* uk = dev_n<uint16_t>(T * KD); auto* uv = dev_n<uint16_t>(T * KD);
    k::unpack_kv_fp8_to_f16(kc, vc, layout, P, T, KD, 1/16.f, 1/32.f, uk, uv, nullptr);
    for (bool keys : {true, false}) {
        const auto& input = keys ? key : value;
        auto actual = host(keys ? uk : uv, T * KD);
        auto* blocks = reinterpret_cast<k::Fp4KvBlock*>(keys ? kc : vc);
        int mismatch = 0;
        for (int h=0; h<T*KV; ++h) {
            const size_t token = P + h / KV, head = h % KV;
            const size_t index = paged ? ((token / 16 * KV + head) * 16 + token % 16) : token * KV + head;
            const auto packed = host(blocks + index * (D / 32), D / 32);
            double ss=0;
            for (int d=0; d<D; ++d) ss += (double)input[h*D+d] * input[h*D+d];
            double norm = keys ? std::min(1.,22./(std::sqrt(ss)+1e-6)) : 1./std::sqrt(ss/D+1e-6);
            float external = keys ? 16.f : 32.f;
            for (int g=0; g<D; g+=32) {
                const float* src = input.data()+h*D+g;
                double peak=0;
                for (int d=0; d<32; ++d) peak=std::max(peak,std::abs(src[d]*norm));
                double block_scale=h2f_host(f2h_host(peak*external/6.))/external;
                const auto& block = packed[g / 32];
                mismatch += block.scale != f2h_host(peak * external / 6.);
                for (int d=0; d<32; ++d) {
                    const double target=std::abs(src[d]*norm);
                    int best=0;
                    for (int i=1; i<8; ++i)
                        if (block_scale > 0) {
                            const double error = std::abs(kv_fp4_levels[i] * block_scale - target);
                            const double previous = std::abs(kv_fp4_levels[best] * block_scale - target);
                            if (error < previous || (error == previous && !(i & 1))) best = i;
                        }
                    const int code = (block.pairs[d / 2] >> (4 * (d % 2))) & 15;
                    mismatch += code != (best | (src[d] < 0.f ? 8 : 0));
                    const float wanted=std::copysign(kv_fp4_levels[best]*block_scale,src[d]);
                    mismatch += h2f_host(actual[h*D+g+d]) != h2f_host(f2h_host(wanted));
                }
            }
        }
        expect_true(keys ? "FP4 clipped key CPU pack/unpack" : "FP4 normalized value CPU pack/unpack", mismatch==0);
    }
    for (auto* ptr : {dk, dv, q, r}) release({ptr});
    release({kc, vc, uk, uv});
}

void test_fp8_kv() {
    // 1. All 254 finite E4M3FN byte values -> unpack_kv_fp8_to_f16 bit-for-bit equality vs exact FP16
    {
        std::vector<uint8_t> all_b(256);
        for (int i = 0; i < 256; ++i) all_b[i] = (uint8_t) i;
        uint8_t* db = dev(all_b);
        uint16_t *dk16 = dev_n<uint16_t>(256), *dv16 = dev_n<uint16_t>(256);
        for (int width : {255, 256}) for (float scale : {0.5f, 1.0f, 2.0f}) {
            k::unpack_kv_fp8_to_f16(db, db, 0, 0, 1, width, scale, scale, dk16, dv16, 0);
            const auto hk = host(dk16, 256), hv = host(dv16, 256);
            bool finite_exact = true, nan_ok = true;
            for (int b = 0; b < width; ++b) {
                if ((b & 0x7F) == 0x7F) {
                    nan_ok &= ((hk[b] & 0x7FFF) > 0x7C00) && ((hv[b] & 0x7FFF) > 0x7C00);
                } else {
                    const uint16_t want_bits = f2h_host(scale * e4m3fn_to_f32_host((uint8_t) b));
                    if (hk[b] != want_bits || hv[b] != want_bits) {
                        std::fprintf(stderr, "E4M3 byte %02x (width %d, scale %g): K=%04x V=%04x expected %04x\n", b, width, scale, hk[b], hv[b], want_bits);
                        finite_exact = false;
                    }
                }
            }
            expect_true("e4m3fn_to_f16: finite bytes bit-exact", finite_exact);
            expect_true("e4m3fn_to_f16: 0x7F/0xFF NaN", nan_ok);
        }
        release({db, dk16, dv16});
    }

    // 2. prep_qkv_r_fp8 + unpack_kv_fp8_to_f16 (ring wraparound, linear, and scalar KD=10 tail)
    {
        const int T = 5, H = 48, KV = 8, D = 128, d_rel = 16, kv_cap = 1056;
        const float qk_clip = 22.627416610717773f, rel_clip = 8.0f, eps = 1e-6f, alpha = 0.1f, n_floor = 128000.0f;
        const float k_scale = 16.0f, v_scale = 32.0f, k_inv = 1.0f / k_scale, v_inv = 1.0f / v_scale;
        for (int full : attention_formats) {
            if (full >= 3) continue;
            const int cap = full ? 0 : kv_cap, pos0 = full ? 109 : 1054;
            const int kc_rows = full == 2 ? ((pos0 + T + 15) / 16) * 16 : cap > 0 ? cap : pos0 + T;
            auto q = fvec((size_t) T * H * D, 4.0f), qc = fvec((size_t) H * D, 2.0f),
                 k_in = fvec((size_t) T * KV * D, 4.0f), v_in = fvec((size_t) T * KV * D, 3.0f),
                 r = fvec((size_t) T * H * d_rel, 4.0f), rc = fvec((size_t) H * d_rel, 2.0f);
            for (size_t i = 0; i < (size_t) H * D; ++i) q[i] *= 0.01f;
            std::vector<int> pos = {5, 130000, 200000, 127999, 300000};
            float *dq = dev(q), *dqc = dev(qc), *dk = dev(k_in), *dv = dev(v_in), *dr = dev(r), *drc = dev(rc);
            uint8_t *dkc = dev_n<uint8_t>((size_t) kc_rows * KV * D),
                    *dvc = dev_n<uint8_t>((size_t) kc_rows * KV * D);
            int* dpos = dev(pos);
            k::prep_qkv_r_fp8(dq, dqc, dk, dkc, dv, dvc, dr, drc, T, H, KV, D, d_rel, qk_clip, rel_clip, eps,
                              full ? dpos : nullptr, alpha, n_floor, pos0, test_fp8_layout(full, cap), k_scale, v_scale, 0);
            std::vector<double> wq(q.size()), wr(r.size()), wk((size_t) T * KV * D), wv((size_t) T * KV * D);
            for (int t = 0; t < T; ++t) {
                const double tau =
                    full ? (1.0 + alpha * std::log(std::max(1.0, ((double) pos[t] + 1.0) / n_floor))) : 1.0;
                for (int h = 0; h < H; ++h) {
                    double ss = 0;
                    for (int d = 0; d < D; ++d) {
                        const double v = (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d];
                        ss += v * v;
                    }
                    const double f = std::min(1.0, qk_clip / (std::sqrt(ss) + 1e-6)) * tau;
                    for (int d = 0; d < D; ++d)
                        wq[((size_t) t * H + h) * D + d] =
                            (double) q[((size_t) t * H + h) * D + d] * qc[(size_t) h * D + d] * f;

                    double sr = 0;
                    for (int d = 0; d < d_rel; ++d) {
                        const double v = (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d];
                        sr += v * v;
                    }
                    const double fr = std::min(1.0, rel_clip / (std::sqrt(sr) + 1e-6)) * tau;
                    for (int d = 0; d < d_rel; ++d)
                        wr[((size_t) t * H + h) * d_rel + d] =
                            (double) r[((size_t) t * H + h) * d_rel + d] * rc[(size_t) h * d_rel + d] * fr;
                }
                for (int kv = 0; kv < KV; ++kv) {
                    double sk = 0, sv = 0;
                    for (int d = 0; d < D; ++d) {
                        const double kv_v = k_in[((size_t) t * KV + kv) * D + d];
                        const double vv_v = v_in[((size_t) t * KV + kv) * D + d];
                        sk += kv_v * kv_v;
                        sv += vv_v * vv_v;
                    }
                    const double fk = std::min(1.0, qk_clip / (std::sqrt(sk) + 1e-6));
                    const double rs = 1.0 / std::sqrt(sv / D + eps);
                    for (int d = 0; d < D; ++d) {
                        wk[((size_t) t * KV + kv) * D + d] = k_in[((size_t) t * KV + kv) * D + d] * fk;
                        wv[((size_t) t * KV + kv) * D + d] = v_in[((size_t) t * KV + kv) * D + d] * rs;
                    }
                }
            }
            // Also check unpack_kv_fp8_to_f16 across wrapped slots [pos0 .. pos0+T)
            uint16_t *uk16 = dev_n<uint16_t>((size_t) T * KV * D), *uv16 = dev_n<uint16_t>((size_t) T * KV * D);
            k::unpack_kv_fp8_to_f16(dkc, dvc, test_fp8_layout(full, cap), pos0, T, KV * D, k_inv, v_inv, uk16, uv16, 0);
            const auto hk_un = host(uk16, (size_t) T * KV * D), hv_un = host(uv16, (size_t) T * KV * D);
            const auto kc_raw = host(dkc, (size_t) kc_rows * KV * D), vc_raw = host(dvc, (size_t) kc_rows * KV * D);
            std::vector<float> got_k((size_t) T * KV * D), got_v((size_t) T * KV * D);
            std::vector<double> wk_exact((size_t) T * KV * D), wv_exact((size_t) T * KV * D);
            bool no_nan = true;
            for (int t = 0; t < T; ++t) {
                const int slot = cap > 0 ? (pos0 + t) % cap : (pos0 + t);
                for (size_t j = 0; j < (size_t) KV * D; ++j) {
                    const size_t k = full == 2 ? page_offset(slot, j / D, j % D, KV, false) : (size_t)slot * KV * D + j;
                    const size_t v = full == 2 ? page_offset(slot, j / D, j % D, KV, true) : (size_t)slot * KV * D + j;
                    const uint8_t bk = kc_raw[k], bv = vc_raw[v];
                    no_nan &= ((bk & 0x7F) != 0x7F) && ((bv & 0x7F) != 0x7F);
                    got_k[(size_t) t * KV * D + j] = h2f_host(hk_un[(size_t) t * KV * D + j]);
                    got_v[(size_t) t * KV * D + j] = h2f_host(hv_un[(size_t) t * KV * D + j]);
                    wk_exact[(size_t) t * KV * D + j] = (double) e4m3fn_to_f32_host(bk) * k_inv;
                    wv_exact[(size_t) t * KV * D + j] = (double) e4m3fn_to_f32_host(bv) * v_inv;
                }
            }
            expect_true(full ? "prep_qkv_r_fp8 full no NaN" : "prep_qkv_r_fp8 ring no NaN", no_nan);
            expect(full ? "prep_qkv_r_fp8 full q" : "prep_qkv_r_fp8 ring q", host(dq, wq.size()), wq, 2e-5);
            expect(full ? "prep_qkv_r_fp8 full r" : "prep_qkv_r_fp8 ring r", host(dr, wr.size()), wr, 2e-5);
            // E4M3 quantization accuracy (~1/16 max relative step)
            expect(full ? "prep_qkv_r_fp8 full k (e4m3)" : "prep_qkv_r_fp8 ring k (e4m3)", got_k, wk, 0.065, 2e-3);
            expect(full ? "prep_qkv_r_fp8 full v (e4m3)" : "prep_qkv_r_fp8 ring v (e4m3)", got_v, wv, 0.065, 2e-3);
            // unpack_kv_fp8_to_f16 exact FP16 decode of stored FP8 bytes
            expect(full ? "unpack_kv_fp8 full k" : "unpack_kv_fp8 ring k", got_k, wk_exact, 1e-3, 1e-5);
            expect(full ? "unpack_kv_fp8 full v" : "unpack_kv_fp8 ring v", got_v, wv_exact, 1e-3, 1e-5);
            for (float* p : {dq, dqc, dk, dv, dr, drc}) release({p});
            release({dkc, dvc, dpos, uk16, uv16});
        }

        // Scalar fallback for unpack_kv_fp8_to_f16 (KD=10, non-mult4, ring wraparound)
        const int nk = 7, KD_s = 10, cap_s = 5, k0_s = 3;
        std::vector<uint8_t> ks((size_t) cap_s * KD_s), vs((size_t) cap_s * KD_s);
        for (size_t i = 0; i < ks.size(); ++i) {
            ks[i] = (uint8_t) ((rng() & 0x7Eu) | ((rng() & 1u) << 7));
            vs[i] = (uint8_t) ((rng() & 0x7Eu) | ((rng() & 1u) << 7));
            if ((ks[i] & 0x7F) == 0x7F) ks[i] = 0x7E;
            if ((vs[i] & 0x7F) == 0x7F) vs[i] = 0x7E;
        }
        uint8_t *dks = dev(ks), *dvs = dev(vs);
        uint16_t *uk = dev_n<uint16_t>((size_t) nk * KD_s), *uv = dev_n<uint16_t>((size_t) nk * KD_s);
        k::unpack_kv_fp8_to_f16(dks, dvs, cap_s, k0_s, nk, KD_s, k_inv, v_inv, uk, uv, 0);
        const auto hk = host(uk, (size_t) nk * KD_s), hv = host(uv, (size_t) nk * KD_s);
        std::vector<float> gk(hk.size()), gv(hv.size());
        std::vector<double> wk(hk.size()), wv(hv.size());
        for (int j = 0; j < nk; ++j) {
            const int slot = (k0_s + j) % cap_s;
            for (int d = 0; d < KD_s; ++d) {
                const size_t idx = (size_t) j * KD_s + d;
                gk[idx] = h2f_host(hk[idx]);
                gv[idx] = h2f_host(hv[idx]);
                wk[idx] = (double) e4m3fn_to_f32_host(ks[(size_t) slot * KD_s + d]) * k_inv;
                wv[idx] = (double) e4m3fn_to_f32_host(vs[(size_t) slot * KD_s + d]) * v_inv;
            }
        }
        expect("unpack_kv_fp8 scalar k", gk, wk, 1e-3, 1e-5);
        expect("unpack_kv_fp8 scalar v", gv, wv, 1e-3, 1e-5);
        release({dks, dvs, uk, uv});
    }

    // 3. attn_decode_fp8 & attn_multi_fp8 (linear & ring buffer) vs double reference
    {
        const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 3100, kv_cap = 1056, window = 1024, nq = 8;
        const float k_inv = 1.0f / 16.0f, v_inv = 1.0f / 32.0f, scale = 1.0f / std::sqrt(128.0f);
        std::vector<uint8_t> kc_lin((size_t) ctx * KV * D), vc_lin((size_t) ctx * KV * D);
        for (size_t i = 0; i < kc_lin.size(); ++i) {
            uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
            if ((bk & 0x7F) >= 0x7E) bk = (bk & 0x80u) | 0x60u;
            if ((bv & 0x7F) >= 0x7E) bv = (bv & 0x80u) | 0x60u;
            kc_lin[i] = bk;
            vc_lin[i] = bv;
        }
        std::vector<uint8_t> kc_ring((size_t) kv_cap * KV * D, 0), vc_ring((size_t) kv_cap * KV * D, 0);
        for (int pos = 1476; pos < 2510; ++pos) {
            const int slot = pos % kv_cap;
            std::memcpy(kc_ring.data() + (size_t) slot * KV * D, kc_lin.data() + (size_t) pos * KV * D,
                        (size_t) KV * D);
            std::memcpy(vc_ring.data() + (size_t) slot * KV * D, vc_lin.data() + (size_t) pos * KV * D,
                        (size_t) KV * D);
        }
        auto q = fvec((size_t) nq * H * D, 3.0f), r = fvec((size_t) nq * H * d_rel, 2.0f),
             sink = fvec((size_t) nq * H, 3.0f), proj = fvec((size_t) 1024 * d_rel);
        uint8_t *dk_lin = dev(kc_lin), *dv_lin = dev(vc_lin), *dk_ring = dev(kc_ring), *dv_ring = dev(vc_ring);
        float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        float *scratch = dev_n<float>((size_t) nq * H * (ctx + 32 * 130)), *out = dev_n<float>((size_t) nq * H * D),
              *one = dev_n<float>((size_t) H * D);

        for (int ring = 0; ring < 2; ++ring) {
            const int k0 = ring ? 1476 : 0, nk = ring ? 1024 : 2500, qpos = 2499, w = ring ? window : (1 << 30);
            k::attn_decode_fp8(dq, dr, ds, ring ? dk_ring : dk_lin, ring ? dv_ring : dv_lin, ring ? kv_cap : 0, H, KV,
                               D, k0, nk, qpos, scale * k_inv, v_inv, dp, d_rel, 1024, w, scratch, one, 0);
            std::vector<double> want((size_t) H * D);
            double mag = 0;   // the largest sum of |terms| of an output (the tolerance below)
            for (int h = 0; h < H; ++h) {
                const int g = h / (H / KV);
                std::vector<double> l(nk);
                double m = sink[h];
                for (int j = 0; j < nk; ++j) {
                    const int dist = qpos - (k0 + j);
                    if (dist < 0 || dist >= w) {
                        l[j] = -INFINITY;
                        continue;
                    }
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += (double) q[h * D + d] *
                             (double) e4m3fn_to_f32_host(kc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                    a *= (double) (scale * k_inv);
                    if (dist < 1024)
                        for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                    l[j] = a;
                    m = std::max(m, a);
                }
                double sum = std::exp(sink[h] - m);
                for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
                for (int d = 0; d < D; ++d) {
                    double o = 0, oa = 0;
                    for (int j = 0; j < nk; ++j) {
                        const double t = std::exp(l[j] - m) / sum *
                                         (double) e4m3fn_to_f32_host(vc_lin[((size_t) (k0 + j) * KV + g) * D + d]);
                        o += t;
                        oa += std::fabs(t);
                    }
                    want[(size_t) h * D + d] = o * (double) v_inv;
                    mag = std::max(mag, oa * (double) v_inv);
                }
            }
            // (V spans E4M3's whole range, so an output is terms up to ~448 that cancel - F32 rounds each at ~6e-8,
            //  over up to 2500 of them ~sqrt(2500) x 6e-8: the absolute part follows the terms' size, 3e-6 of it)
            expect(ring ? "attn_decode_fp8 ring 1056" : "attn_decode_fp8 linear 2500", host(one, want.size()), want,
                   2e-4, std::max(1e-6, 3e-6 * mag));
        }

        const int pos0 = 2500, k0 = pos0 - window + 1, nk = pos0 + nq - k0;
        k::attn_multi_fp8(dq, dr, ds, dk_ring, dv_ring, kv_cap, nq, H, KV, k0, nk, pos0, scale * k_inv, v_inv, dp,
                           d_rel, 1024, window, scratch, out, 0);
        const auto got = host(out, (size_t) nq * H * D);
        std::vector<double> want((size_t) nq * H * D);
        for (int qi = 0; qi < nq; ++qi) {
            const int qp = pos0 + qi, kq = qp - window + 1;
            k::attn_decode_fp8(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H, dk_ring,
                               dv_ring, kv_cap, H, KV, D, kq, qp + 1 - kq, qp, scale * k_inv, v_inv, dp, d_rel, 1024,
                               window, scratch, one, 0);
            const auto o1 = host(one, (size_t) H * D);
            for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
        }
        expect("attn_multi_fp8 ring 1056", got, want, 1e-4, 1e-6);

        for (uint8_t* p : {dk_lin, dv_lin, dk_ring, dv_ring}) release({p});
        for (float* p : {dq, dr, ds, dp, scratch, out, one}) release({p});
    }

    // 4. attn_decode_fp8 with nk > 16384 (exercises P = 128 splits and kv_cap = 0 full-layer path)
    for (int unit_kv = 0; unit_kv < 2; ++unit_kv) {
        const int H = 48, KV = 8, D = 128, d_rel = 16, nk = 16640, qpos = nk - 1, w = 1 << 30;
        const float k_inv = unit_kv ? 1.0f : 1.0f / 16.0f, v_inv = unit_kv ? 1.0f : 1.0f / 32.0f;
        const float scale = 1.0f / std::sqrt(128.0f);
        std::vector<uint8_t> kc((size_t) nk * KV * D), vc((size_t) nk * KV * D);
        for (size_t i = 0; i < kc.size(); ++i) {
            uint8_t bk = (uint8_t) (rng() & 0xFFu), bv = (uint8_t) (rng() & 0xFFu);
            if (unit_kv) {
                // Realistic unit-scale KV magnitudes (<= 4.0) without 1e4x random-walk cancellation across 16640 keys.
                bk = (bk & 0x80u) | (uint8_t) (rng() % 0x49u);
                bv = (bv & 0x80u) | (uint8_t) (rng() % 0x49u);
            } else {
                if ((bk & 0x7F) >= 0x7E) bk = (bk & 0x80u) | 0x60u;
                if ((bv & 0x7F) >= 0x7E) bv = (bv & 0x80u) | 0x60u;
            }
            kc[i] = bk;
            vc[i] = bv;
        }
        auto q = fvec((size_t) H * D, 3.0f), r = fvec((size_t) H * d_rel, 2.0f), sink = fvec((size_t) H, 3.0f),
             proj = fvec((size_t) 1024 * d_rel);
        uint8_t *dk = dev(kc), *dv = dev(vc);
        float *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        float *scratch = dev_n<float>((size_t) H * (nk + 128 * 130)), *one = dev_n<float>((size_t) H * D);
        k::attn_decode_fp8(dq, dr, ds, dk, dv, 0, H, KV, D, 0, nk, qpos, scale * k_inv, v_inv, dp, d_rel, 1024, w,
                           scratch, one, 0);
        std::vector<double> want((size_t) H * D);
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KV);
            std::vector<double> l(nk);
            double m = sink[h];
            for (int j = 0; j < nk; ++j) {
                const int dist = qpos - j;
                double a = 0;
                for (int d = 0; d < D; ++d)
                    a += (double) q[h * D + d] *
                         (double) e4m3fn_to_f32_host(kc[((size_t) j * KV + g) * D + d]);
                a *= (double) (scale * k_inv);
                if (dist < 1024)
                    for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                l[j] = a;
                m = std::max(m, a);
            }
            double sum = std::exp(sink[h] - m);
            for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int j = 0; j < nk; ++j)
                    o += std::exp(l[j] - m) / sum *
                         (double) e4m3fn_to_f32_host(vc[((size_t) j * KV + g) * D + d]);
                want[(size_t) h * D + d] = o * (double) v_inv;
            }
        }
        expect(unit_kv ? "attn_decode_fp8 P=128 unit (nk=16640)" : "attn_decode_fp8 P=128 raw (nk=16640)",
               host(one, want.size()), want, 2e-4, unit_kv ? 1e-6 : 5e-6);
#ifdef STRATA_USE_HIP
        int* position = dev(std::vector<int>{qpos});
        k::attn_decode_fp8_at(dq, dr, ds, dk, dv, 0, H, KV, position, 128, scale * k_inv, v_inv,
                              dp, d_rel, 1024, w, scratch, one, 0);
        expect(unit_kv ? "resident E4M3 attention, unit values" : "resident E4M3 attention, full range",
               host(one, want.size()), want, 2e-4, unit_kv ? 1e-6 : 5e-6);
        release({position});
#endif
        for (uint8_t* p : {dk, dv}) release({p});
        for (float* p : {dq, dr, ds, dp, scratch, one}) release({p});
    }
}

void test_round6() {
    // Fused attention residuals with optional weights and both output formats.
    for (int C : {4608, 517, 2304}) {
        const int T = C == 2304 ? 1 : 7, n = 9;
        for (int nq : {1, 3}) {
            std::vector<std::vector<float>> st(n);
            k::ResStates rs{};
            std::vector<float*> ds(n);
            for (int i = 0; i < n; ++i) st[i] = fvec((size_t) T * C, 1.0f + 0.5f * i), ds[i] = dev(st[i]), rs.st[i] = ds[i];
            auto q = fvec((size_t) nq * C, 2.5f);
            float* dq = dev(q);
            std::vector<std::vector<float>> w(nq);
            std::vector<float*> dw(nq, nullptr), dy32(nq, nullptr);
            std::vector<uint16_t*> dy16(nq, nullptr);
            k::RmsOuts outs{};
            for (int qi = 0; qi < nq; ++qi) {
                if (!(nq == 3 && qi == 1)) {   // leave qi==1 weight null in nq==3 case to exercise null-weight path
                    w[qi] = fvec(C, 2.0f);
                    dw[qi] = dev(w[qi]);
                }
                outs.w[qi] = dw[qi];
                dy16[qi] = dev_n<uint16_t>((size_t) T * C);
                outs.y16[qi] = dy16[qi];
                if (!(nq == 3 && qi == 2)) {   // leave qi==2 y32 null in nq==3 case
                    dy32[qi] = dev_n<float>((size_t) T * C);
                    outs.y32[qi] = dy32[qi];
                }
            }
            const float res_eps = 1e-6f, rms_eps = 1e-5f;
            k::attn_res_rms_f16(rs, n, dq, nq, T, C, res_eps, rms_eps, outs, 0);
            for (int qi = 0; qi < nq; ++qi) {
                std::vector<double> want_y((size_t) T * C);
                for (int t = 0; t < T; ++t) {
                    std::vector<double> r(n);
                    for (int i = 0; i < n; ++i) {
                        double ss = 0;
                        for (int d = 0; d < C; ++d) ss += (double) st[i][(size_t) t * C + d] * st[i][(size_t) t * C + d];
                        r[i] = 1.0 / std::sqrt(ss / C + res_eps);
                    }
                    std::vector<double> sc(n);
                    double m = -1e300, sum = 0;
                    for (int i = 0; i < n; ++i) {
                        double dot = 0;
                        for (int d = 0; d < C; ++d) dot += (double) q[(size_t) qi * C + d] * st[i][(size_t) t * C + d] * r[i];
                        sc[i] = dot / std::sqrt((double) C);
                        m = std::max(m, sc[i]);
                    }
                    for (int i = 0; i < n; ++i) sum += (sc[i] = std::exp(sc[i] - m));
                    std::vector<double> o(C);
                    double oo = 0;
                    for (int d = 0; d < C; ++d) {
                        double a = 0;
                        for (int i = 0; i < n; ++i) a += sc[i] / sum * st[i][(size_t) t * C + d] * r[i];
                        o[d] = a;
                        oo += a * a;
                    }
                    const double ir = 1.0 / std::sqrt(oo / C + rms_eps);
                    for (int d = 0; d < C; ++d)
                        want_y[(size_t) t * C + d] = o[d] * ir * (dw[qi] ? (double) w[qi][d] : 1.0);
                }
                const auto h16 = host(dy16[qi], want_y.size());
                std::vector<float> got16(want_y.size());
                for (size_t j = 0; j < want_y.size(); ++j) got16[j] = h2f_host(h16[j]);
                char label[96];
                std::snprintf(label, sizeof label, "attn_res_rms_f16 C=%d nq=%d qi=%d y16", C, nq, qi);
                expect(label, got16, want_y, 2e-3, 1e-4);
                if (dy32[qi]) {
                    std::snprintf(label, sizeof label, "attn_res_rms_f16 C=%d nq=%d qi=%d y32", C, nq, qi);
                    expect(label, host(dy32[qi], want_y.size()), want_y, 1e-4, 5e-6);
                }
            }
            for (auto* p : ds) release({p});
            for (int qi = 0; qi < nq; ++qi) release({dw[qi], dy16[qi], dy32[qi]});
            release({dq});
        }
    }

    // 2. conv4_res: static/dynamic with and without residual across T=1 and T=41
    for (int T : {1, 41}) {
        const int C = 1024, heads = 8;
        auto x = fvec((size_t) T * C), w = fvec((size_t) C * 4), dyn = fvec((size_t) T * heads * 4),
             res = fvec((size_t) T * C, 2.0f);
        float *dx = dev(x), *dw = dev(w), *dd = dev(dyn), *dres = dev(res), *dy = dev_n<float>((size_t) T * C);
        for (int with_dyn : {0, 1}) {
            const auto base = conv_ref(x, w, with_dyn ? &dyn : nullptr, heads, T, C);
            for (int with_res : {0, 1}) {
                std::vector<double> want(base.size());
                for (size_t i = 0; i < base.size(); ++i) want[i] = base[i] + (with_res ? (double) res[i] : 0.0);
                float* hist = dev_n<float>((size_t) 3 * C);
                k::conv4_res(dx, hist, dw, with_dyn ? dd : nullptr, heads, T, C, with_res ? dres : nullptr, dy, 0);
                char label[80];
                std::snprintf(label, sizeof label, "conv4_res T=%d %s %s", T, with_dyn ? "dyn" : "static",
                              with_res ? "+res" : "no-res");
                expect(label, host(dy, want.size()), want, 2e-5);
                release({hist});
            }
        }
        release({dx, dw, dd, dres, dy});
    }

    // 3. attn_decode_* & attn_multi_* across split boundaries nk = 10, 64, 65, 200, 1056, 2500, 16640 (F32, F16, FP8)
    {
        const int H = 48, KV = 8, D = 128, d_rel = 16, ctx = 16640, nq = 4;
        const float scale = 1.0f / std::sqrt(128.0f);
        auto kc32 = fvec((size_t) ctx * KV * D, 1.5f), vc32 = fvec((size_t) ctx * KV * D, 1.5f);
        std::vector<uint16_t> kc16(kc32.size()), vc16(vc32.size());
        std::vector<uint8_t> kc8(kc32.size()), vc8(vc32.size());
        for (size_t i = 0; i < kc32.size(); ++i) {
            kc16[i] = f2h_host(kc32[i]);
            vc16[i] = f2h_host(vc32[i]);
            kc8[i] = (uint8_t) ((rng() & 0x80u) | (uint8_t) (rng() % 0x49u));   // unit-scale finite E4M3 <= 4.0
            vc8[i] = (uint8_t) ((rng() & 0x80u) | (uint8_t) (rng() % 0x49u));
        }
        auto q = fvec((size_t) nq * H * D, 2.0f), r = fvec((size_t) nq * H * d_rel, 1.5f),
             sink = fvec((size_t) nq * H, 2.5f), proj = fvec((size_t) 1024 * d_rel, 1.0f);
        float *dk32 = dev(kc32), *dv32 = dev(vc32), *dq = dev(q), *dr = dev(r), *ds = dev(sink), *dp = dev(proj);
        uint16_t *dk16 = dev(kc16), *dv16 = dev(vc16);
        uint8_t *dk8 = dev(kc8), *dv8 = dev(vc8);
        float *scratch = dev_n<float>((size_t) nq * H * (ctx + 128 * 130)), *one = dev_n<float>((size_t) H * D),
              *multi = dev_n<float>((size_t) nq * H * D);

        for (int nk : {10, 64, 65, 200, 1056, 2500, 16640}) {
            const int qpos = nk - 1, w = nk == 1056 ? 512 : (1 << 30);   // exercise sliding mask on nk=1056 too
            for (int kind = 0; kind < 3; ++kind) {
                if (kind == 0)
                    k::attn_decode(dq, dr, ds, dk32, dv32, H, KV, D, 0, nk, qpos, scale, dp, d_rel, 1024, w, scratch,
                                   one, 0);
                else if (kind == 1)
                    k::attn_decode_f16(dq, dr, ds, dk16, dv16, 0, H, KV, D, 0, nk, qpos, scale, dp, d_rel, 1024, w,
                                       scratch, one, 0);
                else
                    k::attn_decode_fp8(dq, dr, ds, dk8, dv8, 0, H, KV, D, 0, nk, qpos, scale, 1.0f, dp, d_rel, 1024, w,
                                       scratch, one, 0);
                std::vector<double> want((size_t) H * D);
                for (int h = 0; h < H; ++h) {
                    const int g = h / (H / KV);
                    std::vector<double> l(nk);
                    double m = sink[h];
                    for (int j = 0; j < nk; ++j) {
                        const int dist = qpos - j;
                        if (dist < 0 || dist >= w) {
                            l[j] = -INFINITY;
                            continue;
                        }
                        double a = 0;
                        for (int d = 0; d < D; ++d) {
                            const size_t idx = ((size_t) j * KV + g) * D + d;
                            const double kv = kind == 0 ? kc32[idx]
                                            : kind == 1 ? h2f_host(kc16[idx])
                                                        : e4m3fn_to_f32_host(kc8[idx]);
                            a += (double) q[h * D + d] * kv;
                        }
                        a *= scale;
                        if (dist < 1024)
                            for (int e = 0; e < d_rel; ++e) a += (double) r[h * d_rel + e] * proj[(size_t) dist * d_rel + e];
                        l[j] = a;
                        m = std::max(m, a);
                    }
                    double sum = std::exp(sink[h] - m);
                    for (int j = 0; j < nk; ++j) sum += std::exp(l[j] - m);
                    for (int d = 0; d < D; ++d) {
                        double o = 0;
                        for (int j = 0; j < nk; ++j) {
                            const size_t idx = ((size_t) j * KV + g) * D + d;
                            const double vv = kind == 0 ? vc32[idx]
                                            : kind == 1 ? h2f_host(vc16[idx])
                                                        : e4m3fn_to_f32_host(vc8[idx]);
                            o += std::exp(l[j] - m) / sum * vv;
                        }
                        want[(size_t) h * D + d] = o;
                    }
                }
                const char* knames[] = {"f32", "f16", "fp8"};
                char label[64];
                std::snprintf(label, sizeof label, "attn_decode_%s nk=%d", knames[kind], nk);
                expect(label, host(one, want.size()), want, 2e-4, 1e-6);
            }
        }
        for (int nk : {64, 65, 1056, 2500}) {
            const int pos0 = nk - nq, w = nk == 1056 ? 512 : (1 << 30);
            for (int kind = 0; kind < 3; ++kind) {
                if (kind == 0)
                    k::attn_multi(dq, dr, ds, dk32, dv32, nq, H, KV, 0, nk, pos0, scale, dp, d_rel, 1024, w, scratch,
                                  multi, 0);
                else if (kind == 1)
                    k::attn_multi_f16(dq, dr, ds, dk16, dv16, 0, nq, H, KV, 0, nk, pos0, scale, dp, d_rel, 1024, w,
                                      scratch, multi, 0);
                else
                    k::attn_multi_fp8(dq, dr, ds, dk8, dv8, 0, nq, H, KV, 0, nk, pos0, scale, 1.0f, dp, d_rel, 1024, w,
                                      scratch, multi, 0);
                const auto got = host(multi, (size_t) nq * H * D);
                std::vector<double> want((size_t) nq * H * D);
                for (int qi = 0; qi < nq; ++qi) {
                    const int qp = pos0 + qi;
                    if (kind == 0)
                        k::attn_decode(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                       dk32, dv32, H, KV, D, 0, qp + 1, qp, scale, dp, d_rel, 1024, w, scratch, one, 0);
                    else if (kind == 1)
                        k::attn_decode_f16(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                           dk16, dv16, 0, H, KV, D, 0, qp + 1, qp, scale, dp, d_rel, 1024, w, scratch,
                                           one, 0);
                    else
                        k::attn_decode_fp8(dq + (size_t) qi * H * D, dr + (size_t) qi * H * d_rel, ds + (size_t) qi * H,
                                           dk8, dv8, 0, H, KV, D, 0, qp + 1, qp, scale, 1.0f, dp, d_rel, 1024, w,
                                           scratch, one, 0);
                    const auto o1 = host(one, (size_t) H * D);
                    for (size_t j = 0; j < o1.size(); ++j) want[(size_t) qi * H * D + j] = o1[j];
                }
                const char* knames[] = {"f32", "f16", "fp8"};
                char label[64];
                std::snprintf(label, sizeof label, "attn_multi_%s nk=%d", knames[kind], nk);
                expect(label, got, want, 1e-4, 1e-6);
            }
        }
        for (float* p : {dk32, dv32, dq, dr, ds, dp, scratch, one, multi}) release({p});
        release({dk16, dv16, dk8, dv8});
    }

    // 4. route: E=384 (route_warp_kernel) and E=600 (fallback route_kernel), with exact score ties
    for (int E : {384, 600}) {
        const int T = 64, K = 8;
        auto lg = fvec((size_t) T * E, 6.0f), bias = fvec(E, 0.2f);
        // Inject 10 exact duplicate top-scoring ties so K=8 must deterministically pick the 8 lowest indices
        for (int e : {5, 37, 101, 199, 255, 320, 350, 360, 370, 383}) bias[e] = 5.0f;
        for (int t = 0; t < T; ++t) {
            const float shared_lg = 2.0f + 0.05f * (t & 7);
            for (int e : {5, 37, 101, 199, 255, 320, 350, 360, 370, 383}) lg[(size_t) t * E + e] = shared_lg;
        }
        float *dl = dev(lg), *db = dev(bias), *dw = dev_n<float>((size_t) T * K);
        int* dt = dev_n<int>((size_t) T * K);
        k::route(dl, db, T, E, K, 8.0f, dt, dw, 0);
        const auto top = host(dt, (size_t) T * K);
        const auto w = host(dw, (size_t) T * K);
        bool picks_ok = true;
        std::vector<double> want_w((size_t) T * K);
        for (int t = 0; t < T; ++t) {
            std::vector<std::pair<float, int>> v(E);
            for (int e = 0; e < E; ++e) {
                const float sg = 1.0f / (1.0f + std::exp(-lg[(size_t) t * E + e]));
                v[e] = {-(sg + bias[e]), e};
            }
            std::sort(v.begin(), v.end());
            std::vector<int> pick(K);
            for (int i = 0; i < K; ++i) pick[i] = v[i].second;
            std::sort(pick.begin(), pick.end());
            double sum = 0;
            for (int i = 0; i < K; ++i) sum += 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]]));
            for (int i = 0; i < K; ++i) {
                picks_ok &= top[(size_t) t * K + i] == pick[i];
                want_w[(size_t) t * K + i] =
                    1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + pick[i]])) / (sum + 1e-20) * 8.0;
            }
        }
        char label[64];
        std::snprintf(label, sizeof label, "route ties E=%d picks", E);
        expect_true(label, picks_ok);
        if (picks_ok) {
            std::snprintf(label, sizeof label, "route ties E=%d weights", E);
            expect(label, w, want_w, 2e-5);
        }
        release({dl, db, dw, dt});
    }

    // 5. experts_gate_up_swiglu & experts_down (row_dot_2 + odd C=513 + launch thresholds S=1,3,5,8)
    {
        const int C = 513, F = 256, E = 3, S_max = 8;   // odd C exercises tail row in row_dot_2
        const int types[][3] = {{k::kNVFP4, k::kNVFP4, k::kNVFP4},
                                {k::kQ8_0, k::kQ8_0, k::kQ8_0},
                                {k::kQ5_K, k::kQ5_K, k::kQ5_K},
                                {k::kQ6_K, k::kQ6_K, k::kQ6_K},
                                {k::kIQ3_XXS, k::kIQ3_XXS, k::kNVFP4},
                                {k::kIQ4_XS, k::kIQ4_XS, k::kQ6_K}};
        const char* tnames[] = {"NVFP4", "Q8_0", "Q5_K", "Q6_K", "IQ3_XXS/NVFP4", "IQ4_XS/Q6_K"};
        for (int ti = 0; ti < 6; ++ti) {
            const int tg = types[ti][0], tu = types[ti][1], td = types[ti][2];
            const int Cin = 512;   // weight row width for gate/up (multiple of 256 for Q5_K/Q6_K)
            auto g = rand_blocks(tg, (size_t) E * F * Cin), u = rand_blocks(tu, (size_t) E * F * Cin),
                 d = rand_blocks(td, (size_t) E * C * F);
            const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E;
            const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
            std::vector<uint8_t> pool(slot * E);
            for (int e = 0; e < E; ++e) {
                std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
                std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
                std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
            }
            uint8_t* dpool = dev(pool);
            const int pick[S_max] = {2, 0, 2, 1, 0, 1, 2, 0}, tok[S_max] = {0, 1, 1, 2, 0, 2, 1, 0};
            std::vector<const uint8_t*> blob(S_max);
            for (int i = 0; i < S_max; ++i) blob[i] = dpool + pick[i] * slot;
            const uint8_t** dblob = nullptr;
            CU(cudaMalloc(&dblob, S_max * sizeof(void*)));
            CU(cudaMemcpy(dblob, blob.data(), S_max * sizeof(void*), cudaMemcpyHostToDevice));
            auto x = fvec((size_t) 3 * Cin, 1.5f), sg = fvec(S_max, 0.5f), su = fvec(S_max, 0.5f);
            float *dx = dev(x), *dgu = dev_n<float>((size_t) S_max * 2 * F), *dsg = dev(sg), *dsu = dev(su),
                  *dy = dev_n<float>((size_t) S_max * C);
            int* dtok = dev(std::vector<int>(tok, tok + S_max));
            std::vector<uint16_t> h((size_t) S_max * F);
            std::vector<float> hf(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                h[i] = (uint16_t) ((rng() & 0x3BFF) | ((rng() & 1) << 15));
                hf[i] = h2f_host(h[i]);
            }
            uint16_t *dh_in = dev(h), *dh_ref = dev_n<uint16_t>((size_t) S_max * F),
                     *dh_got = dev_n<uint16_t>((size_t) S_max * F);
            std::vector<double> want_y((size_t) S_max * C);
            std::vector<float> row(F);
            for (int i = 0; i < S_max; ++i) {
                const int e = pick[i];
                for (int r = 0; r < C; ++r) {
                    k::decode_host(td, d.data() + (size_t) e * eb2 + (size_t) r * (eb2 / C), F, row.data());
                    double a = 0;
                    for (int c = 0; c < F; ++c) a += (double) row[c] * hf[(size_t) i * F + c];
                    want_y[(size_t) i * C + r] = a;
                }
            }
            for (int S : {1, 3, 5, 8}) {
                k::experts_gate_up(dblob, dtok, S, tg, tu, eb0, dx, Cin, F, dgu, 0);
                k::swiglu_cap_f32(dgu, S, F, dsg, dsu, 20.0f, 20.0f, dh_ref, 0);
                k::experts_gate_up_swiglu(dblob, dtok, S, tg, tu, eb0, dx, Cin, F, dsg, dsu, 20.0f, 20.0f, dh_got, 0);
                const auto r16 = host(dh_ref, (size_t) S * F), g16 = host(dh_got, (size_t) S * F);
                std::vector<float> fg(g16.size());
                std::vector<double> wr(r16.size());
                for (size_t j = 0; j < g16.size(); ++j) fg[j] = h2f_host(g16[j]), wr[j] = h2f_host(r16[j]);
                char label[80];
                std::snprintf(label, sizeof label, "r6 experts_gu_swiglu S=%d %s", S, tnames[ti]);
                expect(label, fg, wr, 2e-3, 1e-3);

                k::experts_down(dblob, S, td, eb0 + eb1, dh_in, C, F, dy, 0);
                std::snprintf(label, sizeof label, "r6 experts_down C=513 S=%d %s", S, tnames[ti]);
                expect(label, host(dy, (size_t) S * C),
                       std::vector<double>(want_y.begin(), want_y.begin() + (size_t) S * C), 2e-4, 2e-3);
            }
            for (void* p : {(void*) dpool, (void*) dblob, (void*) dx, (void*) dgu, (void*) dsg, (void*) dsu,
                            (void*) dy, (void*) dtok, (void*) dh_in, (void*) dh_ref, (void*) dh_got})
                release({p});
        }
    }
}

}  // namespace

#ifdef STRATA_USE_HIP
void test_rdna3_dense_output_prefill() {
    constexpr int N = 2304, K = 3072, T = 513;
    std::mt19937 gen(8033);
    std::vector<uint16_t> weights((size_t)N * K), x((size_t)T * K);
    auto fill = [&](std::vector<uint16_t>& values) {
        for (auto& v : values) v = f2h_host(((int)(gen() % 511) - 255) / 256.f);
    };
    fill(weights);
    fill(x);
    auto* dw = dev(weights);
    auto* dx = dev(x);
    auto* dy = dev_n<float>((size_t)T * N);
    k::rdna3_dense_gemm(dw, N, K, dx, T, dy, N, 0);
    const auto result = host(dy, (size_t)T * N);
    std::vector<float> actual;
    std::vector<double> expected;
    for (int t : {0, 15, 64, 511, 512})
        for (int n : {0, 31, 63, 64, 1153, N - 1}) {
            double dot = 0;
            for (int k = 0; k < K; ++k) dot += (double)h2f_host(weights[n * K + k]) * h2f_host(x[t * K + k]);
            actual.push_back(result[t * N + n]);
            expected.push_back(dot);
        }
    expect("3072-wide prefill output projection vs FP64", actual, expected, 2e-4, 5e-4);
    release({dw, dx, dy});
}

void test_rdna3_small_dense() {
    constexpr int K = 2304, N = 128;
    const auto x = fvec(K);
    std::vector<uint16_t> weights(N * K);
    for (auto& w : weights) w = f2h_host(frand());
    auto* dx = dev(x);
    auto* dw = dev(weights);
    auto* dy = dev_n<float>(N);
    std::vector<double> ref(N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < K; ++c) ref[r] += h2f_host(weights[r * K + c]) * (double)x[c];
    for (int rows : {32, 128}) {
        k::mv_f16(dw, rows, K, dx, dy, 0);
        expect("RDNA3 small dense matvec", host(dy, rows), std::vector<double>(ref.begin(), ref.begin() + rows),
               2e-4, 1e-3);
        if (std::getenv("IK2_F16_MV") && std::atoi(std::getenv("IK2_F16_MV")) == 2) {
            const auto actual = host(dy, rows);
            bool exact = true;
            for (int r = 0; r < rows; ++r) {
                float lanes[32] = {};
                for (int i = 0; i < K; ++i)
                    lanes[i % 32] = std::fma(h2f_host(weights[r * K + i]), x[i], lanes[i % 32]);
                for (int distance = 16; distance; distance /= 2)
                    for (int i = 0; i < distance; ++i) lanes[i] += lanes[i + distance];
                exact &= actual[r] == lanes[0];
            }
            expect_true("RDNA3 small dense matvec preserves per-lane FP32 sums", exact);
        }
    }
    release({dy, dw, dx});
}

void test_rdna3_dense_gemm() {
    constexpr int K = 192, N = 141, T = 79, ld = N + 9;
    auto x = fvec(T * K), w = fvec(N * K);
    std::vector<uint16_t> xh(x.size()), wh(w.size());
    for (size_t i = 0; i < x.size(); ++i) xh[i] = f2h_host(x[i]);
    for (size_t i = 0; i < w.size(); ++i) wh[i] = f2h_host(w[i]);
    auto *dx = dev(xh), *dw = dev(wh);
    auto* y = dev(std::vector<float>(T * ld, -77.f));
    k::rdna3_dense_gemm(dw, N, K, dx, T, y, ld, 0);
    std::vector<double> ref(T * ld, -77.);
    for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) {
        double v = 0;
        for (int k = 0; k < K; ++k) v += (double)h2f_host(wh[n * K + k]) * h2f_host(xh[t * K + k]);
        ref[t * ld + n] = v;
    }
    expect("RDNA3 dense WMMA vs FP64, partial tiles and padded rows", host(y, T * ld), ref, 1e-4, 2e-5);
    std::vector<int8_t> q(N * K);
    std::vector<uint16_t> scales(N * K / 32);
    for (auto& weight : q) weight = (int)(rng() % 256) - 128;
    for (auto& d : scales) d = f2h_host(frand(0.02f));
    for (size_t i = 0; i < wh.size(); ++i) wh[i] = f2h_host(q[i] * h2f_host(scales[i / 32]));
    CU(cudaMemcpy(dw, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    k::rdna3_dense_gemm(dw, N, K, dx, T, y, ld, 0);
    const auto dequantized = host(y, T * ld);
    auto *dq = dev(q);
    auto *ds = dev(scales);
    k::rdna3_planar_gemm(dq, ds, N, K, dx, T, y, ld, 0);
    expect_true("planar Q8 WMMA equals CPU dequantization and dense WMMA", host(y, T * ld) == dequantized);
    release({dq, ds});
    release({dw, dx, y});
}

void test_rdna3_expert_groups() {
    constexpr int E = 64, K = 4;
    const auto gs = fvec(E), us = fvec(E), ds = fvec(E);
    auto *dg = dev(gs), *du = dev(us), *dd = dev(ds);
    for (int tile_rows : {16, 32, 64}) for (int T : {2, 17, 129}) {
        const int slots = T * K, capacity = k::expert_tile_capacity(slots, E, tile_rows);
        std::vector<int> top(slots);
        for (int t = 0; t < T; ++t) {
            // Expert zero spans many tiles. Most other experts see a mix of one or two rows.
            for (int j = 0; j < K; ++j)
                top[t * K + j] = j == 2 ? 0 : 1 + (t + 7 * j) % (E - 1);
        }
        const auto weights = fvec(slots);
        auto *dt = dev(top), *tokens = dev_n<int>(slots), *inv = dev_n<int>(slots);
        auto *dw = dev(weights), *gate = dev_n<float>(slots), *up = dev_n<float>(slots), *weight = dev_n<float>(slots);
        auto* tiles = dev_n<k::ExpertTile>(capacity);
        for (bool scales : {false, true}) {
            k::rdna3_expert_groups(dt, dw, slots, K, E, scales ? dg : nullptr, scales ? du : nullptr,
                                   scales ? dd : nullptr, {tiles, capacity, tile_rows}, tokens, inv, gate, up, weight, 0);
            std::vector<int> experts(slots, -1);
            bool correct = true;
            for (const auto tile : host(tiles, capacity)) {
                correct &= tile.expert >= 0 && tile.expert < E && tile.first >= 0 && tile.first <= tile.end &&
                           tile.end <= slots && tile.end - tile.first <= tile_rows;
                if (!correct) break;
                for (int r = tile.first; r < tile.end; ++r) {
                    correct &= experts[r] == -1;
                    experts[r] = tile.expert;
                }
            }
            const auto in = host(tokens, slots), out = host(inv, slots);
            const auto g = host(gate, slots), u = host(up, slots), w = host(weight, slots);
            std::vector<int> uses(slots);
            for (int t = 0; t < T && correct; ++t) {
                std::vector<int> picks(top.begin() + t * K, top.begin() + (t + 1) * K);
                std::sort(picks.begin(), picks.end());
                for (int j = 0; j < K; ++j) {
                    const int row = out[t * K + j], e = picks[j];
                    if (row < 0 || row >= slots) { correct = false; break; }
                    const auto it = std::find(top.begin() + t * K, top.begin() + (t + 1) * K, e);
                    correct &= ++uses[row] == 1 && in[row] == t && experts[row] == e &&
                               g[row] == (scales ? gs[e] : 1.f) && u[row] == (scales ? us[e] : 1.f) &&
                               w[row] == weights[it - top.begin()] * (scales ? ds[e] : 1.f);
                }
            }
            expect_true("RDNA3 expert tiles, row permutations, route scales and ordered reduction", correct);
        }
        release({dt, tokens, inv, dw, gate});
        release({up, weight, tiles});
    }
    release({dg, du, dd});
}

// Three nonempty experts separated by an empty one, including a partial token tile and a partial output workgroup.
void test_rdna3_expert_gemm(int type, int K, const char* name) {
    constexpr int E = 4, N = 48, T = 94;
    const std::vector<int> starts{0, 1, 1, 74, T};
    std::vector<uint16_t> x(T * K);
    for (auto& v : x) v = f2h_host(frand(0.75f));
    auto* dx = dev(x);
    auto* dy = dev_n<float>(T * N);
    auto raw = rand_blocks(type, (size_t)E * 2 * N * K);
    auto* weights = dev(raw);
    const size_t matrix_bytes = raw.size() / (2 * E);
    std::vector<const uint8_t*> blob(E);
    for (int e = 0; e < E; ++e) blob[e] = weights + 2 * e * matrix_bytes;
    auto** db = dev(blob);
    std::vector<double> products[2];
    for (int matrix = 0; matrix < 2; ++matrix) {
        const size_t offset = matrix * matrix_bytes;
        std::vector<double> expected(T * N);
        std::vector<float> w(N * K);
        for (int e = 0; e < E; ++e) {
            const uint8_t* block = raw.data() + 2 * e * matrix_bytes + offset;
            if (type == k::kNVFP4) {
                for (size_t i = 0; i < w.size(); ++i) w[i] = nvfp4_def(block, i);
            } else k::decode_host(type, block, N * K, w.data());
            for (int t = starts[e]; t < starts[e + 1]; ++t)
                for (int n = 0; n < N; ++n)
                    for (int i = 0; i < K; ++i)
                        expected[t * N + n] += (double)h2f_host(x[t * K + i]) * h2f_host(f2h_host(w[n * K + i]));
        }
        for (int tile_rows : {16, 32, 64}) {
            std::vector<k::ExpertTile> tiles;
            for (int e = 0; e < E; ++e)
                for (int row = starts[e]; row < starts[e+1]; row += tile_rows)
                    tiles.push_back({e, row, std::min(row + tile_rows, starts[e+1])});
            tiles.push_back({0, 0, 0});
            auto* dt = dev(tiles);
            k::rdna3_expert_gemm(type, db, offset, {dt, (int)tiles.size(), tile_rows}, dx, K, N, dy, 0);
            expect(name, host(dy, T * N), expected, 1e-4, 1e-3);
            release({dt});
        }
        products[matrix] = std::move(expected);
    }
    std::vector<float> sg(T), su(T);
    std::vector<double> expected(T * N);
    for (int t = 0; t < T; ++t) {
        sg[t] = ((t % 13) - 6) * 0.0125f;
        su[t] = ((t % 9) - 4) * 0.015625f;
        for (int n = 0; n < N; ++n) {
            const int i = t * N + n;
            const double g = products[0][i] * sg[t], u = products[1][i] * su[t];
            expected[i] = 12. * std::tanh(g / 12.) / (1. + std::exp(-g)) * 19. * std::tanh(u / 19.);
        }
    }
    std::vector<k::ExpertTile> ranges;
    for (int e = 0; e < E; ++e)
        for (int row = starts[e]; row < starts[e + 1]; row += 64)
            ranges.push_back({e, row, std::min(row + 64, starts[e + 1])});
    ranges.push_back({0, 0, 0});
    auto* dt = dev(ranges);
    const k::ExpertTiles tiles{dt, (int)ranges.size(), 64};
    auto* dsg = dev(sg);
    auto* dsu = dev(su);
    auto* tmp = dev_n<float>(T * 2 * N);
    auto* split = dev_n<uint16_t>(T * N);
    auto* fused = dev_n<uint16_t>(T * N);
    k::rdna3_expert_gemm(type, db, 0, tiles, dx, K, 2 * N, tmp, 0);
    k::swiglu_cap_f32(tmp, T, N, dsg, dsu, 12.f, 19.f, split, 0);
    auto unsorted = x;
    std::vector<int> token_map(T);
    for (int t = 0; t < T; ++t) {
        token_map[t] = T - 1 - t;
        std::copy_n(x.data() + t*K, K, unsorted.data() + token_map[t]*K);
    }
    auto* tokens = dev(token_map);
    auto* input = dev(unsorted);
    k::rdna3_expert_swiglu(type, db, tiles, input, K, N, dsg, dsu, 12.f, 19.f, fused, tokens, 0);
    release({tokens, input});
    auto result = host(fused, T * N);
    expect_true("WMMA fused expert SwiGLU matches separate projections and activation", result == host(split, T * N));
    std::vector<float> floats(result.size());
    std::transform(result.begin(), result.end(), floats.begin(), h2f_host);
    expect("WMMA fused expert SwiGLU vs FP64", floats, expected, 5e-4, 2e-4);
    release({dt, dsg, dsu, tmp, split, fused});
    release({db});
    release({weights});
    release({dy});
    release({dx});
}
#endif

// One decode token has one input shared by its selected experts. Include zero
// groups and signed activations whose low limb changes the dot product.
void test_expert_limbs() {
    constexpr int C = 256, F = 128, S = 3;
    for (int ty : {k::kNVFP4, k::kQ8_0, k::kIQ4_NL}) {
        const bool q8 = ty == k::kQ8_0;
        const int block = k::block_values(ty), bytes = k::block_bytes(ty);
        const size_t matrix_bytes = (size_t)C * F / block * bytes;
        auto pool = rand_blocks(ty, (size_t)S * 3 * C * F);
        for (size_t off = 0; off < pool.size(); off += bytes) {
            if (ty != k::kNVFP4) {
                const uint16_t scale = f2h_host(1.0f / 256);
                std::memcpy(pool.data() + off, &scale, sizeof(scale));
            } else {
                std::fill_n(pool.data() + off, 4, uint8_t(0x20));
            }
        }
        auto* device_pool = dev(pool);
        std::vector<const uint8_t*> ptrs;
        for (int j = 0; j < S; ++j)
            ptrs.push_back(device_pool + 3 * j * matrix_bytes);
        auto** blobs = dev(ptrs);
        auto x = fvec(C);
        std::fill_n(x.begin(), 16, 0.0f);
        auto* input = dev(x);
        auto* tokens = dev(std::vector<int>(S, 0));
        auto* output = dev_n<float>(S * C);
        auto* ref = dev_n<uint16_t>(S * F);
        auto* got = dev_n<uint16_t>(S * F);
        // Each group of sixteen activations has two signed-byte limbs and one FP32 scale.
        auto* scratch = dev_n<float>(9 * std::max(C, S * F) / 16);
        k::experts_gate_up_swiglu(blobs, tokens, S, ty, ty, matrix_bytes, input, C, F, nullptr, nullptr, 20, 20, ref,
                                  0);
        k::expert_limb_mv(true, blobs, matrix_bytes, C, F, S, input, got, output, nullptr, nullptr, 20, 20, scratch, 0,
                          ty);
        const auto h_ref = host(ref, S * F), h_got = host(got, S * F);
        std::vector<double> expected(h_ref.size());
        std::vector<float> actual(h_ref.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = h2f_host(h_ref[i]);
            actual[i] = h2f_host(h_got[i]);
        }
        expect(ty == k::kIQ4_NL ? "IQ4_NL limb SwiGLU" : q8 ? "Q8 limb SwiGLU" : "NVFP4 limb SwiGLU", actual, expected, 2e-3, 5e-3);
        k::experts_down(blobs, S, ty, 2 * matrix_bytes, ref, C, F, output, 0);
        const auto reference_down = host(output, S * C);
        k::expert_limb_mv(false, blobs, 2 * matrix_bytes, C, F, S, input, ref, output, nullptr, nullptr, 20, 20,
                          scratch, 0, ty);
        expect(ty == k::kIQ4_NL ? "IQ4_NL limb down" : q8 ? "Q8 limb down" : "NVFP4 limb down", host(output, S * C),
               std::vector<double>(reference_down.begin(), reference_down.end()), 1e-3, 1e-2);
        // Resident short-prefill callers have an FP16 mirror of each normalized input.
        // Give the reference the very same rounded activations and test indirect output rows.
        std::vector<uint16_t> half_inputs(S * C);
        std::vector<float> rounded(S * C);
        for (size_t j = 0; j < rounded.size(); ++j) {
            half_inputs[j] = f2h_host(frand());
            rounded[j] = h2f_host(half_inputs[j]);
        }
        auto* x16 = dev(half_inputs);
        auto* x32 = dev(rounded);
        auto* counts = dev(std::vector<int>{2, 1});
        std::vector<int> grouped_tokens(2 * k::kGroupMax), grouped_rows(2 * k::kGroupMax);
        grouped_tokens[0] = 0;
        grouped_tokens[1] = 2;
        grouped_tokens[k::kGroupMax] = 1;
        grouped_rows[0] = 2;
        grouped_rows[1] = 0;
        grouped_rows[k::kGroupMax] = 1;
        auto* gt = dev(grouped_tokens);
        auto* gr = dev(grouped_rows);
        auto* per_token = dev(std::vector<int>{2, 1, 0});
        auto** per_blob = dev(std::vector<const uint8_t*>{ptrs[0], ptrs[1], ptrs[0]});
        k::experts_gate_up_swiglu(per_blob, per_token, S, ty, ty, matrix_bytes, x32, C, F, nullptr, nullptr, 20, 20,
                                  ref, 0);
        k::experts_gate_up_swiglu_g(blobs, counts, gt, gr, 2, ty, ty, matrix_bytes, x32, C, F, nullptr, nullptr, 20, 20,
                                    got, 0, x16);
        const auto gh = host(got, S * F), rh = host(ref, S * F);
        for (size_t j = 0; j < gh.size(); ++j) {
            actual[j] = h2f_host(gh[j]);
            expected[j] = h2f_host(rh[j]);
        }
        expect(ty == k::kIQ4_NL ? "IQ4_NL grouped FP16 input" : q8 ? "Q8 grouped FP16 input" : "NVFP4 grouped FP16 input", actual, expected, 2e-3, 5e-3);
        release({x16});
        release({x32});
        release({counts});
        release({gt});
        release({gr});
        release({per_token});
        release({per_blob});
        release({scratch});
        release({got});
        release({ref});
        release({output});
        release({tokens});
        release({input});
        release({blobs});
        release({device_pool});
    }
}

// One token's NVFP4 experts through the tensor cores against the scalar kernels over the same blocks (every scale
// byte, the subnormal ones and 0x7F with them): gate | up with the input as two FP16 limbs, down over the FP16 h.
void test_experts_mma(int C, int F, int S) {
    const int ty = k::kNVFP4;
    const size_t matrix_bytes = (size_t) C * F / k::block_values(ty) * k::block_bytes(ty);
    auto pool = rand_blocks(ty, (size_t) S * 3 * C * F);
    for (size_t off = 0, i = 0; off < pool.size(); off += 36, ++i) {
        if (C == 2304) {
            // Exercise repeated K splits at model width with scales spanning 1/8 through 1.
            for (size_t j = 0; j < 4; ++j) pool[off + j] = uint8_t(32 + (pool[off + j] % 25));
        } else if (i % 7 == 0) {
            pool[off + i % 4] = uint8_t(i % 13 == 0 ? 0x7F : i % 8);
        }
    }
    auto* device_pool = dev(pool);
    std::vector<const uint8_t*> ptrs;
    for (int j = 0; j < S; ++j) ptrs.push_back(device_pool + 3 * j * matrix_bytes);
    auto** blobs = dev(ptrs);
    const auto x = fvec(C, 3.0f);
    auto* input = dev(x);
    auto* tokens = dev(std::vector<int>(S, 0));
    auto sgv = fvec(S), suv = fvec(S);
    auto *sg = dev(sgv), *su = dev(suv);
    auto* limbs = dev_n<uint16_t>(2 * C);
    auto *ref = dev_n<uint16_t>(S * F), *got = dev_n<uint16_t>(S * F);
    auto *y_ref = dev_n<float>(S * C), *y_got = dev_n<float>(S * C);
    k::experts_gate_up_swiglu(blobs, tokens, S, ty, ty, matrix_bytes, input, C, F, sg, su, 20, 20, ref, 0);
    k::to_f16_limbs(input, C, limbs, 0);
    k::experts_gate_up_swiglu_mma(blobs, S, matrix_bytes, limbs, C, F, sg, su, 20, 20, got, 0);
    const auto h_ref = host(ref, S * F), h_got = host(got, S * F);
    std::vector<double> expected(h_ref.size());
    std::vector<float> actual(h_ref.size());
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = h2f_host(h_ref[i]), actual[i] = h2f_host(h_got[i]);
    expect("experts_gate_up_swiglu_mma", actual, expected, 1e-3, 1e-3);
    k::experts_down(blobs, S, ty, 2 * matrix_bytes, ref, C, F, y_ref, 0);
    k::experts_down_mma(blobs, S, 2 * matrix_bytes, ref, C, F, y_got, 0);
    const auto want = host(y_ref, S * C);
    expect("experts_down_mma", host(y_got, S * C), std::vector<double>(want.begin(), want.end()), 2e-5, 0.25);   // (random scales reach 240: sums of terms of 1e4 and more that cancel)
    std::vector<double> cpu_down(S * C);
    for (int slot = 0; slot < S; ++slot) {
        const uint8_t* gate = pool.data() + 3 * slot * matrix_bytes;
        const uint8_t* up = gate + matrix_bytes;
        const uint8_t* down = up + matrix_bytes;
        for (int r = 0; r < F; ++r) {
            double g = 0, u = 0;
            for (int k = 0; k < C; ++k) {
                g += nvfp4_def(gate, r * C + k) * x[k];
                u += nvfp4_def(up, r * C + k) * x[k];
            }
            g *= sgv[slot];
            u *= suv[slot];
            expected[slot * F + r] = 20. * std::tanh(g / 20.) / (1. + std::exp(-g)) * 20. * std::tanh(u / 20.);
        }
        for (int r = 0; r < C; ++r)
            for (int k = 0; k < F; ++k)
                cpu_down[slot * C + r] += nvfp4_def(down, r * F + k) * h2f_host(h_ref[slot * F + k]);
    }
    expect("NVFP4 token WMMA SwiGLU vs FP64", actual, expected, 1e-3, 1e-3);
    expect("NVFP4 token WMMA down vs FP64", host(y_got, S * C), cpu_down, 2e-5, 0.25);
    expect_true("experts_mma_fits", k::experts_mma_fits(ty, ty, ty, 2304, 1152) && !k::experts_mma_fits(ty, k::kQ8_0, ty, 2304, 1152));
    CU(cudaFree(y_got)); CU(cudaFree(y_ref)); CU(cudaFree(got)); CU(cudaFree(ref)); CU(cudaFree(limbs));
    CU(cudaFree(sg)); CU(cudaFree(su)); CU(cudaFree(tokens)); CU(cudaFree(input));
    CU(cudaFree(blobs)); CU(cudaFree(device_pool));
}

void test_q8_verify_tile() {
    constexpr int N = 35, K = 256, ld = 39;
    std::vector<int8_t> weights(N * K);
    for (auto& q : weights)
        q = (int8_t)(rng() & 255);
    std::vector<uint16_t> scales(N * K / 32);
    for (auto& d : scales)
        d = f2h_host(0.015625f * std::fabs(frand()));
    auto* dw = dev(weights);
    auto* ds = dev(scales);
    for (int T : {2, 8, 13}) {
        std::vector<uint16_t> x(T * K);
        for (auto& v : x)
            v = f2h_host(frand());
        auto* dx = dev(x);
        auto* dy = dev(std::vector<float>(T * ld));
        k::q8_tile(dw, ds, N, K, dx, T, dy, ld, 0);
        bool half_weights = true;
#ifdef STRATA_USE_HIP
        const char* mv = std::getenv("IK2_Q8_TILE_MV");
        half_weights = T > 8 || (mv && !std::atoi(mv));
#endif
        std::vector<double> wanted(T * ld);
        double max_weight_error = 0;
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < N; ++r) {
                double rounding_error = 0;
                for (int k = 0; k < K; ++k) {
                    const int i = r * K + k;
                    const double w = weights[i] * (double)h2f_host(scales[i / 32]);
                    const double v = h2f_host(x[t * K + k]);
                    wanted[t * ld + r] += w * v;
                    if (half_weights) rounding_error += (h2f_host(f2h_host(w)) - w) * v;
                }
                // Maximum change to any exact dot from rounding just its weights.
                max_weight_error = std::max(max_weight_error, std::fabs(rounding_error));
            }
        }
        expect("short Q8 tensor-core tile", host(dy, T * ld), wanted, half_weights ? 6e-4 : 2e-6,
               max_weight_error + (half_weights ? 5e-4 : 1e-5));
        release({dx});
        release({dy});
    }
    release({dw});
    release({ds});
}

#ifndef STRATA_USE_HIP
void test_linear_block_verify(int type, int T) {
    constexpr int C = 96, F = 64;
    auto raw = rand_blocks(type, 3 * C * F);
    const size_t matrix = raw.size() / 3;
    std::vector<float> weights(3 * C * F);
    k::decode_host(type, raw.data(), weights.size(), weights.data());
    auto x = fvec(T * C, 0.1f);
    // Every token picks expert zero. Wide batches also pick seven singleton experts apiece.
    const int K = T > k::kGroupMax ? 8 : 1;
    const int S = T * K, G = std::max(2, S);
    const int chunks = (T + k::kGroupMax - 1) / k::kGroupMax;
    std::vector<int> tokens(G * k::kGroupMax, -1), rows(tokens.size(), -1), count(G);
    auto gate_scale = fvec(S), up_scale = fvec(S);
    std::vector<double> hidden(S * F), down(S * C);
    for (int slot = 0; slot < S; ++slot) {
        const int t = slot / K, pick = slot % K;
        const int group = pick ? chunks + t * (K - 1) + pick - 1 : t / k::kGroupMax;
        const int entry = group * k::kGroupMax + count[group]++;
        tokens[entry] = T - 1 - t;
        rows[entry] = (slot + 1) % S;
        for (int f = 0; f < F; ++f) {
            double g = 0, u = 0;
            for (int c = 0; c < C; ++c) {
                const double v = x[tokens[entry] * C + c];
                g += v * weights[f * C + c];
                u += v * weights[(F + f) * C + c];
            }
            g *= gate_scale[rows[entry]];
            u *= up_scale[rows[entry]];
            hidden[rows[entry] * F + f] = 4 * std::tanh(g / 4) / (1 + std::exp(-g)) * 25 * std::tanh(u / 25);
        }
    }
    auto* w = dev(raw);
    std::vector<const uint8_t*> ptrs(G);
    for (int g = 0; g < G; ++g)
        if (count[g]) ptrs[g] = w;
    auto** blobs = dev(ptrs);
    auto* counts = dev(count);
    auto* dt = dev(tokens);
    auto* dr = dev(rows);
    auto* dx = dev(x);
    auto* limbs = dev_n<uint16_t>(T * 2 * C);
    auto* dh = dev_n<uint16_t>(S * F);
    auto* dy = dev_n<float>(S * C);
    auto* sg = dev(gate_scale);
    auto* su = dev(up_scale);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamCreate(&stream));
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    k::f16_limbs_rows(dx, T, C, limbs, stream);
    k::experts_block_gu_linear(type, blobs, counts, dt, dr, G, T, matrix, C, F, limbs, sg, su, 4, 25, dh, stream);
    k::experts_block_down_linear(type, blobs, counts, dr, G, 2 * matrix, C, F, dh, dy, stream);
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    CU(cudaGraphLaunch(exec, stream));
    CU(cudaStreamSynchronize(stream));
    CU(cudaGraphExecDestroy(exec));
    CU(cudaGraphDestroy(graph));
    CU(cudaStreamDestroy(stream));
    auto h = host(dh, S * F);
    std::vector<float> actual(h.size());
    std::transform(h.begin(), h.end(), actual.begin(), h2f_host);
    expect("block verify gate/up vs FP64", actual, hidden, 1e-3, 2e-3);
    for (int t = 0; t < S; ++t)
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < F; ++f)
                down[t * C + c] += double(actual[t * F + f]) * weights[2 * C * F + c * F + f];
    expect("block verify down vs FP64", host(dy, down.size()), down, 1e-4, 0.01);
    release({sg, su});
    release({w, blobs, counts, dt, dr});
    release({dx, limbs, dh, dy});
}
#endif

void test_draft_attn_online() {
    constexpr int H = 4, Hkv = 2, D = 128, nq = 8, cap = 64, P = 90, window = 50;
    const int k_lo = std::max(0, P - window), nc = P - k_lo, nk = nc + nq;
    const float scale = 1.0f / std::sqrt((float) D);
    auto q = fvec((size_t) nq * H * D, 1.5f);
    auto kc = fvec((size_t) cap * Hkv * D, 1.0f), vc = fvec((size_t) cap * Hkv * D, 1.0f);
    std::vector<uint16_t> kc16(kc.size()), vc16(vc.size());
    for (size_t i = 0; i < kc.size(); ++i) {
        kc[i] = h2f_host(kc16[i] = f2h_host(kc[i]));
        vc[i] = h2f_host(vc16[i] = f2h_host(vc[i]));
    }
    auto kb = fvec((size_t) nq * Hkv * D, 1.0f), vb = fvec((size_t) nq * Hkv * D, 1.0f);
    float *dq = dev(q), *dkc = dev(kc), *dvc = dev(vc), *dkb = dev(kb), *dvb = dev(vb);
    int* dP = dev(std::vector<int>{P});
    float *scratch = dev_n<float>((size_t) 64 * H * 8 * 130), *out = dev_n<float>((size_t) nq * H * D);
    std::vector<double> want((size_t) nq * H * D, 0.0);
    auto row_window = want;
    for (int h = 0; h < H; ++h) {
        const int g = h / (H / Hkv);
        for (int qi = 0; qi < nq; ++qi) {
            std::vector<double> sc(nk);
            double mx = -INFINITY;
            for (int j = 0; j < nk; ++j) {
                const float* kr = j < nc ? kc.data() + ((size_t) ((k_lo + j) % cap) * Hkv + g) * D
                                         : kb.data() + ((size_t) (j - nc) * Hkv + g) * D;
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += (double) q[((size_t) qi * H + h) * D + d] * kr[d];
                sc[j] = dot * scale;
                mx = std::max(mx, sc[j]);
            }
            double den = 0.0;
            for (double& s : sc) { s = std::exp(s - mx); den += s; }
            double row_den = 0;
            for (int j = qi; j < nk; ++j) row_den += sc[j];
            for (int d = 0; d < D; ++d) {
                double row_acc = 0;
                double acc = 0.0;
                for (int j = 0; j < nk; ++j) {
                    const float* vr = j < nc ? vc.data() + ((size_t) ((k_lo + j) % cap) * Hkv + g) * D
                                             : vb.data() + ((size_t) (j - nc) * Hkv + g) * D;
                    acc += sc[j] * vr[d];
                    if (j >= qi) row_acc += sc[j] * vr[d];
                }
                want[((size_t) qi * H + h) * D + d] = acc / den;
                row_window[((size_t) qi * H + h) * D + d] = row_acc / row_den;
            }
        }
    }
    k::draft_attn(dq, dkc, dvc, k_lo, P, dkb, dvb, nq, H, Hkv, scale, scratch, out, 0, cap);
    expect("draft_attn online (ring wrap)", host(out, want.size()), want, 1e-4, 2e-5);
    k::draft_attn_graph(dq, dkc, dvc, dP, window, dkb, dvb, nq, H, Hkv, scale, 4, scratch, out, 0, cap);
    expect("draft_attn_graph online (d_P, S=4)", host(out, want.size()), want, 1e-4, 2e-5);
    k::dflash2_attn(dq, dkc, dvc, dP, window, dkb, dvb, nq, H, Hkv, scale, 4, scratch, out, 0, cap);
    expect("DFlash2 per-row window after wrap", host(out, want.size()), row_window, 1e-4, 2e-5);
    auto* dk16 = dev(kc16);
    auto* dv16 = dev(vc16);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamCreate(&stream));
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    k::draft_attn_f16_graph(dq, dk16, dv16, dP, window, dkb, dvb, nq, H, Hkv, scale, 4, scratch, out, stream, cap);
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    for (int pos : {0, P}) {
        CU(cudaMemcpy(dP, &pos, sizeof(pos), cudaMemcpyHostToDevice));
        CU(cudaGraphLaunch(exec, stream));
        CU(cudaStreamSynchronize(stream));
        const auto result = host(out, want.size());
        if (pos == P) expect("FP16 draft graph replay after ring wrap vs FP64", result, want, 1e-4, 2e-5);
        else {
            k::draft_attn(dq, dkc, dvc, 0, 0, dkb, dvb, nq, H, Hkv, scale, scratch, out, 0, cap);
            const auto eager = host(out, want.size());
            expect("FP16 draft graph at empty context", result, std::vector<double>(eager.begin(), eager.end()), 1e-4, 2e-5);
        }
    }
    CU(cudaGraphExecDestroy(exec));
    CU(cudaGraphDestroy(graph));
    CU(cudaStreamDestroy(stream));
    release({dk16, dv16});
    // Also verify vectorized multi-token mv_f16_rows (T=5, K=256)
    constexpr int N = 35, K = 256, T = 5;
    std::vector<uint16_t> W(N * K);
    for (auto& v : W) v = f2h_host(frand(0.5f));
    auto x = fvec(T * K, 1.5f);
    uint16_t* dW = dev(W);
    float *dx = dev(x), *dy = dev_n<float>(T * N);
    k::mv_f16_rows(dW, N, K, dx, K, T, dy, N, 0);
    std::vector<double> wy(T * N, 0.0);
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < N; ++r)
            for (int i = 0; i < K; ++i) wy[t * N + r] += (double) h2f_host(W[r * K + i]) * x[t * K + i];
    expect("mv_f16_rows vectorized (T=5)", host(dy, wy.size()), wy, 1e-4, 1e-4);
    for (void* p : {(void*) dq, (void*) dkc, (void*) dvc, (void*) dkb, (void*) dvb, (void*) dP, (void*) scratch,
                    (void*) out, (void*) dW, (void*) dx, (void*) dy})
        release({p});
}

#ifdef STRATA_USE_HIP
void test_rdna3_speculative_kernels() {
    // 1. Grouped block-verify experts across NVFP4, Q8_0, and IQ4_NL at T=5 (group sizes 1, 2, 3, 5)
    constexpr int C = 256, F = 128, E = 4, T = 5, K = 4, S = T * K;
    const int types[3] = {k::kNVFP4, k::kQ8_0, k::kIQ4_NL};
    const char* tnames[3] = {"NVFP4", "Q8_0", "IQ4_NL"};
    for (int ti = 0; ti < 3; ++ti) {
        const int ty = types[ti];
        auto g = rand_blocks(ty, (size_t) E * F * C), u = rand_blocks(ty, (size_t) E * F * C),
             d = rand_blocks(ty, (size_t) E * C * F);
        const size_t eb0 = g.size() / E, eb1 = u.size() / E, eb2 = d.size() / E, slot = (eb0 + eb1 + eb2 + 255) / 256 * 256;
        std::vector<uint8_t> pool(slot * E);
        for (int e = 0; e < E; ++e) {
            std::memcpy(pool.data() + e * slot, g.data() + e * eb0, eb0);
            std::memcpy(pool.data() + e * slot + eb0, u.data() + e * eb1, eb1);
            std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data() + e * eb2, eb2);
        }
        uint8_t* dpool = dev(pool);
        const int want_ng[E] = {1, 2, 3, 5};
        std::vector<const uint8_t*> gb(E);
        std::vector<int> gn(E, 0), gtok(E * k::kGroupMax, 0), grow(E * k::kGroupMax, 0);
        std::vector<const uint8_t*> sblob;
        std::vector<int> stok;
        for (int e = 0; e < E; ++e) {
            gb[e] = dpool + e * slot;
            for (int t = 0; t < want_ng[e]; ++t) {
                const int r = (int) sblob.size();
                sblob.push_back(dpool + e * slot);
                stok.push_back(t);
                gtok[e * k::kGroupMax + gn[e]] = t;
                grow[e * k::kGroupMax + gn[e]] = r;
                ++gn[e];
            }
        }
        const int n = (int) sblob.size();
        const uint8_t **dblob = dev(sblob), **dgb = dev(gb);
        int *dtok = dev(stok), *dgn = dev(gn), *dgt = dev(gtok), *dgr = dev(grow);
        auto x = fvec((size_t) T * C, 1.5f);
        std::vector<uint16_t> x16(x.size());
        for (size_t i = 0; i < x.size(); ++i) x16[i] = f2h_host(x[i]), x[i] = h2f_host(x16[i]);
        auto sgv = fvec(n), suv = fvec(n);
        float *dx = dev(x), *dsg = dev(sgv), *dsu = dev(suv);
        uint16_t *dx16 = dev(x16), *h_ref = dev_n<uint16_t>((size_t) n * F), *h_got = dev_n<uint16_t>((size_t) n * F);
        float *y_ref = dev_n<float>((size_t) n * C), *y_got = dev_n<float>((size_t) n * C);
        k::experts_gate_up_swiglu(dblob, dtok, n, ty, ty, eb0, dx, C, F, dsg, dsu, 12.f, 19.f, h_ref, 0);
        k::rdna3_experts_block_gu(ty, dgb, dgn, dgt, dgr, E, T, eb0, C, F, dx16, dsg, dsu, 12.f, 19.f, h_got, 0);
        const auto hr = host(h_ref, (size_t) n * F), hg = host(h_got, (size_t) n * F);
        std::vector<double> wh(hr.size());
        std::vector<float> gh(hg.size());
        for (size_t i = 0; i < hr.size(); ++i) wh[i] = h2f_host(hr[i]), gh[i] = h2f_host(hg[i]);
        char name[80];
        std::snprintf(name, sizeof name, "rdna3_experts_block_gu %s (T=5)", tnames[ti]);
        expect(name, gh, wh, 5e-3, 5e-3);
        k::experts_down(dblob, n, ty, eb0 + eb1, h_ref, C, F, y_ref, 0);
        k::rdna3_experts_block_down(ty, dgb, dgn, dgr, E, T, eb0 + eb1, C, F, h_ref, y_got, 0);
        const auto yr = host(y_ref, (size_t) n * C);
        std::snprintf(name, sizeof name, "rdna3_experts_block_down %s (T=5)", tnames[ti]);
        expect(name, host(y_got, yr.size()), std::vector<double>(yr.begin(), yr.end()), 5e-3, 5e-2);
        for (void* p : {(void*) dpool, (void*) dblob, (void*) dgb, (void*) dtok, (void*) dgn, (void*) dgt, (void*) dgr,
                        (void*) dx, (void*) dsg, (void*) dsu, (void*) dx16, (void*) h_ref, (void*) h_got,
                        (void*) y_ref, (void*) y_got})
            release({p});
    }
    // 2. Small-T BF16 GEMM (T=5)
    {
        constexpr int N = 35, K = 256, T = 5, ld = 40;
        std::vector<uint16_t> W(N * K), X(T * K);
        auto to_bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); };
        auto from_bf = [](uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; };
        for (auto& v : W) v = to_bf(frand(0.5f));
        for (auto& v : X) v = to_bf(frand(1.0f));
        uint16_t *dW = dev(W), *dX = dev(X);
        float* dY = dev_n<float>(T * ld);
        k::rdna3_bf16_gemm(dW, N, K, dX, T, dY, ld, 0);
        std::vector<double> want(T * ld, 0.0);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < N; ++r)
                for (int i = 0; i < K; ++i) want[t * ld + r] += (double) from_bf(W[r * K + i]) * from_bf(X[t * K + i]);
        expect("rdna3_bf16_gemm (T=5)", host(dY, want.size()), want, 1e-4, 1e-4);
        release({dW, dX, dY});
    }
    // 3. Fused separate-input sliding QKV/R/G planar Q8 (T=5)
    {
        constexpr int HD = 32, KD = 16, N = HD + 2 * KD + 20, K = 256, T = 5;
        std::vector<int8_t> q(N * K);
        for (auto& v : q) v = (int8_t) (rng() & 255);
        std::vector<uint16_t> d(N * (K / 32)), xq(T * K), xk(T * K), xv(T * K);
        for (auto& v : d) v = f2h_host(0.015625f * std::fabs(frand()));
        for (auto& v : xq) v = f2h_host(frand());
        for (auto& v : xk) v = f2h_host(frand());
        for (auto& v : xv) v = f2h_host(frand());
        int8_t* dq = dev(q);
        uint16_t *dd = dev(d), *dxq = dev(xq), *dxk = dev(xk), *dxv = dev(xv);
        float* dy = dev_n<float>(T * N);
        k::rdna3_q8_tile_qkv_sep(dq, dd, N, K, HD, KD, dxq, dxk, dxv, T, dy, N, 0);
        std::vector<double> want(T * N, 0.0);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < N; ++r) {
                const auto& xin = (r < HD || r >= HD + 2 * KD) ? xq : (r < HD + KD ? xk : xv);
                for (int i = 0; i < K; ++i)
                    want[t * N + r] += (double) ((float) q[r * K + i] * h2f_host(d[(r * K + i) / 32])) * h2f_host(xin[t * K + i]);
            }
        expect("rdna3_q8_tile_qkv_sep (T=5)", host(dy, want.size()), want, 2e-4, 2e-4);
        release({dq, dd, dxq, dxk, dxv, dy});
    }
    (void) S;
}
#endif

#ifndef STRATA_USE_HIP
void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    const int rc = _putenv_s(name, value);
#else
    const int rc = setenv(name, value, 1);
#endif
    if (rc) {
        std::fprintf(stderr, "cannot set %s\n", name);
        std::exit(1);
    }
}

void check_q8_down(int C, int F, size_t matrix, const std::vector<int>& picks,
                   const std::vector<uint8_t>& weights, const std::vector<float>& hidden,
                   const std::vector<double>& reference, const std::vector<float>& direct,
                   const std::vector<float>& ref_y, const std::vector<float>& actual_y) {
    std::vector<float> decoded(C * F), direct_error(direct.size()), pipeline_error(direct.size());
    for (size_t slot = 0; slot < picks.size(); ++slot) {
        const auto* down = weights.data() + (3 * picks[slot] + 2) * matrix;
        k::decode_host(k::kQ8_0, down, decoded.size(), decoded.data());
        for (int c = 0; c < C; ++c) {
            double sum = 0, propagation = 0, magnitude = 0;
            for (int i = 0; i < F; ++i) {
                const double weight = decoded[c * F + i];
                const size_t hi = slot * F + i;
                sum += weight * reference[hi];
                propagation += weight * (hidden[hi] - reference[hi]);
                magnitude += std::fabs(weight) * std::max(std::fabs(double(hidden[hi])), std::fabs(reference[hi]));
            }
            // Cancellation can make y tiny. Normalize by sum |w*h| for the
            // FP32 dot error, and account for the actual FP16 SwiGLU steps.
            const size_t j = slot * C + c;
            const double norm = std::max(magnitude, 1e-30);
            direct_error[j] = std::max(std::fabs(direct[j] - sum), std::fabs(ref_y[j] - sum)) / norm;
            pipeline_error[j] = ((double(actual_y[j]) - ref_y[j]) - propagation) / norm;
        }
    }
    const std::vector<double> zero(direct.size(), 0);
    // F binary32 rounding steps per dot, with two dots in the pipeline comparison.
    expect("wide Q8_0 grouped down (same hidden)", direct_error, zero, 0, F * 0x1p-23);
    expect("wide Q8_0 grouped down (end to end)", pipeline_error, zero, 0, 2 * F * 0x1p-23);
}

// Replay the same graph with fully distinct picks, then with hot experts. This also
// checks that a smaller group count clears the unused groups from the previous replay.
void test_batch_groups(int T, int ty) {
    constexpr int K = 8, C = 320, F = 64;
    const int S = T * K, E = S + 3;
    const size_t matrix = C * F / k::block_values(ty) * k::block_bytes(ty);
    // This seed crosses an FP16 SwiGLU rounding boundary in the T=16 hot Q8 group.
    std::mt19937 gen(20261168);
    auto samples = [&](size_t count) {
        std::vector<float> values(count);
        for (auto& v : values) v = float(int(gen() >> 8) - (1 << 23)) * 0x1p-23f;
        return values;
    };
    std::vector<uint8_t> weights(E * 3 * matrix);
    for (auto& byte : weights) byte = uint8_t(gen());
    for (size_t off = 0; off < weights.size(); off += k::block_bytes(ty)) {
        if (ty == k::kNVFP4) std::fill_n(weights.data() + off, 4, uint8_t(0x18));
        else {
            const uint16_t scale = ty == k::kQ8_0 ? f2h_host((128 + (gen() & 255)) * 0x1p-16f) : f2h_host(1.f / 256);
            std::memcpy(weights.data() + off, &scale, 2);
        }
    }
    auto* pool = dev(weights);
    std::vector<const uint8_t*> table(E);
    for (int e = 0; e < E; ++e) table[e] = pool + e * 3 * matrix;
    auto** tab = dev(table);
    auto scales = samples(E), weights_in = samples(S);
    auto* scale = dev(scales);
    auto* w = dev(weights_in);
    auto* top = dev_n<int>(S);
    auto** blobs = dev_n<const uint8_t*>(S);
    auto* n = dev_n<int>(S);
    auto* tok = dev_n<int>(S * k::kGroupMax);
    auto* row = dev_n<int>(S * k::kGroupMax);
    auto* sg = dev_n<float>(S);
    auto* su = dev_n<float>(S);
    auto* wt = dev_n<float>(S);
    auto* inv = dev_n<int>(S);
    auto input = samples(T * C);
    // MMA consumes x16. Compare projections of the same numbers in both paths.
    const char* mma = std::getenv("IK2_GROUP_MMA");
    if (mma && (std::atoi(mma) & 1))
        for (auto& v : input) v = h2f_host(f2h_host(v));
    auto* x = dev(input);
    auto* x16 = dev_n<uint16_t>(T * C);
    k::to_f16(x, T * C, x16, 0);
    auto* h = dev_n<uint16_t>(S * F);
    auto* y = dev_n<float>(S * C);
    auto* href = dev_n<uint16_t>(S * F);
    auto* yref = dev_n<float>(S * C);
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamCreate(&stream));
    CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    k::group_route(top, w, S, K, tab, scale, nullptr, scale, blobs, n, tok, row, sg, su, wt, inv, stream);
    k::experts_gate_up_swiglu_g(blobs, n, tok, row, S, ty, ty, matrix, x, C, F, sg, su, 20, 20, h, stream, x16);
    k::experts_down_g(blobs, n, row, S, ty, 2 * matrix, h, C, F, y, stream);
    CU(cudaStreamEndCapture(stream, &graph));
    CU(cudaGraphInstantiate(&exec, graph, 0));
    for (bool skew : {false, true}) {
        std::vector<int> picks(S), tokens(S);
        std::vector<const uint8_t*> per_slot(S);
        std::vector<std::vector<int>> members(E);
        for (int i = 0; i < S; ++i) {
            const int t = i / K, rank = i % K;
            // Expert E-1 receives 19 picks at T=32. Experts 1..7 receive all T.
            const int e = skew ? (rank ? rank : t < 19 ? E - 1 : t + K) : i + 2;
            picks[i] = e;
            members[e].push_back(i);
            per_slot[i] = table[e];
            tokens[i] = t;
        }
        CU(cudaMemcpy(top, picks.data(), S * sizeof(int), cudaMemcpyHostToDevice));
        CU(cudaGraphLaunch(exec, stream));
        auto counts = host(n, S), rows = host(row, S * k::kGroupMax), toks = host(tok, S * k::kGroupMax);
        auto ptrs = host(blobs, S);
        auto gates = host(sg, S), ups = host(su, S), scaled = host(wt, S);
        auto inverse = host(inv, S);
        bool ok = true;
        int groups = 0;
        for (int i = 0; i < S; ++i) {
            int e = picks[i];
            ok &= gates[i] == scales[e] && ups[i] == 1 && scaled[i] == weights_in[i] * scales[e] && inverse[i] == i;
            if (members[e].front() != i) continue;
            for (int j = 0; j < int(members[e].size()); ++j) {
                if (j % k::kGroupMax == 0) {
                    ok &= ptrs[groups] == table[e] && counts[groups] == std::min(k::kGroupMax, int(members[e].size()) - j);
                    ++groups;
                }
                int idx = (groups - 1) * k::kGroupMax + j % k::kGroupMax;
                ok &= rows[idx] == members[e][j] && toks[idx] == members[e][j] / K;
            }
        }
        expect_true("wide route chunks, scales and slot order", ok && std::all_of(counts.begin() + groups, counts.end(),
                                                                                  [](int v) { return v == 0; }));
        auto** refs = dev(per_slot);
        auto* rt = dev(tokens);
        k::experts_gate_up_swiglu(refs, rt, S, ty, ty, matrix, x, C, F, sg, su, 20, 20, href, 0);
        k::experts_down(refs, S, ty, 2 * matrix, href, C, F, yref, 0);
        auto actual_h = host(h, S * F), ref_h = host(href, S * F);
        auto actual_y = host(y, S * C), ref_y = host(yref, S * C);
        if (ty == k::kIQ4_NL) {
            expect_true("wide IQ4_NL grouped SwiGLU bits", actual_h == ref_h);
            expect_true("wide IQ4_NL grouped down bits", !std::memcmp(actual_y.data(), ref_y.data(), ref_y.size() * sizeof(float)));
        } else {
            std::vector<float> hidden(actual_h.size());
            std::vector<double> reference(ref_h.size());
            std::transform(actual_h.begin(), actual_h.end(), hidden.begin(), h2f_host);
            std::transform(ref_h.begin(), ref_h.end(), reference.begin(), h2f_host);
            if (ty == k::kQ8_0) {
                // Allow one normal FP16 step or a small FP32 dot error near cancellation.
                expect("wide Q8_0 grouped SwiGLU", hidden, reference, 0x1p-10, 3e-5);
                k::experts_down_g(blobs, n, row, S, ty, 2 * matrix, href, C, F, y, 0);
                const auto direct = host(y, S * C);
                check_q8_down(C, F, matrix, picks, weights, hidden, reference, direct, ref_y, actual_y);
            } else {
                expect("wide NVFP4 grouped SwiGLU", hidden, reference, 2e-3, 5e-3);
                expect("wide NVFP4 grouped down", actual_y, std::vector<double>(ref_y.begin(), ref_y.end()), 2e-3, 5e-3);
            }
        }
        release({refs, rt});
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    release({pool, tab, scale, w, top, blobs, n});
    release({tok, row, sg, su, wt, inv, x, x16});
    release({h, y, href, yref});
}

// An expert spanning five tiles, two empty experts, and a different pick order on alternate tokens.
void test_blackwell_prefill_routes() {
    constexpr int T = 129, K = 2, S = T*K, E = 5, C = 64;
    std::vector<int> picks(S);
    std::vector<float> weights(S), scales{1, 2, 3, 4, 5};
    std::vector<uint16_t> input(T*C);
    for (int t = 0; t < T; ++t) {
        picks[2*t + (t&1)] = 3;
        picks[2*t + 1 - (t&1)] = t % 3 ? 0 : 2;
    }
    for (int i = 0; i < S; ++i) weights[i] = (1 + i % 4) * .125f;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < C; ++j)
            input[t*C+j] = f2h_host(j % 16 == t % 16 ? std::ldexp(6.f, t % 5 - 3) : 0.f);
    const size_t matrix = C * 36;
    std::vector<uint8_t> matrices(E*matrix, 0);
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < C; ++r) {
            auto* block = matrices.data() + e*matrix + r*36;
            std::fill_n(block, 4, uint8_t(0x38));
            block[4 + r/16*8 + r%8] = uint8_t((e+1) << ((r & 8) ? 4 : 0));
        }
    auto *pool = dev(matrices);
    std::vector<const uint8_t*> blobs(E);
    for (int e = 0; e < E; ++e) blobs[e] = pool + e*matrix;
    auto **table = dev(blobs);
    auto *p = dev(picks), *inv = dev_n<int>(S);
    auto *w = dev(weights), *sc = dev(scales), *sg = dev_n<float>(S), *su = dev_n<float>(S), *wt = dev_n<float>(S);
    auto* plan = dev_n<uint8_t>(k::blackwell_group_bytes(k::blackwell_group_capacity(S, E)));
    auto batch = k::blackwell_prefill_groups(p, w, S, K, E, table, sc, nullptr, sc, plan, sg, su, wt, inv, 0);
    std::vector<int> tile_sizes;
    for (int e = 0; e < E; ++e) {
        const int count = std::count(picks.begin(), picks.end(), e);
        for (int first = 0; first < count; first += 32)
            tile_sizes.push_back(std::min(32, count - first));
    }
    tile_sizes.resize(batch.size, 0);
    expect_true("resident prefill groups contain 32 picks before the tail", host(batch.counts, batch.size) == tile_sizes);
    auto* x = dev(input);
    auto* packed = dev_n<k::BlackwellActivation>(S);
    auto* y = dev_n<float>(S*C);
    auto* hidden = dev_n<uint16_t>(S*C);
    auto* packed_h = dev_n<k::BlackwellActivation>(S);
    k::blackwell_pack_rows(x, T, C, packed, 0);
    // Reuse each expert's diagonal matrix for all three projections.
    k::blackwell_expert_product(true, batch, 0, C, C, x, packed, sg, su, 20, 16, hidden, nullptr, 0, packed_h);
    k::blackwell_expert_product(false, batch, 0, C, C, hidden, packed_h, nullptr, nullptr, 0, 0, nullptr, y, 0);
    std::vector<double> expected(S*C);
    const auto order = host(inv, S);
    const auto gate = host(sg, S), up = host(su, S), down = host(wt, S);
    bool maps = true;
    for (int i = 0; i < S; ++i) {
        const int e = picks[i], rank = picks[i^1] < e;
        maps &= order[(i/2)*2 + rank] == i && gate[i] == scales[e] && up[i] == 1 && down[i] == weights[i]*scales[e];
        for (int r = 0; r < C; ++r) {
            const double diagonal = nvfp4_def(matrices.data()+e*matrix, r*C+r);
            const double u = diagonal * h2f_host(input[(i/K)*C+r]);
            if (u == 0) continue;
            const double g = scales[e] * u;
            const double h = h2f_host(f2h_host(20 * std::tanh(g/20) / (1 + std::exp(-g)) * 16 * std::tanh(u/16)));
            // The single nonzero per block-16 always rounds to E2M1 six.
            const double outer = std::ldexp(0.25, std::ilogb(h));
            const double target = h / (6 * outer);
            int best = 1;
            for (int code = 2; code < 127; ++code) {
                const double err = std::abs(e4m3fn_to_f32_host(code) - target);
                const double prev = std::abs(e4m3fn_to_f32_host(best) - target);
                if (err < prev || (err == prev && !(code & 1))) best = code;
            }
            expected[i*C+r] = diagonal * 6 * outer * e4m3fn_to_f32_host(best);
        }
    }
    expect_true("Blackwell resident expert order and router scales", maps);
    expect("Blackwell routed SwiGLU, hidden quantization and down vs FP64", host(y, S*C), expected, 0, 0);
    CU(cudaFree(hidden)); CU(cudaFree(packed_h));
    CU(cudaFree(y)); CU(cudaFree(packed)); CU(cudaFree(x)); CU(cudaFree(plan)); CU(cudaFree(wt)); CU(cudaFree(su));
    CU(cudaFree(sg)); CU(cudaFree(sc)); CU(cudaFree(w)); CU(cudaFree(inv)); CU(cudaFree(p)); CU(cudaFree(table)); CU(cudaFree(pool));
}

// A forced peak of six fixes each vector's UE4M3 scale. Off-grid inputs
// distinguish native quantization from the FP16 fallback.
enum class BlackwellPath { Inline, Verify, Prefill };

void test_blackwell_group(int members, int K, int N, bool gate, BlackwellPath path) {
    const int blocks = K / 64;
    const size_t matrix = (size_t)N * blocks * 36;
    auto weights = rand_blocks(k::kNVFP4, (size_t)2 * N * K);
    for (size_t i = 0; i < weights.size(); i += 36)
        for (int j = 0; j < 4; ++j)
            weights[i+j] = (i / 36 + j) % 11 == 0 ? 0x7f : uint8_t(rng() % 30);
    if (members == 1) weights[0] = weights[matrix] = 255;  // legacy scale, exactly representable input
    const float levels[] = {0, .5f, 1.125f, 1.5f, 1.875f, 3, 4, 6};
    const double rounded[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    std::vector<uint16_t> x(members * K);
    std::vector<double> quant(x.size());
    for (int m = 0; m < members; ++m) {
        for (int k = 0; k < K; ++k) {
            const int code = k % 16 == 0 ? 7 : rng() % 8;
            const float scale = std::ldexp(1.f + (k / 16 % 4) * .125f, (m + k / 16) % 6 - 4) *
                                ((rng() & 1) ? 1.f : -1.f);
            x[m*K+k] = f2h_host(scale * (members == 1 ? rounded[code] : levels[code]));
            quant[m*K+k] = scale * rounded[code];
        }
    }
    const bool prefill = path == BlackwellPath::Prefill;
    std::vector<int> order(std::max(members, k::kGroupMax)), token(order.size());
    for (int i = 0; i < members; ++i) {
        order[i] = prefill ? i : members - 1 - i;
        token[i] = prefill ? members - 1 - i : i;
    }
    auto *dx = dev(x), *dh = dev_n<uint16_t>(members*N);
    auto* dy = dev_n<float>(members*N);
    auto* dw = dev(weights);
    auto** blob = dev(std::vector<const uint8_t*>{dw});
    auto* count = dev(std::vector<int>{members});
    auto *row = dev(order), *tok = dev(token);
    if (path != BlackwellPath::Inline) {
        auto* packed = dev_n<k::BlackwellActivation>(members * blocks);
        k::blackwell_pack_rows(dx, members, K, packed, 0);
        k::BlackwellExpertBatch batch = prefill ? k::BlackwellExpertBatch{dw, nullptr, nullptr, tok, nullptr, members, true}
                                                : k::BlackwellExpertBatch{nullptr, blob, count, tok, row, 1, false};
        const int hidden_blocks = prefill && gate && N % 64 == 0 ? members * (N / 64) : 0;
        auto* fused = hidden_blocks ? dev_n<k::BlackwellActivation>(hidden_blocks) : nullptr;
        k::blackwell_expert_product(gate, batch, matrix, gate ? K : N, gate ? N : K, dx, packed,
                                    nullptr, nullptr, 20, 20, dh, dy, 0, fused);
        if (fused) {
            auto* separate = dev_n<k::BlackwellActivation>(hidden_blocks);
            k::blackwell_pack_rows(dh, members, N, separate, 0);
            const auto a = host(fused, hidden_blocks), b = host(separate, hidden_blocks);
            expect_true("fused SwiGLU NVFP4 codes and scales", !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])));
            CU(cudaFree(separate));
            CU(cudaFree(fused));
        }
        CU(cudaFree(packed));
    } else if (gate) {
        k::experts_gate_up_swiglu_g(blob, count, tok, row, 1, k::kNVFP4, k::kNVFP4, matrix, nullptr,
                                    K, N, nullptr, nullptr, 20, 20, dh, 0, dx);
    } else {
        k::experts_down_g(blob, count, row, 1, k::kNVFP4, matrix, dx, N, K, dy, 0);
    }
    std::vector<float> actual;
    if (gate) {
        for (uint16_t h : host(dh, members*N)) actual.push_back(h2f_host(h));
    } else {
        actual = host(dy, members*N);
    }
    std::vector<double> expected(actual.size());
    for (int m = 0; m < members; ++m) {
        for (int r = 0; r < N; ++r) {
            double a = 0, b = 0;
            for (int k = 0; k < K; ++k) {
                const double v = quant[(gate ? token[m] : m)*K+k];
                a += v * nvfp4_def(weights.data(), r*K+k);
                b += v * nvfp4_def(weights.data()+matrix, r*K+k);
            }
            if (gate) expected[order[m]*N+r] = 20 * std::tanh(a/20) / (1 + std::exp(-a)) * 20 * std::tanh(b/20);
            else expected[m*N+r] = b;
        }
    }
    expect(gate ? "Blackwell FP4 SwiGLU vs FP64" : "Blackwell FP4 down vs FP64",
           actual, expected, gate ? 6e-4 : 1e-5, gate ? 5e-5 : 1e-5);
    CU(cudaFree(dx)); CU(cudaFree(dh)); CU(cudaFree(dy)); CU(cudaFree(dw));
    CU(cudaFree(blob)); CU(cudaFree(count)); CU(cudaFree(row)); CU(cudaFree(tok));
}
#endif

int main(int argc, char** argv) {
    int dev_count = 0;
    CU(cudaGetDeviceCount(&dev_count));
    cudaDeviceProp p{};
    CU(cudaGetDeviceProperties(&p, 0));
#ifdef STRATA_USE_HIP
    std::printf("ik2_kernels_test on %s (%s)\n", p.name, p.gcnArchName);
    if (argc == 2 && !std::strcmp(argv[1], "--fp4-batch")) {
        for (int depth : {256, 2048})
            for (int rows_per_slot : {1, 4}) test_sequence_attention(depth, rows_per_slot, true);
        test_sequence_attention(257, 8, false, false, true);
        test_fp4_kv_write();
        test_fp4_kv_write(true);
        test_attn_flash();
        test_attn_decode_mma();
        test_attn_multi();
        return g_fail ? 1 : 0;
    }
#else
    std::printf("ik2_kernels_test on %s (sm_%d%d)\n", p.name, p.major, p.minor);
#endif
#ifndef STRATA_USE_HIP
    if (argc == 2 && !std::strcmp(argv[1], "--batch-mma")) {
        set_test_env("IK2_GROUP_MMA", "3");
        for (int T : {16, 32}) {
            test_batch_groups(T, k::kNVFP4);
            test_batch_groups(T, k::kQ8_0);
            for (int type : {k::kIQ4_NL, k::kQ8_0}) test_linear_block_verify(type, T);
        }
        return g_fail ? 1 : 0;
    }
    if (argc == 2 && !std::strcmp(argv[1], "--blackwell")) {
        if (p.major != 12 || p.minor > 1) return 77;
        set_test_env("IK2_BLACKWELL_NVFP4", "1");
        expect_true("native Blackwell NVFP4 dispatch", k::use_blackwell_nvfp4());
        for (bool gate : {false, true}) {
            test_blackwell_group(1, 64, gate ? 8 : 16, gate, BlackwellPath::Inline);
            test_blackwell_group(7, 320, gate ? 24 : 48, gate, BlackwellPath::Inline);
            for (int n = 1; n <= 8; ++n)
                test_blackwell_group(n, 320, gate ? 24 : 48, gate, BlackwellPath::Verify);
            for (int n : {1, 7, 19, 33, 64, 129})
                test_blackwell_group(n, 320, gate ? 128 : 144, gate, BlackwellPath::Prefill);
        }
        test_blackwell_prefill_routes();
        return g_fail ? 1 : 0;
    }
#endif
#ifndef STRATA_USE_HIP
    test_served_copy();
#endif
    test_rms();
    test_project_out();
    test_l2clip();
    test_conv();
    test_attn_res(9, 4608);
    test_attn_res(1, 2304);
    test_attn_probs();
    test_route(384);
#ifdef STRATA_USE_HIP
    test_fp4_kv_write(true);
    test_route(128);
    test_project_route();
#endif
    test_moe_pieces();
    test_decode();
    test_iq_ref();
    test_experts_mv();
#ifdef STRATA_USE_HIP
    test_iq4nl_gate_cancellation();
    test_rdna3_loop_gate();
    test_rdna3_attention_output_projection();
    test_rdna3_equal_attention_scores();
    test_rdna3_dense_output_prefill();
    test_rdna3_small_dense();
    test_rdna3_dense_gemm();
    test_rdna3_expert_groups();
    test_rdna3_expert_gemm(k::kIQ4_NL, 96, "RDNA3 WMMA IQ4_NL expert GEMM");
    test_rdna3_expert_gemm(k::kQ8_0, 96, "RDNA3 WMMA Q8_0 expert GEMM");
    test_rdna3_expert_gemm(k::kNVFP4, 192, "RDNA3 WMMA NVFP4 expert GEMM");
#endif
    test_attn_decode();
    test_mv_blocks();
    test_experts_grouped();
    test_attn_multi();
    test_argmax();
    test_attn_f16();
    test_round2_fused();
    test_fp8_kv();
    test_fp4_kv_write();
    test_round6();
    test_attn_flash();
    test_attn_decode_mma();
    test_cand_topk();
    test_token_path();
    test_expert_limbs();
    test_experts_mma(256, 128, 5);
    test_q8_verify_tile();
    test_draft_attn_online();
#ifndef STRATA_USE_HIP
    for (int t = 2; t <= 8; ++t)
        for (int type : {k::kIQ4_NL, k::kQ8_0}) test_linear_block_verify(type, t);
#endif
#ifdef STRATA_USE_HIP
    test_experts_mma(2304, 1152, 1);
    test_rdna3_speculative_kernels();
#endif
#ifdef STRATA_USE_HIP
    test_sequence_attention(257, 8, false, false, true);
    test_sequence_attention(238, 8, false, true, false, true);
#endif
    test_sequence_attention(256, 1, false);
    test_sequence_attention(256, 4, false);
#ifndef STRATA_USE_HIP
    test_sequence_attention(238, 8, false, true);
#endif
    test_deep_attention();
    test_fp4_sequence_replay();
    test_sequence_kernels();
    test_window_rollback();
    test_sparse_decode();
#ifndef STRATA_USE_HIP
    for (int T : {16, 32, 64})
        for (int ty : {k::kIQ4_NL, k::kQ8_0, k::kNVFP4}) test_batch_groups(T, ty);
#endif
    std::printf(g_fail ? "FAILED: %d check(s)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
