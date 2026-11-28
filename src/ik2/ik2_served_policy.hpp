#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace ik2 {

// Rank expert indices descending by saved profile count (tie-broken by expert index) and return the top `hot`.
inline std::vector<int> rank_hot_experts(const uint64_t* layer_counts, int experts, int hot) {
    const int k = std::clamp(hot, 0, experts);
    std::vector<int> ord(experts);
    std::iota(ord.begin(), ord.end(), 0);
    if (layer_counts) {
        bool any = false;
        for (int e = 0; e < experts; ++e)
            if (layer_counts[e] > 0) {
                any = true;
                break;
            }
        if (any) {
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
                if (layer_counts[a] != layer_counts[b]) return layer_counts[a] > layer_counts[b];
                return a < b;
            });
        }
    }
    ord.resize(k);
    return ord;
}

// Allocate per-layer hot expert slot counts within `hot_budget` bytes by greedily maximizing
// expected decode time saved per byte of VRAM: looped layers (visited multiple times per token)
// and higher-frequency profile experts receive proportionally more VRAM slots.
inline std::vector<int> allocate_served_hot_slots(const std::vector<std::size_t>& slot_bytes,
                                                  const std::vector<int>& visits_per_token, int experts,
                                                  std::size_t hot_budget, const uint64_t* counts = nullptr) {
    const int n_layer = int(slot_bytes.size());
    std::vector<int> hot(n_layer, 0);
    if (n_layer == 0 || experts <= 0 || hot_budget == 0) return hot;

    std::size_t full_need = 0;
    for (int l = 0; l < n_layer; ++l) full_need += std::size_t(experts) * slot_bytes[l];
    if (hot_budget >= full_need) {
        std::fill(hot.begin(), hot.end(), experts);
        return hot;
    }

    // Zipf prior blended with empirical profile frequencies so unseen prompts still get balanced coverage.
    std::vector<double> zipf(experts, 0.0);
    double zipf_sum = 0.0;
    for (int k = 0; k < experts; ++k) {
        zipf[k] = 1.0 / std::pow(double(k + 1), 0.85);
        zipf_sum += zipf[k];
    }
    for (double& v : zipf) v /= zipf_sum;

    std::vector<std::vector<double>> gain(n_layer, std::vector<double>(experts, 0.0));
    for (int l = 0; l < n_layer; ++l) {
        std::vector<double> emp;
        double sum = 0.0;
        if (counts) {
            emp.resize(experts);
            for (int e = 0; e < experts; ++e) {
                emp[e] = double(counts[std::size_t(l) * experts + e]);
                sum += emp[e];
            }
            if (sum > 0.0) {
                std::sort(emp.begin(), emp.end(), std::greater<double>());
                for (double& v : emp) v /= sum;
            }
        }
        const double vis = double(std::max(1, visits_per_token[l]));
        const double mb = double(std::max<std::size_t>(1, slot_bytes[l])) / 1048576.0;
        // Bandwidth term (slot_bytes cancels) + small per-miss fixed-overhead term (~0.5 MiB equivalent).
        const double byte_factor = 1.0 + 0.5 / std::max(0.5, mb);
        for (int k = 0; k < experts; ++k) {
            const double p = (sum > 0.0) ? (0.7 * emp[k] + 0.3 * zipf[k]) : zipf[k];
            gain[l][k] = vis * p * byte_factor;
        }
    }

    std::size_t rem = hot_budget;
    for (;;) {
        int best_l = -1;
        double best_g = -1.0;
        for (int l = 0; l < n_layer; ++l) {
            if (hot[l] >= experts || slot_bytes[l] > rem) continue;
            const double g = gain[l][hot[l]];
            if (g > best_g) {
                best_g = g;
                best_l = l;
            }
        }
        if (best_l < 0) break;
        ++hot[best_l];
        rem -= slot_bytes[best_l];
    }
    return hot;
}

// Each reserve is dedicated to one router pick of one physical layer visit.
// A miss temporarily makes that expert resident for the rest of the token.
struct ServedSlots {
    std::vector<int> owner, reserve;
    int hot;

    ServedSlots(int slots, int visits_times_k, const std::vector<int>& initial_hot = {})
        : owner(slots, -1), reserve(visits_times_k), hot(slots - visits_times_k) {
        if (!initial_hot.empty()) {
            const int n = std::min(hot, int(initial_hot.size()));
            for (int i = 0; i < n; ++i) owner[i] = initial_hot[i];
        } else {
            for (int i = 0; i < hot; ++i) owner[i] = i;
        }
        for (int i = 0; i < visits_times_k; ++i) reserve[i] = hot + i;
    }

    void write_table(const uint8_t** table, int experts, const uint8_t* cold, const uint8_t* cache, std::size_t bytes) const {
        for (int e = 0; e < experts; ++e) table[e] = cold + e * bytes;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] >= 0) table[owner[s]] = cache + s * bytes;
        for (int i = 0; i < int(reserve.size()); ++i) table[experts + i] = cache + reserve[i] * bytes;
    }

    void refill(const uint32_t* frequency) {
        std::vector<int> used;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] >= 0) used.push_back(s);
        std::sort(used.begin(), used.end(), [&](int a, int b) {
            if (frequency[owner[a]] != frequency[owner[b]]) return frequency[owner[a]] > frequency[owner[b]];
            return owner[a] < owner[b];
        });
        for (int i = hot; i < int(used.size()); ++i) owner[used[i]] = -1;
        int next = 0;
        for (int s = 0; s < int(owner.size()); ++s)
            if (owner[s] < 0) reserve[next++] = s;
    }
};
}
