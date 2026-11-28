#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
