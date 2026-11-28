#include "ik2_served_policy.hpp"
#include <cstdio>

int main() {
    // Two hot slots and a spare pick on each of three visits.
    ik2::ServedSlots slots(5, 3);
    uint8_t cold[7]{}, cache[5]{};
    const uint8_t* table[10]{};
    slots.write_table(table, 7, cold, cache, 1);
    if (table[0] != cache || table[1] != cache + 1 || table[5] != cold + 5 ||
        table[7] != cache + 2 || table[9] != cache + 4) return 1;

    slots.owner[slots.reserve[0]] = 5;
    slots.owner[slots.reserve[2]] = 3;
    uint32_t frequency[]{4, 1, 0, 4, 0, 9, 0};
    slots.refill(frequency);
    slots.write_table(table, 7, cold, cache, 1);
    if (table[5] != cache + 2 || table[0] != cache || table[1] != cold + 1 ||
        table[3] != cold + 3 || slots.reserve != std::vector<int>{1, 3, 4}) return 1;

    // Profile-warmed initialization places the hottest experts in slots 0..hot-1.
    const uint64_t profile[]{10, 500, 20, 900, 500, 0, 1};
    const auto ranked = ik2::rank_hot_experts(profile, 7, 3);
    if (ranked != std::vector<int>{3, 1, 4}) return 2;
    ik2::ServedSlots warmed(6, 3, ranked);
    warmed.write_table(table, 7, cold, cache, 1);
    if (table[3] != cache || table[1] != cache + 1 || table[4] != cache + 2 ||
        table[0] != cold || table[2] != cold + 2 || table[7] != cache + 3) return 3;

    // With only reserves every address must revert to its mapped source between tokens.
    ik2::ServedSlots tiny(2, 2);
    tiny.owner = {3, 5};
    tiny.refill(frequency);
    tiny.write_table(table, 7, cold, cache, 1);
    if (table[3] != cold + 3 || table[5] != cold + 5 || table[7] != cache || table[8] != cache + 1) return 4;
    std::puts("served expert table, profile warming and LFU refill passed");
}
