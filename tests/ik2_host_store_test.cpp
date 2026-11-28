#include "ik2_host_store.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static uint8_t pattern_byte(int l, int e, int r, size_t off) {
    return (uint8_t)(((uint64_t)l * 131u + (uint64_t)e * 17u + (uint64_t)r * 31u + off * 7u + 13u) & 0xFFu);
}

int main() {
    // 1. Verify host RAM detection returns sane nonzero numbers.
    const auto ram = ik2::tier::query_host_ram();
    if (ram.total == 0 || ram.avail == 0 || ram.avail > ram.total) {
        std::fprintf(stderr, "unexpected host RAM info: total=%zu avail=%zu\n", ram.total, ram.avail);
        return 1;
    }

    // 2. Build a synthetic shard with unaligned offsets across 4096-byte sector boundaries.
    const fs::path tmp = fs::temp_directory_path() / "ik2_host_store_test.bin";
    const int n_layers = 2, E = 8;
    const size_t eb_small[3] = {513, 513, 771};     // fused gate_up + down, deliberately unaligned
    const size_t eb_large[3] = {2111, 2111, 3007};  // larger size class spanning multiple sectors
    std::vector<uint8_t> file_bytes(37, 0xA5);      // unaligned header prefix
    std::vector<ik2::tier::DiskExpertLoc> locs((size_t)n_layers * E);

    for (int l = 0; l < n_layers; ++l) {
        const size_t* eb = (l == 0) ? eb_small : eb_large;
        for (int e = 0; e < E; ++e) {
            auto& d = locs[(size_t)l * E + e];
            // Write gate + up contiguously (like fused gate_up), then a small gap, then down.
            for (int r = 0; r < 2; ++r) {
                d.file_idx[r] = 0;
                d.off[r] = file_bytes.size();
                d.bytes[r] = eb[r];
                for (size_t i = 0; i < eb[r]; ++i) file_bytes.push_back(pattern_byte(l, e, r, i));
            }
            file_bytes.insert(file_bytes.end(), 19, 0x5A);
            d.file_idx[2] = 0;
            d.off[2] = file_bytes.size();
            d.bytes[2] = eb[2];
            for (size_t i = 0; i < eb[2]; ++i) file_bytes.push_back(pattern_byte(l, e, 2, i));
        }
    }
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(file_bytes.data()), (std::streamsize)file_bytes.size());
        if (!out) return 2;
    }

    auto verify_expert = [&](int l, int e, const uint8_t* ptr, size_t bytes) -> bool {
        const size_t* eb = (l == 0) ? eb_small : eb_large;
        if (bytes != eb[0] + eb[1] + eb[2]) return false;
        size_t pos = 0;
        for (int r = 0; r < 3; ++r) {
            for (size_t i = 0; i < eb[r]; ++i) {
                if (ptr[pos + i] != pattern_byte(l, e, r, i)) return false;
            }
            pos += eb[r];
        }
        return true;
    };

    // 3. Test DirectFile concurrent unaligned reads.
    {
        ik2::tier::DirectFile df;
        df.open(tmp.string(), true);
        ik2::tier::IoPool pool(4);
        std::atomic<int> bad{0};
        pool.parallel_for(n_layers * E, [&](int idx) {
            const int l = idx / E, e = idx % E;
            const auto& d = locs[idx];
            std::vector<uint8_t> buf(d.slot_bytes());
            df.pread_exact(d.off[0], d.bytes[0] + d.bytes[1], buf.data());
            df.pread_exact(d.off[2], d.bytes[2], buf.data() + d.bytes[0] + d.bytes[1]);
            if (!verify_expert(l, e, buf.data(), buf.size())) ++bad;
        });
        if (bad != 0) {
            std::fprintf(stderr, "DirectFile parallel unaligned read failed (%d bad)\n", bad.load());
            fs::remove(tmp);
            return 3;
        }
    }

    // 4. Test HostExpertStore tiered RAM cache (capacity = ~4 experts per layer) + disk eviction & stats.
    {
        ik2::tier::HostExpertStore store;
        const size_t sb0 = (eb_small[0] + eb_small[1] + eb_small[2] + 255) / 256 * 256;
        const size_t sb1 = (eb_large[0] + eb_large[1] + eb_large[2] + 255) / 256 * 256;
        const size_t ram_budget = 4 * sb0 + 4 * sb1;
        uint64_t waited_fence = 0;
        store.init({tmp.string()}, n_layers, E, locs, ram_budget, 4, true,
                   [](size_t n) { return static_cast<uint8_t*>(ik2::tier::aligned_alloc_bytes(64, n)); },
                   [](uint8_t* p) { ik2::tier::aligned_free_bytes(p); },
                   [&](uint64_t f) { waited_fence = std::max(waited_fence, f); }, 8);

        if (store.all_in_ram || store.total_ram_slots() != 8) {
            std::fprintf(stderr, "expected partial RAM cache with 8 slots, got all=%d slots=%d\n",
                         (int)store.all_in_ram, store.total_ram_slots());
            fs::remove(tmp);
            return 4;
        }

        // Warm RAM skipping experts 0,1 (pretending they are already in VRAM) using profile counts.
        std::vector<uint64_t> counts((size_t)n_layers * E, 1);
        counts[0 * E + 6] = 500;
        counts[0 * E + 7] = 400;
        counts[0 * E + 2] = 300;
        counts[0 * E + 5] = 200;
        store.warm_ram(counts, [](int, int e) { return e < 2; });
        if (!store.has_in_ram(0, 6) || !store.has_in_ram(0, 7) || !store.has_in_ram(0, 2) || !store.has_in_ram(0, 5) ||
            store.has_in_ram(0, 0) || store.has_in_ram(0, 3)) {
            std::fprintf(stderr, "warm_ram did not prioritize non-VRAM hot experts\n");
            fs::remove(tmp);
            return 5;
        }

        // Fetch a mixed batch: 2 RAM hits (6, 7) + 2 Disk misses (3, 4).
        ik2::tier::HostExpertStore::BatchItem batch[] = {{0, 6}, {0, 3}, {0, 7}, {0, 4}};
        int verified = 0;
        store.fetch_batch(batch, 4, 1, [&](int i, const uint8_t* ptr, size_t bytes) {
            if (verify_expert(batch[i].l, batch[i].e, ptr, bytes)) ++verified;
        });
        if (verified != 4 || store.ram_hits != 2 || store.disk_hits != 2) {
            std::fprintf(stderr, "mixed batch failed: verified=%d ram=%llu disk=%llu\n", verified,
                         (unsigned long long)store.ram_hits, (unsigned long long)store.disk_hits);
            fs::remove(tmp);
            return 6;
        }

        // Fetch again: experts 3 and 4 should now be RAM hits!
        store.fetch_batch(batch, 4, 2, [&](int i, const uint8_t* ptr, size_t bytes) {
            if (verify_expert(batch[i].l, batch[i].e, ptr, bytes)) ++verified;
        });
        if (verified != 8 || store.ram_hits != 6 || store.disk_hits != 2) {
            std::fprintf(stderr, "second batch should hit RAM: verified=%d ram=%llu disk=%llu\n", verified,
                         (unsigned long long)store.ram_hits, (unsigned long long)store.disk_hits);
            fs::remove(tmp);
            return 7;
        }

        // Force eviction of a fenced RAM slot by fetching remaining cold experts (0, 1, 2, 5).
        ik2::tier::HostExpertStore::BatchItem cold_batch[] = {{0, 0}, {0, 1}, {0, 2}, {0, 5}};
        store.fetch_batch(cold_batch, 4, 3, [&](int i, const uint8_t* ptr, size_t bytes) {
            if (verify_expert(cold_batch[i].l, cold_batch[i].e, ptr, bytes)) ++verified;
        });
        if (verified != 12 || waited_fence < 2) {
            std::fprintf(stderr, "eviction/fence wait failed: verified=%d fence=%llu\n", verified,
                         (unsigned long long)waited_fence);
            fs::remove(tmp);
            return 8;
        }

        // Regression test: re-warming RAM after prior fetches must clear stale loc[] mappings so
        // previously cached experts do not read overwritten slots.
        auto wr = store.warm_ram(counts, [](int l, int e) { return l == 0 && (e == 0 || e == 1 || e == 2); });
        if (wr.loaded != 8 || store.has_in_ram(0, 0) || store.has_in_ram(0, 1) || store.has_in_ram(0, 2)) {
            std::fprintf(stderr, "re-warm_ram did not clear stale RAM mappings cleanly (loaded=%d)\n", wr.loaded);
            fs::remove(tmp);
            return 10;
        }
        int re_verified = 0;
        store.fetch_batch(cold_batch, 4, 4, [&](int i, const uint8_t* ptr, size_t bytes) {
            if (verify_expert(cold_batch[i].l, cold_batch[i].e, ptr, bytes)) ++re_verified;
        });
        if (re_verified != 4) {
            std::fprintf(stderr, "re-fetch after re-warm_ram returned corrupted data (%d/4)\n", re_verified);
            fs::remove(tmp);
            return 11;
        }

        // Test stream_from_disk + surplus RAM backing (preserve_existing=true in warm_ram).
        ik2::tier::HostExpertStore store2;
        const size_t budget2 = 6 * sb0 + 6 * sb1; // 6 RAM slots per pool; 5 in VRAM -> 3 non-VRAM + 3 surplus!
        store2.init({tmp.string()}, n_layers, E, locs, budget2, 2, true,
                    [](size_t n) { return static_cast<uint8_t*>(ik2::tier::aligned_alloc_bytes(64, n)); },
                    [](uint8_t* p) { ik2::tier::aligned_free_bytes(p); }, nullptr);
        if (store2.surplus_ram_slots_for_layer(0, 5) != 3) {
            std::fprintf(stderr, "surplus_ram_slots_for_layer expected 3, got %d\n",
                         store2.surplus_ram_slots_for_layer(0, 5));
            fs::remove(tmp);
            return 12;
        }
        std::vector<std::pair<int, int>> stream_items = {{0, 4}, {0, 3}, {0, 2}, {0, 1}, {0, 0}};
        int stream_verified = 0, chunks = 0;
        store2.stream_from_disk(
            stream_items,
            [&](int i, const uint8_t* ptr, size_t bytes) {
                if (verify_expert(stream_items[i].first, stream_items[i].second, ptr, bytes)) ++stream_verified;
                if (i < 3) store2.insert_free_ram(stream_items[i].first, stream_items[i].second, ptr, 1);
            },
            [&]() { ++chunks; });
        auto wr2 = store2.warm_ram(counts, [](int l, int e) { return l == 0 && e < 5; }, true);
        // Layer 0 had 3 surplus backed + 3 non-VRAM loaded; layer 1 had 6 loaded -> 9 total loaded from disk.
        if (stream_verified != 5 || chunks != 1 || wr2.loaded != 9 || !store2.has_in_ram(0, 4) ||
            !store2.has_in_ram(0, 5) || !store2.has_in_ram(0, 6) || !store2.has_in_ram(0, 7)) {
            std::fprintf(stderr, "surplus RAM backing failed: sv=%d chunks=%d loaded=%d\n", stream_verified, chunks,
                         wr2.loaded);
            fs::remove(tmp);
            return 13;
        }
    }

    // 5. Test zero-RAM staging mode (ram_budget = 0 -> pure disk reads through staging buffers).
    {
        ik2::tier::HostExpertStore store;
        store.init({tmp.string()}, n_layers, E, locs, 0, 2, true,
                   [](size_t n) { return static_cast<uint8_t*>(ik2::tier::aligned_alloc_bytes(64, n)); },
                   [](uint8_t* p) { ik2::tier::aligned_free_bytes(p); }, nullptr, 4);
        ik2::tier::HostExpertStore::BatchItem batch[] = {{1, 0}, {1, 3}, {1, 7}};
        int verified = 0;
        const bool used_stage = store.fetch_batch(batch, 3, 1, [&](int i, const uint8_t* ptr, size_t bytes) {
            if (verify_expert(batch[i].l, batch[i].e, ptr, bytes)) ++verified;
        });
        if (!used_stage || verified != 3 || store.ram_hits != 0 || store.disk_hits != 3) {
            std::fprintf(stderr, "zero-RAM staging failed: used_stage=%d verified=%d ram=%llu disk=%llu\n",
                         (int)used_stage, verified, (unsigned long long)store.ram_hits,
                         (unsigned long long)store.disk_hits);
            fs::remove(tmp);
            return 9;
        }
    }

    fs::remove(tmp);
    std::puts("direct I/O and 3-tier host expert store passed");
    return 0;
}
