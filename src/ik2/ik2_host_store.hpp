#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ik2::tier {

inline void* aligned_alloc_bytes(size_t align, size_t bytes) {
    if (bytes == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(bytes, align);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
#endif
}

inline void aligned_free_bytes(void* p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct HostRamInfo {
    size_t total = 0;
    size_t avail = 0;
};

inline HostRamInfo query_host_ram() {
    HostRamInfo info{};
#ifdef _WIN32
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        info.total = (size_t)st.ullTotalPhys;
        info.avail = (size_t)st.ullAvailPhys;
    }
#else
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) info.total = (size_t)kb * 1024;
            else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) info.avail = (size_t)kb * 1024;
        }
        std::fclose(f);
    }
    if (info.total == 0) {
        const long pages = ::sysconf(_SC_PHYS_PAGES);
        const long pg = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && pg > 0) info.total = (size_t)pages * (size_t)pg;
    }
    if (info.avail == 0) info.avail = info.total / 2;
#endif
    return info;
}

// Default RAM budget for experts when auto-sizing: leave OS & desktop headroom so the machine stays responsive.
inline size_t default_host_ram_budget(const HostRamInfo& info = query_host_ram()) {
    if (info.avail == 0) return 0;
#ifdef _WIN32
    const size_t min_headroom = 4ULL << 30;
    const size_t pct_headroom = (info.total / 100) * 15;
#else
    const size_t min_headroom = 2ULL << 30;
    const size_t pct_headroom = (info.total / 100) * 10;
#endif
    const size_t headroom = std::max(min_headroom, pct_headroom);
    return info.avail > headroom ? info.avail - headroom : 0;
}

// Thread-safe random-access file reader that bypasses the OS page cache when supported
// (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED on Windows, O_DIRECT on Linux).
class DirectFile {
public:
    DirectFile() = default;
    ~DirectFile() { close(); }
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;
    DirectFile(DirectFile&& o) noexcept { move_from(std::move(o)); }
    DirectFile& operator=(DirectFile&& o) noexcept {
        if (this != &o) {
            close();
            move_from(std::move(o));
        }
        return *this;
    }

    void open(const std::string& path, bool prefer_direct = true) {
        close();
        path_ = path;
        direct_ = false;
        sector_ = 4096;
#ifdef _WIN32
        DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
        HANDLE h = INVALID_HANDLE_VALUE;
        if (prefer_direct) {
            h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            flags | FILE_FLAG_NO_BUFFERING, nullptr);
            if (h != INVALID_HANDLE_VALUE) direct_ = true;
        }
        if (h == INVALID_HANDLE_VALUE) {
            h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
        }
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("DirectFile: cannot open " + path);
        LARGE_INTEGER li{};
        if (!GetFileSizeEx(h, &li)) {
            CloseHandle(h);
            throw std::runtime_error("DirectFile: GetFileSizeEx failed for " + path);
        }
        size_ = (uint64_t)li.QuadPart;
        if (direct_) {
            // Query logical sector size for the volume; fall back to 4096 if unavailable.
            char root[4] = {};
            if (path.size() >= 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/')) {
                root[0] = path[0];
                root[1] = ':';
                root[2] = '\\';
                DWORD spc = 0, bps = 0, fc = 0, tc = 0;
                if (GetDiskFreeSpaceA(root, &spc, &bps, &fc, &tc) && bps >= 512 && (bps & (bps - 1)) == 0)
                    sector_ = std::max<size_t>(4096, bps);
            }
        }
        handle_ = h;
#else
        int fd = -1;
#ifdef O_DIRECT
        if (prefer_direct) {
            fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd >= 0) direct_ = true;
        }
#endif
        if (fd < 0) fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("DirectFile: cannot open " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("DirectFile: fstat failed for " + path);
        }
        size_ = (uint64_t)st.st_size;
        fd_ = fd;
#endif
    }

    void close() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
        direct_ = false;
    }

    bool is_open() const {
#ifdef _WIN32
        return handle_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }
    bool is_direct() const { return direct_; }
    size_t sector_size() const { return sector_; }
    uint64_t file_size() const { return size_; }
    const std::string& path() const { return path_; }

    // Thread-safe exact read of `bytes` at 64-bit `offset` into `dst` (arbitrary alignment).
    void pread_exact(uint64_t offset, size_t bytes, uint8_t* dst) const {
        if (bytes == 0) return;
        if (offset + bytes > size_) throw std::runtime_error("DirectFile: read past EOF in " + path_);

        // Max single OS read chunk (64 MiB keeps Win32 DWORD and bounce memory bounded).
        constexpr size_t kMaxChunk = 16ULL << 20;
        if (!direct_) {
            size_t done = 0;
            while (done < bytes) {
                const size_t n = std::min(kMaxChunk, bytes - done);
                const size_t got = raw_pread(offset + done, n, dst + done);
                if (got == 0) throw std::runtime_error("DirectFile: unexpected EOF in " + path_);
                done += got;
            }
#ifndef _WIN32
            ::posix_fadvise(fd_, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#endif
            return;
        }

        // Direct unbuffered read: align file offset down and length up to sector_ boundaries.
        const size_t align = sector_;
        const uint64_t a_off = offset & ~(uint64_t(align) - 1);
        const size_t head = (size_t)(offset - a_off);
        const uint64_t end = offset + bytes;
        const uint64_t a_end = (end + align - 1) & ~(uint64_t(align) - 1);

        // Thread-local aligned bounce buffer avoids per-read heap churn and stays page-aligned.
        struct Bounce {
            uint8_t* ptr = nullptr;
            size_t cap = 0;
            ~Bounce() { aligned_free_bytes(ptr); }
            uint8_t* ensure(size_t align_b, size_t need) {
                if (cap < need) {
                    aligned_free_bytes(ptr);
                    ptr = static_cast<uint8_t*>(aligned_alloc_bytes(align_b, need));
                    if (!ptr) throw std::bad_alloc();
                    cap = need;
                }
                return ptr;
            }
        };
        thread_local Bounce bounce;

        size_t copied = 0;
        uint64_t cur_a = a_off;
        while (copied < bytes) {
            const size_t rem_a = (size_t)(a_end - cur_a);
            const size_t chunk_a = std::min(kMaxChunk, rem_a);
            uint8_t* buf = bounce.ensure(align, chunk_a);
            size_t got_total = 0;
            // On the final chunk, EOF may be unaligned if file_size() is not a multiple of sector_.
            const uint64_t max_possible = cur_a < size_ ? std::min<uint64_t>(chunk_a, size_ - cur_a) : 0;
            while (got_total < max_possible) {
                const size_t req = chunk_a - got_total;
                const size_t got = raw_pread(cur_a + got_total, req, buf + got_total);
                if (got == 0) break;
                got_total += got;
            }
            const size_t slice_off = (cur_a == a_off) ? head : 0;
            const size_t avail_in_chunk = got_total > slice_off ? got_total - slice_off : 0;
            const size_t take = std::min(bytes - copied, avail_in_chunk);
            if (take == 0) throw std::runtime_error("DirectFile: short direct read in " + path_);
            std::memcpy(dst + copied, buf + slice_off, take);
            copied += take;
            cur_a += chunk_a;
        }
    }

private:
    size_t raw_pread(uint64_t offset, size_t bytes, uint8_t* dst) const {
#ifdef _WIN32
        struct EventHolder {
            HANDLE ev = nullptr;
            EventHolder() : ev(CreateEventA(nullptr, TRUE, FALSE, nullptr)) {}
            ~EventHolder() {
                if (ev) CloseHandle(ev);
            }
        };
        thread_local EventHolder holder;
        if (!holder.ev) throw std::runtime_error("DirectFile: CreateEvent failed");
        ResetEvent(holder.ev);
        OVERLAPPED ov{};
        ov.Offset = (DWORD)(offset & 0xFFFFFFFFULL);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = holder.ev;
        DWORD got = 0;
        const DWORD req = (DWORD)std::min<size_t>(bytes, 0x7FFFF000ULL);
        BOOL ok = ReadFile(handle_, dst, req, &got, &ov);
        if (!ok) {
            const DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                if (!GetOverlappedResult(handle_, &ov, &got, TRUE)) {
                    const DWORD err2 = GetLastError();
                    if (err2 == ERROR_HANDLE_EOF) return 0;
                    throw std::runtime_error("DirectFile: GetOverlappedResult error " + std::to_string(err2) +
                                             " on " + path_);
                }
            } else if (err == ERROR_HANDLE_EOF) {
                return 0;
            } else {
                throw std::runtime_error("DirectFile: ReadFile error " + std::to_string(err) + " on " + path_);
            }
        }
        return (size_t)got;
#else
        for (;;) {
            const ssize_t r = ::pread(fd_, dst, bytes, (off_t)offset);
            if (r >= 0) return (size_t)r;
            if (errno == EINTR) continue;
            throw std::runtime_error("DirectFile: pread error " + std::to_string(errno) + " on " + path_);
        }
#endif
    }

    void move_from(DirectFile&& o) noexcept {
        path_ = std::move(o.path_);
        size_ = o.size_;
        sector_ = o.sector_;
        direct_ = o.direct_;
#ifdef _WIN32
        handle_ = o.handle_;
        o.handle_ = INVALID_HANDLE_VALUE;
#else
        fd_ = o.fd_;
        o.fd_ = -1;
#endif
        o.size_ = 0;
        o.direct_ = false;
    }

    std::string path_;
    uint64_t size_ = 0;
    size_t sector_ = 4096;
    bool direct_ = false;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

// Simple persistent worker pool for parallel disk expert reads without per-token thread creation overhead.
class IoPool {
public:
    explicit IoPool(int n_threads = 0) { start(n_threads); }
    ~IoPool() { stop(); }
    IoPool(const IoPool&) = delete;
    IoPool& operator=(const IoPool&) = delete;

    void start(int n_threads) {
        stop();
        if (n_threads <= 1) return;
        stop_ = false;
        workers_.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        task = std::move(q_.front());
                        q_.pop_front();
                    }
                    try {
                        task();
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(err_mu_);
                        if (!err_) err_ = std::current_exception();
                    }
                    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lk(mu_);
                        done_cv_.notify_all();
                    }
                }
            });
        }
    }

    void stop() {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            if (w.joinable()) w.join();
        workers_.clear();
    }

    int size() const { return (int)workers_.size(); }

    template <typename Fn>
    void parallel_for(int n, Fn&& fn) {
        if (n <= 0) return;
        if (n == 1 || workers_.empty()) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        err_ = nullptr;
        pending_.store(n, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; ++i) q_.emplace_back([&fn, i]() { fn(i); });
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_cv_.wait(lk, [this]() { return pending_.load(std::memory_order_acquire) == 0; });
        }
        if (err_) {
            auto e = err_;
            err_ = nullptr;
            std::rethrow_exception(e);
        }
    }

private:
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<std::function<void()>> q_;
    std::atomic<int> pending_{0};
    bool stop_ = false;
    std::mutex err_mu_;
    std::exception_ptr err_;
};

// Immutable on-disk location of one expert's three tensors (gate, up, down) in a GGUF shard.
struct DiskExpertLoc {
    int file_idx[3] = {-1, -1, -1};
    uint64_t off[3] = {0, 0, 0};
    size_t bytes[3] = {0, 0, 0};
    size_t slot_bytes() const { return bytes[0] + bytes[1] + bytes[2]; }
};

// Host RAM cache + Direct-I/O Disk tier for experts (Tier 2 + Tier 3 beneath VRAM Tier 1).
// Memory allocation and DMA fence synchronization are injected via callbacks so this class
// is 100% unit-testable without CUDA and works identically with cudaHostAlloc / mmap+register.
struct HostExpertStore {
    using AllocFn = std::function<uint8_t*(size_t)>;
    using FreeFn = std::function<void(uint8_t*)>;
    using WaitFenceFn = std::function<void(uint64_t)>;

    struct Pool {
        size_t slot_bytes = 0;
        int n_slots = 0;
        int n_layers = 0;
        int clock_hand = 0;
        uint8_t* data = nullptr;
        std::vector<int> owner;       // slot -> expert idx (l * E + e), -1 if free
        std::vector<uint8_t> freq;    // clock / LFU score [0..7]
        std::vector<uint64_t> stamp;  // last access tick (protects current-batch slots)
        std::vector<uint64_t> fence;  // DMA fence sequence (must wait before overwriting)
    };

    int n_layers = 0;
    int E = 0;
    bool active = false;
    bool all_in_ram = false;
    uint64_t tick = 0;
    uint64_t ram_hits = 0;
    uint64_t disk_hits = 0;
    uint64_t disk_bytes = 0;

    std::vector<DirectFile> files;
    std::vector<DiskExpertLoc> disk;   // [n_layers * E]
    std::vector<int> layer_pool;       // [n_layers]
    std::vector<int> loc;              // [n_layers * E] -> slot in pool, -1 if disk-only
    std::vector<Pool> pools;
    // Small fallback staging buffer used only when a pool has 0 RAM slots (pure VRAM + Disk).
    std::vector<uint8_t*> stage_bufs;
    size_t stage_bytes = 0;
    IoPool io_pool;

    AllocFn alloc_fn;
    FreeFn free_fn;
    WaitFenceFn wait_fence_fn;

    HostExpertStore() = default;
    ~HostExpertStore() { release(); }
    HostExpertStore(const HostExpertStore&) = delete;
    HostExpertStore& operator=(const HostExpertStore&) = delete;

    void release() {
        io_pool.stop();
        if (free_fn) {
            for (auto& p : pools)
                if (p.data) free_fn(p.data), p.data = nullptr;
            for (auto* b : stage_bufs)
                if (b) free_fn(b);
        }
        pools.clear();
        stage_bufs.clear();
        files.clear();
        disk.clear();
        layer_pool.clear();
        loc.clear();
        active = false;
        all_in_ram = false;
    }

    size_t total_ram_bytes() const {
        size_t b = 0;
        for (const auto& p : pools) b += (size_t)p.n_slots * p.slot_bytes;
        return b;
    }
    int total_ram_slots() const {
        int s = 0;
        for (const auto& p : pools) s += p.n_slots;
        return s;
    }

    bool has_in_ram(int l, int e) const { return active && loc[(size_t)l * E + e] >= 0; }

    // Configure per-size-class RAM pools to fit within `ram_budget`, open DirectFile handles,
    // and start the parallel disk I/O worker pool.
    void init(const std::vector<std::string>& shard_paths, int n_layers_in, int E_in,
              std::vector<DiskExpertLoc> disk_locs, size_t ram_budget, int io_threads, bool prefer_direct,
              AllocFn alloc_cb, FreeFn free_cb, WaitFenceFn wait_fence_cb = nullptr, int max_stage = 32) {
        release();
        n_layers = n_layers_in;
        E = E_in;
        disk = std::move(disk_locs);
        alloc_fn = std::move(alloc_cb);
        free_fn = std::move(free_cb);
        wait_fence_fn = std::move(wait_fence_cb);
        layer_pool.assign(n_layers, -1);
        loc.assign((size_t)n_layers * E, -1);

        files.resize(shard_paths.size());
        for (size_t i = 0; i < shard_paths.size(); ++i) files[i].open(shard_paths[i], prefer_direct);

        // Group layers by exact expert byte size (rounded up to 256 B for GPU alignment).
        std::map<size_t, std::vector<int>> by_sb;
        size_t max_sb = 0;
        size_t total_model_exp = 0;
        for (int l = 0; l < n_layers; ++l) {
            const size_t raw = disk[(size_t)l * E].slot_bytes();
            const size_t sb = (raw + 255) / 256 * 256;
            by_sb[sb].push_back(l);
            max_sb = std::max(max_sb, sb);
            total_model_exp += (size_t)E * sb;
        }

        pools.resize(by_sb.size());
        std::vector<int> pmax(pools.size(), 0);
        size_t pi = 0;
        for (const auto& kv : by_sb) {
            pools[pi].slot_bytes = kv.first;
            pools[pi].n_layers = (int)kv.second.size();
            for (int l : kv.second) layer_pool[l] = (int)pi;
            pmax[pi] = pools[pi].n_layers * E;
            ++pi;
        }

        all_in_ram = (ram_budget >= total_model_exp);
        if (all_in_ram) {
            for (size_t p = 0; p < pools.size(); ++p) pools[p].n_slots = pmax[p];
        } else {
            // Distribute RAM slots proportionally across layer size classes.
            auto used_for = [&](double spl) {
                size_t u = 0;
                for (size_t p = 0; p < pools.size(); ++p) {
                    const int ns = std::clamp((int)std::floor(pools[p].n_layers * spl), 0, pmax[p]);
                    u += (size_t)ns * pools[p].slot_bytes;
                }
                return u;
            };
            double lo = 0.0, hi = (double)E;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (used_for(mid) <= ram_budget) lo = mid;
                else hi = mid;
            }
            size_t used = 0;
            for (size_t p = 0; p < pools.size(); ++p) {
                pools[p].n_slots = std::clamp((int)std::floor(pools[p].n_layers * lo), 0, pmax[p]);
                used += (size_t)pools[p].n_slots * pools[p].slot_bytes;
            }
            size_t rem = ram_budget > used ? ram_budget - used : 0;
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
        }

        // Allocate pinned host buffers per pool.
        for (auto& P : pools) {
            if (P.n_slots <= 0) continue;
            P.data = alloc_fn((size_t)P.n_slots * P.slot_bytes);
            if (!P.data) throw std::bad_alloc();
            P.owner.assign(P.n_slots, -1);
            P.freq.assign(P.n_slots, 0);
            P.stamp.assign(P.n_slots, 0);
            P.fence.assign(P.n_slots, 0);
        }

        // Allocate small staging ring if any pool has fewer slots than a batch (e.g. 0 RAM slots).
        bool need_stage = false;
        for (const auto& P : pools)
            if (P.n_slots < max_stage) need_stage = true;
        if (need_stage && max_sb > 0) {
            stage_bytes = max_sb;
            stage_bufs.resize(max_stage, nullptr);
            for (int i = 0; i < max_stage; ++i) {
                stage_bufs[i] = alloc_fn(stage_bytes);
                if (!stage_bufs[i]) throw std::bad_alloc();
            }
        }

        io_pool.start(std::max(1, io_threads));
        active = true;
    }

    // Read one expert's 3 tensors contiguously into `dst` (gate | up | down).
    // Coalesces adjacent same-file roles (such as fused gate_up GGUF tensors) into one direct read.
    void read_expert_from_disk(int l, int e, uint8_t* dst) const {
        const auto& d = disk[(size_t)l * E + e];
        if (d.file_idx[0] == d.file_idx[1] && d.off[1] == d.off[0] + d.bytes[0]) {
            const size_t gu = d.bytes[0] + d.bytes[1];
            files[d.file_idx[0]].pread_exact(d.off[0], gu, dst);
            files[d.file_idx[2]].pread_exact(d.off[2], d.bytes[2], dst + gu);
            return;
        }
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            files[d.file_idx[r]].pread_exact(d.off[r], d.bytes[r], dst + off);
            off += d.bytes[r];
        }
    }

    // Copy an already-read expert buffer into a free RAM slot (used during VRAM warmup to back
    // eviction-candidate VRAM experts in surplus RAM slots at zero extra disk I/O).
    bool insert_free_ram(int l, int e, const uint8_t* src, uint8_t init_freq = 1) {
        if (!active || l < 0 || l >= n_layers || e < 0 || e >= E) return false;
        const int idx = l * E + e;
        if (loc[idx] >= 0) return true;
        const int p = layer_pool[l];
        Pool& P = pools[p];
        for (int s = 0; s < P.n_slots; ++s) {
            if (P.owner[s] < 0) {
                std::memcpy(P.data + (size_t)s * P.slot_bytes, src, disk[idx].slot_bytes());
                P.owner[s] = idx;
                P.freq[s] = init_freq;
                P.stamp[s] = 0;
                P.fence[s] = 0;
                loc[idx] = s;
                return true;
            }
        }
        return false;
    }

    int surplus_ram_slots_for_layer(int l, int vram_count_in_pool) const {
        if (!active || l < 0 || l >= n_layers) return 0;
        const Pool& P = pools[layer_pool[l]];
        const int total_in_pool = P.n_layers * E;
        const int non_vram = std::max(0, total_in_pool - vram_count_in_pool);
        return std::max(0, P.n_slots - non_vram);
    }

    // Pre-populate the RAM pools in parallel with the highest-priority experts not already in VRAM.
    // `skip_in_vram(l, e)` returns true if the expert is already warm in VRAM so we maximize the
    // disjoint union (VRAM U RAM) and never read VRAM experts twice from disk at startup.
    // When `preserve_existing` is true, experts already backed in RAM (e.g. surplus VRAM-eviction
    // backups inserted during VRAM warmup) are kept in their slots and only free slots are filled.
    // Returns {experts_loaded, bytes_loaded}.
    struct WarmResult {
        int loaded = 0;
        size_t bytes = 0;
    };
    WarmResult warm_ram(const std::vector<uint64_t>& counts, const std::function<bool(int, int)>& skip_in_vram = nullptr,
                        bool preserve_existing = false) {
        WarmResult res;
        if (!active) return res;
        struct LoadJob {
            int l, e, p, slot;
            uint8_t* dst;
            size_t bytes;
        };
        std::vector<LoadJob> jobs;
        for (size_t p = 0; p < pools.size(); ++p) {
            Pool& P = pools[p];
            if (P.n_slots <= 0) continue;
            if (!preserve_existing) {
                // Clear any pre-existing RAM slot mappings cleanly so `loc` can never go stale.
                for (int s = 0; s < P.n_slots; ++s) {
                    if (P.owner[s] >= 0) {
                        if (P.fence[s] != 0 && wait_fence_fn) {
                            wait_fence_fn(P.fence[s]);
                            P.fence[s] = 0;
                        }
                        loc[P.owner[s]] = -1;
                        P.owner[s] = -1;
                        P.freq[s] = 0;
                        P.stamp[s] = 0;
                    }
                }
                P.clock_hand = 0;
            }

            std::vector<int> free_slots;
            free_slots.reserve(P.n_slots);
            for (int s = 0; s < P.n_slots; ++s)
                if (P.owner[s] < 0) free_slots.push_back(s);
            if (free_slots.empty()) continue;

            std::vector<int> cand;
            cand.reserve(P.n_layers * E);
            for (int l = 0; l < n_layers; ++l) {
                if (layer_pool[l] != (int)p) continue;
                for (int e = 0; e < E; ++e) {
                    const int idx = l * E + e;
                    if (loc[idx] >= 0) continue;
                    if (!all_in_ram && skip_in_vram && skip_in_vram(l, e)) continue;
                    cand.push_back(idx);
                }
            }
            if (!counts.empty()) {
                std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return counts[a] > counts[b]; });
            }
            const int take = std::min((int)cand.size(), (int)free_slots.size());
            for (int i = 0; i < take; ++i) {
                const int s = free_slots[i];
                const int idx = cand[i];
                const int l = idx / E, e = idx % E;
                P.owner[s] = idx;
                P.freq[s] = (!counts.empty() && counts[idx] > 0) ? 2 : 1;
                P.stamp[s] = 0;
                P.fence[s] = 0;
                loc[idx] = s;
                const size_t sb = disk[idx].slot_bytes();
                jobs.push_back({l, e, (int)p, s, P.data + (size_t)s * P.slot_bytes, sb});
                res.bytes += sb;
            }
        }
        // Sort by on-disk offset so startup reads sweep sequentially across each GGUF shard.
        std::sort(jobs.begin(), jobs.end(), [&](const LoadJob& a, const LoadJob& b) {
            const auto& da = disk[(size_t)a.l * E + a.e];
            const auto& db = disk[(size_t)b.l * E + b.e];
            if (da.file_idx[0] != db.file_idx[0]) return da.file_idx[0] < db.file_idx[0];
            return da.off[0] < db.off[0];
        });
        io_pool.parallel_for((int)jobs.size(), [&](int i) {
            const auto& J = jobs[i];
            read_expert_from_disk(J.l, J.e, J.dst);
        });
        res.loaded = (int)jobs.size();
        return res;
    }

    // Stream `n` experts directly from disk in file-offset order through temporary pinned bounce
    // buffers without touching RAM cache pools (used for fast parallel VRAM warmup).
    template <typename ReadyFn, typename SyncFn>
    void stream_from_disk(const std::vector<std::pair<int, int>>& items, ReadyFn&& on_ready, SyncFn&& on_chunk_done) {
        const int n = (int)items.size();
        if (n <= 0) return;
        const int chunk = std::max(1, std::min(n, 16));
        size_t max_b = 0;
        for (int i = 0; i < n; ++i)
            max_b = std::max(max_b, disk[(size_t)items[i].first * E + items[i].second].slot_bytes());
        std::vector<uint8_t*> bufs(chunk, nullptr);
        for (int c = 0; c < chunk; ++c) {
            bufs[c] = alloc_fn(max_b);
            if (!bufs[c]) {
                for (int k = 0; k < c; ++k) free_fn(bufs[k]);
                throw std::bad_alloc();
            }
        }
        try {
            std::vector<int> order(n);
            for (int i = 0; i < n; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const auto& da = disk[(size_t)items[a].first * E + items[a].second];
                const auto& db = disk[(size_t)items[b].first * E + items[b].second];
                if (da.file_idx[0] != db.file_idx[0]) return da.file_idx[0] < db.file_idx[0];
                return da.off[0] < db.off[0];
            });
            for (int base = 0; base < n; base += chunk) {
                const int m = std::min(chunk, n - base);
                io_pool.parallel_for(m, [&](int j) {
                    const auto& it = items[order[base + j]];
                    read_expert_from_disk(it.first, it.second, bufs[j]);
                });
                for (int j = 0; j < m; ++j) {
                    const int orig = order[base + j];
                    const auto& it = items[orig];
                    on_ready(orig, bufs[j], disk[(size_t)it.first * E + it.second].slot_bytes());
                }
                on_chunk_done();
            }
        } catch (...) {
            for (auto* b : bufs)
                if (b) free_fn(b);
            throw;
        }
        for (auto* b : bufs)
            if (b) free_fn(b);
    }

    int max_stage_batch() const { return stage_bufs.empty() ? 0 : (int)stage_bufs.size(); }
    bool pool_needs_stage(int l, int batch_n) const {
        if (!active || l < 0 || l >= n_layers) return false;
        return pools[layer_pool[l]].n_slots < batch_n;
    }

    // Allocate or evict a RAM slot in pool `p` (never evicts slots touched in current `cur_tick`).
    // Returns -1 if pool has 0 slots or every slot is pinned in the current batch.
    int alloc_ram_slot(Pool& P, uint64_t cur_tick) {
        if (P.n_slots <= 0) return -1;
        // Pass 1: find an empty slot or run LFU-clock scan.
        for (int pass = 0; pass < 9; ++pass) {
            for (int step = 0; step < P.n_slots; ++step) {
                const int s = P.clock_hand;
                P.clock_hand = (P.clock_hand + 1) % P.n_slots;
                if (P.owner[s] < 0) return s;
                if (P.stamp[s] == cur_tick) continue;
                if (P.freq[s] == 0) {
                    if (P.fence[s] != 0 && wait_fence_fn) {
                        wait_fence_fn(P.fence[s]);
                        P.fence[s] = 0;
                    }
                    loc[P.owner[s]] = -1;
                    P.owner[s] = -1;
                    return s;
                }
                --P.freq[s];
            }
        }
        return -1;
    }

    // Batch fetch `n` experts (all VRAM misses) into host buffers and invoke `on_ready(i, host_ptr, slot_bytes)`
    // for each item so the caller can issue `cudaMemcpyAsync` immediately.
    // RAM hits are dispatched first without waiting for disk; disk misses are read in parallel!
    // Returns true if any temporary staging buffer was used (caller must synchronize stream before next batch).
    struct BatchItem {
        int l = 0;
        int e = 0;
    };

    template <typename ReadyFn>
    bool fetch_batch(const BatchItem* items, int n, uint64_t fence_seq, ReadyFn&& on_ready) {
        if (n <= 0) return false;
        const uint64_t cur_tick = ++tick;

        struct DiskJob {
            int item_idx;
            int l, e;
            int pool_idx;
            int slot;        // >= 0 if placed in RAM pool, < 0 if using staging buffer
            bool dup;        // duplicate of an earlier job in the same batch (already read)
            uint8_t* dst;
            size_t bytes;
        };
        std::vector<DiskJob> disk_jobs;
        disk_jobs.reserve(n);
        int stage_next = 0;

        // Pass 1: Serve RAM hits immediately and mark them protected for cur_tick.
        for (int i = 0; i < n; ++i) {
            const int l = items[i].l, e = items[i].e;
            const int idx = l * E + e;
            const int p = layer_pool[l];
            Pool& P = pools[p];
            const int s = loc[idx];
            if (s >= 0) {
                ++ram_hits;
                P.stamp[s] = cur_tick;
                if (P.freq[s] < 7) ++P.freq[s];
                P.fence[s] = fence_seq;
                on_ready(i, P.data + (size_t)s * P.slot_bytes, disk[idx].slot_bytes());
            } else {
                disk_jobs.push_back({i, l, e, p, -1, false, nullptr, disk[idx].slot_bytes()});
            }
        }
        if (disk_jobs.empty()) return false;

        // Pass 2: Assign RAM slots (or staging buffers if pool has 0/saturated slots) for all disk misses.
        bool used_stage = false;
        for (auto& J : disk_jobs) {
            Pool& P = pools[J.pool_idx];
            const int idx = J.l * E + J.e;
            // Deduplicate if the same expert appears twice in one batch.
            if (loc[idx] >= 0) {
                J.slot = loc[idx];
                J.dup = true;
                J.dst = P.data + (size_t)J.slot * P.slot_bytes;
                continue;
            }
            const int s = alloc_ram_slot(P, cur_tick);
            if (s >= 0) {
                P.owner[s] = idx;
                P.stamp[s] = cur_tick;
                P.freq[s] = 1;
                P.fence[s] = fence_seq;
                loc[idx] = s;
                J.slot = s;
                J.dst = P.data + (size_t)s * P.slot_bytes;
            } else {
                if (stage_bufs.empty() || stage_next >= (int)stage_bufs.size())
                    throw std::runtime_error("HostExpertStore: staging buffer exhausted (chunk batch <= max_stage)");
                J.slot = -1;
                J.dst = stage_bufs[stage_next++];
                used_stage = true;
            }
        }

        // Pass 3: Read all unique disk-cold experts in parallel with DirectFile (sorted by offset).
        std::vector<int> read_idx;
        read_idx.reserve(disk_jobs.size());
        for (int j = 0; j < (int)disk_jobs.size(); ++j)
            if (!disk_jobs[j].dup) read_idx.push_back(j);
        std::sort(read_idx.begin(), read_idx.end(), [&](int a, int b) {
            const auto& da = disk[(size_t)disk_jobs[a].l * E + disk_jobs[a].e];
            const auto& db = disk[(size_t)disk_jobs[b].l * E + disk_jobs[b].e];
            if (da.file_idx[0] != db.file_idx[0]) return da.file_idx[0] < db.file_idx[0];
            return da.off[0] < db.off[0];
        });
        io_pool.parallel_for((int)read_idx.size(), [&](int k) {
            const auto& J = disk_jobs[read_idx[k]];
            read_expert_from_disk(J.l, J.e, J.dst);
        });

        // Pass 4: Dispatch H2D copies for all disk-loaded experts and update tier counters.
        for (const auto& J : disk_jobs) {
            if (!J.dup) {
                ++disk_hits;
                disk_bytes += J.bytes;
            } else {
                ++ram_hits;
            }
            on_ready(J.item_idx, J.dst, J.bytes);
        }
        return used_stage;
    }
};

} // namespace ik2::tier
