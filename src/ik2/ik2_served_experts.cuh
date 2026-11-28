// GPU-served expert tables shared by startup and token graph replay.
struct ServedLayer {
    ik2::ServedSlots slots;
    uint8_t *cold, *cache;
    size_t bytes;
    int offset;
};

struct ServedExperts {
    std::vector<ServedLayer> layers;
    const uint8_t **device, **host;
    int *records, *published;
    int entries = 0;
    uint64_t misses = 0, copies = 0, refills = 0, loop_reuses = 0;

    void build_table(Layer& ly, ServedLayer& s, int E) {
        auto tab = host + s.offset;
        s.slots.write_table(tab, E, s.cold, s.cache, s.bytes);
        ly.tab_h.assign(tab, tab + E);
    }

    void init(Model& m, size_t budget) {
        const auto& c = m.c;
        entries = c.n_layer * c.E + c.n_visit * c.K;
        device = dalloc<const uint8_t*>(entries);
        CU(cudaMallocHost(&host, entries * sizeof(*host)));
        records = dalloc<int>(c.n_visit * c.K);
        CU(cudaMallocHost(&published, c.n_visit * c.K * sizeof(int)));
        size_t reserve_bytes = 0, layer_sum = 0;
        for (auto& v : m.V) reserve_bytes += c.K * m.L[v.l].slot_bytes;
        for (auto& ly : m.L) layer_sum += ly.slot_bytes;
        if (budget < reserve_bytes) die("--cache-gb cannot hold the per-visit expert reserves");
        const int hot = int(std::min<size_t>(c.E, (budget - reserve_bytes) / layer_sum));
        int offset = 0;
        for (int l = 0; l < c.n_layer; ++l) {
            auto& ly = m.L[l];
            int reserve = c.K * (l >= c.core0 && l < c.core1 ? c.n_pass : 1);
            int n = hot + reserve;
            uint8_t* cold;
            CU(cudaHostGetDevicePointer(&cold, const_cast<uint8_t*>(ly.host[0]), 0));
            if (ly.slot_bytes % 16) die("GPU-served experts require 16-byte aligned expert blobs");
            layers.push_back({ik2::ServedSlots(n, reserve), cold, dalloc<uint8_t>(n * ly.slot_bytes), ly.slot_bytes, offset});
            auto& s = layers.back();
            if (hot) CU(cudaMemcpy(s.cache, ly.host[0], hot * s.bytes, cudaMemcpyHostToDevice));
            build_table(ly, s, c.E);
            ly.tab_blob = device + offset;
            for (int r = 0; r < 3; ++r)
                if (!ly.scale[r].empty()) ly.tab_sc[r] = dev_f32(ly.scale[r]);
            offset += c.E + reserve;
        }
        CU(cudaMemcpy(device, host, entries * sizeof(*host), cudaMemcpyHostToDevice));
        std::printf("GPU-served experts: %d hot slots per layer, %d visit reserves\n", hot, c.n_visit * c.K);
    }

    void finish(Model& m, Work& w) {
        const auto& c = m.c;
        if (w.use.empty()) w.use.assign(c.n_layer * c.E, 0), w.use_tot.assign(c.n_layer, 0);
        bool changed = false;
        std::vector<bool> dirty(c.n_layer);
        for (int v = 0; v < c.n_visit; ++v) {
            int l = m.V[v].l;
            auto& slots = layers[l].slots;
            for (int k = 0; k < c.K; ++k) {
                int record = published[v * c.K + k];
                int e = record < 0 ? ~record : record;
                ++w.use[l * c.E + e];
                if (record >= 0 && m.L[l].tab_h[e] == layers[l].cold + e * layers[l].bytes)
                    ++loop_reuses;
                if (record < 0) {
                    ++misses;
                    ++copies;
                    slots.owner[slots.reserve[m.V[v].pass * c.K + k]] = e;
                    dirty[l] = changed = true;
                }
            }
            w.use_tot[l] += c.K;
        }
        if (!changed) return;
        for (int l = 0; l < c.n_layer; ++l) {
            if (!dirty[l]) continue;
            auto& s = layers[l];
            for (int slot : s.slots.reserve) refills += s.slots.owner[slot] >= 0;
            s.slots.refill(w.use.data() + l * c.E);
            build_table(m.L[l], s, c.E);
        }
        CU(cudaMemcpyAsync(device, host, entries * sizeof(*host), cudaMemcpyHostToDevice, g_s));
    }
};

extern ServedExperts served;
