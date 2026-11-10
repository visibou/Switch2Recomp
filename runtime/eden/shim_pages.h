// One bit per 4 KiB guest page, set and read from several threads.
#pragma once
#include <atomic>
#include <memory>
#include "common/common_types.h"

struct ShimPages {
    std::unique_ptr<std::atomic<u64>[]> bits;
    u64 pages = 0;

    void Init(u64 mem_size) {
        pages = mem_size >> 12;
        bits.reset(new std::atomic<u64>[(pages + 63) / 64]());
    }
    bool Test(u64 page) const {
        return page < pages && (bits[page >> 6].load(std::memory_order_relaxed) >> (page & 63)) & 1;
    }
    static void Set(ShimPages& p, u64 addr, u64 size, bool on) {
        if (!p.bits) return;
        const u64 first = addr >> 12, last = std::min((addr + size - 1) >> 12, p.pages - 1);
        for (u64 g = first; g <= last; ++g) {
            const u64 m = 1ull << (g & 63);
            if (on) p.bits[g >> 6].fetch_or(m, std::memory_order_relaxed);
            else p.bits[g >> 6].fetch_and(~m, std::memory_order_relaxed);
        }
    }
};

extern ShimPages g_shim_cached;

// 64 KiB chunks that ever held a cached page: the write watch is only read over these (reading it over the whole
// 128 GiB reservation cost ~50 ms per submission; with 2 MiB chunks it read ~2 GB, growing, per submission).
inline constexpr u64 ShimChunkBits = 16;
extern std::unique_ptr<std::atomic<u64>[]> g_shim_cached_chunks;
extern u64 g_shim_nchunks;

// CPU writes the buffer cache must see, handed to the GPU thread by System::GatherGPUDirtyMemory.
void ShimCollectDirty(u64 addr, u64 size);
