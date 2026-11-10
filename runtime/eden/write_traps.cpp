// CPU writes to guest memory that Eden caches, found by page protection instead of the write watch.
//
// Pages Eden caches (buffers, textures, shaders: RasterizerMarkRegionCached) are made read-only. The first write to
// one faults: the handler notes the page as dirty and makes it writable again, so a page costs one fault per
// submission however often it is written. Each submission takes the dirty pages, Eden invalidates them, and the ones
// still cached are protected again. Reading the write watch over everything Eden caches cost ~10 ms per submission
// (~40 ms a frame in MKW), growing with the cache; this costs what is really written.
//
// Any thread's write is a CPU write here, Eden's own included (its flushes of GPU data into guest memory): Eden
// re-reads those pages, which hold the same data. Windows only; kept apart from Eden's headers (windows.h).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <windows.h>
#include "eden/write_traps.h"

namespace {
using u64 = std::uint64_t;
constexpr u64 PageBits = 12, PageSize = 1ull << PageBits;

std::uint8_t* g_base = nullptr;
u64 g_pages = 0;
std::unique_ptr<std::atomic<u64>[]> g_cached;    // Eden caches the page
std::unique_ptr<std::atomic<u64>[]> g_trapped;   // read-only by us right now
std::mutex g_lock;                               // protection changes and the dirty list (not taken in the handler)

// Dirty pages, written by the fault handler without locks. Past its capacity, everything trapped is invalidated.
constexpr u64 DirtyCap = 1u << 20;
std::unique_ptr<std::atomic<u64>[]> g_dirty;
std::atomic<u64> g_ndirty{0};
std::atomic<bool> g_overflow{false};
std::atomic<u64> g_stat_faults{0}, g_stat_protected{0}, g_stat_overflows{0};

bool Test(const std::unique_ptr<std::atomic<u64>[]>& bits, u64 page) {
    return (bits[page >> 6].load(std::memory_order_acquire) >> (page & 63)) & 1;
}
void Set(const std::unique_ptr<std::atomic<u64>[]>& bits, u64 page, bool on) {
    const u64 m = 1ull << (page & 63);
    if (on) bits[page >> 6].fetch_or(m, std::memory_order_acq_rel);
    else bits[page >> 6].fetch_and(~m, std::memory_order_acq_rel);
}
// Clears the bit; true if it was set (exactly one thread wins a page).
bool TakeBit(const std::unique_ptr<std::atomic<u64>[]>& bits, u64 page) {
    const u64 m = 1ull << (page & 63);
    return (bits[page >> 6].fetch_and(~m, std::memory_order_acq_rel) & m) != 0;
}

// Read-only for each run of cached, committed, untrapped pages in [first, last]. Under g_lock. One VirtualQuery per
// memory region, not per page (Eden marks gigabytes cached).
void ProtectRange(u64 first, u64 last) {
    u64 run = 0, count = 0;
    const auto flush = [&] {
        if (!count) return;
        DWORD old;
        VirtualProtect(g_base + (run << PageBits), count << PageBits, PAGE_READONLY, &old);
        g_stat_protected += count;
        count = 0;
    };
    u64 region_end = 0;   // first page past the region last queried
    bool committed = false;
    for (u64 p = first; p <= last; ++p) {
        if (p >= region_end) {
            MEMORY_BASIC_INFORMATION mi{};
            if (!VirtualQuery(g_base + (p << PageBits), &mi, sizeof mi)) { flush(); break; }
            committed = mi.State == MEM_COMMIT;
            region_end = (u64(static_cast<std::uint8_t*>(mi.BaseAddress) - g_base) + mi.RegionSize) >> PageBits;
            if (!committed) { flush(); p = region_end - 1; continue; }
        }
        if (!Test(g_cached, p) || Test(g_trapped, p)) { flush(); continue; }
        Set(g_trapped, p, true);
        if (!count) run = p;
        ++count;
    }
    flush();
}

LONG WINAPI OnFault(EXCEPTION_POINTERS* e) {
    const auto* r = e->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;   // only writes
    const auto* a = reinterpret_cast<const std::uint8_t*>(r->ExceptionInformation[1]);
    if (a < g_base || a >= g_base + (g_pages << PageBits)) return EXCEPTION_CONTINUE_SEARCH;
    const u64 page = u64(a - g_base) >> PageBits;
    if (TakeBit(g_trapped, page)) {
        g_stat_faults.fetch_add(1, std::memory_order_relaxed);
        const u64 i = g_ndirty.fetch_add(1, std::memory_order_acq_rel);
        if (i < DirtyCap) g_dirty[i].store(page, std::memory_order_release);
        else g_overflow.store(true, std::memory_order_release);
        DWORD old;
        VirtualProtect(g_base + (page << PageBits), PageSize, PAGE_READWRITE, &old);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // Another thread untrapped it a moment ago: the page is writable again (or about to be)
    MEMORY_BASIC_INFORMATION mi;
    if (Test(g_cached, page) && VirtualQuery(g_base + (page << PageBits), &mi, sizeof mi) &&
        mi.State == MEM_COMMIT && mi.Protect == PAGE_READWRITE)
        return EXCEPTION_CONTINUE_EXECUTION;
    return EXCEPTION_CONTINUE_SEARCH;
}
} // namespace

void WriteTrapsInit(void* base, u64 size) {
    g_base = static_cast<std::uint8_t*>(base);
    g_pages = size >> PageBits;
    const u64 words = (g_pages + 63) / 64;
    g_cached.reset(new std::atomic<u64>[words]());
    g_trapped.reset(new std::atomic<u64>[words]());
    g_dirty.reset(new std::atomic<u64>[DirtyCap]());
    AddVectoredExceptionHandler(1, OnFault);
}

void WriteTrapsSetCached(u64 addr, u64 size, bool cached) {
    if (!g_base || !size || addr >= (g_pages << PageBits)) return;
    const u64 first = addr >> PageBits, last = std::min((addr + size - 1) >> PageBits, g_pages - 1);
    std::scoped_lock lk{g_lock};
    if (cached) {
        for (u64 p = first; p <= last; ++p) Set(g_cached, p, true);
        ProtectRange(first, last);
        // Eden read these pages just before telling us: a write between its read and the protection above would be
        // lost. The write watch (the guest memory is reserved with it) has every write before the protection: those
        // pages count as written (re-reading one written earlier is harmless). A write after the protection faults.
        static PVOID addrs[4096];
        for (u64 p = first; p <= last;) {
            const u64 n = std::min<u64>(last - p + 1, 1u << 16);
            for (;;) {   // each call with the reset flag hands out the next pages
                ULONG_PTR count = 4096;
                ULONG gran = 0;
                if (GetWriteWatch(WRITE_WATCH_FLAG_RESET, g_base + (p << PageBits), n << PageBits, addrs, &count, &gran) != 0)
                    break;
                for (ULONG_PTR k = 0; k < count; ++k) {
                    const u64 page = u64(static_cast<std::uint8_t*>(addrs[k]) - g_base) >> PageBits;
                    const u64 i = g_ndirty.fetch_add(1, std::memory_order_acq_rel);
                    if (i < DirtyCap) g_dirty[i].store(page, std::memory_order_release);
                    else g_overflow.store(true, std::memory_order_release);
                }
                if (count < 4096) break;
            }
            p += n;
        }
        return;
    }
    u64 run = 0, count = 0;
    const auto flush = [&] {
        if (!count) return;
        DWORD old;
        VirtualProtect(g_base + (run << PageBits), count << PageBits, PAGE_READWRITE, &old);
        count = 0;
    };
    for (u64 p = first; p <= last; ++p) {
        Set(g_cached, p, false);
        if (!TakeBit(g_trapped, p)) { flush(); continue; }
        if (!count) run = p;
        ++count;
    }
    flush();
}

void WriteTrapsCollect(void (*invalidate)(u64 addr, u64 size, void* user), void* user) {
    if (!g_base) return;
    std::vector<u64> pages;
    bool overflow;
    {
        std::scoped_lock lk{g_lock};
        const u64 n = std::min<u64>(g_ndirty.exchange(0, std::memory_order_acq_rel), DirtyCap);
        overflow = g_overflow.exchange(false, std::memory_order_acq_rel);
        pages.reserve(n);
        for (u64 i = 0; i < n; ++i) pages.push_back(g_dirty[i].load(std::memory_order_acquire));
    }
    if (overflow) {   // too many to list: every cached page that is not trapped any more was written
        ++g_stat_overflows;
        pages.clear();
        for (u64 p = 0; p < g_pages; ++p)
            if (Test(g_cached, p) && !Test(g_trapped, p)) pages.push_back(p);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    // Protect again first, then let Eden re-read: a write after the protection faults and is in the next
    // collection; protecting after the invalidation lost a write made in between (Eden had re-read the page
    // already, the page was writable: MKW hung on stale data).
    {
        std::scoped_lock lk{g_lock};
        for (size_t i = 0; i < pages.size();) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j - 1] + 1) ++j;
            ProtectRange(pages[i], pages[j - 1]);
            i = j;
        }
    }
    for (size_t i = 0; i < pages.size();) {   // coalesce runs
        size_t j = i + 1;
        while (j < pages.size() && pages[j] == pages[j - 1] + 1) ++j;
        invalidate(pages[i] << PageBits, u64(j - i) << PageBits, user);
        i = j;
    }
}

void WriteTrapsStats(std::uint64_t* faults, std::uint64_t* protected_pages, std::uint64_t* overflows) {
    *faults = g_stat_faults.load();
    *protected_pages = g_stat_protected.load();
    *overflows = g_stat_overflows.load();
}
