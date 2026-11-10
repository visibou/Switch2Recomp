// The part of Eden's core that video_core calls, answered by the runtime instead of Eden's emulator.
// Guest memory: the runtime keeps the game's whole address space in one reservation (g_mem, guest address =
// offset), so Eden's "physical" memory is that reservation and every device address maps 1:1 onto it.
#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#if !SWR_EDEN_EGLANTINE
#include "core/hle/dlss_trace.h"
#endif
#include "core/hle/kernel/board/nintendo/nx/k_system_control.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"

#include "eden/shim_pages.h"
#include "eden/write_traps.h"

u8* g_shim_mem_base = nullptr;   // set by eden_gpu_start before the System is made
size_t g_shim_mem_size = 0;
std::optional<Tegra::GPU> g_shim_gpu;   // filled by VideoCore::CreateGPU (one System per process)
ShimPages g_shim_cached;          // guest pages Eden caches (see RasterizerMarkRegionCached)
std::mutex g_shim_dirty_lock;
std::vector<std::pair<u64, u64>> g_shim_dirty;   // CPU-written ranges for the GPU thread
void ShimCollectDirty(u64 addr, u64 size) {
    std::scoped_lock lk{g_shim_dirty_lock};
    if (!g_shim_dirty.empty() && g_shim_dirty.back().first + g_shim_dirty.back().second == addr)
        g_shim_dirty.back().second += size;
    else
        g_shim_dirty.emplace_back(addr, size);
}
std::unique_ptr<std::atomic<u64>[]> g_shim_cached_chunks;
u64 g_shim_nchunks = 0;

namespace Common {
class HostMemory::Impl {};
HostMemory::HostMemory(size_t backing_size_, size_t virtual_size_)
    : backing_size{backing_size_}, virtual_size{virtual_size_} {
    backing_base = g_shim_mem_base;
}
HostMemory::~HostMemory() = default;
#if SWR_EDEN_EGLANTINE
// Eglantine's working-set hint for rarely read memory: the runtime keeps guest memory as it is.
void HostMemory::HintCold(size_t, size_t, std::optional<u64>) const {}
#endif
} // namespace Common

namespace Core {
DeviceMemory::DeviceMemory() : buffer{g_shim_mem_size, 0} {}
DeviceMemory::~DeviceMemory() = default;

namespace Memory {
struct Memory::Impl {};
Memory::Memory(Core::System& system_) : system{system_} {}
Memory::~Memory() = default;
u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    const u64 a = GetInteger(vaddr);
    return a < g_shim_mem_size ? g_shim_mem_base + a : nullptr;
}
// Eden marks the pages it caches (buffers, textures, shaders): CPU writes to them must reach it as invalidations.
// The bridge checks this map against the runtime's write watch (eden_gpu.cpp).
#if SWR_EDEN_EGLANTINE
u32 Memory::Read32(Common::ProcessAddress vaddr) {
    const u64 a = GetInteger(vaddr);
    u32 v = 0;
    if (a + 4 <= g_shim_mem_size) std::memcpy(&v, g_shim_mem_base + a, 4);
    return v;
}
// Eglantine arms read traps so the next CPU read of GPU-written pages flushes them first (reactive flush). The
// runtime runs the game's code natively and has no such traps yet: CPU reads of GPU output see what is in memory.
void Memory::RasterizerArmReadTraps(Common::ProcessAddress, u64) {}
#endif
void Memory::RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached) {
    const u64 a = GetInteger(vaddr);
    if (!size || a >= g_shim_mem_size) return;
    ShimPages::Set(g_shim_cached, a, std::min<u64>(size, g_shim_mem_size - a), cached);
    WriteTrapsSetCached(a, std::min<u64>(size, g_shim_mem_size - a), cached);
    if (cached && g_shim_cached_chunks) {
        const u64 last = std::min((a + size - 1) >> ShimChunkBits, g_shim_nchunks - 1);
        for (u64 c = a >> ShimChunkBits; c <= last; ++c)
            g_shim_cached_chunks[c >> 6].fetch_or(1ull << (c & 63), std::memory_order_relaxed);
    }
}
} // namespace Memory

namespace Timing {
// As in Eden's core_timing.cpp (the event queue is never used here: video_core only reads the GPU clock).
struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;
    heap_t::handle_type handle{};
    friend bool operator>(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }
    friend bool operator<(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};
CoreTiming::CoreTiming() = default;
CoreTiming::~CoreTiming() = default;
u64 CoreTiming::GetGPUTicks() const {
    // The GPU timer runs at 614.4 MHz: ns * 384 / 625.
    const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch()).count();
    return ns / 625 * 384 + ns % 625 * 384 / 625;
}
} // namespace Timing

#if !SWR_EDEN_EGLANTINE
namespace HLE::DlssTrace {   // Eden-NS2's diagnostic write watch: off
std::atomic<u64> g_watch_address{0};
std::atomic<const unsigned char*> g_watch_host{nullptr};
void WriteWatch(const char*, const void*, size_t) {}
} // namespace HLE::DlssTrace
#endif

PerfStats::PerfStats(u64 title_id_) : title_id{title_id_} {}
PerfStats::~PerfStats() = default;
void PerfStats::EndGameFrame() {}

struct System::Impl {
    explicit Impl(System& system) : memory{system}, perf_stats{0} {}
    Core::DeviceMemory device_memory;
    Memory::Memory memory;
    Timing::CoreTiming core_timing;
    PerfStats perf_stats;
    std::unique_ptr<Tegra::Host1x::Host1x> host1x;
};

System::System() : impl{std::make_unique<Impl>(*this)} {
    impl->host1x = std::make_unique<Tegra::Host1x::Host1x>(*this);
}
System::~System() {
    g_shim_gpu.reset();
    impl->host1x.reset();
}
bool System::IsShuttingDown() const { return false; }
bool System::IsPoweredOn() const { return true; }
void System::GatherGPUDirtyMemory(std::function<void(PAddr, size_t)>& callback) {
    std::vector<std::pair<u64, u64>> ranges;
    {
        std::scoped_lock lk{g_shim_dirty_lock};
        ranges.swap(g_shim_dirty);
    }
    for (const auto& [a, n] : ranges) callback(a, n);
}
Memory::Memory& System::ApplicationMemory() { return impl->memory; }
const Memory::Memory& System::ApplicationMemory() const { return impl->memory; }
Tegra::GPU& System::GPU() { return *g_shim_gpu; }
const Tegra::GPU& System::GPU() const { return *g_shim_gpu; }
Tegra::Host1x::Host1x& System::Host1x() { return *impl->host1x; }
const Tegra::Host1x::Host1x& System::Host1x() const { return *impl->host1x; }
DeviceMemory& System::DeviceMemory() { return impl->device_memory; }
const DeviceMemory& System::DeviceMemory() const { return impl->device_memory; }
Timing::CoreTiming& System::CoreTiming() { return impl->core_timing; }
const Timing::CoreTiming& System::CoreTiming() const { return impl->core_timing; }
PerfStats& System::GetPerfStats() { return impl->perf_stats; }
const PerfStats& System::GetPerfStats() const { return impl->perf_stats; }
void System::RegisterHostThread() {}
} // namespace Core

namespace Kernel::Board::Nintendo::Nx {
// Sizes Eden's device-address -> physical table: all of guest memory.
std::size_t KSystemControl::Init::GetIntendedMemorySize() {
    return g_shim_mem_size;
}
} // namespace Kernel::Board::Nintendo::Nx
