// The part of Eden's core that video_core calls, answered by our runtime instead of Eden's emulator.
// Step 1 (link test): guest memory is a plain reservation; later it becomes the runtime's g_mem.
#include <chrono>
#include <optional>
#include <windows.h>
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/frontend/emu_window.h"
#include "core/hle/dlss_trace.h"
#include "core/hle/kernel/board/nintendo/nx/k_system_control.h"
#include "core/memory.h"
#include "core/perf_stats.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"

// Physical memory the GPU side sees. Device addresses map onto offsets in this buffer.
u8* g_shim_mem_base = nullptr;
size_t g_shim_mem_size = 0x200000000ull; // 8 GiB reserved, committed on demand by the runtime
// The GPU that VideoCore::CreateGPU fills (one System per process).
std::optional<Tegra::GPU> g_shim_gpu;

namespace Common {
class HostMemory::Impl {};
HostMemory::HostMemory(size_t backing_size_, size_t virtual_size_)
    : backing_size{backing_size_}, virtual_size{virtual_size_} {
    if (!g_shim_mem_base)
        g_shim_mem_base = static_cast<u8*>(VirtualAlloc(nullptr, g_shim_mem_size, MEM_RESERVE, PAGE_NOACCESS));
    backing_base = g_shim_mem_base;
}
HostMemory::~HostMemory() = default;
} // namespace Common

namespace Core {
DeviceMemory::DeviceMemory() : buffer{g_shim_mem_size, 0} {}
DeviceMemory::~DeviceMemory() = default;

namespace Memory {
struct Memory::Impl {};
Memory::Memory(Core::System& system_) : system{system_} {}
Memory::~Memory() = default;
u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    return g_shim_mem_base + GetInteger(vaddr);
}
void Memory::RasterizerMarkRegionCached(Common::ProcessAddress, u64, bool) {}
} // namespace Memory

namespace Timing {
// As in Eden's core_timing.cpp (the queue is never used here: video_core only reads the GPU clock).
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

PerfStats::PerfStats(u64 title_id_) : title_id{title_id_} {}
PerfStats::~PerfStats() = default;
void PerfStats::EndGameFrame() {}

namespace HLE::DlssTrace {
std::atomic<u64> g_watch_address{0};
std::atomic<const unsigned char*> g_watch_host{nullptr};
void WriteWatch(const char*, const void*, size_t) {}
} // namespace HLE::DlssTrace

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
void System::GatherGPUDirtyMemory(std::function<void(PAddr, size_t)>&) {}
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
std::size_t KSystemControl::Init::GetIntendedMemorySize() {
    return g_shim_mem_size;
}
} // namespace Kernel::Board::Nintendo::Nx
