// C bridge between the runtime and Eden's GPU (video_core). See eden_gpu.h.
//
// How the runtime's GPU side maps onto Eden (the same calls Eden's own nvdrv makes):
// - one GPU address space (the runtime has one GMMU) = one Tegra::MemoryManager. Device address = guest address,
//   so a GMMU map is a device map of the guest range (identity) plus a GMMU map onto it.
// - one runtime channel (keyed by its syncpoint) = one Eden channel, Switch 2 ("native") ABI.
// - a submission = the GPFIFO entries, then the channel's syncpoint increments after the work (with wait for idle).
//   The runtime's syncpoint counters stay the game-visible ones: every increment Eden completes (its "host" value,
//   reached when the GPU work is done) raises the runtime's counter by one.
// - CPU writes: Eden reports the guest pages it caches; dirty pages from the runtime's write watch that are cached
//   are invalidated before each submission.
// - present: the guest image the game presents, as Eden's framebuffer layer (Switch 2 block-linear tiling).
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "eden/eden_gpu.h"
#include "eden/shim_pages.h"
#include "eden/write_traps.h"
#include "trace.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/puller.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/host1x/host1x.h"
#include "video_core/nx2.h"
#if !SWR_EDEN_EGLANTINE
#include "video_core/macro.h"
#endif
#include "video_core/memory_manager.h"
#include "video_core/pte_kind.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

extern u8* g_shim_mem_base;
extern size_t g_shim_mem_size;
extern std::optional<Tegra::GPU> g_shim_gpu;

extern "C" void syncpt_raise(u32 id, u32 value);                       // runtime (nvdrv.c)
extern "C" u32 syncpt_read(u32 id);
extern "C" int plat_ww_collect(u8* base, u64 size, void (*cb)(u64));   // runtime (plat.c)
extern "C" int plat_ww_query(u8* base, u64 size, int reset, void (*cb)(u64));
extern "C" void plat_ww_reset(u8* base, u64 size);

namespace {
class GameWindow final : public Core::Frontend::EmuWindow {
public:
    GameWindow(void* hwnd, u32 w, u32 h) {
        window_info.type = Core::Frontend::WindowSystemType::Windows;
        window_info.render_surface = hwnd;
        UpdateCurrentFramebufferLayout(w, h);
    }
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<Core::Frontend::GraphicsContext>();   // Vulkan: no context to share
    }
    bool IsShown() const override { return true; }
    void OnFrameDisplayed() override { ++displayed; }
    std::atomic<u64> displayed{0};
};

std::unique_ptr<GameWindow> g_window;
std::unique_ptr<Core::System> g_system;
u64 g_program_id = 0;

std::mutex g_lock;                         // the runtime calls from many threads; keeps maps and submits in order
std::unique_ptr<Tegra::MemoryManager> g_gmmu;
Core::Asid g_asid{};
ShimPages g_dev_mapped;                    // guest pages already in Eden's device page table
std::unordered_map<u32, std::shared_ptr<Tegra::Control::ChannelState>> g_channels;
int g_ww = -1;                             // write watch usable (plat_ww_collect)
// CPU writes: the write watch (default), read once per batch of queued submissions (QueueLoop), or write traps
// (SWR_EDEN_WRITE_TRAPS=1, experimental: MKW still hangs with them, a write is missed somewhere).
// File reads into trapped pages go through a bounce buffer (plat_pread).
const bool g_use_write_watch = std::getenv("SWR_EDEN_WRITE_TRAPS") == nullptr;
u64 g_stat_submits = 0, g_stat_inval = 0, g_stat_ww_ns = 0, g_stat_submit_ns = 0, g_stat_ww_bytes = 0, g_stat_batches = 0;

// Eden keeps its log and caches in the paths below; point them all at data_dir so nothing goes to %APPDATA%.
// SetEdenPath ignores a folder that does not exist yet, so the folders are made first.
bool SetPaths(const std::filesystem::path& dir) {
    using Common::FS::EdenPath;
    std::error_code ec;
    std::filesystem::create_directories(dir / "log", ec);
    std::filesystem::create_directories(dir / "cache" / "shader", ec);
    if (!std::filesystem::is_directory(dir / "log") || !std::filesystem::is_directory(dir / "cache" / "shader"))
        return false;
    for (auto p : {EdenPath::EdenDir, EdenPath::CrashDumpsDir, EdenPath::DumpDir, EdenPath::LoadDir,
                   EdenPath::ScreenshotsDir, EdenPath::PostPresetDir, EdenPath::PostShaderDir,
                   EdenPath::ConfigDir}) {
        Common::FS::SetEdenPath(p, dir);
    }
    Common::FS::SetEdenPath(EdenPath::LogDir, dir / "log");
    Common::FS::SetEdenPath(EdenPath::CacheDir, dir / "cache");
    Common::FS::SetEdenPath(EdenPath::ShaderDir, dir / "cache" / "shader");
    return Common::FS::GetEdenPath(EdenPath::LogDir) == dir / "log";
}

// Device address = guest address: put the guest pages of [addr, addr+size) into Eden's device page table once.
void DeviceMap(u64 addr, u64 size) {
    auto& smmu = g_system->Host1x().MemoryManager();
    const u64 end = std::min<u64>((addr + size + 0xFFF) & ~0xFFFull, g_shim_mem_size);
    u64 run = 0, run_start = 0;
    for (u64 a = addr & ~0xFFFull; a < end; a += 0x1000) {
        if (!g_dev_mapped.Test(a >> 12)) {
            if (!run) run_start = a;
            run += 0x1000;
            continue;
        }
        if (run) smmu.Map(run_start, run_start, run, g_asid);
        run = 0;
    }
    if (run) smmu.Map(run_start, run_start, run, g_asid);
    ShimPages::Set(g_dev_mapped, addr, end > addr ? end - addr : 0, true);
}

// CPU writes to guest pages Eden caches -> invalidate. Reading the write watch costs ~2-3 ms per GB on Windows (it walks
// the range; several threads at once are no faster), and MKW's cached memory is ~1.3 GB: read before
// every batch of submissions it held Eden's GPU thread ~20 ms a frame. So it is read two ways:
// - before each batch (InvalidateDirty, with g_lock, on the critical path): only the hot chunks, those where a write
//   was found in the last HotScans full reads, read and reset;
// - once per present on its own thread (FullScanLoop): everything cached, read without reset; the pages found are
//   reset and then invalidated together under g_lock (a write after the reset is seen next time, one before it is in
//   what Eden reads after the invalidation). It also finds the chunks that turn hot.
// A write to a cold chunk Eden already caches is invalidated by the next full read (about a frame later); memory Eden
// has not cached yet is read as it is whenever Eden first uses it. SWR_EDEN_FULL_WW=1: everything before each batch.
// 2 MiB chunks of GPU-only memory (Switch 2 sparse regions: render targets, texture pools). The CPU cannot write
// those on the console, so the write watch is not read there (MKW has ~25 GB of them, most of what
// Eden caches: reading the write watch over them cost ~10 ms a submission).
std::unique_ptr<std::atomic<u64>[]> g_gpu_only_chunks;
std::unique_ptr<std::atomic<u32>[]> g_chunk_hot;   // full read index when a write was last found in the chunk
std::atomic<u32> g_full_scans{1};
constexpr u32 HotScans = 120;
const bool g_full_ww = std::getenv("SWR_EDEN_FULL_WW") != nullptr;
u64 g_stat_hot_bytes = 0, g_stat_full_ns = 0, g_stat_full = 0, g_stat_full_bytes = 0;

struct DirtyCollector {
    std::vector<std::pair<u64, u64>> dirty;   // coalesced guest ranges Eden caches
    u64 base = 0;                             // guest address of the chunk run being read
    u32 scan = 0;                             // full read index, for g_chunk_hot
    bool count_late = false;                  // full read: count cached chunks written while cold
    u64 late = 0;                             // ... found (Eden may have used their old contents for a frame)
};
u64 g_stat_late = 0;
thread_local DirtyCollector* t_collect = nullptr;

void CollectDirty(u64 off) {
    DirtyCollector& c = *t_collect;
    off += c.base;
    const u64 ch = off >> ShimChunkBits;
    const u32 prev = g_chunk_hot[ch].load(std::memory_order_relaxed);
    g_chunk_hot[ch].store(c.scan, std::memory_order_relaxed);
    if (!g_shim_cached.Test(off >> 12)) return;
    if (c.count_late && (prev == 0 || c.scan - prev > HotScans)) ++c.late;
    if (!c.dirty.empty() && c.dirty.back().first + c.dirty.back().second == off) c.dirty.back().second += 0x1000;
    else c.dirty.emplace_back(off, 0x1000);
}

// Reads the write watch over each run of chunks that are cached, not GPU-only and pass `want`. Returns false when
// the write watch failed.
template <typename Want>
bool ScanChunks(DirtyCollector& c, bool reset, u64& bytes, Want&& want) {
    t_collect = &c;
    const auto scan = [&](u64 i) {
        return ((g_shim_cached_chunks[i >> 6].load(std::memory_order_relaxed) &
                 ~g_gpu_only_chunks[i >> 6].load(std::memory_order_relaxed)) >> (i & 63) & 1) && want(i);
    };
    for (u64 ch = 0; ch < g_shim_nchunks;) {   // each run of chunks to read
        const u64 w = g_shim_cached_chunks[ch >> 6].load(std::memory_order_relaxed) &
                      ~g_gpu_only_chunks[ch >> 6].load(std::memory_order_relaxed);
        if (!w) { ch = (ch | 63) + 1; continue; }
        if (!scan(ch)) { ++ch; continue; }
        u64 e = ch + 1;
        // Runs a few chunks apart are read as one: each read has a fixed cost (the hot set is ~100 MiB in a few
        // thousand pieces), and reading the chunks between them costs less than another call.
        constexpr u64 MaxGap = 16;
        for (;;) {
            while (e < g_shim_nchunks && scan(e)) ++e;
            u64 next = e;
            while (next < g_shim_nchunks && next - e < MaxGap && !scan(next)) ++next;
            if (next >= g_shim_nchunks || next - e >= MaxGap) break;
            e = next;
        }
        c.base = ch << ShimChunkBits;
        bytes += (e - ch) << ShimChunkBits;
        if (plat_ww_query(g_shim_mem_base + c.base, (e - ch) << ShimChunkBits, reset, CollectDirty)) return false;
        ch = e;
    }
    return true;
}

void WriteWatchFailed() {
    g_ww = 0;
    LOG_WARNING(HW_GPU, "Switch2Recomp: no write watch, CPU writes to GPU data are not seen");
}

// Before each batch, with g_lock held.
void InvalidateDirty() {
    static const bool off = std::getenv("SWR_EDEN_NO_WW") != nullptr;   // experiment: no CPU-write reports
    if (off) return;
    if (!g_use_write_watch) {   // write traps (write_traps.cpp)
        const auto t0 = std::chrono::steady_clock::now();
        WriteTrapsCollect([](u64 a, u64 n, void*) { g_shim_gpu->InvalidateRegion(a, n); ++g_stat_inval; }, nullptr);
        g_stat_ww_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    if (!g_ww) return;
    static DirtyCollector c;
    c.dirty.clear();
    c.scan = g_full_scans.load(std::memory_order_relaxed);
    const auto t0 = std::chrono::steady_clock::now();
    u64 bytes = 0;
    // Hot chunks and the 16 after each (1 MiB): the game streams per-frame data through ring buffers, and the first
    // write of each lap lands in a chunk gone cold since the last lap (~3 such chunks a frame were found only by the
    // full read, a frame late: Eden could draw that frame with the previous lap's data).
    constexpr u64 Lookahead = 16;
    const auto hot = [&](u64 i) {
        const u32 h = g_chunk_hot[i].load(std::memory_order_relaxed);
        return h != 0 && c.scan - h < HotScans;
    };
    const bool ok = ScanChunks(c, true, bytes, [&](u64 i) {
        if (g_full_ww) return true;
        for (u64 k = 0; k <= Lookahead && k <= i; ++k) {
            if (hot(i - k)) return true;
        }
        return false;
    });
    g_stat_ww_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    g_stat_ww_bytes += bytes;
    g_stat_hot_bytes += bytes;
    if (!ok) return WriteWatchFailed();
    g_ww = 1;
    // Invalidated now, before the submission that may read them. Deferring GPU-modified ranges to the GPU thread
    // (OnCPUWrite + GatherGPUDirtyMemory, as Eglantine's own guest memory does) let a flush of GPU-written command
    // pages copy the GPU's older words over commands the game had just written (MKW stalled at ~140 frames).
    // Eden's own writes into guest memory (flushes of GPU data) are reported too; re-reading them is harmless.
    for (const auto& [a, n] : c.dirty) g_shim_gpu->InvalidateRegion(a, n);
    g_stat_inval += c.dirty.size();
}

// The full read, once per present (FullScanRequest), on its own thread.
std::mutex g_full_lock;
std::condition_variable_any g_full_cv;
bool g_full_wanted = false;
std::jthread g_full_thread;

void FullScanRequest() {
    if (g_full_ww || !g_use_write_watch) return;
    {
        std::scoped_lock lk{g_full_lock};
        g_full_wanted = true;
    }
    g_full_cv.notify_one();
}

void FullScanLoop(std::stop_token stop) {
    swr_trace_thread_name("bridge write watch (full)");
    DirtyCollector c;
    for (;;) {
        {
            std::unique_lock lk{g_full_lock};
            g_full_cv.wait(lk, stop, [] { return g_full_wanted; });
            if (stop.stop_requested()) return;
            g_full_wanted = false;
        }
        if (g_ww != 1) continue;
        const u64 t_trace = swr_trace_ts();
        const auto t0 = std::chrono::steady_clock::now();
        c.dirty.clear();
        c.count_late = true;
        c.late = 0;
        c.scan = g_full_scans.fetch_add(1, std::memory_order_relaxed) + 1;
        u64 bytes = 0;
        const bool ok = ScanChunks(c, false, bytes, [](u64) { return true; });
        {
            std::scoped_lock lk{g_lock};
            if (!ok) {
                WriteWatchFailed();
            } else {
                for (const auto& [a, n] : c.dirty) plat_ww_reset(g_shim_mem_base + a, n);
                for (const auto& [a, n] : c.dirty) g_shim_gpu->InvalidateRegion(a, n);
                g_stat_inval += c.dirty.size();
            }
        }
        g_stat_late += c.late;
        swr_trace_counter("write watch: cold chunks written", static_cast<long long>(c.late));
        g_stat_full_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        g_stat_full_bytes += bytes;
        ++g_stat_full;
        swr_trace_span("full write watch read", t_trace);
    }
}

Tegra::Control::ChannelState& Channel(u32 syncpt) {
    auto& ch = g_channels[syncpt];
    if (!ch) {
        ch = g_shim_gpu->AllocateChannel();
        ch->memory_manager = g_gmmu.get();
#if !SWR_EDEN_EGLANTINE
        // Eden-NS2 marks Switch 2 channels per channel; its NVN2 channels share one TSG's Turing+ macros
        ch->native_abi = true;
        static const auto macros = [] {
            auto m = std::make_shared<Tegra::NativeMacroContext>();
            m->engine.native_turing = true;
            m->compute_engine.native_turing = true;
            return m;
        }();
        ch->native_macro_context = macros;
#endif
        g_shim_gpu->InitChannel(*ch, g_program_id);
        LOG_INFO(HW_GPU, "Switch2Recomp: channel for syncpoint {} -> Eden channel {}", syncpt, ch->bind_id);
    }
    return *ch;
}

Tegra::CommandHeader FenceAction(Tegra::Engines::Puller::FenceOperation op, u32 id) {
    Tegra::Engines::Puller::FenceAction a{};
    a.op.Assign(op);
    a.syncpoint_id.Assign(id);
    return {a.raw};
}
// Syncpoint mirror: every value Eden's GPU reaches (its "host" value: the work is done) raises the runtime's
// counter. One action per syncpoint waits for the next value; it fires under Eden's syncpoint lock, so it only
// queues the id, and this thread reads the value, raises the runtime's counter and waits for the next one.
constexpr u32 NumSyncpoints = 192;   // Host1x syncpoints (SyncpointManager::NUM_MAX_SYNCPOINTS)
std::mutex g_sp_lock;
std::condition_variable g_sp_cv;
std::vector<u32> g_sp_fired;
std::jthread g_sp_thread;

void ArmSyncpoint(u32 id, u32 next) {
    g_system->Host1x().GetSyncpointManager().RegisterHostAction(id, next, [id] {
        std::scoped_lock lk{g_sp_lock};
        g_sp_fired.push_back(id);
        g_sp_cv.notify_one();
    });
}

void SyncpointMirror(std::stop_token stop) {
    auto& sm = g_system->Host1x().GetSyncpointManager();
    std::vector<u32> ids;
    while (!stop.stop_requested()) {
        {
            std::unique_lock lk{g_sp_lock};
            g_sp_cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return !g_sp_fired.empty(); });
            ids.swap(g_sp_fired);
        }
        for (const u32 id : ids) {
            const u32 v = sm.GetHostSyncpointValue(id);
            syncpt_raise(id, v);
            ArmSyncpoint(id, v + 1);
        }
        ids.clear();
    }
}
} // namespace

// Present pacing. The console's compositor frees an image only after its frame's fence, which keeps a game from
// lapping the GPU; here every image is free at once. Each present therefore waits until Eden's GPU thread has
// processed everything submitted up to the previous present, on every channel (Eglantine's nx2:vi2 229 waits on the
// present's own fence only, which covers one channel). MKW alternates two blocks of queue semaphores
// by frame and clears one before reusing it; with only one channel paced, a lagging channel then waited for a
// release the game had already cleared (Eglantine hung at ~140 frames): its channels lag within one frame, so each
// present waits for its own frame on all channels. SWR_EDEN_FRAMES_AHEAD=1: wait for the previous one only (CPU
// and GPU overlap; hangs MKW under Eglantine).
u32 g_expect[NumSyncpoints]{};   // per syncpoint: the guest value once all work submitted so far is processed

// How long Eden's GPU thread and the split's back thread have waited for work since the start (frame pacing).
extern "C" void eden_gpu_wait_ns(unsigned long long* front, unsigned long long* back) {
    *front = VideoCommon::g_swr_wait_ns[0].load(std::memory_order_relaxed);
    *back = VideoCommon::g_swr_wait_ns[1].load(std::memory_order_relaxed);
}

extern "C" void eden_gpu_present_wait(unsigned fence_id, unsigned fence_value) {
    static std::vector<std::pair<u32, u32>> previous;   // (syncpoint, value) of every channel at the last present
    if (!g_shim_gpu) return;
    auto& sm = g_system->Host1x().GetSyncpointManager();
    const auto wait = [&sm](u32 id, u32 value) {
        if (id >= NumSyncpoints || sm.IsReadyGuest(id, value)) return;
        struct Wake { std::mutex m; std::condition_variable cv; bool done = false; };
        auto wake = std::make_shared<Wake>();
        sm.RegisterGuestAction(id, value, [wake] {
            { std::scoped_lock lk{wake->m}; wake->done = true; }
            wake->cv.notify_all();
        });
        std::unique_lock lk{wake->m};
        if (!wake->cv.wait_for(lk, std::chrono::seconds(4), [&] { return wake->done; })) {
            static int said;
            if (said++ < 16) LOG_WARNING(HW_GPU, "Switch2Recomp: present: syncpoint {} not processed to {} in 4 s (at {})", id, value, sm.GetGuestSyncpointValue(id));
        }
    };
    std::vector<std::pair<u32, u32>> now;
    {
        std::scoped_lock lk{g_lock};
        for (const auto& [syncpt, ch] : g_channels)
            if (syncpt < NumSyncpoints) now.emplace_back(syncpt, g_expect[syncpt]);
    }
    // default: Eglantine waits for this frame (its channels lag within a frame), Eden-NS2 overlaps one frame
    static const bool strict = std::getenv("SWR_EDEN_FRAMES_AHEAD") ? std::getenv("SWR_EDEN_FRAMES_AHEAD")[0] == '0'
                                                                    : SWR_EDEN_EGLANTINE != 0;
    for (const auto& [id, value] : strict ? now : previous) wait(id, value);
    (void)fence_id; (void)fence_value;   // covered: the present's fence is on one of the channels
    previous = std::move(now);
}

extern "C" void eden_gpu_syncpt_cpu_incr(unsigned id) {
    if (!g_shim_gpu || id >= NumSyncpoints) return;
    auto& sm = g_system->Host1x().GetSyncpointManager();
    {
        std::scoped_lock lk{g_lock};
        ++g_expect[id];
    }
    sm.IncrementGuest(id);
    sm.IncrementHost(id);
}

namespace {
void StartQueue();   // the submission worker (below)
void StopQueue();
float g_resolution = 1.0f;   // internal render scale (runtime ini `resolution`), set before the start
}

extern "C" void eden_gpu_set_resolution(float scale) {
    g_resolution = scale;
}

extern "C" int eden_gpu_start(void* hwnd, int width, int height, void* guest_mem, unsigned long long guest_size,
                              const char* data_dir, unsigned long long program_id) {
    g_shim_mem_base = static_cast<u8*>(guest_mem);
    g_shim_mem_size = guest_size;
    g_program_id = program_id;
    // never fall back to Eden's default folders (%APPDATA%\eden): no Eden then
    if (!SetPaths(std::filesystem::u8path(data_dir))) return 0;
    // Texture cache thresholds: Eden follows the driver's budget (it shrinks when other programs use video memory),
    // which fixed limits did not (1536/2304 MiB, with other programs open the game went over the budget and the
    // driver stalled). EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB still set them.
    if (const char* f = std::getenv("SWR_EDEN_LOG_FILTER")) Settings::values.log_filter.SetValue(f);   // e.g. "*:Info HW_GPU:Debug"
    Common::Log::Initialize();
    Common::Log::Start();
    // Defaults from Eden-NS2's notes for Switch 2 games: low GPU accuracy, asynchronous shader builds.
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    Settings::values.use_asynchronous_gpu_emulation.SetValue(true);
    Settings::values.use_asynchronous_shaders.SetValue(!std::getenv("SWR_EDEN_SYNC_SHADERS"));   // SWR_EDEN_SYNC_SHADERS=1: build before drawing
    Settings::values.gpu_accuracy.SetValue(Settings::GpuAccuracy::Low);
    Settings::UpdateGPUAccuracy();
    // Eglantine's NX2_NOTES rows 99/107/128 for title 1 (MKW) on 6-8 GB laptop GPUs: ASTC decoded on the GPU and
    // kept as BC3 (RGBA8 is 4x bigger and ran out of VRAM), no reactive flushing (its read traps stuttered).
    Settings::values.accelerate_astc.SetValue(Settings::AstcDecodeMode::Gpu);
    Settings::values.astc_recompression.SetValue(Settings::AstcRecompression::Bc3);
    if (const char* a = std::getenv("SWR_EDEN_ASTC")) {   // cpu | async | gpu, and ",raw" for no BC3
        const std::string v{a};
        Settings::values.accelerate_astc.SetValue(v.starts_with("cpu") ? Settings::AstcDecodeMode::Cpu
                                                  : v.starts_with("async") ? Settings::AstcDecodeMode::CpuAsynchronous
                                                                           : Settings::AstcDecodeMode::Gpu);
        if (v.find("raw") != std::string::npos) Settings::values.astc_recompression.SetValue(Settings::AstcRecompression::Uncompressed);
        if (v.find("bc1") != std::string::npos) Settings::values.astc_recompression.SetValue(Settings::AstcRecompression::Bc1);
    }
    Settings::values.use_reactive_flushing.SetValue(false);
    // the runtime's `resolution` option: Eden's render-target scaling (1/4x .. 2x)
    {
        using R = Settings::ResolutionSetup;
        const float r = g_resolution;
        Settings::values.resolution_setup.SetValue(r <= 0.3f ? R::Res1_4X : r <= 0.6f ? R::Res1_2X : r <= 0.85f ? R::Res3_4X
                                                   : r <= 1.1f ? R::Res1X : r <= 1.3f ? R::Res5_4X : r <= 1.6f ? R::Res3_2X : R::Res2X);
    }
    // the large physical table (device_memory_manager.h, see runtime/eden/patches) is used with this layout
    Settings::values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_12Gb);

    // Switch 2 (NVN2) command streams: Ampere classes, Turing+ macros shared by the process's channels (one TSG),
    // the console's GOB swizzle. Eglantine's nvhost_ctrl_gpu sets this when an NX2 driver asks for the GPU.
#if SWR_EDEN_EGLANTINE
    Tegra::g_nx2_ampere = true;   // Eden-NS2 marks channels instead (native_abi, Channel())
#else
    if (std::getenv("SWR_EDEN_NX2_FLAG")) Tegra::g_nx2_ampere = true;   // experiment
#endif
#if SWR_EDEN_EGLANTINE
    Tegra::Nx2ClearSemaphoreRegions();
#endif
    Tegra::Nx2ClearGpuWrittenPages();
    g_shim_cached.Init(guest_size);
    if (!g_use_write_watch) WriteTrapsInit(guest_mem, guest_size);
    g_shim_nchunks = guest_size >> ShimChunkBits;
    g_shim_cached_chunks.reset(new std::atomic<u64>[(g_shim_nchunks + 63) / 64]());
    g_gpu_only_chunks.reset(new std::atomic<u64>[(g_shim_nchunks + 63) / 64]());
    g_chunk_hot.reset(new std::atomic<u32>[g_shim_nchunks]());
    g_dev_mapped.Init(guest_size);
    g_window = std::make_unique<GameWindow>(hwnd, u32(width), u32(height));
    g_system = std::make_unique<Core::System>();
    VideoCore::CreateGPU(g_shim_gpu, *g_window, *g_system);
    if (!g_shim_gpu) {
        g_system.reset();
        g_window.reset();
        return 0;
    }
    g_asid = g_system->Host1x().MemoryManager().RegisterProcess(&g_system->ApplicationMemory());
    // the runtime's GMMU: 40-bit GPU addresses, 4 KiB pages (Switch 2 maps no big pages)
    g_gmmu = std::make_unique<Tegra::MemoryManager>(*g_system, 40, 1ULL << 34, 16, 12);
    g_shim_gpu->InitAddressSpace(*g_gmmu);
    // Eden's shader/pipeline disk cache for this game (<data dir>/cache/shader/<program id>/vulkan.bin): pipelines
    // built in earlier runs are built again now, before the game runs, instead of stuttering in the race; new ones
    // are appended as they are built. SWR_EDEN_NO_DISK_CACHE=1: off.
    if (!std::getenv("SWR_EDEN_NO_DISK_CACHE") && program_id) {
        const auto t0 = std::chrono::steady_clock::now();
        std::stop_source stop;
        g_shim_gpu->Renderer().ReadRasterizer()->LoadDiskResources(program_id, stop.get_token(),
                                                                   [](VideoCore::LoadCallbackStage, size_t, size_t) {});
        LOG_INFO(HW_GPU, "Switch2Recomp: shader disk cache loaded in {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    g_shim_gpu->Start();
    {   // Eden's syncpoints start where the runtime's are, then follow them (SyncpointMirror)
        auto& sm = g_system->Host1x().GetSyncpointManager();
        for (u32 id = 0; id < NumSyncpoints; ++id) {
            for (u32 v = syncpt_read(id); sm.GetHostSyncpointValue(id) < v;) {
                sm.IncrementGuest(id);
                sm.IncrementHost(id);
            }
            ArmSyncpoint(id, sm.GetHostSyncpointValue(id) + 1);
            g_expect[id] = sm.GetGuestSyncpointValue(id);
        }
        g_sp_thread = std::jthread(SyncpointMirror);
        StartQueue();
        if (g_use_write_watch && !g_full_ww) g_full_thread = std::jthread(FullScanLoop);
    }
    LOG_INFO(HW_GPU, "Switch2Recomp: Eden GPU started ({}x{}, guest memory {} GiB, program {:016X})", width,
             height, guest_size >> 30, program_id);
    return 1;
}

extern "C" void eden_gpu_map(unsigned long long gva, unsigned long long cpu, unsigned long long size) {
    if (!g_shim_gpu || !size) return;
    // device addresses are guest addresses: Eden's device space has 36 bits (64 GiB)
    constexpr u64 dev_end = 1ULL << Tegra::MaxwellDeviceTraits::device_virtual_bits;
    if (cpu + size > std::min<u64>(g_shim_mem_size, dev_end)) {
        static int said;
        if (said++ < 8) LOG_ERROR(HW_GPU, "Switch2Recomp: GPU map of guest {:#x}+{:#x} is past the device space", cpu, size);
        return;
    }
    std::scoped_lock lk{g_lock};
    DeviceMap(cpu, size);
    g_gmmu->Map(gva, cpu, size, Tegra::PTEKind::PITCH, false);
}

// GPU-accessible memory the game made (nx2 gpu worker 101/102). Two-page regions hold NVN2 queue semaphores:
// GPU releases land in the second page, waits read the first; Eglantine's puller mirrors them (video_core/nx2.h).
extern "C" void eden_gpu_memory(unsigned long long start, unsigned long long size, int created) {
    static const bool off = std::getenv("SWR_EDEN_NO_SEMREGIONS") != nullptr;   // experiment
    if (!g_shim_gpu || size != 0x2000 || off) return;
    LOG_INFO(HW_GPU, "Switch2Recomp: semaphore region {:#x} {}", start, created ? "added" : "removed");
    if (created) Tegra::Nx2AddSemaphoreRegion(start);
    else Tegra::Nx2RemoveSemaphoreRegion(start);
}

// GPU-only memory (whole 2 MiB chunks inside [cpu, cpu+size)): left out of the CPU-write scan
extern "C" void eden_gpu_gpu_only(unsigned long long cpu, unsigned long long size, int on) {
    if (!g_gpu_only_chunks || !size) return;
    const u64 first = (cpu + (1ull << ShimChunkBits) - 1) >> ShimChunkBits, end = (cpu + size) >> ShimChunkBits;
    for (u64 c = first; c < end && c < g_shim_nchunks; ++c) {
        const u64 m = 1ull << (c & 63);
        if (on) g_gpu_only_chunks[c >> 6].fetch_or(m); else g_gpu_only_chunks[c >> 6].fetch_and(~m);
    }
}

extern "C" void eden_gpu_unmap(unsigned long long gva, unsigned long long size) {
    if (!g_shim_gpu || !size) return;
    std::scoped_lock lk{g_lock};
    g_gmmu->Unmap(gva, size);
}

namespace {
// Submissions and presents run on a bridge worker thread, in the order the game made them. The game's thread
// only copies the entries and returns: before, it also read the write watch and pushed to Eden itself (~3-4 ms
// a submission, 4 a frame, all in series with the game's frame). SWR_EDEN_SYNC_SUBMIT=1: on the caller's thread.
struct Job {
    bool present = false;
    std::vector<u64> entries;
    u32 syncpt = 0, incs = 0, wait_id = ~0u, wait_val = 0;
#if SWR_EDEN_EGLANTINE
    std::vector<Tegra::CommandHeader> words;   // the command words as submitted (snapshot)
    std::vector<u32> offsets;
#endif
    Tegra::FramebufferConfig fb{};
};

void SubmitNow(Job& job, bool invalidate = true) {
    using Op = Tegra::Engines::Puller::FenceOperation;
    using Tegra::BufferMethods;
    using Tegra::SubmissionMode;
    const auto t_submit = std::chrono::steady_clock::now();
    std::scoped_lock lk{g_lock};
    if (invalidate) InvalidateDirty();
    auto& ch = Channel(job.syncpt);
    auto& gpu = *g_shim_gpu;
    if (job.wait_id != ~0u) {
        boost::container::small_vector<Tegra::CommandHeader, 512> l{
            Tegra::BuildCommandHeader(BufferMethods::SyncpointPayload, 1, SubmissionMode::Increasing),
            Tegra::CommandHeader{job.wait_val},
            Tegra::BuildCommandHeader(BufferMethods::SyncpointOperation, 1, SubmissionMode::Increasing),
            FenceAction(Op::Acquire, job.wait_id)};
        gpu.PushGPUEntries(ch.bind_id, Tegra::CommandList{std::move(l)});
    }
    if (!job.entries.empty()) {
        Tegra::CommandList list(job.entries.size());
        std::memcpy(list.command_lists.data(), job.entries.data(), job.entries.size() * sizeof(u64));
#if SWR_EDEN_EGLANTINE
        list.submitted_words = std::move(job.words);
        list.submitted_offsets = std::move(job.offsets);
#endif
        gpu.PushGPUEntries(ch.bind_id, std::move(list));
    }
    if (job.incs && job.syncpt < NumSyncpoints) {
        boost::container::small_vector<Tegra::CommandHeader, 512> l{
            Tegra::BuildCommandHeader(BufferMethods::WaitForIdle, 1, SubmissionMode::Increasing), Tegra::CommandHeader{0u},
            Tegra::BuildCommandHeader(BufferMethods::SyncpointPayload, 1, SubmissionMode::Increasing), Tegra::CommandHeader{0u}};
        for (u32 k = 0; k < job.incs; ++k) {
            l.push_back(Tegra::BuildCommandHeader(BufferMethods::SyncpointOperation, 1, SubmissionMode::Increasing));
            l.push_back(FenceAction(Op::Increment, job.syncpt));
        }
        gpu.PushGPUEntries(ch.bind_id, Tegra::CommandList{std::move(l)});
    }
    g_stat_submit_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t_submit).count();
    if (++g_stat_submits % 200 == 0 && !g_use_write_watch) {
        u64 faults, prot, ovf;
        WriteTrapsStats(&faults, &prot, &ovf);
        LOG_INFO(HW_GPU, "Switch2Recomp: write traps: {} faults, {} pages protected, {} overflows", faults, prot, ovf);
    }
    if (g_stat_submits % 200 == 0)
        LOG_INFO(HW_GPU, "Switch2Recomp: {} submissions ({:.2f} ms each, write watch {:.2f} ms), {} ranges invalidated, {} frames composited, {} MiB scanned each, {} batches; full reads: {} ({:.2f} ms, {} MiB each, {} cold chunks found written)",
                 g_stat_submits, g_stat_submit_ns / 1e6 / g_stat_submits, g_stat_ww_ns / 1e6 / g_stat_submits, g_stat_inval,
                 g_window->displayed.load(), (g_stat_ww_bytes / g_stat_submits) >> 20, g_stat_batches, g_stat_full,
                 g_stat_full ? g_stat_full_ns / 1e6 / g_stat_full : 0.0, g_stat_full ? (g_stat_full_bytes / g_stat_full) >> 20 : 0, g_stat_late);
}

// SWR_FRAME_BURST=<start s>,<frames>: from that second on, every composited frame (as many as the screenshot path
// keeps up with) to captures/burst_NNNN.ppm at a quarter of the window size, to look at frame-to-frame problems.
void BurstCapture() {
    static double start = -1;
    static int left = 0, index = 0;
    static std::atomic<bool> busy{false};
    if (start < 0) {
        const char* e = std::getenv("SWR_FRAME_BURST");
        if (!e) { start = 1e30; return; }
        start = std::atof(e);
        const char* c = std::strchr(e, ',');
        left = c ? std::atoi(c + 1) : 60;
    }
    if (left <= 0 || swr_run_ns() / 1e9 < start || busy.exchange(true)) return;
    const auto layout = g_window->GetFramebufferLayout();
    if (!layout.width || !layout.height) { busy = false; return; }
    auto bits = std::make_shared<std::vector<u8>>(size_t(layout.width) * layout.height * 4);
    const int n = index++;
    --left;
    g_shim_gpu->Renderer().RequestScreenshot(bits->data(), [bits, layout, n](bool) {
        const u32 w = layout.width / 4, h = layout.height / 4;
        char name[64];
        std::snprintf(name, sizeof name, "captures/burst_%04d.ppm", n);
        std::filesystem::create_directories("captures");
        if (FILE* f = std::fopen(name, "wb")) {
            std::fprintf(f, "P6\n%u %u\n255\n", w, h);
            std::vector<u8> row(size_t(w) * 3);
            for (u32 y = 0; y < h; ++y) {
                const u8* src = bits->data() + size_t(y * 4) * layout.width * 4;
                for (u32 x = 0; x < w; ++x) {   // BGRA -> RGB
                    row[x * 3] = src[x * 16 + 2];
                    row[x * 3 + 1] = src[x * 16 + 1];
                    row[x * 3 + 2] = src[x * 16];
                }
                std::fwrite(row.data(), 1, row.size(), f);
            }
            std::fclose(f);
        }
        busy = false;
    }, layout);
}

void PresentNow(Job& job) {
    {
        std::scoped_lock lk{g_lock};
        DeviceMap(job.fb.address, u64(job.fb.stride) * 4 * job.fb.height);
    }
    std::vector<Tegra::FramebufferConfig> layers{job.fb};
    BurstCapture();
    g_shim_gpu->RequestComposite(std::move(layers), {});
}

std::mutex g_q_lock;
std::condition_variable g_q_cv;
std::deque<Job> g_queue;
std::jthread g_q_thread;
const bool g_sync_submit = std::getenv("SWR_EDEN_SYNC_SUBMIT") != nullptr;

// Everything queued is taken at once and the CPU writes are looked for once for all of it: that read starts after
// every job in the batch was queued, so it sees every write made before them (one read of the write watch per
// batch, not per submission: ~6 ms each over MKW's cached memory).
void QueueLoop(std::stop_token stop) {
    swr_trace_thread_name("bridge worker");
    std::deque<Job> batch;
    for (;;) {
        {
            std::unique_lock lk{g_q_lock};
            g_q_cv.wait(lk, [&] { return !g_queue.empty() || stop.stop_requested(); });
            if (g_queue.empty()) return;
            batch.swap(g_queue);
        }
        {
            std::scoped_lock lk{g_lock};
            const u64 t0 = swr_trace_ts();
            InvalidateDirty();
            swr_trace_span("write watch", t0);
            ++g_stat_batches;
        }
        for (auto& job : batch) {
            const u64 t0 = swr_trace_ts();
            if (job.present) PresentNow(job);
            else SubmitNow(job, false);
            swr_trace_span(job.present ? "present to Eden" : "submit to Eden", t0);
        }
        batch.clear();
    }
}

void StartQueue() {
    g_q_thread = std::jthread(QueueLoop);
}

void StopQueue() {
    if (!g_q_thread.joinable()) return;
    g_q_thread.request_stop();
    g_q_cv.notify_all();
    g_q_thread.join();
}

void Enqueue(Job&& job) {
    if (g_sync_submit) {
        if (job.present) PresentNow(job);
        else SubmitNow(job);
        return;
    }
    {
        std::scoped_lock lk{g_q_lock};
        g_queue.push_back(std::move(job));
    }
    g_q_cv.notify_one();
}
} // namespace

extern "C" void eden_gpu_submit(const unsigned long long* entries, unsigned n, unsigned syncpt, unsigned incs,
                                unsigned wait_id, unsigned wait_val) {
    if (!g_shim_gpu) return;
    Job job;
    job.entries.assign(entries, entries + n);
    job.syncpt = syncpt;
    job.incs = incs;
    job.wait_id = wait_id;
    job.wait_val = wait_val;
#if SWR_EDEN_EGLANTINE
    // Keep the words as submitted (as Eglantine's nvhost_gpu does for NX2): MKW refills command
    // memory it already submitted before the GPU thread gets to it, which ran the next frame's commands.
    if (!std::getenv("SWR_EDEN_NO_SNAPSHOT")) {
        size_t offset = 0;
        for (const u64 raw : job.entries) {
            Tegra::CommandListHeader e{};
            e.raw = raw;
            job.offsets.push_back(static_cast<u32>(offset));
            job.words.resize(offset + e.size);
            g_gmmu->ReadBlockUnsafe(e.addr, job.words.data() + offset, size_t(e.size) * sizeof(u32));
            offset += e.size;
        }
    }
#endif
    {
        // the value each syncpoint reaches once everything queued so far is processed (present pacing)
        std::scoped_lock lk{g_lock};
        if (incs && syncpt < NumSyncpoints) g_expect[syncpt] += incs;
        Channel(syncpt);   // known to the pacing from its first submission on
    }
    Enqueue(std::move(job));
}

extern "C" volatile unsigned g_swr_present_seq;   // svc_ns2.c: presents so far (frames.csv column n)

extern "C" void eden_gpu_present_image(unsigned long long addr, unsigned w, unsigned h, unsigned pitch,
                                       unsigned layout, unsigned bhl, unsigned fmt, int crop_right, int crop_bottom) {
    if (!g_shim_gpu) return;
    (void)layout; (void)bhl; (void)fmt;   // Switch 2 presents RGBA8 block-linear images (svc_ns2.c)
    Job job;
    job.present = true;
    job.fb = Tegra::FramebufferConfig{
        .address = addr,
        .offset = 0,
        .width = w,
        .height = h,
        .stride = pitch / 4,
        .pixel_format = Service::android::PixelFormat::Rgba8888,
        .transform_flags = Service::android::BufferTransformFlags::Unset,
        // the part of the image the game shows (vi2 present word 7: MKW draws 1920x1080 or less into
        // its larger window images); none: all of it
        .crop_rect = crop_right > 0 && crop_bottom > 0 ? Common::Rectangle<int>{0, 0, crop_right, crop_bottom}
                                                       : Common::Rectangle<int>{},
        .blending = Tegra::BlendMode::Opaque,
        .native_tiling = true,
        .swr_present_id = g_swr_present_seq,   // composites.csv, captures/frames.bin
    };
    Enqueue(std::move(job));
    FullScanRequest();   // a full read of the write watch per frame
}

// The next composited frame as <path>.ppm (the GL renderer's screenshot format). Returns a hash of the pixels,
// 0 when no frame came within 10 s.
extern "C" unsigned long long eden_gpu_screenshot(const char* path) {
    if (!g_shim_gpu || !g_window) return 0;
    const auto layout = g_window->GetFramebufferLayout();
    if (!layout.width || !layout.height) return 0;
    auto bits = std::make_shared<std::vector<u8>>(size_t(layout.width) * layout.height * 4);
    auto state = std::make_shared<std::pair<std::mutex, std::condition_variable>>();
    auto done = std::make_shared<int>(0);
    g_shim_gpu->Renderer().RequestScreenshot(bits->data(), [state, done, bits](bool) {
        std::scoped_lock lk{state->first};
        *done = 1;
        state->second.notify_all();
    }, layout);
    {
        std::unique_lock lk{state->first};
        if (!state->second.wait_for(lk, std::chrono::seconds(10), [&] { return *done != 0; })) return 0;
    }
    u64 hash = 1469598103934665603ull;   // FNV-1a over the pixels
    for (const u8 b : *bits) hash = (hash ^ b) * 1099511628211ull;
    if (path) {
        const std::string p = std::string(path) + ".ppm";
        if (FILE* f = std::fopen(p.c_str(), "wb")) {
            std::fprintf(f, "P6\n%u %u\n255\n", layout.width, layout.height);
            std::vector<u8> row(size_t(layout.width) * 3);
            for (u32 y = 0; y < layout.height; ++y) {
                const u8* src = bits->data() + size_t(y) * layout.width * 4;
                for (u32 x = 0; x < layout.width; ++x) {   // BGRA -> RGB
                    row[x * 3] = src[x * 4 + 2];
                    row[x * 3 + 1] = src[x * 4 + 1];
                    row[x * 3 + 2] = src[x * 4];
                }
                std::fwrite(row.data(), 1, row.size(), f);
            }
            std::fclose(f);
        }
    }
    return hash ? hash : 1;
}

extern "C" void eden_gpu_present(void) {
    if (g_shim_gpu) g_shim_gpu->RequestComposite({}, {});
}

extern "C" void eden_gpu_resize(int width, int height) {
    if (g_window) g_window->UpdateCurrentFramebufferLayout(u32(width), u32(height));
}

extern "C" void eden_gpu_stop(void) {
    if (g_full_thread.joinable()) {
        g_full_thread.request_stop();
        g_full_cv.notify_all();
        g_full_thread.join();
    }
    StopQueue();
    if (g_shim_gpu) g_shim_gpu->NotifyShutdown();
    if (g_sp_thread.joinable()) {
        g_sp_thread.request_stop();
        g_sp_thread.join();
    }
    g_channels.clear();
    g_gmmu.reset();
    g_system.reset();   // destroys the GPU before Host1x
    g_window.reset();
    Common::Log::Stop();
}
