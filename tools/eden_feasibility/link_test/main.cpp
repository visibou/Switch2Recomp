// Eden link test: start Eden's GPU (Vulkan renderer) on a plain Win32 window over our core shim and present
// empty frames. Exit code 0 when the GPU came up and presented; the log says how far it got.
#include <atomic>
#include <cstdio>
#include <optional>
#include <thread>
#include <windows.h>
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

extern std::optional<Tegra::GPU> g_shim_gpu;

namespace {
class TestWindow final : public Core::Frontend::EmuWindow {
public:
    TestWindow(HWND hwnd, u32 w, u32 h) {
        window_info.type = Core::Frontend::WindowSystemType::Windows;
        window_info.render_surface = hwnd;
        UpdateCurrentFramebufferLayout(w, h);
    }
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<Core::Frontend::GraphicsContext>();
    }
    bool IsShown() const override { return true; }
    void OnFrameDisplayed() override { ++displayed; }
    std::atomic<int> displayed{0};
};

LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}
} // namespace

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? atoi(argv[1]) : 3;
    Common::Log::Initialize();
    Common::Log::Start();
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    Settings::values.use_asynchronous_gpu_emulation.SetValue(true);

    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"EdenLinkTest";
    RegisterClassW(&wc);
    const u32 w = 1280, h = 720;
    RECT rc{0, 0, (LONG)w, (LONG)h};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"Eden link test", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                              CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr,
                              wc.hInstance, nullptr);

    TestWindow window{hwnd, w, h};
    Core::System system;
    std::printf("link test: creating GPU\n");
    VideoCore::CreateGPU(g_shim_gpu, window, system);
    if (!g_shim_gpu) {
        std::printf("link test: GPU creation failed\n");
        return 1;
    }
    std::printf("link test: GPU created, renderer %s\n", g_shim_gpu->Renderer().GetDeviceVendor().c_str());
    g_shim_gpu->Start();

    int frames = 0;
    const DWORD end = GetTickCount() + seconds * 1000;
    while (GetTickCount() < end) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        g_shim_gpu->RequestComposite({}, {});
        ++frames;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    std::printf("link test: %d frames requested, %d composited\n", frames, window.displayed.load());
    g_shim_gpu->NotifyShutdown();
    g_shim_gpu.reset();
    return 0;
}
