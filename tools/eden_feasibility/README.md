# Eden GPU backend: feasibility build

Compiles Eden-NS2's `shader_recompiler` and `video_core` (their own CMakeLists, so their source lists) with the
project's MinGW gcc, everything they link to replaced by include-only stand-ins. Nothing is linked: the undefined
symbols of the two libraries are the list a `core` shim has to provide (FEASIBILITY_EDEN_GPU.md).

Inputs (none of them are in the repo):
- `EDEN_SRC`: a clone of https://github.com/0x2593/Eden-NS2 (tested at its main branch, 2026-11-02)
- `DEPS`: fmt 12.1.0, Vulkan-Headers v1.4.345, Vulkan-Utility-Libraries v1.4.345, VulkanMemoryAllocator v3.3.0,
  eden-emulator/sirit v1.0.5, SPIRV-Headers vulkan-sdk-1.4.341.0, nlohmann/json v3.12.0, frozen, herumi/xbyak,
  Boost 1.89.0 headers (`boost/boost_1_89_0`), FFmpeg headers (`ffmpeg/ucrt64/include`, MSYS2 UCRT64 ffmpeg 9.0.2),
  each in a folder named as in CMakeLists.txt
- `GLSLANG`: glslang.exe (KhronosGroup glslang 16.6.0 Windows release)
- FidelityFX-FSR (GPUOpen-Effects) copied to `externals/FidelityFX-FSR` next to this CMakeLists.txt
- Ninja (1.13.2)

    cmake -S . -B b -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja.exe> -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
      -DEDEN_SRC=<eden-ns2> -DDEPS=<deps> -DGLSLANG=<glslang.exe>
    ninja -C b -k 0 shader_recompiler video_core

Result 2026-11-02: all 467 targets compile with gcc 16.2.0 (libshader_recompiler.a 39 MB, libvideo_core.a 28 MB).

## Step 1: link test (`link_test/`, 2026-11-02)

`ninja -C b eden_link_test` builds everything video_core links to (sirit from source, the Eden `common` subset,
emu_window/framebuffer_layout, bc_decoder, stb_dxt, glad, VMA) and links `eden_link_test.exe` (44 MB, static):
- `core_shim.cpp`: the emulator side video_core calls (`Core::System`, `Core::Memory::Memory`, `Core::DeviceMemory` +
  `Common::HostMemory` over one reservation, the GPU clock, PerfStats, DlssTrace, KSystemControl).
- `ffmpeg_stub.c`: FFmpeg stand-ins, so video decoding (NVDEC) reports "no decoder".
- `main.cpp`: a Win32 window, `VideoCore::CreateGPU` (Vulkan, async GPU), `GPU::Start`, empty `RequestComposite`
  calls, shutdown. `eden_link_test.exe [seconds]`.

Run it from a folder that has a `user/cache` subfolder: Eden then keeps its log and cache in `user/` there (portable
mode) and not in %APPDATA%. Result: RTX 3050 Laptop, NVIDIA 617.42, Vulkan 1.4.351; 5 s run, 302 frames requested,
302 composited, no errors in `user/log/eden_log.txt`, exit 0.
