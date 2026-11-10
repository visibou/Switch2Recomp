# Eden GPU backend (optional, `SWR_GPU=eden`)

Eden-NS2's GPU side (`video_core` + `shader_recompiler`: Vulkan renderer and Maxwell/Ampere shader translation) as a
second GPU backend of the runtime. The game's code stays recompiled; services stay ours. Eden replaces only what
`gpu_gl.c` does. Plan and status: FEASIBILITY_EDEN_GPU.md, ROADMAP.md section 7 (Phase 2).

## Files
- `CMakeLists.txt`: builds Eden's libraries with their own CMakeLists, plus sirit, the `common` subset, bc_decoder,
  stb_dxt, VMA, and `swr_eden` (ours).
- `core_shim.cpp`: the emulator side that video_core calls (`Core::System`, memory, the GPU clock, ...). Eden's
  "physical" memory is the runtime's guest address space (`g_mem`): a device address maps 1:1 onto it.
- `eden_gpu.cpp` / `eden_gpu.h`: the C bridge the runtime calls (`runtime/gpu_eden.c`).
- `ffmpeg_stub.c`: no FFmpeg yet (Eden's NVDEC video decoding reports "no decoder").
- `vma.cpp`: VulkanMemoryAllocator implementation with Eden's configuration.
- `externals/FidelityFX-FSR/ffx-fsr`: AMD FSR 1 headers (MIT), used by Eden's host shaders.

## Inputs (not in the repo): the `external` folder (tools/s2r_paths.py finds it)
- `eden-ns2/`: github.com/0x2593/Eden-NS2 at commit `b2cef0c0f2e8517673233e22c929918e2c60e654` (only `src/common`,
  `src/core`, `src/video_core`, `src/shader_recompiler`, `externals/{bc_decoder,glad,stb,FidelityFX-FSR}`).
- `deps/`: fmt 12.1.0, Vulkan-Headers v1.4.345, Vulkan-Utility-Libraries v1.4.345, VulkanMemoryAllocator v3.3.0,
  eden-emulator/sirit v1.0.5, SPIRV-Headers vulkan-sdk-1.4.341.0, nlohmann/json v3.12.0, frozen, herumi/xbyak,
  `boost/boost_1_89_0` (release headers), `ffmpeg/ucrt64/include` (MSYS2 UCRT64 ffmpeg headers).
- `tools/`: `glslang/bin/glslang.exe` (glslang 16.6.0), `ninja.exe` (1.13.2).

## Source trees (EDEN_FLAVOR)
- `eden-ns2` (default): github.com/0x2593/Eden-NS2 at b2cef0c0, in `external/eden-ns2`. Runs MKW to races.
- `eglantine` (experimental): Djihads80/Eglantine build-130, in `external/eglantine-130`. It adds task/mesh shaders
  (VK_EXT_mesh_shader), VP9 movies, CPU reads of compute output and many GPU-thread speedups. With our runtime
  it stalls in MKW at ~140 frames: two channels wait on queue semaphores under its global Switch 2 mode
  (`g_nx2_ampere`). Eden-NS2 does the same with that mode on. See CHANGES_SWITCH2RECOMP.md for what was ruled
  out. One local patch: runtime/eden/patches/README.md.

`EDEN_FLAVOR=eglantine sh tools/build_eden.sh` builds it into `build/eden_eglantine`. The runtime links the one
named by `EDEN_FLAVOR` (default eden-ns2).

## How the runtime connects (eden_gpu.cpp)
- GMMU map/unmap (nvdrv.c `gmmu_map`/`gmmu_unmap`): device-map the guest range (identity), then map it in Eden's
  one `Tegra::MemoryManager`.
- `gpu_submit` (gpu.c): GPFIFO entries to the channel's Eden channel (Switch 2 ABI, shared TSG macro context), then
  the syncpoint increments after the work. Each completed increment raises the runtime's counter.
- Before each submission: write-watched guest pages that Eden caches are invalidated (only chunks with cached pages
  are read).
- `vi_show_addr` (main.c): the presented image as an Eden framebuffer layer. Screenshots take Eden's next frame.
- Presents wait (svc_ns2.c `vi2_present` -> `eden_gpu_present_wait`) until Eden processed the previous frame.
- Syncpoints: completed GPU values raise the runtime's counters (watcher thread), CPU increments go to Eden too.
- GPU sparse regions (svc_ns2.c 214) are all backed, lazily (committed on first touch, below 64 GB).

## Build and use
    sh tools/build_eden.sh      # once (about 6 min), into build/eden_eden-ns2
    sh tools/build_rt.sh        # the runtime links build/eden when it is there

Then set `gpu=eden` in the game's .ini (or `SWR_GPU=eden` in the environment). Otherwise, or if Eden does not
start, the OpenGL renderer runs.
Eden's log and caches go to `<game>_eden/` next to the game exe.

Licence: Eden is GPL-3.0-or-later (yuzu parts GPL-2.0-or-later). The project is AGPL-3.0 and not released.
