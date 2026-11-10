# SwitchRecomp developer guide (for AI assistants and contributors)

SwitchRecomp is a static recompiler for Nintendo Switch games: AArch64 code becomes C (Python front end), GCC turns
that C into a native module, and a C runtime emulates Horizon OS (kernel, IPC, services) and the Maxwell GPU
(OpenGL 4.6). This guide covers how to build it, how the parts fit together, how to read the diagnostics users send,
and the bugs that are still open.

Rules of the project:
- The version stays **v190** (`SWR_RUNTIME_VERSION` in `runtime/main.c`). Changes go in `CHANGELOG_v190.txt`,
  in English.
- No code from GPL-2.0-only or other incompatible projects. yuzu, Eden and similar emulators may be read to understand
  hardware behavior, but their code is not copied.
- Every source file keeps the header `SwitchRecomp - Copyright 2026 MZ Labs` (AGPL v3 with NOTICE.txt).
- Packages stay under 20 MB. Python, GCC and SDL2 are downloaded by `Install.bat` / `Install.sh`, not shipped.
- A fix is not done until it is explained by a mechanism (log, trace or replay), not by trial and error.

---

## 1. What to install

### Linux (Debian/Ubuntu names; use your distro's equivalents)

```sh
sudo apt install build-essential gcc python3 \
    libsdl2-dev libx11-dev libgl-dev libegl-dev \
    mingw-w64 gcc-mingw-w64-x86-64-win32 \
    squashfs-tools p7zip-full zip imagemagick xdotool wine
```

- `gcc`, `python3`: the recompiler and the Linux runtime.
- `libsdl2-dev`, `libx11-dev`, `libgl-dev`: the Linux x11 runtime (audio uses SDL2).
- `x86_64-w64-mingw32-gcc` (the **win32 thread model** variant): cross-builds the Windows runtime and tool.
- `squashfs-tools`: builds the AppImage. `p7zip-full`: opens users' capture `.7z` files.
- `imagemagick`: converts the `.ppm` images in traces (`convert x.ppm x.png`, `montage`).
- `libegl-dev` and Mesa (llvmpipe): offscreen replay of captured shaders (section 5.4).
- `wine` (optional): runs the Windows build to check it starts.

### Windows

Run `Install.bat`. It downloads Python 3.14 (NuGet) and GCC (conda-forge) into the SwitchRecomp folder and checks
each download against its SHA-256. `tool/get_deps.py` holds the pinned URLs and hashes.

---

## 2. Building

All commands run from the `SwitchRecomp` folder.

```sh
# runtimes (one per target/backend)
python3 switchrecomp.py --build-runtime --target linux --backend x11      --cc gcc
python3 switchrecomp.py --build-runtime --target win64 --backend win32    --cc x86_64-w64-mingw32-gcc
python3 switchrecomp.py --build-runtime --target linux --backend headless --cc gcc
python3 switchrecomp.py --build-runtime --target win64 --backend headless --cc x86_64-w64-mingw32-gcc
# outputs: bin/runtime_linux_x11, bin/runtime_win64_win32.exe, bin/runtime_*_headless[.exe]

# the GUI tool (SwitchRecomp.exe / SwitchRecomp.bin)
tool/build_tool.sh SwitchRecomp.exe        # Windows tool, cross-built
tool/build_tool_linux.sh SwitchRecomp.bin  # Linux tool
```

A game folder holds `<Game>` (or `<Game>.exe`), which is a copy of the runtime, plus `<Game>_code.so` /
`<Game>_code.dll` (the recompiled game), `<Game>_data/`, `<Game>.ini` and `<Game>.log`. **To test a runtime
change on a game, copy the new runtime over the game's executable** and keep the same name. The tool's
"Update all games" does that for every game. Windows Explorer sometimes renames the copy; check the name.

Packaging (what the release has):
- Windows zip: the whole `SwitchRecomp` folder, without `gcc/`, `python/`, `games/*/*` and `__pycache__`.
- Linux tar.gz: the Linux package folder (`SwitchRecomp` launcher script + `SwitchRecomp.bin`, `lib/`, `bin/`).
- AppImage: the Linux package + `AppRun`, built with `mksquashfs -comp zstd` on top of an AppImage runtime.

Regression checks (builds, unit tests, homebrew GPU tests, games) are in REGRESSION.md.

---

## 3. Where things are

| Path | What it does |
| --- | --- |
| `switchrecomp.py`, `a64.py`, `codegen.py`, `brt.py`, `nxloader.py`, `tools_sym.py` | Recompiler: loads NSO/NRO, decodes AArch64 (`a64.py`), emits C (`codegen.py`), compiles the module |
| `runtime/main.c` | Startup, `.ini` options (`INI_OPTS` table, `INI_ALIAS` for old Spanish keys), stall watchdog, `threads.txt` |
| `runtime/kernel.c`, `svc_*.c`, `ipc.c` | Horizon kernel, SVCs, IPC and HLE services (fs, hid, vi, am, audio, nvdrv...) |
| `runtime/nvdrv.c`, `host1x.c` | nvhost devices, syncpoints, GPFIFO submits, NVDEC/VIC |
| `runtime/gpu.c` | GPU thread front end: pushbuffer parsing, `gpu_submit`, semaphores, syncpoint waits |
| `runtime/gpu_engines.c`, `gpu_3d.c` | Maxwell engines (3D, compute, 2D, DMA, inline-to-memory, macros) |
| `runtime/gpu_gl.c` | The OpenGL renderer (~4000 lines): textures, render targets, draws, compute, traces, profiler |
| `runtime/shader.c` | Maxwell shader → GLSL translator (`shader_to_glsl`) |
| `runtime/glq.c`, `glq_gen.c` | GL call queue: the GPU thread records GL calls, a GL thread runs them. Driver lock for Intel |
| `runtime/shader_proc.c` | Optional helper process that compiles shaders (Intel crash isolation) |
| `runtime/astc.c`, `bcn/` | Texture decoders |
| `runtime/audio.c` | audren mixer, audout, output (waveOut / SDL2) |
| `runtime/backend_win32.c`, `backend_x11.c`, `backend_headless.c` | Window, input, F-key menus |
| `runtime/lsfg.c` | Lossless Scaling frame generation integration |
| `tool/` | The GUI tool, `Install` scripts, `get_deps.py` |
| `tool/dev/replay.c` | Offline replay of one captured draw (section 5.4) |

### How a frame flows
1. The game writes GPU commands and calls `nvdrv` SubmitGpfifo → `gpu_submit()` queues the work for the GPU thread
   (a submit with a wait fence is waited on by the GPU thread, not by the game thread).
2. The GPU thread parses the pushbuffer (`gpu.c`) and runs engine methods (`gpu_engines.c`, `gpu_3d.c`).
3. Draws, clears and dispatches become GL calls in `gpu_gl.c`. Those calls go through `glq.c` to the GL thread.
4. Shaders: `shader.c` translates on first use and the compiler thread compiles. A draw whose shader isn't ready is
   skipped ("waiting for shader") and drawn once it is.
5. Present: `vi` / `present_addr` shows the frame, then `frame_stats()` and the profiler run.

### GPU concepts that cause most bugs
- **Textures and render targets share guest addresses.** `tex_from_tic()` finds the texture for a TIC entry. If a
  render target (RT) exists at that address, the GPU copy is used instead of guest memory. Arrays, cubemaps and mip
  chains that the game renders level by level into separate RTs are assembled by `layered_from_rts()`
  (`gpu: ... assembled from its render targets` in the log). Missing mips are box-generated (`lvl_gen`).
- **Compute image writes (SUST)** bind a TIC; `img_level()` picks the mip from TIC word 7. Env
  `SWR_NO_IMG_LEVEL=1` writes everything to mip 0; `SWR_IMG_LEVEL_2D_ONLY=1` does that only for cubemaps and arrays.
- **Constant buffers** are uploaded into a persistent ring (`upload_cbufs_ring`). Reads past the bound size must
  return 0 (zero tail), or shaders read leftovers from other draws.
- **Shaders with jumps** run as `while(true){ switch(pc){...} }` with a loop guard (`loop_limit=` in the `.ini`,
  default 16384). A shader that hits the guard returns zeros and costs seconds per frame on small GPUs.
- **Intel on Windows**: "driver exclusive mode" (the GL thread and compiler take turns inside the driver). The Intel
  driver crashes or hangs when two threads are inside it.

---

## 4. Diagnostics users send

| File | Where | What to look for |
| --- | --- | --- |
| `<Game>.log` | game folder | The first lines (runtime version, GPU, driver checks). `warning: 6 s without a new image` blocks with the GPU thread, GL thread, compile thread and **GPU progress** (which pass the GPU didn't finish, with its shaders). `gpu:` lines about assembled textures, depth copies, failed shaders |
| `threads.txt` | game folder | Written every 15 s. Every guest thread's state, plus the GPU/GL/compile thread lines and GPU progress. A stuck game thread waiting on a mutex/condvar/event points to the blocked subsystem |
| `performance.txt` | game folder | Every 5 s: FPS, CPU, GPU thread breakdown, **GPU time of one frame** per render target / compute shader, blocking GL calls. Shows where frame time goes |
| `captures/traces/N/` | game folder, after **F5** | One frame, fully listed (section 5) |
| `captures/rt_*.ppm`, `screenshot_*.ppm` | after F5 | The render targets and the presented image |

Useful keys in game: **F5** trace of the next frame, **F1** settings, **F7** record video, **F11** fullscreen.

Useful `.ini` options (the game's `.ini` lists them all with comments): `shadows`, `resolution`, `loop_limit`,
`driver_exclusive`, `shader_thread`, `shader_process`, `memory_gb`, `log=2` (verbose), `diag=1`.

Environment switches (`SWR_*`, set before starting the game) turn individual fixes off to bisect a regression. The
ones used most: `SWR_NO_IMG_LEVEL`, `SWR_IMG_LEVEL_2D_ONLY`, `SWR_NO_CUBE_MIPGEN`, `SWR_NO_UBO_RING`, `SWR_NO_GLQ`,
`SWR_SYNC_SHADERS`, `SWR_NO_GPU_PROF`, `SWR_NO_GPU_WATCH`, `SWR_LAYDBG`, `SWR_RTDBG`, `SWR_DUMP_SHADERS`,
`SWR_CS_SYNC`. `grep -n 'getenv("SWR_' runtime/*.c` lists all of them with the code they change.

---

## 5. Reading an F5 trace

`captures/traces/1/trace.txt` lists every operation of one frame in order:
- `#N CLEAR ...`: clears (target address, color, depth, layer).
- `#N DRAW vs=<hash> fs=<hash> ...`: then its render targets (`rt0 <addr> WxH fmt`), depth (`z ...`), blend state,
  every texture (`fs texK: ... tic fmt ... addr=... -> RT #n` or `-> tex #n`), every constant buffer
  (values and the offset of its bytes in `cbdata.bin`), and `result rt0: dNNNN.ppm`.
- `COMPUTE <hash> grid ... buffers ...`: a dispatch. The `image K: <addr> ... mip M (tic levels a..b)` lines just
  **before** a COMPUTE line belong to that dispatch.
- `DMA`, `BLIT2D`: copies.
- At the end: `tex #n` for every texture used. Cubemaps and arrays list the **mean and max of every mip**, and shadow
  arrays list min/mean/max **per layer**. A black or NaN level shows up there.

Files next to it:
- `dNNNN.ppm`: the result of draw N, at reduced size, **color on the left, alpha on the right**.
- `sh_<hash>_vs.glsl` / `_fs.glsl` / `cs_<hash>.glsl`: the translated shaders. `.bin`: the Maxwell code.
- `texNNN_<addr>_WxH_f<fmt>[_rt].ppm`: thumbnails of the textures.
- `cbdata.bin`: the bytes of every constant buffer the trace lists.

### 5.1 Finding where an image goes wrong
1. Make a contact sheet: `montage -label '%f' d04*.ppm -tile 6x -geometry 320x90+2+2 sheet.png`.
2. Find the first draw whose result looks wrong, and read its entry in `trace.txt`.
3. Check its inputs: the textures it samples (`-> RT #n`), and which earlier draw or compute wrote them (search for
   that address).
4. Check the shader: the GLSL of its `fs=` hash. The G-buffer flags, the constants it reads, and its loops.

### 5.2 Constants
`fs cbN @addr bound=... data=cbdata.bin+OFF (SIZE bytes)` → read with Python
`struct.unpack_from('<4f', data, OFF + 16*index)` to get `cbN.d[index]`.

### 5.3 Comparing two runs
Take a trace with the fix and one with it switched off (an `SWR_*` switch), and compare the same draw number:
results (`convert dNNNN.ppm -format '%[fx:mean]' info:` on each half), texture mips at the end of the trace, and
constants.

### 5.4 Replaying a draw offline (no game needed)
`tool/dev/replay.c` draws one captured fullscreen pass with Mesa's software renderer, so you can change inputs and
see what the shader does:

```sh
gcc -O1 -o replay tool/dev/replay.c -lEGL -lGL -lm
# the shader: replace binding=113 (render scale UBO) with binding=80, llvmpipe has fewer UBO slots
sed 's/binding=113/binding=80/' captures/traces/1/sh_<fs>_fs.glsl > fs.glsl
EGL_PLATFORM=surfaceless ./replay fs.glsl captures/traces/1/sh_<vs>_vs.glsl captures/traces/1/cbdata.bin config.txt out.ppm
```

`config.txt`, one line per input (the bindings come from the GLSL: textures 32+K for fs `texK`, UBO bindings from
the `layout(std140, binding=...)` lines):

```
size 320 180
ubo 21 681b0 1c00                       # binding, offset in cbdata.bin, size (hex)
ubof 80 1 1 1 1                         # a UBO with one vec4
ssbo 8                                  # an empty storage buffer
tex2d 32 captures/traces/1/tex167_..._rt.ppm   # a texture from a trace thumbnail
texcube 39 c 0.5 0.5 0.5 1              # a constant cubemap (CUBE_MIP0ONLY=1: only mip 0 filled)
texcubearr 38 c 0.5 0.5 0.5 1
tex2darr 42 c 1 1 1 1
```

It prints the mean/max of the output. Changing one input at a time shows which one the result depends on: that is
how the Frontiers lighting was traced to the sky cubemap.

---

## 6. Open bugs (status at the time of this package)

### Sonic Frontiers (Linux, AMD RX 470): world lighting black
- The G-buffer is correct. The deferred lighting pass (`fs=1b0fb8fff1cc3963`, around draw #456) outputs black
  except emissive objects. In the build without the cubemap mip fix, it outputs nearly white, and the final image
  looks washed out and blue.
- An offline replay shows that pass's output scales directly with the sky cubemap `5ff960000` (128x128, 7 mips,
  sampled as `tex7` with `textureLod`). With only mip 0 filled, the result is fully black.
- Compute shaders build the cubemap mips (`7bdab8afd8b212db` writes mip 0 of a probe cube, `755f7a0611293bb6`
  writes mips 1-6 in one dispatch, `48c3e0b42e09bacf` prefilters). The image-level fix now writes each mip where the
  TIC says. Find out why the result is black: check the mip means at the end of an F5 trace, with and without
  `SWR_IMG_LEVEL_2D_ONLY=1`.
- Also check: the shadow cascades (`5f0580000` + n*0x400000, array assembled from 4 depth RTs; per-layer min/max at
  the end of the trace), and whether the cubemap's sampled TIC has a base level different from 0.
- Reference: Sky Sanctuary should be bright and green with sharp sun shadows.

### Super Mario Odyssey and Tomodachi Life on Intel HD Gen11 (Windows): freeze when 3D starts
- When the 3D scene starts, the frame never completes on the GPU (GL thread stuck in the driver > 12 s, audio keeps
  playing). Tomodachi measured about 6 s of GPU time in one pass on its 3D map.
- Leading theory: a shader runs into the loop guard on every pixel. That can come from garbage constants read past
  a buffer's end (the zero-tail fix for indexed reads is in), or from a translation bug.
- What to ask for: the log and `threads.txt` from the current build. Its GPU watchdog prints the unfinished pass and
  its shaders. Also a run with `loop_limit=2048` in the game's `.ini`: if the freeze goes away, a shader is looping.
  Fix that shader's translation or data.

### Official GBA emulator (NSO): fixed
SubmitGpfifo fences are waited on by the GPU thread, and new channels start with color masks enabled. It shows the
Game Boy Advance menu now.

---

## 7. How code, comments and text are written

Everything in the project should read like one person wrote it. New code that looks generated stands out right away
next to the old code, so match what is already there.

### 7.1 Code
- **Plain C (C99 with GNU extensions)** in the runtime and **Python 3** in the recompiler. No C++ except where a
  library needs it (`lsfg_spvc.cpp`, `bcn/`).
- **Dense layout, like the surrounding code.** Short functions on one line, several statements per line when they
  belong together, braces on the same line, no blank lines inside functions:
  ```c
  static u32 hkey(u64 gva){ return (u32)((gva>>8)^(gva>>20))&(TEXH-1); }
  if(!t || t->target!=GL_TEXTURE_2D) return 0;
  u32 lw=t->w>>l, lh=t->h>>l; if(!lw) lw=1; if(!lh) lh=1;
  ```
- **Types:** `u8 u16 u32 u64 i32 i64`. GL names are `GLuint`, `GLenum` and so on.
- **Names:** lowercase with underscores, short and specific (`tex_from_tic`, `bl_level_size`, `cs_cb_read`). Globals
  start with `g_`, counters for the stats with `st_`. Locals can be one or two letters when their use is obvious
  (`t` texture, `r` registers, `c` context, `w`/`h` sizes).
- **Logging:** `LOG(...)` for facts the user's log needs, limited with a static counter so it can't flood:
  ```c
  { static int said; if(said<4){ said++; LOG("gpu: compute readback in flight for 4 s, letting %d held fence write(s) go\n",n); } }
  ```
  Log lines start with the subsystem (`gpu:`, `audio:`, `nvdrv:`) and say what happened in plain words.
- **Switches:** a new behavior that could break something gets an `SWR_NO_...` environment switch to turn it off,
  read once into a static.
- **No new dependencies**, no new files for small things, no abstraction layers, no defensive checks for cases that
  can't happen, no dead code left behind. Change only what the fix needs.
- **Every new file** starts with the three-line MZ Labs header (copy it from any `runtime/*.c`).

### 7.2 Comments
- **Lowercase, short, in English**, written like notes from the person who debugged it. A comment says **why** the
  code does something, or what broke without it, usually naming the game:
  ```c
  /* reads past the bound size give 0 on hw. the ring had an older upload there: tomodachi's mii shaders read past
     a small bound buffer and got leftovers from other draws */
  /* glGen*: object names don't need the GL thread's answer right away. I grab 256 at a time, so making textures and
     buffers mid frame doesn't drain the queue every time */
  ```
- First person ("I", "we") is fine. Contractions are fine.
- **No comments that repeat the code** (`/* increment the counter */`), no section-by-section narration, no
  "Note:", "IMPORTANT:", "TODO: consider", no emoji, no banners other than the existing
  `/* ============================================================ name */` ones.
- **Don't describe history in the code** ("this was changed to...", "previously..."). Comments describe the code as
  it is. History goes in `CHANGELOG_v190.txt`.
- **No bug IDs or chat references** ("as discussed", "per the user's request", "fix #3").

### 7.3 Changelog, release notes and messages (the humanizer rules)
The changelog, Discord announcements and anything the user publishes have to sound like a person wrote them. Before
handing text over, check it for these AI patterns and rewrite them out:

1. **No em dashes (—) or en dashes (–).** Use a period, comma, colon or parentheses.
2. **No inflated words:** crucial, pivotal, vital, robust, seamless, comprehensive, leverage, delve, enhance,
   foster, showcase, underscore, testament, landscape, tapestry, vibrant, groundbreaking, game-changer, elevate.
3. **No sales tone** ("boasts", "stunning", "breathtaking", "a major step forward", "exciting times ahead").
4. **No "not just X, it's Y"** or "not only... but also" constructions, and no forced groups of three.
5. **No "-ing" add-ons that fake depth** ("..., ensuring smooth gameplay", "..., highlighting the progress").
6. **Plain verbs:** "is", "has", "fixes", not "serves as", "stands as", "features".
7. **No filler or hedging:** "in order to" → "to"; cut "it's important to note that", "basically", "could
   potentially".
8. **No chatbot leftovers:** "I hope this helps", "Let me know if...", "Great question", "Here's a...".
9. **No bold mini-headings in lists** ("**Performance:** improved..."), no emoji bullets, no Title Case Headings.
10. **No vague endings** ("the future looks bright"). End on the last concrete fact.
11. **Straight quotes** ("), not curly ones.
12. **Concrete over general:** name the game, the symptom and the GPU ("Odyssey froze on its first 3D frame on Intel
    HD"), not "improved stability on some hardware".

Changelog entries follow the existing style: one fact per line or short paragraph, past symptom then what changed:
```
Intel on Windows: Tomodachi Life froze a few seconds after it started using compute shaders. The GL thread checked
the compute fences without the driver lock while the compile thread was inside the driver. It takes the lock now.
```

Discord and Threads posts are in English, short, in the user's voice: what's new, what's fixed, what still doesn't
work. If the user gives a sample of their own writing, match its tone, sentence length and punctuation over these
rules.

---

## 8. Working method

1. Reproduce from the user's files: log first, then `threads.txt` / `performance.txt`, then the F5 trace.
2. Find the first wrong thing (first wrong draw, first stalled thread, first slow pass) and its inputs.
3. Prove the cause: an `SWR_*` switch, a replay, or a trace comparison. Say what is proven and what is a guess.
4. Fix it in `runtime/`, build all four runtimes (section 2), add a line to `CHANGELOG_v190.txt`.
5. Send the user the runtime to copy over the game's executable, with exactly what to test and what to send back.
