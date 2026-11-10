# Debugging the native GPU path

Replaying a five-minute recording to look at one flash gives little evidence. These tools work on the running game and on what it saved. They apply to `SWR_GPU=nvn` (the Vulkan renderer in `runtime/nvn/vk`).

Test on `games/MKWTest`, never on the installed game: it uses a saved-game copy, and a replay brings its own save.

## Keys in the game

| Key | What it does |
|---|---|
| F5 | Saves the last 120 frames and what each one drew (the flight recorder). The window title says so for 3 seconds |
| F3 | Screenshot |
| Shift+Esc | Quits. Esc alone does nothing now, because it closed the game by accident |

`NXRT_VK_FLIGHT=0` turns the recorder off. Use it for benchmarks.

## `tools/gamectl.py`: control the running game

Commands go through the named pipe `\\.\pipe\swr_ctl` (`runtime/nvn/vk/vk_ctl.c`). Only one game can own it.

```bash
python tools/gamectl.py start --replay rec_20261103_214452 --seconds 500 --name test
```

```bash
python tools/gamectl.py wait-frame 3300
```

| Command | Use |
|---|---|
| `status` | Frame, pause state, GPU fence counts, last recording |
| `flight [label]` | The same as F5; prints the folder |
| `flashes [N] [BLOCKS]` | Scores the last N frames for one-frame flashes, in memory: the largest connected piece must pass BLOCKS 8x8 blocks (default 60; distant panels are under 140). Use this in loops: a recording is 550 MB, and 40 GB of them stopped a test by filling the disk |
| `pause`, `resume`, `step [N]` | Holds the game at a present, or advances it N frames |
| `hide KEY...`, `show KEY...`, `showall`, `hidden` | Skips the draws of a vertex or pixel shader while the game runs. Best way to test a suspect |
| `set NAME VALUE`, `get NAME`, `switches` | Renderer switches. `no_mesh`, `wb_whole`, `pending_merge` |
| `passview [N]` | Thumbnails of every pass of the next N frames, depth included |
| `drawstats [N]` | Pixel shader runs and samples of every draw of the next N frames, on the GPU. Finds a draw that suddenly fills the screen. The samples, the last column, count depth-only draws too; -1 where a game query was open |
| `watch N KEY...` | Compares the buffers of a shader's draws: guest memory at submit, the GPU's copy, guest memory when the GPU finished |
| `rawdump N SERIAL...` | Named images at the end of each of the next N frames, for `tools/image_raw.py` |
| `gputime` | One frame timed on the GPU, pass by pass, in the log: totals, then the 28 slowest, with what sat idle before each submission |
| `passdump [N]` | Every bind and draw of N frames, in the log. About 45,000 lines a frame |
| `peek ADDR [WORDS]` | Guest memory |
| `wpage ADDR` | Logs the first write of each burst to a guest page, up to 600 |
| `wtrace ADDR LEN [huge]` | Logs every write into a guest range with its value, host pc and guest registers, up to 3,000. `huge` keeps floats of 1e5 or more and NaN |
| `bp RVA [N]` | Breakpoint in `MKW_code.dll`, RVA = address from `nm` minus its ImageBase. Each hit logs the guest registers and the memory the decoder reads. Keep N small: a thread passes the breakpoint unseen while another steps over it |
| `rdoc` | A RenderDoc capture; start with `--rdoc`. Not tried on this game yet |
| `quit` | Ends the game |

## `tools/flight.py`: read a recording

Recordings go to `<exe folder>/flight/<time>[_label]/`, 550 MB each. Delete old ones when you need the space.

```bash
python tools/flight.py games/MKWTest/flight/20261106_130134
```

This lists the programs that came and went between frames and the frames where part of the picture changed for one frame only.

| Option | Use |
|---|---|
| `--sheet F0 F1` | Contact sheet of up to 24 frames |
| `--png F` | One frame |
| `--draws F [KEY]` | Every draw of a frame, or of one shader |
| `--counts` | Indirect draws whose arguments the CPU read differently from the GPU |
| `--pv F` | On a `pv_` folder: every thumbnail of a frame |
| `--pvdiff A B` | On a `pv_` folder: two frames compared pass by pass |

## NVIDIA Nsight Aftermath: GPU faults and freezes

`NXRT_VK_AFTERMATH=1` turns on GPU crash dumps (`runtime/nvn/vk/vk_aftermath.c`). It needs `GFSDK_Aftermath_Lib.x64.dll` beside the exe: copy it from `external\aftermath\lib\x64` (NVIDIA's Nsight Aftermath SDK, under NVIDIA's licence, not included) into `games/MKWTest` only. It is slower, so never use it for benchmarks.

```bash
python tools/race_runs.py am --runs 4 --rig --aftermath --stop-on-freeze
```

```bash
python tools/aftermath_read.py
```

- Files go to `games/MKWTest/aftermath/<run time>/`: `dump*.nv-gpudmp`, the driver's `shader-*.nvdbg` debug info, `vendor_fault.bin` and `shaders.tsv`, which maps each module's Aftermath hash to our shader key. `aftermath/spv` keeps every module: `<hash>.spv` as given to the driver, `<hash>.src.spv` before the line markers.
- Line markers: an `OpLine` before each instruction, whose line number is the instruction's word offset in `<hash>.src.spv`. `aftermath_read.py` prints the SPIR-V around it. `NXRT_VK_AFTERMATH_LINES=0` leaves them out.
- `--aftermath` uses its own pipeline cache, `games/MKWTest/vk_am`, seeded with the prewarm list, so the user's cache is not filled with marked modules.
- After a loss the game waits up to `NXRT_VK_AFTERMATH_WAIT_S` (300) for the dump. `race_runs.py` leaves a frozen game alone until the log says whether a dump came.

## Shader faults: find the draw, reproduce it outside the game, shrink it

Made on the night of 2026-11-07 to 08 for the GPU freeze, general for any shader.

- **`NXRT_VK_DRAWLOG=KEY[,KEY]`**: the first draw of those shaders is logged with instances, vertices, vertex-stage bindings and the vertex input (formats, streams, strides, rates, divisors, buffer ranges), then a count every 5 s.
- **`python tools/drawtest.py NAME --key KEY [--cache DIR] [--spv-dir DIR]`**: one replay, watched until that shader's first draw and 25 s after: SURVIVED, FROZEN or NOT DRAWN. When a shader's first draw hangs the GPU every time, this tests an idea in about 6 minutes.
- **`NXRT_VK_SPV_DIR=DIR`** replaces `<key>.spv` modules from DIR. **Use an absolute path** (drawtest.py does it for you): a relative one replaces nothing, silently. Check for "SPIR-V replaced" in the log. In Git Bash, `"$T\dir\$v"` leaves `$v` unexpanded: build such strings as `"$T"'\dir'"$v"`.
- **`NXRT_VK_PIPE_STATS=KEY[,KEY]`**: pipelines using those shaders are compiled with `VK_KHR_pipeline_executable_properties`: register count and binary size per stage in the log. NVIDIA (617.42) gives statistics but no machine code.
- **`tools/drawrepro/drawrepro.c`** (build: `x86_64-w64-mingw32-gcc -O1 -I ../external/deps/Vulkan-Headers/include tools/drawrepro/drawrepro.c -o drawrepro.exe`): one draw of a vertex + fragment SPIR-V pair outside the game, everything they bind made from their SPIR-V, every word of every buffer and image set to `--fill F` or `--fillbits HEX`. Prints OK, TIMEOUT or DEVICE_LOST in seconds. The modules as the driver got them are in `games/MKWTest/aftermath/spv/<hash>.src.spv` after an Aftermath run (`shaders.tsv` maps hashes to keys).
- **`spirv-reduce`** (built in `external/spirv-tools-build/tools`) with an interestingness test that runs drawrepro (`games/MKWTest/reduce/hangs.bat`) shrinks a hanging module to what still hangs.
- **`python tools/abshot.py NAME --frames 100,400 --env ...`** then **`--stack A B`**: screenshots at the same frame counts after the race intro for two setups, stacked. Counting frames keeps the moments the same when the speed differs. A pixel measure over a region (as done for the optimizing-mode glow) turns it into an automatic bisection.
- `NXRT_VK_NO_INVARIANT=1`: strips Invariant from vertex shader positions (test only).

## How to chase a visual bug

1. Get the moment: F5, or `flight` on a timer.
2. Look at the frames, then compare a bad frame with the good ones around it.
3. `passview` over a bad frame, then look for the first pass that differs. Depth shows real geometry: the panels below looked like a lighting bug until depth showed them.
4. `hide` the suspects in groups inside one run, with windows of nothing hidden around them. `tools/hide_test.py` does this: it retries until a run has flashes, and writes no recordings.
5. `watch` the shader that remains.

A run proves nothing until it shows the bug: check for it in the same run before reading any other result, or compare with a hidden-nothing window. The flat panels come in stretches, and many runs have none.

## What these tools have found so far

- Draws used a base instance read from guest memory before the GPU wrote it, so GPU-driven objects took other objects' data. Fixed by `VsBaseInstance` in `tools/nxshader/nxs_spv.py`.
- The flat panels, dark frames, vertical streaks and half-screen splits are one bug: real giant geometry, which covers the sky and blocks the sun. Hiding pixel shader `4841d29d86cd486b` stops them.
- Constants of the screen-space passes match the GPU's copy, and the 128x128x5 array stays normal during flashes, so nxnative's issue #1186 has a different cause here.
- The GPU freeze leaves the GPU idle (0%, P8), so it isn't a shader in a loop.
