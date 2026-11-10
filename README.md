# Switch2Recomp

> **This project is an independent open-source research tool. It is not affiliated with, endorsed, sponsored, or supported by Nintendo. All trademarks belong to their respective owners. This project contains no copyrighted code, assets, or cryptographic keys.**

Switch2Recomp recompiles Switch 2 games to run natively on Windows. It isn't an emulator: the game's ARM64 code is translated to C and compiled with GCC. Its graphics (NVN) are replayed on Vulkan, with its shaders translated ahead of time.

Requires a decrypted executable file provided by the user. Everything is built locally, on the user's PC. Nothing built from a game is distributed, and what the tools make from a game must stay on the PC it was made on.

**Status: early developer preview.** The one supported game is **MKW 1.0.0**, through **[MKW Recomp](https://github.com/visibou/mkwrecomp)**, a separate repository built on this one. Other games may build and start, but nothing beyond that is supported yet, so please don't open issues for them.

**Switch2Recomp by visibou**, a modified version of [SwitchRecomp by MZ Labs](NOTICE.txt), changed from 2026-11-01 on.

## MKW Recomp

To play MKW, use **[MKW Recomp](https://github.com/visibou/mkwrecomp)**. It has a builder window, checks the version of the files you provide, and builds the game locally in about 50 minutes. Switch2Recomp is included in it as a submodule, so you don't need to clone this repository separately.

## Developing Switch2Recomp

What you need: Windows 10 or 11 (64-bit), [Git](https://git-scm.com), [Python 3.10+](https://www.python.org) and the [Vulkan SDK](https://vulkan.lunarg.com).

```
git clone https://github.com/visibou/Switch2Recomp
cd Switch2Recomp
Install.bat                        (downloads Python and GCC into this folder, once)
python -m pip install zstandard
python tools\get_external.py       (fetches the outside dependencies into external\, ~650 MB, once)
python tools\s2r_paths.py          (shows what was found)
```

Switch2Recomp builds with GCC only. `tools/recompile_game.py` recompiles a dumped game; `tools/port_game.py` adds its shader cache; `docs/RECOMPILE_A_GAME.md` explains both.

## How it works

| Part | Where |
|---|---|
| ARM64 to C translation, build driver | `switchrecomp.py`, `codegen.py`, `a64.py`, `nxloader.py` |
| Runtime: the Switch 2 system services the game calls, on Windows | `runtime/` |
| NVN replayed on Vulkan | `runtime/nvn/` (`vk/` is the Vulkan backend) |
| Shader translation (Switch 2 shaders to SPIR-V) | `tools/nxshader/`, with nxnative's translator |
| Textures converted ahead of play (ASTC to BC) | `tools/texprep/` |
| Recompile any dumped game (experimental) | `tools/recompile_game.py`, `tools/port_game.py` |
| Where every tool finds GCC, Python and dependencies | `tools/s2r_paths.py`, `tools/env.sh` |

`docs/` holds development notes: plans, handoffs between work sessions, debugging tools and measurements. They are working notes rather than user documentation.

## Outside dependencies

`tools/get_external.py` fetches these into `external/` at the exact versions Switch2Recomp is tested with:

- [nxnative](https://github.com/seedds/nxnative) (MIT): game shader profiles, the shader split, and the shader translator, which is built on Ryujinx's shader code (MIT).
- [pokopia-native](https://github.com/xFractals/pokopia-native) (MIT; its `ampere-lower` is GPL): shader installer pieces.
- capstone, lz4 and backports.zstd from PyPI.

`Install.bat` downloads Python (from nuget.org) and GCC / MinGW-w64 (from conda-forge), checking each download against a fixed SHA-256 (`tool/get_deps.py`). The Vulkan SDK supplies glslang, spirv-val, spirv-opt and the Vulkan headers.

## License and credits

Switch2Recomp is free software under the **GNU Affero General Public License v3** ([LICENSE.txt](LICENSE.txt)), with the additional terms in [NOTICE.txt](NOTICE.txt). Anyone who conveys it, or a work based on it, must:

- keep the SwitchRECOMP startup screen;
- keep the NOTICE file;
- keep the credit "SwitchRecomp by MZ Labs";
- publish their source under the same license.

Credits:

- **SwitchRecomp** by MZ Labs: the recompiler and runtime this project is built on. Some fixes from Mythrax-ZS's fork are under the Apache License 2.0.
- **nxnative** by seedds and **pokopia-native** by xFractals (MIT): NVN, shader and texture layout work this project builds on (`runtime/nvn/s2tex` comes from pokopia-native).
- **astcenc** (Arm, Apache 2.0), **bc7enc** (Rich Geldreich, MIT / public domain), **stb_dxt** (public domain), **RenderDoc's API header** (MIT), in `runtime/nvn/vk/third_party/`.
- The components listed in [README.txt](README.txt): OpenH264, Opus, SDL2, SPIRV-Cross, texture2ddecoder, fonts.

Each component keeps its own license; see its folder.
