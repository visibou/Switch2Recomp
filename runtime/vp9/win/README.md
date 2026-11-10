# libvpx for Windows

`libvpx.a` is libvpx 1.15.2 from MSYS2's UCRT64 package `mingw-w64-ucrt-x86_64-libvpx-1.15.2-1-any.pkg.tar.zst`
(repo.msys2.org/mingw/ucrt64), static, built for the UCRT C runtime like the conda-forge mingw-w64 GCC this project
uses. Its decoder ABI (VPX_IMAGE_ABI_VERSION 5, VPX_CODEC_ABI_VERSION 4 + image, VPX_DECODER_ABI_VERSION 3 + codec)
matches the 1.14 headers in `../include/vpx/`. BSD-3-Clause, see `../LICENSE-libvpx.txt` and `../PATENTS-libvpx.txt`.
