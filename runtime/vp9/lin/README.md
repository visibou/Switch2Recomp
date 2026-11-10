# libvpx for Linux

`libvpx.a` is libvpx 1.14.0 built from the Ubuntu source package `libvpx_1.14.0-1ubuntu2.3` (upstream tarball
SHA-256 11150687870e4c94479abafe5d1ed38790e68dc7bbe08880fd5fb55138e4c6f3, with Ubuntu's security patches for
CVE-2024-5197, CVE-2025-5283 and CVE-2026-2447 applied). Only the VP9 decoder is built:

    ./configure --target=x86_64-linux-gcc --enable-static --disable-shared --enable-pic --disable-vp8 \
      --disable-vp9-encoder --enable-vp9-decoder --disable-examples --disable-tools --disable-docs \
      --disable-unit-tests --enable-multithread --as=nasm

BSD-3-Clause, see `../LICENSE-libvpx.txt` and `../PATENTS-libvpx.txt`. The headers in `../include/vpx/` come from
the same source.
