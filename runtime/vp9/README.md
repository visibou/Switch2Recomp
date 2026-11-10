# VP9 decoder

`vp9video.inc` (included by `host1x.c`) plays the VP9 WebM movies games decode through NVDEC. It is adapted from
[Metroid Dread Recomp](https://github.com/DaniSald/Metroid-Dread-Recomp/tree/main/runtime) (AGPL-3.0). libvpx is
BSD-3-Clause (`LICENSE-libvpx.txt`, `PATENTS-libvpx.txt`).

`lin/libvpx.a` and `win/libvpx.a` are built from source (see the README in each folder). `switchrecomp.py
--build-runtime` links the one for the target when it exists and defines `SWR_HAVE_VP9`. Without it the runtime
still builds and VP9 movies stay black.
