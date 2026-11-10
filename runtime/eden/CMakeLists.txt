# Eglantine's (Eden-NS2 family) GPU side (video_core + shader_recompiler, Vulkan) as an optional GPU backend of the runtime.
# Built once with tools/build_eden.sh into build/eden; switchrecomp.py links build/eden/*.a when they exist and the
# runtime uses it with SWR_GPU=eden. Eden's own CMakeLists give the source lists; what they link to is built here.
# Host shaders need externals/FidelityFX-FSR next to this file (Eden looks for it under CMAKE_SOURCE_DIR).
cmake_minimum_required(VERSION 3.22)
project(swr_eden CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXT "${CMAKE_CURRENT_SOURCE_DIR}/../../../external" CACHE PATH "external folder: eden-ns2, deps, tools")
# The Eden-family tree: eglantine-130 (Eglantine build-130: task/mesh shaders, NX2 fixes) or eden-ns2 (Eden-NS2
# b2cef0c0). The bridge builds against either (SWR_EDEN_EGLANTINE).
set(EDEN_FLAVOR "eden-ns2" CACHE STRING "eden-ns2 (default, runs MKW) or eglantine (experimental: stalls in MKW, see README)")
if(EDEN_FLAVOR STREQUAL "eden-ns2")
  set(E "${EXT}/eden-ns2")
  set(SWR_EGL 0)
else()
  set(E "${EXT}/eglantine-130")
  set(SWR_EGL 1)
endif()
set(D "${EXT}/deps")
set(ARCHITECTURE_x86_64 ON)
set(ENABLE_OPENGL OFF)
set(ENABLE_RESHADE OFF)
set(ENABLE_LSFG OFF)
set(YUZU_USE_BUNDLED_FFMPEG OFF)
set(GLSLANGVALIDATOR "${EXT}/tools/glslang/bin/glslang.exe")
add_compile_options(-w)
# As Eden's own build (DetectArchitecture.cmake, CMakeLists.txt): without ARCHITECTURE_x86_64 the x86-64 macro JIT
# and the GPU split's macros on its front thread were compiled out (the back thread did all the work)
add_compile_definitions(ARCHITECTURE_x86_64=1)
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-msse4.1> $<$<COMPILE_LANGUAGE:C,CXX>:-mcx16>)
# The CPU level Eden is compiled for (-DSWR_EDEN_MARCH=x86-64-v3 = AVX2, e.g. a Ryzen 5800H; empty: only the above).
# Measured 2026-11-04 on the 5800H: no gain (driving replay 26.7 vs 26.4-26.6 FPS, race start 29.6 vs 29.5): Eden's
# GPU thread halves spend their time waiting on each other, not computing. Off, so the build runs on any x86-64 CPU.
set(SWR_EDEN_MARCH "" CACHE STRING "-march for Eden's code (empty: SSE4.1 only)")
if(SWR_EDEN_MARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-march=${SWR_EDEN_MARCH}>)
endif()
function(create_target_directory_groups)
endfunction()

# include-only targets Eden's CMakeLists link to
add_library(common INTERFACE)
target_include_directories(common INTERFACE "${E}/src" "${CMAKE_BINARY_DIR}/gen" "${D}/fmt/include"
                           "${D}/boost/boost_1_89_0" "${D}/json/include" "${D}/frozen/include")
target_compile_definitions(common INTERFACE FMT_HEADER_ONLY)
add_library(core INTERFACE)
target_link_libraries(core INTERFACE common)
foreach(t stb bc_decoder glad vkh vku vma xbyak_t dyn_t)
  add_library(${t} INTERFACE)
endforeach()
target_include_directories(stb INTERFACE "${E}/externals/stb")
target_include_directories(bc_decoder INTERFACE "${E}/externals/bc_decoder")
target_include_directories(glad INTERFACE "${E}/externals/glad/include")
target_include_directories(vkh INTERFACE "${D}/Vulkan-Headers/include")
target_include_directories(vku INTERFACE "${D}/Vulkan-Utility-Libraries/include")
target_include_directories(vma INTERFACE "${D}/VulkanMemoryAllocator/include")
target_include_directories(xbyak_t INTERFACE "${D}/xbyak")
add_library(xbyak::xbyak ALIAS xbyak_t)
add_library(dynarmic::dynarmic ALIAS dyn_t)
add_library(Vulkan::Headers ALIAS vkh)
add_library(Vulkan::UtilityHeaders ALIAS vku)
add_library(GPUOpen::VulkanMemoryAllocator ALIAS vma)
add_library(fmt::fmt ALIAS common)
add_library(Boost::headers ALIAS common)
add_library(Boost::context ALIAS common)

# sirit (SPIR-V emitter) from source
file(GLOB SIRIT_SRC "${D}/sirit/src/*.cpp" "${D}/sirit/src/instructions/*.cpp")
add_library(sirit_lib STATIC ${SIRIT_SRC})
target_include_directories(sirit_lib PRIVATE "${D}/sirit/src")
target_include_directories(sirit_lib PUBLIC "${D}/sirit/include" "${D}/SPIRV-Headers/include")
add_library(sirit::sirit ALIAS sirit_lib)

set(FFmpeg_INCLUDE_DIR "${D}/ffmpeg/ucrt64/include")
set(FFmpeg_LIBRARIES "")
add_subdirectory("${E}/src/shader_recompiler" shader_recompiler)
add_subdirectory("${E}/src/video_core" video_core)

# small externals + the part of Eden's common video_core uses
add_library(eden_ext STATIC "${E}/externals/bc_decoder/bc_decoder.cpp" "${E}/externals/stb/stb_dxt.cpp" vma.cpp)
target_link_libraries(eden_ext PRIVATE common vkh vma)
target_include_directories(eden_ext PRIVATE "${E}/externals/bc_decoder" "${E}/externals/stb")
set(C "${E}/src/common")
add_library(eden_common STATIC
  ${C}/assert.cpp ${C}/address_space.cpp ${C}/cityhash.cpp ${C}/cpu_features.cpp ${C}/dynamic_library.cpp
  ${C}/fs/file.cpp ${C}/fs/fs.cpp ${C}/fs/fs_util.cpp ${C}/fs/path_util.cpp ${C}/logging.cpp
  ${C}/multi_level_page_table.cpp ${C}/settings.cpp ${C}/settings_common.cpp ${C}/settings_input.cpp
  ${C}/sparse_large_vector.cpp ${C}/steady_clock.cpp ${C}/stream.cpp ${C}/string_util.cpp ${C}/thread.cpp
  ${C}/time_zone.cpp ${C}/x64/rdtsc.cpp ${C}/windows/timer_resolution.cpp
  ${E}/src/core/frontend/emu_window.cpp ${E}/src/core/frontend/framebuffer_layout.cpp)
target_link_libraries(eden_common PUBLIC common xbyak_t)

# our side: the core shim over the runtime's guest memory and the C bridge the runtime calls
add_library(swr_eden STATIC core_shim.cpp eden_gpu.cpp write_traps.cpp ffmpeg_stub.c)
target_link_libraries(swr_eden PRIVATE video_core common vkh vku vma)
target_include_directories(swr_eden PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(swr_eden PRIVATE SWR_EDEN_EGLANTINE=${SWR_EGL})
