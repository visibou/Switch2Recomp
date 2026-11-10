# M5 track B: the Vulkan backend (docs/m5-vulkan.md), built into nxrt_core and chosen with nxrt --gfx vulkan. Needs the Vulkan SDK (headers and
# glslangValidator; the loader is opened at run time) and SDL3's development package; without them nxrt builds with the null backend only.
if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/Include/vulkan/vulkan.h")
  set(_vk_default "$ENV{VULKAN_SDK}")
else()
  set(_vk_default "C:/VulkanSDK/1.4.363.0")
endif()
set(NXRT_VULKAN_SDK "${_vk_default}" CACHE PATH "Vulkan SDK (headers, glslangValidator)")
set(NXRT_SDL3_DIR "${CMAKE_SOURCE_DIR}/../toolchains/SDL3-3.4.18" CACHE PATH "SDL3 development package (SDL3-devel-<version>-VC, unpacked)")
find_program(NXRT_GLSLANG glslangValidator HINTS "${NXRT_VULKAN_SDK}/Bin" NO_DEFAULT_PATH)
if(NOT EXISTS "${NXRT_VULKAN_SDK}/Include/vulkan/vulkan.h" OR NOT EXISTS "${NXRT_SDL3_DIR}/include/SDL3/SDL.h" OR NOT NXRT_GLSLANG)
  message(STATUS "nxrt: the Vulkan backend is not built (Vulkan SDK at ${NXRT_VULKAN_SDK}, SDL3 at ${NXRT_SDL3_DIR}): --gfx null only")
  return()
endif()
find_package(SDL3 CONFIG REQUIRED PATHS "${NXRT_SDL3_DIR}/cmake" NO_DEFAULT_PATH)
message(STATUS "nxrt: Vulkan backend with SDL3 ${SDL3_VERSION} (${NXRT_SDL3_DIR}) and the Vulkan SDK headers (${NXRT_VULKAN_SDK})")

# The backend's own shaders, compiled to SPIR-V headers
set(_spv)
foreach(sh present.vert present.frag test_fill.comp test_pervertex.vert test_pervertex_cull.vert test_bind.frag clear.vert clear_f.frag clear_u.frag clear_i.frag test_dual.frag test_tri.vert test_ballot.comp test_ballot_used.comp)
  string(REPLACE "." "_" var ${sh})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${var}.h
    COMMAND ${NXRT_GLSLANG} -V --target-env vulkan1.3 --vn ${var}_spv -o ${CMAKE_CURRENT_BINARY_DIR}/${var}.h ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${sh}
    DEPENDS shaders/${sh} VERBATIM)
  list(APPEND _spv ${CMAKE_CURRENT_BINARY_DIR}/${var}.h)
endforeach()

# Third party (third_party/*/LICENSE*, docs/PROVENANCE.md): astc-encoder 5.7.0 (Apache-2.0), bc7enc_rdo b943862 (MIT / Unlicense)
file(GLOB _astcenc_src CONFIGURE_DEPENDS third_party/astcenc/*.cpp)
add_library(nxrt_astcenc STATIC ${_astcenc_src})
target_compile_definitions(nxrt_astcenc PRIVATE ASTCENC_NEON=0 ASTCENC_SVE=0 ASTCENC_SSE=41 ASTCENC_AVX=2 ASTCENC_X86_GATHERS=0 ASTCENC_POPCNT=1 ASTCENC_F16C=1)
target_compile_options(nxrt_astcenc PRIVATE -mavx2 -mfma -mpopcnt -mf16c -w)   # DESIGN D6: x86-64-v3 is the minimum CPU
add_library(nxrt_bc7enc STATIC third_party/bc7enc/bc7enc.cpp third_party/bc7enc/bc7decomp.cpp)
target_compile_options(nxrt_bc7enc PRIVATE -w)

add_library(nxrt_vk OBJECT
  gfx_vk.c vk_fn.c vk_dev.c vk_window.c vk_submit.c vk_defer.c vk_present.c vk_pcache.c vk_mem.c vk_fmt.c vk_tex.c vk_astc.cpp
  vk_json.c vk_aftermath.c vk_shader.c vk_pipe.c vk_replay.c vk_shot.c ${_spv})
target_include_directories(nxrt_vk PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} "${NXRT_VULKAN_SDK}/Include" "${NXRT_SDL3_DIR}/include"
  ${CMAKE_CURRENT_SOURCE_DIR}/../gfx ${CMAKE_CURRENT_SOURCE_DIR}/../host ${CMAKE_CURRENT_SOURCE_DIR}/../front ${CMAKE_SOURCE_DIR}/src/s2tex)
target_compile_definitions(nxrt_vk PRIVATE NXRT_GFX_VK VK_USE_PLATFORM_WIN32_KHR)
set_target_properties(nxrt_vk PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(nxrt_vk PRIVATE -fms-extensions $<$<COMPILE_LANGUAGE:C>:-Wno-sign-compare -Wno-missing-field-initializers -Wno-missing-braces>)

# Into the runtime: gfx_select.c lists the backend, the executables link SDL3 (delay-loaded: nxrt starts without SDL3.dll under --gfx null)
target_sources(nxrt_core PRIVATE $<TARGET_OBJECTS:nxrt_vk>)
# M6 (docs/m6-input-gameplay.md): the SDL3 front end (keyboard, mouse look, controllers) and the SDL3 audio output, built with the runtime's own includes
target_sources(nxrt_core PRIVATE ../front/front_sdl.c ../front/audio_sdl.c)
target_include_directories(nxrt_core PRIVATE ../front)
target_compile_definitions(nxrt_core PRIVATE NXRT_GFX_VK)
target_link_libraries(nxrt_core PUBLIC nxrt_astcenc nxrt_bc7enc SDL3::SDL3 delayimp shell32 ole32)
target_link_options(nxrt_core INTERFACE "LINKER:/DELAYLOAD:SDL3.dll")
file(COPY "${NXRT_SDL3_DIR}/lib/x64/SDL3.dll" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Tests (docs/m5-vulkan.md §6): skipped (77) without a Vulkan GPU or without the data they need
set(NXRT_VK_TEST_INC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} "${NXRT_VULKAN_SDK}/Include" "${NXRT_SDL3_DIR}/include" ${CMAKE_CURRENT_SOURCE_DIR}/../gfx ${CMAKE_SOURCE_DIR}/src/s2tex)
foreach(t device dirty tex astc pcache replay windows ballot)
  nxrt_executable(nxrt-tests-vk-${t} tests/t_vk_${t}.c)
  target_include_directories(nxrt-tests-vk-${t} PRIVATE ${NXRT_VK_TEST_INC})
  target_compile_options(nxrt-tests-vk-${t} PRIVATE -fms-extensions -Wno-sign-compare -Wno-missing-field-initializers -Wno-missing-braces)
  add_test(NAME nxrt.vk_${t} COMMAND nxrt-tests-vk-${t})
  set_tests_properties(nxrt.vk_${t} PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "NXRT_LOG_DIR=${CMAKE_BINARY_DIR}/nxrt-test-logs;NXRT_VK_CACHE_DIR=${CMAKE_BINARY_DIR}/vk-test-cache")
endforeach()
# The pool windows once more, side by side (no sparse aliasing): a range across a window edge is refused
add_test(NAME nxrt.vk_windows_side COMMAND nxrt-tests-vk-windows)
set_tests_properties(nxrt.vk_windows_side PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "NXRT_VK_NO_ALIAS=1;NXRT_LOG_DIR=${CMAKE_BINARY_DIR}/nxrt-test-logs;NXRT_VK_CACHE_DIR=${CMAKE_BINARY_DIR}/vk-test-cache")
target_link_libraries(nxrt-tests-vk-astc PRIVATE nxrt_astcenc nxrt_bc7enc)
target_sources(nxrt-tests-vk-pcache PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_fill_comp.h)
target_sources(nxrt-tests-vk-ballot PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_ballot_comp.h ${CMAKE_CURRENT_BINARY_DIR}/test_ballot_used_comp.h)
target_sources(nxrt-tests-vk-replay PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test_pervertex_vert.h ${CMAKE_CURRENT_BINARY_DIR}/test_pervertex_cull_vert.h ${CMAKE_CURRENT_BINARY_DIR}/test_bind_frag.h
  ${CMAKE_CURRENT_BINARY_DIR}/test_dual_frag.h ${CMAKE_CURRENT_BINARY_DIR}/test_tri_vert.h)
# M6 (docs/m6-input-gameplay.md): input (scripted input, the front end's mapping) and the SDL3 sound card on SDL's dummy driver
nxrt_executable(nxrt-tests-input ../tests/test_input.c ../tests/guest_test.c)
set(_nxrt_dir ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(nxrt-tests-input PRIVATE ${_nxrt_dir}/plat ${_nxrt_dir}/host ${_nxrt_dir}/hle ${_nxrt_dir}/gen/out ${_nxrt_dir}/front "${NXRT_SDL3_DIR}/include")
add_test(NAME nxrt.input COMMAND nxrt-tests-input)
nxrt_executable(nxrt-tests-audio-sdl ../tests/test_audio_sdl.c)
target_include_directories(nxrt-tests-audio-sdl PRIVATE ${_nxrt_dir}/plat ${_nxrt_dir}/host ${_nxrt_dir}/hle)
add_test(NAME nxrt.audio_sdl COMMAND nxrt-tests-audio-sdl)
nxrt_executable(nxrt-tests-pad-sdl ../tests/test_pad_sdl.c ../tests/guest_test.c)
target_include_directories(nxrt-tests-pad-sdl PRIVATE ${_nxrt_dir}/plat ${_nxrt_dir}/host ${_nxrt_dir}/hle ${_nxrt_dir}/gen/out ${_nxrt_dir}/front)
add_test(NAME nxrt.pad_sdl COMMAND nxrt-tests-pad-sdl)
set_tests_properties(nxrt.input nxrt.audio_sdl nxrt.pad_sdl PROPERTIES ENVIRONMENT "NXRT_LOG_DIR=${CMAKE_BINARY_DIR}/nxrt-test-logs")
set_tests_properties(nxrt.pad_sdl PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(nxrt.vk_tex PROPERTIES ENVIRONMENT "NXRT_LOG_DIR=${CMAKE_BINARY_DIR}/nxrt-test-logs;NXRT_VK_CACHE_DIR=${CMAKE_BINARY_DIR}/vk-test-cache;S2TEX_CORPUS=${CMAKE_SOURCE_DIR}/_local/s2tex/corpus")
