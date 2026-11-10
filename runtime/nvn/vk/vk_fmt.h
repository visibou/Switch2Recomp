// NVN formats -> Vulkan (docs/m5-vulkan.md §4.1): the format each NVN format is created with on this device, the conversion its upload needs, the
// component mapping its views start from. Fallbacks (D24S8 -> D32S8, ASTC -> BC7, 3-component -> 4-component...) are decided once per device here and
// applied to every view through vkf_view_format (research/01 lesson 8).
#pragma once
#include "vk_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

enum vkf_conv {
    VKF_COPY = 0,        // the storage's bytes as they are
    VKF_RGB_TO_RGBA,     // 3-component texels expanded to 4 (alpha = one of the type)
    VKF_ASTC_BC7,        // ASTC decoded and re-encoded to BC7 (vk_astc.cpp)
    VKF_ASTC_RGBA8,      // ASTC decoded to RGBA8 (no BC on the device)
    VKF_D24_D32F,        // 24-bit depth in 32 bits -> float depth
    VKF_D24S8_D32FS8,    // 24-bit depth + stencil -> float depth + stencil
    VKF_S8_DS,           // stencil only, into a depth-stencil format's stencil aspect
    VKF_RGB10A2S_RGBA16, // signed 10:10:10:2 unpacked to 16 bits a component
};
enum { VKF_SRGB = 1, VKF_COMPRESSED = 2, VKF_INT = 4, VKF_I2F = 8, VKF_DEPTH = 16, VKF_STENCIL = 32, VKF_FALLBACK = 64, VKF_UNVERIFIED = 128 };

typedef struct {
    VkFormat vk;                  // created with this format on this device (0: no Vulkan counterpart at all)
    VkFormat ideal;               // the exact counterpart, before device fallbacks
    uint8_t conv, flags;          // vkf_conv, VKF_*
    uint8_t bpb, bw, bh;          // the NVN storage: bytes per block, block footprint
    uint8_t vbpb;                 // bytes per block of vk as uploaded
    VkComponentMapping swz;       // the base component mapping (RGBX alpha = 1, packed 16-bit channel order)
    VkFormatFeatureFlags2 feat;   // optimal-tiling features of vk on this device
    const char *name;
} vkf_info;

void vkf_init(void);                                  // after vk_dev_open: device support and fallbacks
const vkf_info *vkf_get(int nvn_format);              // NULL for NONE, unknown or unsupported
// The format of a view: the view's NVN format through the image's own fallback. A view that reinterprets an image created with a fallback or a
// conversion cannot be honoured: it gets the image's format (logged once per pair)
VkFormat vkf_view_format(int image_nvn_format, int view_nvn_format);
VkImageAspectFlags vkf_aspect(VkFormat f);
VkFormat vkf_unorm_of(VkFormat f);                    // the UNORM twin of an sRGB format (presentation of sRGB content); f itself otherwise
int vkf_is_compressed(int nvn_format);
void vkf_log_table(void);                             // one line per NVN format with its mapping on this device

#ifdef __cplusplus
}
#endif
