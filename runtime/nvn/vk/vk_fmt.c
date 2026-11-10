// The NVN -> Vulkan format table (vk_fmt.h). Written entry by entry in NVN enum order (nvn.h 0x01-0x9C) from the NVN header's names and the Vulkan
// specification's format definitions; nxnative's Metal table (dm/gfx/tex/nvn_fmt_mtl.c, MIT) was the reference for which formats need a conversion
// or a channel swap. Packed 16-bit formats follow the Maxwell texture formats' bit order with red in the low bits; like nxnative's, that order is not
// yet confirmed on real data (VKF_UNVERIFIED) and is to be checked when a game texture first uses one.
#include "vk_fmt.h"
#include "nvn_tex.h"
#include <stdio.h>
#include <string.h>

#define I VK_COMPONENT_SWIZZLE_IDENTITY
#define ONE VK_COMPONENT_SWIZZLE_ONE
#define R_ VK_COMPONENT_SWIZZLE_R
#define G_ VK_COMPONENT_SWIZZLE_G
#define B_ VK_COMPONENT_SWIZZLE_B
#define A_ VK_COMPONENT_SWIZZLE_A
typedef struct { VkFormat ideal, alt; uint8_t conv, flags; VkComponentMapping swz; const char *name; } ent;
#define E(f, fl, nm) {f, 0, VKF_COPY, fl, {I, I, I, I}, nm}
#define X(f, fl, nm) {f, 0, VKF_COPY, fl, {I, I, I, ONE}, nm}                          // alpha reads as one
#undef RGB   // wingdi.h's
#define RGB(f3, f4, fl, nm) {f3, f4, VKF_RGB_TO_RGBA, fl, {I, I, I, ONE}, nm}           // 3 components, expanded to f4
#define SW(f, fl, r, g, b, a, nm) {f, 0, VKF_COPY, fl, {r, g, b, a}, nm}
enum { S = VKF_SRGB, C = VKF_COMPRESSED, N = VKF_INT, F = VKF_I2F, D = VKF_DEPTH, T = VKF_STENCIL, U = VKF_UNVERIFIED };

static const ent T_[0x9D] = {
    [0x01] = E(VK_FORMAT_R8_UNORM, 0, "R8"), E(VK_FORMAT_R8_SNORM, 0, "R8SN"), E(VK_FORMAT_R8_UINT, N, "R8UI"), E(VK_FORMAT_R8_SINT, N, "R8I"),
    [0x05] = E(VK_FORMAT_R16_SFLOAT, 0, "R16F"), E(VK_FORMAT_R16_UNORM, 0, "R16"), E(VK_FORMAT_R16_SNORM, 0, "R16SN"), E(VK_FORMAT_R16_UINT, N, "R16UI"), E(VK_FORMAT_R16_SINT, N, "R16I"),
    [0x0A] = E(VK_FORMAT_R32_SFLOAT, 0, "R32F"), E(VK_FORMAT_R32_UINT, N, "R32UI"), E(VK_FORMAT_R32_SINT, N, "R32I"),
    [0x0D] = E(VK_FORMAT_R8G8_UNORM, 0, "RG8"), E(VK_FORMAT_R8G8_SNORM, 0, "RG8SN"), E(VK_FORMAT_R8G8_UINT, N, "RG8UI"), E(VK_FORMAT_R8G8_SINT, N, "RG8I"),
    [0x11] = E(VK_FORMAT_R16G16_SFLOAT, 0, "RG16F"), E(VK_FORMAT_R16G16_UNORM, 0, "RG16"), E(VK_FORMAT_R16G16_SNORM, 0, "RG16SN"), E(VK_FORMAT_R16G16_UINT, N, "RG16UI"), E(VK_FORMAT_R16G16_SINT, N, "RG16I"),
    [0x16] = E(VK_FORMAT_R32G32_SFLOAT, 0, "RG32F"), E(VK_FORMAT_R32G32_UINT, N, "RG32UI"), E(VK_FORMAT_R32G32_SINT, N, "RG32I"),
    // 3-component formats: sampling them is optional in Vulkan and they cannot be render targets or storage images: always uploaded expanded to 4
    [0x19] = RGB(VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 0, "RGB8"), RGB(VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, 0, "RGB8SN"),
             RGB(VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT, N, "RGB8UI"), RGB(VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT, N, "RGB8I"),
    [0x1D] = RGB(VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 0, "RGB16F"), RGB(VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 0, "RGB16"),
             RGB(VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16A16_SNORM, 0, "RGB16SN"), RGB(VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT, N, "RGB16UI"),
             RGB(VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT, N, "RGB16I"),
    [0x22] = RGB(VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 0, "RGB32F"), RGB(VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT, N, "RGB32UI"),
             RGB(VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT, N, "RGB32I"),
    [0x25] = E(VK_FORMAT_R8G8B8A8_UNORM, 0, "RGBA8"), E(VK_FORMAT_R8G8B8A8_SNORM, 0, "RGBA8SN"), E(VK_FORMAT_R8G8B8A8_UINT, N, "RGBA8UI"), E(VK_FORMAT_R8G8B8A8_SINT, N, "RGBA8I"),
    [0x29] = E(VK_FORMAT_R16G16B16A16_SFLOAT, 0, "RGBA16F"), E(VK_FORMAT_R16G16B16A16_UNORM, 0, "RGBA16"), E(VK_FORMAT_R16G16B16A16_SNORM, 0, "RGBA16SN"),
             E(VK_FORMAT_R16G16B16A16_UINT, N, "RGBA16UI"), E(VK_FORMAT_R16G16B16A16_SINT, N, "RGBA16I"),
    [0x2E] = E(VK_FORMAT_R32G32B32A32_SFLOAT, 0, "RGBA32F"), E(VK_FORMAT_R32G32B32A32_UINT, N, "RGBA32UI"), E(VK_FORMAT_R32G32B32A32_SINT, N, "RGBA32I"),
    [0x31] = E(VK_FORMAT_S8_UINT, T, "STENCIL8"), E(VK_FORMAT_D16_UNORM, D, "DEPTH16"),
    [0x33] = E(VK_FORMAT_X8_D24_UNORM_PACK32, D, "DEPTH24"), E(VK_FORMAT_D32_SFLOAT, D, "DEPTH32F"),
    [0x35] = E(VK_FORMAT_D24_UNORM_S8_UINT, D | T, "DEPTH24_STENCIL8"), E(VK_FORMAT_D32_SFLOAT_S8_UINT, D | T, "DEPTH32F_STENCIL8"),
    [0x37] = X(VK_FORMAT_R8G8B8A8_SRGB, S, "RGBX8_SRGB"), E(VK_FORMAT_R8G8B8A8_SRGB, S, "RGBA8_SRGB"),
    // Packed 16-bit: red in the low bits (A4B4G4R4, A1B5G5R5 and B5G6R5 in Vulkan's most-significant-first names)
    [0x39] = E(VK_FORMAT_A4B4G4R4_UNORM_PACK16, U, "RGBA4"),
             SW(VK_FORMAT_A1R5G5B5_UNORM_PACK16, U, B_, G_, R_, ONE, "RGB5"), SW(VK_FORMAT_A1R5G5B5_UNORM_PACK16, U, B_, G_, R_, A_, "RGB5A1"),
             E(VK_FORMAT_B5G6R5_UNORM_PACK16, U, "RGB565"),
    [0x3D] = E(VK_FORMAT_A2B10G10R10_UNORM_PACK32, 0, "RGB10A2"), E(VK_FORMAT_A2B10G10R10_UINT_PACK32, N, "RGB10A2UI"),
             E(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 0, "R11G11B10F"), E(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 0, "RGB9E5F"),
    [0x41] = E(VK_FORMAT_BC1_RGB_UNORM_BLOCK, C, "RGB_DXT1"), E(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, C, "RGBA_DXT1"), E(VK_FORMAT_BC2_UNORM_BLOCK, C, "RGBA_DXT3"),
             E(VK_FORMAT_BC3_UNORM_BLOCK, C, "RGBA_DXT5"), E(VK_FORMAT_BC1_RGB_SRGB_BLOCK, C | S, "RGB_DXT1_SRGB"), E(VK_FORMAT_BC1_RGBA_SRGB_BLOCK, C | S, "RGBA_DXT1_SRGB"),
             E(VK_FORMAT_BC2_SRGB_BLOCK, C | S, "RGBA_DXT3_SRGB"), E(VK_FORMAT_BC3_SRGB_BLOCK, C | S, "RGBA_DXT5_SRGB"),
    [0x49] = E(VK_FORMAT_BC4_UNORM_BLOCK, C, "RGTC1_UNORM"), E(VK_FORMAT_BC4_SNORM_BLOCK, C, "RGTC1_SNORM"), E(VK_FORMAT_BC5_UNORM_BLOCK, C, "RGTC2_UNORM"),
             E(VK_FORMAT_BC5_SNORM_BLOCK, C, "RGTC2_SNORM"), E(VK_FORMAT_BC7_UNORM_BLOCK, C, "BPTC_UNORM"), E(VK_FORMAT_BC7_SRGB_BLOCK, C | S, "BPTC_UNORM_SRGB"),
             E(VK_FORMAT_BC6H_SFLOAT_BLOCK, C, "BPTC_SFLOAT"), E(VK_FORMAT_BC6H_UFLOAT_BLOCK, C, "BPTC_UFLOAT"),
    // *_UI2F / *_I2F: integer storage read as float without normalising. Vulkan's USCALED / SSCALED formats are vertex-only on most GPUs: the integer format,
    // and the shader translator converts (VKF_I2F; nxnative's Metal table does the same)
    [0x51] = E(VK_FORMAT_R8_UINT, N | F, "R8_UI2F"), E(VK_FORMAT_R8_SINT, N | F, "R8_I2F"), E(VK_FORMAT_R16_UINT, N | F, "R16_UI2F"), E(VK_FORMAT_R16_SINT, N | F, "R16_I2F"),
             E(VK_FORMAT_R32_UINT, N | F, "R32_UI2F"), E(VK_FORMAT_R32_SINT, N | F, "R32_I2F"),
    [0x57] = E(VK_FORMAT_R8G8_UINT, N | F, "RG8_UI2F"), E(VK_FORMAT_R8G8_SINT, N | F, "RG8_I2F"), E(VK_FORMAT_R16G16_UINT, N | F, "RG16_UI2F"), E(VK_FORMAT_R16G16_SINT, N | F, "RG16_I2F"),
             E(VK_FORMAT_R32G32_UINT, N | F, "RG32_UI2F"), E(VK_FORMAT_R32G32_SINT, N | F, "RG32_I2F"),
    [0x5D] = RGB(VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT, N | F, "RGB8_UI2F"), RGB(VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT, N | F, "RGB8_I2F"),
             RGB(VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT, N | F, "RGB16_UI2F"), RGB(VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT, N | F, "RGB16_I2F"),
             RGB(VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT, N | F, "RGB32_UI2F"), RGB(VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT, N | F, "RGB32_I2F"),
    [0x63] = E(VK_FORMAT_R8G8B8A8_UINT, N | F, "RGBA8_UI2F"), E(VK_FORMAT_R8G8B8A8_SINT, N | F, "RGBA8_I2F"), E(VK_FORMAT_R16G16B16A16_UINT, N | F, "RGBA16_UI2F"),
             E(VK_FORMAT_R16G16B16A16_SINT, N | F, "RGBA16_I2F"), E(VK_FORMAT_R32G32B32A32_UINT, N | F, "RGBA32_UI2F"), E(VK_FORMAT_R32G32B32A32_SINT, N | F, "RGBA32_I2F"),
    [0x69] = {VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_R16G16B16A16_SNORM, VKF_COPY, 0, {I, I, I, I}, "RGB10A2SN"},
             {VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_R16G16B16A16_SINT, VKF_COPY, N, {I, I, I, I}, "RGB10A2I"},
             E(VK_FORMAT_A2B10G10R10_UINT_PACK32, N | F, "RGB10A2_UI2F"),
             {VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_R16G16B16A16_SINT, VKF_COPY, N | F, {I, I, I, I}, "RGB10A2_I2F"},
    [0x6D] = X(VK_FORMAT_R8G8B8A8_UNORM, 0, "RGBX8"), X(VK_FORMAT_R8G8B8A8_SNORM, 0, "RGBX8SN"), X(VK_FORMAT_R8G8B8A8_UINT, N, "RGBX8UI"), X(VK_FORMAT_R8G8B8A8_SINT, N, "RGBX8I"),
    [0x71] = X(VK_FORMAT_R16G16B16A16_SFLOAT, 0, "RGBX16F"), X(VK_FORMAT_R16G16B16A16_UNORM, 0, "RGBX16"), X(VK_FORMAT_R16G16B16A16_SNORM, 0, "RGBX16SN"),
             X(VK_FORMAT_R16G16B16A16_UINT, N, "RGBX16UI"), X(VK_FORMAT_R16G16B16A16_SINT, N, "RGBX16I"),
    [0x76] = X(VK_FORMAT_R32G32B32A32_SFLOAT, 0, "RGBX32F"), X(VK_FORMAT_R32G32B32A32_UINT, N, "RGBX32UI"), X(VK_FORMAT_R32G32B32A32_SINT, N, "RGBX32I"),
    // 0x79-0x94 ASTC: computed below
    [0x95] = E(VK_FORMAT_R5G6B5_UNORM_PACK16, U, "BGR565"), X(VK_FORMAT_A1R5G5B5_UNORM_PACK16, U, "BGR5"), E(VK_FORMAT_A1R5G5B5_UNORM_PACK16, U, "BGR5A1"),
             E(VK_FORMAT_B5G5R5A1_UNORM_PACK16, U, "A1BGR5"),
    [0x99] = X(VK_FORMAT_B8G8R8A8_UNORM, 0, "BGRX8"), E(VK_FORMAT_B8G8R8A8_UNORM, 0, "BGRA8"), X(VK_FORMAT_B8G8R8A8_SRGB, S, "BGRX8_SRGB"), E(VK_FORMAT_B8G8R8A8_SRGB, S, "BGRA8_SRGB"),
};
static const char *const astc_names[14] = {"4x4", "5x4", "5x5", "6x5", "6x6", "8x5", "8x6", "8x8", "10x5", "10x6", "10x8", "10x10", "12x10", "12x12"};
static const VkFormat astc_unorm[14] = {VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_5x4_UNORM_BLOCK, VK_FORMAT_ASTC_5x5_UNORM_BLOCK, VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    VK_FORMAT_ASTC_6x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x5_UNORM_BLOCK, VK_FORMAT_ASTC_8x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
    VK_FORMAT_ASTC_10x6_UNORM_BLOCK, VK_FORMAT_ASTC_10x8_UNORM_BLOCK, VK_FORMAT_ASTC_10x10_UNORM_BLOCK, VK_FORMAT_ASTC_12x10_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_UNORM_BLOCK};

static vkf_info g_fi[0x9D];
static char g_astc_name[28][24];

static VkFormatFeatureFlags feats(VkFormat f) { VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(g_vk.pd, f, &p); return p.optimalTilingFeatures; }
static uint8_t vk_bpb(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_S8_UINT: return 1;
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_D16_UNORM: case VK_FORMAT_A4B4G4R4_UNORM_PACK16:
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16: case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK: return 16;
    default: break;
    }
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) return 16;
    return 4;   // everything else in the table is 32 bits a texel
}

static void set(int nvn, VkFormat vk, VkFormat ideal, uint8_t conv, uint8_t flags, VkComponentMapping swz, const char *name) {
    vkf_info *e = &g_fi[nvn]; int bw = 1, bh = 1;
    e->bpb = (uint8_t)nvn_fmt_bpb(nvn, &bw, &bh); e->bw = (uint8_t)bw; e->bh = (uint8_t)bh;
    e->vk = vk; e->ideal = ideal; e->conv = conv; e->flags = flags; e->swz = swz; e->name = name;
    e->vbpb = vk ? vk_bpb(vk) : 0;
    e->feat = vk ? feats(vk) : 0;
}

void vkf_init(void) {
    memset(g_fi, 0, sizeof g_fi);
    for (int f = 1; f <= 0x9C; f++) {
        if (f >= 0x79 && f <= 0x94) {   // ASTC: native when the device samples it (not on desktop GPUs), else BC7 through the transcoder, else RGBA8
            int k = (f - 0x79) % 14, srgb = f >= 0x87;
            VkFormat native = (VkFormat)(astc_unorm[k] + (srgb ? 1 : 0));   // each _SRGB_BLOCK follows its _UNORM_BLOCK
            snprintf(g_astc_name[f - 0x79], sizeof g_astc_name[0], "RGBA_ASTC_%s%s", astc_names[k], srgb ? "_SRGB" : "");
            VkComponentMapping id = {I, I, I, I};
            if (g_vk.has.astc_ldr && (feats(native) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) set(f, native, native, VKF_COPY, C | (srgb ? S : 0), id, g_astc_name[f - 0x79]);
            else if (g_vk.has.bc) set(f, srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK, native, VKF_ASTC_BC7, C | VKF_FALLBACK | (srgb ? S : 0), id, g_astc_name[f - 0x79]);
            else set(f, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, native, VKF_ASTC_RGBA8, C | VKF_FALLBACK | (srgb ? S : 0), id, g_astc_name[f - 0x79]);
            continue;
        }
        const ent *t = &T_[f];
        if (!t->ideal) continue;
        VkFormat use = t->ideal; uint8_t conv = t->conv, fl = t->flags; VkComponentMapping swz = t->swz;
        if (conv == VKF_RGB_TO_RGBA) use = t->alt;
        VkFormatFeatureFlags need = (fl & (VKF_DEPTH | VKF_STENCIL)) ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((feats(use) & need) != need) {
            fl |= VKF_FALLBACK;
            switch (f) {
            case 0x31: use = (feats(VK_FORMAT_D24_UNORM_S8_UINT) & need) ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT; conv = VKF_S8_DS; break;   // stencil only
            case 0x33: use = VK_FORMAT_D32_SFLOAT; conv = VKF_D24_D32F; break;                                     // research/01 lesson 8: D24 -> D32
            case 0x35: use = VK_FORMAT_D32_SFLOAT_S8_UINT; conv = VKF_D24S8_D32FS8; break;                       // D24S8 -> D32S8 (AMD)
            case 0x39: use = VK_FORMAT_R4G4B4A4_UNORM_PACK16; swz = (VkComponentMapping){A_, B_, G_, R_}; fl &= ~VKF_FALLBACK; break;   // the same bits, read reversed
            case 0x69: case 0x6A: case 0x6C: use = t->alt; conv = VKF_RGB10A2S_RGBA16; break;
            default: break;   // kept, and logged by vkf_log_table: such a texture is created only if the device can
            }
        }
        set(f, use, t->ideal, conv, fl, swz, t->name);
    }
}

const vkf_info *vkf_get(int f) { return f > 0 && f <= 0x9C && g_fi[f].vk ? &g_fi[f] : NULL; }
int vkf_is_compressed(int f) { return f > 0 && f <= 0x9C && (g_fi[f].flags & VKF_COMPRESSED); }

VkFormat vkf_view_format(int image_f, int view_f) {
    const vkf_info *a = vkf_get(image_f), *b = vkf_get(view_f);
    if (!a) return VK_FORMAT_UNDEFINED;
    if (!b || view_f == image_f) return a->vk;
    // the same conversion and size, differing only in sRGB encoding (ASTC UNORM / sRGB through BC7, BC1 / BC1 sRGB...): the view's own format
    if (a->conv == b->conv && a->vbpb == b->vbpb && a->bw == b->bw && a->bh == b->bh && !((a->flags ^ b->flags) & ~VKF_SRGB & (VKF_DEPTH | VKF_STENCIL | VKF_COMPRESSED | VKF_INT)))
        return b->vk;
    // Converted or substituted images can only be viewed in their own format; same-size reinterpretations of exact formats keep the view's format
    if (a->conv != VKF_COPY || b->conv != VKF_COPY || a->vbpb != b->vbpb || ((a->flags ^ b->flags) & (VKF_DEPTH | VKF_STENCIL | VKF_COMPRESSED))) {
        if (a->conv == VKF_COPY && b->conv == VKF_COPY && a->ideal == b->ideal) return a->vk;
        static int nlog; if (nlog++ < 16) VKLOG("view format %s of a %s image: viewed as %s (the image's format on this device)", b->name, a->name, a->name);
        return a->vk;
    }
    return b->vk;
}

VkImageAspectFlags vkf_aspect(VkFormat f) {
    switch (f) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT: return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

VkFormat vkf_unorm_of(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
    case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
    case VK_FORMAT_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
    default: return f;
    }
}

void vkf_log_table(void) {
    int exact = 0, conv = 0, fb = 0, none = 0;
    for (int f = 1; f <= 0x9C; f++) {
        const vkf_info *e = &g_fi[f];
        if (!e->vk) { none++; continue; }
        if (e->flags & VKF_FALLBACK) fb++; else if (e->conv) conv++; else exact++;
        int ok = (e->flags & (VKF_DEPTH | VKF_STENCIL)) ? (e->feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 : (e->feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        if ((e->flags & VKF_FALLBACK) || !ok)
            VKLOG("format %#x %s -> %d%s%s (conversion %d)", f, e->name, e->vk, e->flags & VKF_FALLBACK ? " (fallback)" : "", ok ? "" : " [not sampled by this device]", e->conv);
    }
    VKLOG("format table: %d exact, %d converted on upload, %d device fallbacks, %d without a counterpart", exact, conv, fb, none);
}
