/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <SDL_ttf.h>
#include "gfx.h"

static SDL_Renderer *g_r;
static TTF_Font *g_fonts[FONT_COUNT];

Color color_mix(Color a, Color b, float t)
{
    return (Color){
        (Uint8)(a.r + (b.r - a.r) * t), (Uint8)(a.g + (b.g - a.g) * t),
        (Uint8)(a.b + (b.b - a.b) * t), (Uint8)(a.a + (b.a - a.a) * t),
    };
}

/* ---- shape cache: white textures with alpha coverage, tinted when drawn ---- */

typedef float (*SdfFn)(float x, float y, const float *p);

typedef struct {
    int kind;
    float p[4];
    int w, h;
    SDL_Texture *tex;
    Uint32 used;
} Shape;

/* shapes get drawn at a few fixed sizes (animations scale them) so the cache stays small. textures are kept and
   never freed and made again, that churn is what made it slow */
static Shape g_shapes[256];
static int g_nshapes;
static Uint32 g_shape_clock;

static Shape *shape_slot(void)
{
    static Shape spare;
    if (g_nshapes < (int)(sizeof(g_shapes) / sizeof(g_shapes[0])))
        return &g_shapes[g_nshapes++];
    if (spare.tex)
        SDL_DestroyTexture(spare.tex);
    return &spare;
}

/* coverage from a signed distance in pixels, 4x4 samples per pixel so thin strokes stay smooth */
static SDL_Texture *rasterize(int w, int h, SdfFn fn, const float *p)
{
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_ABGR8888);
    if (!s)
        return NULL;
    Uint32 *px = s->pixels;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float cover = 0;
            for (int sy = 0; sy < 4; sy++)
                for (int sx = 0; sx < 4; sx++) {
                    float d = fn(x + (sx + 0.5f) / 4, y + (sy + 0.5f) / 4, p);
                    if (d <= 0)
                        cover += 1;
                }
            Uint32 a = (Uint32)(cover / 16 * 255 + 0.5f);
            px[y * (s->pitch / 4) + x] = (a << 24) | 0xFFFFFF;
        }
    SDL_Texture *t = SDL_CreateTextureFromSurface(g_r, s);
    SDL_FreeSurface(s);
    if (t)
        SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
    return t;
}

static SDL_Texture *shape_get(int kind, const float *p, int w, int h, SdfFn fn)
{
    for (int i = 0; i < g_nshapes; i++) {
        Shape *s = &g_shapes[i];
        if (s->kind == kind && s->w == w && s->h == h && !memcmp(s->p, p, sizeof(s->p))) {
            s->used = ++g_shape_clock;
            return s->tex;
        }
    }
    SDL_Texture *t = rasterize(w, h, fn, p);
    *shape_slot() = (Shape){ kind, { p[0], p[1], p[2], p[3] }, w, h, t, ++g_shape_clock };
    return t;
}

static void draw_tinted(SDL_Texture *t, const SDL_Rect *src, float x, float y, float w, float h, Color c)
{
    if (!t)
        return;
    SDL_SetTextureColorMod(t, c.r, c.g, c.b);
    SDL_SetTextureAlphaMod(t, c.a);
    SDL_FRect dst = { x, y, w, h };
    SDL_RenderCopyF(g_r, t, src, &dst);
}

/* ---- signed distance functions, in pixels ---- */

static float len2(float x, float y) { return sqrtf(x * x + y * y); }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

static float sd_box(float px, float py, float hx, float hy, float r)
{
    float qx = fabsf(px) - hx + r, qy = fabsf(py) - hy + r;
    return len2(fmaxf(qx, 0), fmaxf(qy, 0)) + fminf(fmaxf(qx, qy), 0) - r;
}

static float sd_segment(float px, float py, float ax, float ay, float bx, float by, float r)
{
    float pax = px - ax, pay = py - ay, bax = bx - ax, bay = by - ay;
    float h = clampf((pax * bax + pay * bay) / (bax * bax + bay * bay), 0, 1);
    return len2(pax - bax * h, pay - bay * h) - r;
}

/* arc of radius R around (cx,cy) between angles a0..a1 (radians, y down), stroke t, round caps */
static float sd_arc(float px, float py, float cx, float cy, float R, float a0, float a1, float t)
{
    float dx = px - cx, dy = py - cy;
    float a = atan2f(dy, dx);
    while (a < a0)
        a += 2 * (float)M_PI;
    if (a <= a1)
        return fabsf(len2(dx, dy) - R) - t / 2;
    float e0 = len2(dx - R * cosf(a0), dy - R * sinf(a0));
    float e1 = len2(dx - R * cosf(a1), dy - R * sinf(a1));
    return fminf(e0, e1) - t / 2;
}

/* p = { size(w), height, radius, thickness } */
static float sdf_round_rect(float x, float y, const float *p)
{
    return sd_box(x - p[0] / 2, y - p[1] / 2, p[0] / 2, p[1] / 2, p[2]);
}

static float sdf_round_frame(float x, float y, const float *p)
{
    float outer = sd_box(x - p[0] / 2, y - p[1] / 2, p[0] / 2, p[1] / 2, p[2]);
    float inner = sd_box(x - p[0] / 2, y - p[1] / 2, p[0] / 2 - p[3], p[1] / 2 - p[3], fmaxf(p[2] - p[3], 0));
    return fmaxf(outer, -inner);
}


/* soft shadow around a w x h rounded box, kept in the cache as kind 100 */
static SDL_Texture *shadow_tex(int w, int h, float radius, float spread)
{
    float p[4] = { (float)w, (float)h, radius, spread };
    for (int i = 0; i < g_nshapes; i++) {
        Shape *s = &g_shapes[i];
        if (s->kind == 100 && !memcmp(s->p, p, sizeof(p))) {
            s->used = ++g_shape_clock;
            return s->tex;
        }
    }
    int pad = (int)ceilf(spread * 2);
    int tw = w + pad * 2, th = h + pad * 2;
    SDL_Surface *s = SDL_CreateRGBSurfaceWithFormat(0, tw, th, 32, SDL_PIXELFORMAT_ABGR8888);
    if (!s)
        return NULL;
    Uint32 *px = s->pixels;
    for (int y = 0; y < th; y++)
        for (int x = 0; x < tw; x++) {
            float d = sd_box(x + 0.5f - tw / 2.0f, y + 0.5f - th / 2.0f, w / 2.0f, h / 2.0f, radius);
            float a = d <= 0 ? 1 : expf(-(d * d) / (spread * spread * 0.5f));
            px[y * (s->pitch / 4) + x] = ((Uint32)(a * 255) << 24) | 0xFFFFFF;
        }
    SDL_Texture *t = SDL_CreateTextureFromSurface(g_r, s);
    SDL_FreeSurface(s);
    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND);
    *shape_slot() = (Shape){ 100, { p[0], p[1], p[2], p[3] }, tw, th, t, ++g_shape_clock };
    return t;
}

/* ---- icons, drawn in a unit square scaled to p[0] pixels ---- */

static float icon_gear(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s - 0.5f, v = y / s - 0.5f;
    /* outlined gear: a thin ring with eight teeth around a big hole */
    float d = len2(u, v) - 0.31f;
    for (int k = 0; k < 8; k++) {
        float a = k * (float)M_PI / 4;
        float ca = cosf(a), sa = sinf(a);
        float ru = u * ca + v * sa, rv = -u * sa + v * ca;
        d = fminf(d, sd_box(ru - 0.37f, rv, 0.08f, 0.06f, 0.015f));
    }
    d = fmaxf(d, -(len2(u, v) - 0.215f));
    return d * s;
}

static float icon_power(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s, v = y / s;
    float gap = 0.62f; /* half opening at the top, radians */
    float top = -(float)M_PI / 2;
    float arc = sd_arc(u, v, 0.5f, 0.54f, 0.34f, top + gap, top - gap + 2 * (float)M_PI, 0.085f);
    float bar = sd_segment(u, v, 0.5f, 0.10f, 0.5f, 0.50f, 0.0425f);
    return fminf(arc, bar) * s;
}

static float icon_wifi(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s, v = y / s;
    float a0 = -(float)M_PI * 0.75f, a1 = -(float)M_PI * 0.25f;
    float d = 1e9f;
    const float radii[3] = { 0.36f, 0.62f, 0.88f };
    for (int i = 0; i < 3; i++)
        d = fminf(d, sd_arc(u, v, 0.5f, 0.97f, radii[i], a0, a1, 0.13f));
    return d * s;
}

static float icon_handheld(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s, v = y / s; /* 1 wide, about 0.44 tall, centered at v = 0.5 */
    float body = sd_box(u - 0.5f, v - 0.5f, 0.5f, 0.215f, 0.075f);
    float screen = sd_box(u - 0.5f, v - 0.5f, 0.30f, 0.155f, 0.0f);
    float d = fmaxf(body, -screen);
    const float dots[][2] = {
        { 0.085f, 0.37f },                                      /* - button */
        { 0.085f, 0.565f }, { 0.050f, 0.60f }, { 0.120f, 0.60f }, { 0.085f, 0.635f }, /* d-pad */
        { 0.915f, 0.40f }, { 0.880f, 0.435f }, { 0.950f, 0.435f }, { 0.915f, 0.47f },  /* ABXY */
        { 0.915f, 0.63f },                                      /* right stick */
    };
    const float rad[] = { 0.032f, 0.018f, 0.018f, 0.018f, 0.018f, 0.018f, 0.018f, 0.018f, 0.018f, 0.032f };
    for (unsigned i = 0; i < sizeof(rad) / sizeof(rad[0]); i++)
        d = fmaxf(d, -(len2(u - dots[i][0], v - dots[i][1]) - rad[i]));
    return d * s;
}

static float icon_plus(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s - 0.5f, v = y / s - 0.5f;
    return fminf(sd_box(u, v, 0.42f, 0.11f, 0.02f), sd_box(u, v, 0.11f, 0.42f, 0.02f)) * s;
}

static float icon_minus(float x, float y, const float *p)
{
    float s = p[0];
    return sd_box(x / s - 0.5f, y / s - 0.5f, 0.42f, 0.11f, 0.02f) * s;
}

/* filled triangle (exact distance, Inigo Quilez's formula) */
static float sd_tri(float px, float py, float ax, float ay, float bx, float by, float cx, float cy)
{
    float e0x = bx - ax, e0y = by - ay, e1x = cx - bx, e1y = cy - by, e2x = ax - cx, e2y = ay - cy;
    float v0x = px - ax, v0y = py - ay, v1x = px - bx, v1y = py - by, v2x = px - cx, v2y = py - cy;
    float t0 = fminf(fmaxf((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y), 0), 1);
    float t1 = fminf(fmaxf((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y), 0), 1);
    float t2 = fminf(fmaxf((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y), 0), 1);
    float q0x = v0x - e0x * t0, q0y = v0y - e0y * t0, q1x = v1x - e1x * t1, q1y = v1y - e1y * t1, q2x = v2x - e2x * t2, q2y = v2y - e2y * t2;
    float s = e0x * e2y - e0y * e2x;
    float d = fminf(fminf(q0x * q0x + q0y * q0y, q1x * q1x + q1y * q1y), q2x * q2x + q2y * q2y);
    float sg = fminf(fminf(s * (v0x * e0y - v0y * e0x), s * (v1x * e1y - v1y * e1x)), s * (v2x * e2y - v2y * e2x));
    return -sqrtf(d) * (sg < 0 ? -1.0f : 1.0f);
}

/* circular arrow, the "update" glyph: an open ring with a solid head at the top right */
static float icon_refresh(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s, v = y / s;
    const float pi = (float)M_PI, R = 0.30f;
    float a0 = -pi / 2 + 0.55f;              /* the head end */
    float arc = sd_arc(u, v, 0.5f, 0.5f, R, a0, a0 + 2 * pi - 0.95f, 0.09f);
    /* head: points along the turn at a0, so backwards (counterclockwise) from the open end */
    float ex = 0.5f + R * cosf(a0), ey = 0.5f + R * sinf(a0);
    float tx = sinf(a0), ty = -cosf(a0);      /* tangent pointing to smaller angles */
    float nx = cosf(a0), ny = sinf(a0);
    float hw = 0.13f, hl = 0.17f;
    float head = sd_tri(u, v, ex + nx * hw, ey + ny * hw, ex - nx * hw, ey - ny * hw, ex + tx * hl, ey + ty * hl);
    return fminf(arc, head) * s;
}

/* folder: solid, with the flap line across */
static float icon_folder(float x, float y, const float *p)
{
    float s = p[0];
    float u = x / s - 0.5f, v = y / s - 0.5f;
    float back = sd_box(u, v + 0.03f, 0.40f, 0.28f, 0.05f);
    float tab = sd_box(u + 0.20f, v + 0.27f, 0.20f, 0.07f, 0.04f);
    float d = fminf(back, tab);
    float gap = sd_box(u, v + 0.14f, 0.45f, 0.022f, 0.0f);   /* the line between the tab part and the front */
    return fmaxf(d, -gap) * s;
}

static const SdfFn g_icon_fns[ICON_COUNT] = {
    [ICON_REFRESH] = icon_refresh,
    [ICON_FOLDER] = icon_folder,
    [ICON_GEAR] = icon_gear,
    [ICON_POWER] = icon_power,
    [ICON_WIFI] = icon_wifi,
    [ICON_HANDHELD] = icon_handheld,
    [ICON_PLUS] = icon_plus,
    [ICON_MINUS] = icon_minus,
};

/* ---- text cache ---- */

typedef struct {
    FontId font;
    char str[160];
    SDL_Texture *tex;
    int w, h;
    Uint32 used;
} TextEntry;

static TextEntry g_text[192];
static Uint32 g_text_clock;

static TextEntry *text_get(FontId f, const char *s)
{
    TextEntry *slot = NULL;
    for (unsigned i = 0; i < sizeof(g_text) / sizeof(g_text[0]); i++) {
        TextEntry *e = &g_text[i];
        if (e->tex && e->font == f && !strcmp(e->str, s)) {
            e->used = ++g_text_clock;
            return e;
        }
        if (!slot || (!e->tex && slot->tex) || (e->tex && slot->tex && e->used < slot->used))
            slot = e;
    }
    if (slot->tex)
        SDL_DestroyTexture(slot->tex);
    memset(slot, 0, sizeof(*slot));
    if (!g_fonts[f] || !s[0])
        return NULL;
    SDL_Surface *surf = TTF_RenderUTF8_Blended(g_fonts[f], s, (SDL_Color){ 255, 255, 255, 255 });
    if (!surf)
        return NULL;
    slot->tex = SDL_CreateTextureFromSurface(g_r, surf);
    slot->w = surf->w;
    slot->h = surf->h;
    SDL_FreeSurface(surf);
    if (!slot->tex)
        return NULL;
    SDL_SetTextureBlendMode(slot->tex, SDL_BLENDMODE_BLEND);
    slot->font = f;
    snprintf(slot->str, sizeof(slot->str), "%s", s);
    slot->used = ++g_text_clock;
    return slot;
}

/* ---- public ---- */

static const struct { const char *file; int size; } g_font_specs[FONT_COUNT] = {
    [FONT_CLOCK] = { "BarlowSemiCondensed-Medium.ttf", 29 },
    [FONT_CLOCK_SMALL] = { "Hind-SemiBold.ttf", 20 },
    [FONT_TITLE] = { "Hind-Regular.ttf", 31 },
    [FONT_HINT] = { "Hind-Regular.ttf", 25 },
    [FONT_BODY] = { "Hind-Regular.ttf", 24 },
    [FONT_BODY_BOLD] = { "Hind-SemiBold.ttf", 24 },
    [FONT_HEADER] = { "Hind-Regular.ttf", 30 },
    [FONT_TILE] = { "Hind-Medium.ttf", 26 },
};

bool gfx_init(SDL_Renderer *r, const char *font_dir)
{
    g_r = r;
    if (TTF_Init() < 0)
        return false;
    for (int i = 0; i < FONT_COUNT; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s%s", font_dir, g_font_specs[i].file);
        g_fonts[i] = TTF_OpenFont(path, g_font_specs[i].size);
        if (!g_fonts[i])
            return false;
    }
    return true;
}

void gfx_exit(void)
{
    for (unsigned i = 0; i < sizeof(g_text) / sizeof(g_text[0]); i++)
        if (g_text[i].tex)
            SDL_DestroyTexture(g_text[i].tex);
    for (int i = 0; i < g_nshapes; i++)
        if (g_shapes[i].tex)
            SDL_DestroyTexture(g_shapes[i].tex);
    for (int i = 0; i < FONT_COUNT; i++)
        if (g_fonts[i])
            TTF_CloseFont(g_fonts[i]);
    TTF_Quit();
}

SDL_Renderer *gfx_renderer(void)
{
    return g_r;
}

void gfx_clear(Color c)
{
    SDL_SetRenderDrawColor(g_r, c.r, c.g, c.b, c.a);
    SDL_RenderClear(g_r);
}

void gfx_rect(float x, float y, float w, float h, Color c)
{
    if (w <= 0 || h <= 0)
        return;
    SDL_SetRenderDrawBlendMode(g_r, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_r, c.r, c.g, c.b, c.a);
    SDL_FRect rc = { x, y, w, h };
    SDL_RenderFillRectF(g_r, &rc);
}

/* nine-slice: corners from a texture of size 2r, edges and middle as plain rects */
static void nine_slice(SDL_Texture *t, int r, float x, float y, float w, float h, Color c, float edge)
{
    SDL_Rect q[4] = { { 0, 0, r, r }, { r, 0, r, r }, { 0, r, r, r }, { r, r, r, r } };
    draw_tinted(t, &q[0], x, y, r, r, c);
    draw_tinted(t, &q[1], x + w - r, y, r, r, c);
    draw_tinted(t, &q[2], x, y + h - r, r, r, c);
    draw_tinted(t, &q[3], x + w - r, y + h - r, r, r, c);
    if (edge <= 0) {
        gfx_rect(x + r, y, w - 2 * r, h, c);
        gfx_rect(x, y + r, r, h - 2 * r, c);
        gfx_rect(x + w - r, y + r, r, h - 2 * r, c);
    } else {
        gfx_rect(x + r, y, w - 2 * r, edge, c);
        gfx_rect(x + r, y + h - edge, w - 2 * r, edge, c);
        gfx_rect(x, y + r, edge, h - 2 * r, c);
        gfx_rect(x + w - edge, y + r, edge, h - 2 * r, c);
    }
}

void gfx_round_rect(float x, float y, float w, float h, float radius, Color c)
{
    int r = (int)radius;
    if (r < 1) {
        gfx_rect(x, y, w, h, c);
        return;
    }
    float p[4] = { 2.0f * r, 2.0f * r, (float)r, 0 };
    nine_slice(shape_get(1, p, 2 * r, 2 * r, sdf_round_rect), r, x, y, w, h, c, 0);
}

void gfx_round_frame(float x, float y, float w, float h, float radius, float thickness, Color c)
{
    int r = (int)fmaxf(radius, thickness);
    float p[4] = { 2.0f * r, 2.0f * r, radius, thickness };
    nine_slice(shape_get(2, p, 2 * r, 2 * r, sdf_round_frame), r, x, y, w, h, c, thickness);
}

void gfx_circle_scaled(float cx, float cy, float radius, float scale, Color c)
{
    int d = (int)ceilf(radius * 2);
    float p[4] = { (float)d, (float)d, d / 2.0f, 0 };
    float s = d * scale;
    draw_tinted(shape_get(1, p, d, d, sdf_round_rect), NULL, cx - s / 2, cy - s / 2, s, s, c);
}

void gfx_circle(float cx, float cy, float radius, Color c)
{
    gfx_circle_scaled(cx, cy, radius, 1, c);
}

void gfx_shadow(float x, float y, float w, float h, float radius, float spread, Color c)
{
    SDL_Texture *t = shadow_tex((int)w, (int)h, radius, spread);
    int pad = (int)ceilf(spread * 2);
    draw_tinted(t, NULL, x - pad, y - pad, w + pad * 2, h + pad * 2, c);
}

void gfx_icon_scaled(IconId id, float cx, float cy, float size, float scale, Color c)
{
    if (id >= ICON_COUNT || !g_icon_fns[id])
        return;
    int s = (int)ceilf(size);
    float p[4] = { (float)s, 0, 0, 0 };
    float d = s * scale;
    draw_tinted(shape_get(10 + id, p, s, s, g_icon_fns[id]), NULL, cx - d / 2, cy - d / 2, d, d, c);
}

void gfx_icon(IconId id, float cx, float cy, float size, Color c)
{
    gfx_icon_scaled(id, cx, cy, size, 1, c);
}

void gfx_battery(float x, float y, int percent, bool charging, Color c)
{
    /* 36x20 like the console: 3 px outline, a nub on the right */
    gfx_round_frame(x, y, 32, 20, 2, 3, c);
    gfx_rect(x + 32, y + 6, 4, 8, c);
    float inner = 24.0f * clampf(percent / 100.0f, 0, 1);
    gfx_rect(x + 4, y + 4, inner, 12, c);
    if (charging)
        gfx_rect(x + 4 + inner, y + 4, 24 - inner, 12, with_alpha(c, 0.25f));
}

void gfx_texture(SDL_Texture *t, float x, float y, float w, float h, float alpha)
{
    if (!t)
        return;
    SDL_SetTextureColorMod(t, 255, 255, 255);
    SDL_SetTextureAlphaMod(t, (Uint8)(255 * clampf(alpha, 0, 1)));
    SDL_FRect dst = { x, y, w, h };
    SDL_RenderCopyF(g_r, t, NULL, &dst);
}

float gfx_text_width(FontId f, const char *s)
{
    int w = 0;
    if (g_fonts[f] && s[0])
        TTF_SizeUTF8(g_fonts[f], s, &w, NULL);
    return (float)w;
}

float gfx_text(FontId f, const char *s, float x, float y, Align align, Color c)
{
    TextEntry *e = text_get(f, s);
    if (!e)
        return 0;
    float left = align == ALIGN_CENTER ? x - e->w / 2.0f : align == ALIGN_RIGHT ? x - e->w : x;
    draw_tinted(e->tex, NULL, roundf(left), roundf(y - TTF_FontAscent(g_fonts[f])), e->w, e->h, c);
    return (float)e->w;
}

typedef struct {
    FontId font;
    int max_w;
    char src[160], fit[260];
} FitEntry;

static FitEntry g_fit[16];
static int g_fit_next;

void gfx_text_fit(FontId f, const char *s, float x, float y, float max_w, Align align, Color c)
{
    for (unsigned i = 0; i < sizeof(g_fit) / sizeof(g_fit[0]); i++) {
        FitEntry *e = &g_fit[i];
        if (e->font == f && e->max_w == (int)max_w && e->src[0] && !strcmp(e->src, s)) {
            gfx_text(f, e->fit, x, y, align, c);
            return;
        }
    }
    FitEntry *e = &g_fit[g_fit_next++ % (sizeof(g_fit) / sizeof(g_fit[0]))];
    e->font = f;
    e->max_w = (int)max_w;
    snprintf(e->src, sizeof(e->src), "%s", s);
    if (gfx_text_width(f, s) <= max_w) {
        snprintf(e->fit, sizeof(e->fit), "%s", s);
        gfx_text(f, s, x, y, align, c);
        return;
    }
    char buf[256];
    size_t n = strlen(s);
    if (n > sizeof(buf) - 4)
        n = sizeof(buf) - 4;
    while (n > 0) {
        n--;
        while (n > 0 && (s[n] & 0xC0) == 0x80) /* don't cut a UTF-8 sequence */
            n--;
        memcpy(buf, s, n);
        strcpy(buf + n, "...");
        if (gfx_text_width(f, buf) <= max_w)
            break;
    }
    snprintf(e->fit, sizeof(e->fit), "%s", buf);
    gfx_text(f, buf, x, y, align, c);
}
