/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* drawing helpers on top of SDL_Renderer. anti aliased shapes and icons get rasterized once into textures, text
   goes through a small cache */
#pragma once
#include <SDL.h>
#include <stdbool.h>

typedef struct { Uint8 r, g, b, a; } Color;

static inline Color rgb(int r, int g, int b) { return (Color){ r, g, b, 255 }; }
static inline Color with_alpha(Color c, float a)
{
    c.a = (Uint8)(c.a * (a < 0 ? 0 : a > 1 ? 1 : a));
    return c;
}
Color color_mix(Color a, Color b, float t);

typedef enum {
    FONT_CLOCK,       /* clock digits */
    FONT_CLOCK_SMALL, /* AM / PM */
    FONT_TITLE,       /* game name over the row */
    FONT_HINT,        /* bottom bar */
    FONT_BODY,        /* lists and dialogs */
    FONT_BODY_BOLD,
    FONT_HEADER,
    FONT_TILE,        /* name on tiles without an icon */
    FONT_COUNT
} FontId;

typedef enum {
    ICON_GEAR,
    ICON_POWER,
    ICON_WIFI,
    ICON_HANDHELD,
    ICON_BUTTON_A,
    ICON_BUTTON_B,
    ICON_PLUS,
    ICON_MINUS,
    ICON_REFRESH,
    ICON_FOLDER,
    ICON_COUNT
} IconId;

/* font_dir: folder with the .ttf files, ending in a slash */
bool gfx_init(SDL_Renderer *r, const char *font_dir);
void gfx_exit(void);
SDL_Renderer *gfx_renderer(void);

void gfx_clear(Color c);
void gfx_rect(float x, float y, float w, float h, Color c);
void gfx_round_rect(float x, float y, float w, float h, float radius, Color c);
void gfx_round_frame(float x, float y, float w, float h, float radius, float thickness, Color c);
void gfx_circle(float cx, float cy, float radius, Color c);
/* drawn at radius * scale from the texture made for radius, for animations so there's no new texture per frame */
void gfx_circle_scaled(float cx, float cy, float radius, float scale, Color c);
void gfx_shadow(float x, float y, float w, float h, float radius, float spread, Color c);
void gfx_icon(IconId id, float cx, float cy, float size, Color c);
void gfx_icon_scaled(IconId id, float cx, float cy, float size, float scale, Color c);
void gfx_battery(float x, float y, int percent, bool charging, Color c);
void gfx_texture(SDL_Texture *t, float x, float y, float w, float h, float alpha);

/* text: x is the left edge, right edge or center depending on align; y is the baseline */
typedef enum { ALIGN_LEFT, ALIGN_CENTER, ALIGN_RIGHT } Align;
float gfx_text(FontId f, const char *s, float x, float y, Align align, Color c);
float gfx_text_width(FontId f, const char *s);
/* text clipped to max_w with an ellipsis */
void gfx_text_fit(FontId f, const char *s, float x, float y, float max_w, Align align, Color c);
