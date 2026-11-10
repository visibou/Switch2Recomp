#version 450
// Mode 0: the presented texture, bilinear ("Standard"). Mode 2: the same, upscaled sharp: each of the four source texels around the point is sharpened
// against its four neighbours by how much contrast they leave room for (strong where the picture is flat, gentle at edges, so no halos), then the four
// are blended bilinearly; our own, after the idea of contrast-adaptive sharpening. Used only when the window is larger than the picture.
// Mode 1: the test pattern shown until the replay draws frames: colour bars, a 16:9 grid that shows the letterboxing, and a bar that moves one step per
// game present (so the window visibly runs at the game's pace)
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
layout(set = 0, binding = 0) uniform sampler2D src;
layout(push_constant) uniform PC { vec4 rect; vec4 misc; vec4 sharp; } pc;   // sharp.x: sharpness 0..1

vec3 texel(ivec2 q, ivec2 lo, ivec2 hi) { return texelFetch(src, clamp(q, lo, hi), 0).rgb; }
vec3 sharpened(ivec2 q, ivec2 lo, ivec2 hi, float peak) {
    vec3 c = texel(q, lo, hi), n = texel(q + ivec2(0, -1), lo, hi), s = texel(q + ivec2(0, 1), lo, hi);
    vec3 w = texel(q + ivec2(-1, 0), lo, hi), e = texel(q + ivec2(1, 0), lo, hi);
    vec3 mn = min(c, min(min(n, s), min(w, e))), mx = max(c, max(max(n, s), max(w, e)));
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-5)), 0.0, 1.0));
    vec3 k = amp * peak;   // negative: the neighbours are subtracted
    return clamp((c + (n + s + w + e) * k) / (1.0 + 4.0 * k), 0.0, 1.0);
}

void main() {
    if (pc.misc.x < 0.5) { o = vec4(texture(src, uv).rgb, 1.0); return; }
    if (pc.misc.x > 1.5) {
        vec2 ts = vec2(textureSize(src, 0));
        ivec2 lo = ivec2(floor(min(pc.rect.xy, pc.rect.zw) * ts)), hi = ivec2(ceil(max(pc.rect.xy, pc.rect.zw) * ts)) - 1;   // the crop, either way up
        vec2 p = uv * ts - 0.5; ivec2 i = ivec2(floor(p)); vec2 f = p - floor(p);
        float peak = -1.0 / mix(8.0, 5.0, clamp(pc.sharp.x, 0.0, 1.0));
        vec3 a = sharpened(i, lo, hi, peak), b = sharpened(i + ivec2(1, 0), lo, hi, peak);
        vec3 c = sharpened(i + ivec2(0, 1), lo, hi, peak), d = sharpened(i + ivec2(1, 1), lo, hi, peak);
        o = vec4(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), 1.0);
        return;
    }
    vec2 t = clamp((uv - pc.rect.xy) / (pc.rect.zw - pc.rect.xy), 0.0, 1.0);
    const vec3 bars[8] = vec3[](vec3(0.75), vec3(0.75, 0.75, 0.0), vec3(0.0, 0.75, 0.75), vec3(0.0, 0.75, 0.0),
                                vec3(0.75, 0.0, 0.75), vec3(0.75, 0.0, 0.0), vec3(0.0, 0.0, 0.75), vec3(0.1));
    vec3 c = t.y < 0.62 ? bars[int(min(t.x * 8.0, 7.0))] : vec3(0.06 + 0.5 * t.x);
    vec2 px = t * pc.misc.zw;
    vec2 g = abs(fract(px / 120.0 + 0.5) - 0.5) * 120.0;
    if (min(g.x, g.y) < 1.0) c = mix(c, vec3(1.0), 0.35);
    float fr = mod(pc.misc.y, 120.0) / 120.0;
    if (t.y > 0.66 && t.y < 0.9 && abs(t.x - fr) < 0.012) c = vec3(1.0);
    if (t.y > 0.92 && t.x < mod(pc.misc.y, 60.0) / 60.0) c = vec3(0.95, 0.55, 0.1);
    if (t.x < 0.004 || t.x > 0.996 || t.y < 0.007 || t.y > 0.993) c = vec3(1.0, 0.2, 0.2);
    o = vec4(c, 1.0);
}
