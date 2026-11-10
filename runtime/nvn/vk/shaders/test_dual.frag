#version 450
// Tests (t_vk_replay.c): a pixel shader writing outputs 0 and 1 with constants, as the translator's do: the source of the dual-source and alpha-test variants
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    o0 = vec4(0.25, 0.5, 0.75, 0.6);
    o1 = vec4(0.5, 0.25, 1.0, 0.2);
    if (gl_FragCoord.x < 4.0) return;   // an early return: the alpha test goes before every return
    o0.a = 0.4;
}
