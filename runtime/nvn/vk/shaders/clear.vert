#version 450
// Partial clears (vk_replay.c clear_draw): one triangle covering the viewport; the scissor limits it to the clear's rectangle
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
