#version 450
// The present pass: one triangle covering the viewport (the letterboxed picture rectangle); uv spans the source rectangle (crop, flip)
layout(location = 0) out vec2 uv;
layout(push_constant) uniform PC { vec4 rect; vec4 misc; vec4 sharp; } pc;   // rect: u0 v0 u1 v1; misc: mode, frame, picture width / height; sharp: the fragment shader's
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = mix(pc.rect.xy, pc.rect.zw, p);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
