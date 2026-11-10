// t_vk_replay: the binding kinds a translated fragment shader has (support buffer, a texture read as uint, a constant buffer, a buffer texture)
#version 450
layout(set = 0, binding = 0, std140) uniform support_buffer { vec4 s0; } sb;
layout(set = 0, binding = 1) uniform usampler2D t_tex;
layout(set = 0, binding = 2, std140) uniform fp_c3 { vec4 c[16]; } cb;
layout(set = 0, binding = 3) uniform samplerBuffer t_buf;
layout(location = 0) out vec4 o;
void main() { o = vec4(texelFetch(t_tex, ivec2(0), 0)) * cb.c[0] + texelFetch(t_buf, 1) + sb.s0; }
