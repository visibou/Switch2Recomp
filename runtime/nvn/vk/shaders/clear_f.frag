#version 450
// Partial clears of a float or normalized target: the clear colour; the pipeline's write mask keeps the channels the clear leaves alone
layout(location = 0) out vec4 o;
layout(push_constant) uniform PC { vec4 c; } pc;
void main() { o = pc.c; }
