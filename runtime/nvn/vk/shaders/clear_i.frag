#version 450
// Partial clears of a signed integer target
layout(location = 0) out ivec4 o;
layout(push_constant) uniform PC { ivec4 c; } pc;
void main() { o = pc.c; }
