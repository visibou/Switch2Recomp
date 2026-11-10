#version 450
// Partial clears of an unsigned integer target
layout(location = 0) out uvec4 o;
layout(push_constant) uniform PC { uvec4 c; } pc;
void main() { o = pc.c; }
