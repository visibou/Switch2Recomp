// t_vk_replay: the same with the cull distance written (it cannot be dropped)
#version 450
out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; float gl_CullDistance[1]; };
layout(location = 0) in vec4 pos;
void main() { gl_Position = pos; gl_ClipDistance[0] = pos.x; gl_CullDistance[0] = pos.y; }
