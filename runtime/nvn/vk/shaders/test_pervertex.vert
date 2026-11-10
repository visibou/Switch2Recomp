// t_vk_replay: gl_PerVertex as nxshader's translations declare it (8 clip distances and 1 cull distance, one more than maxCombinedClipAndCullDistances 8),
// the clip distances written, the cull distance not
#version 450
out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; float gl_CullDistance[1]; };
layout(location = 0) in vec4 pos;
void main() {
    gl_Position = pos;
    for (int i = 0; i < 8; i++) gl_ClipDistance[i] = pos.x - float(i);
}
