#version 450 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 tex_coords;

out vec2 TexCoords;
out vec3 originalPos;

void main()
{
    gl_Position = vec4(position, 1.0f);
    TexCoords = tex_coords;
	originalPos = position;
}
