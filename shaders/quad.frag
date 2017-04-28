#version 450 core
in vec2 TexCoords;
out vec4 color;

layout( binding = 0 ) uniform usampler2D ourTexture;

void main()
{
	vec4 c = texture(ourTexture, TexCoords);
	color = c / 255.0;
}
