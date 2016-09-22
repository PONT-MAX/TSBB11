#version 150

in  vec3 in_Position;
in  vec2 inTexCoord;
uniform mat4 tot;
out vec2 intpTexCoord;


void main(void)
{
	gl_Position = tot * vec4(in_Position, 1.0);
    intpTexCoord = inTexCoord;
}
