#version 150


out vec4 out_Color;
in vec2 intpTexCoord;
uniform sampler2D tex,ortho_tex,dhm_tex,dsm_tex,dtm_tex;
//in vec3 pixelPos;
uniform bool is_visible[4];


void main(void)
{
    
    if (is_visible[0] == true) {
        vec4 ortho = texture(ortho_tex,intpTexCoord);
        out_Color = ortho;
    }
    else if (is_visible[1] == true) {
        vec4 dhm = texture(dhm_tex,intpTexCoord);
        out_Color = dhm;
    }
    else if (is_visible[2] == true) {
        vec4 dsm = texture(dsm_tex,intpTexCoord);
        out_Color = dsm;
    }
    else if (is_visible[3] == true) {
        vec4 dtm = texture(dtm_tex,intpTexCoord);
        out_Color = dtm;
    }
    
}
