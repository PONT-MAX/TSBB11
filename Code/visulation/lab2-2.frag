#version 150


out vec4 out_Color;
in vec2 intpTexCoord;
uniform sampler2D ortho_tex,dhm_tex,dsm_tex,dtm_tex,cls_tex;
uniform bool is_visible[5];
uniform float mask_transparency;
float color_scalar = 1.0;


void main(void)
{
    out_Color = vec4(0.0,0.0,0.0,0.0);
    
    if (is_visible[4] == true) {
        color_scalar = mask_transparency;
        vec4 aux = texture(cls_tex,intpTexCoord);
        out_Color = aux*(1-color_scalar);
    }
    
    if (is_visible[0] == true) {
        vec4 ortho = texture(ortho_tex,intpTexCoord);
        out_Color += ortho*color_scalar;
    }
    else if (is_visible[1] == true) {
        vec4 dhm = texture(dhm_tex,intpTexCoord);
        out_Color += vec4(dhm.x,dhm.x,dhm.x,1.0)*color_scalar;
    }
    else if (is_visible[2] == true) {
        vec4 dsm = texture(dsm_tex,intpTexCoord);
        out_Color += vec4(dsm.x,dsm.x,dsm.x,1.0)*color_scalar;
    }
    else if (is_visible[3] == true) {
        vec4 dtm = texture(dtm_tex,intpTexCoord);
        out_Color += vec4(dtm.x,dtm.x,dtm.x,1.0)*color_scalar;
    }
    
}
