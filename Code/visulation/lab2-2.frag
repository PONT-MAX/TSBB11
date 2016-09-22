#version 150


out vec4 out_Color;
in vec2 intpTexCoord;
uniform sampler2D ortho_tex,dhm_tex,dsm_tex,dtm_tex,cls_tex;
uniform bool is_visible[5];
uniform float mask_transparency;
float color_scalar = 1.0;


void main(void)
{
    
    if (is_visible[4] == true) {
        if((is_visible[3] == false) && (is_visible[2] == false) && (is_visible[1] == false) && (is_visible[0] == false) ){
        color_scalar = 0.0;
        }
        else{
            color_scalar = mask_transparency;
        }

        vec4 aux = texture(cls_tex,intpTexCoord);
        if(aux.r < 1.0/255.0){
            aux.b = 1.0;
        }
        else if(aux.r < 2.0/255.0){
            aux.r = 1.0;
            aux.g = 1.0;
        }
        else if(aux.r < 3.0/255.0){
            aux.r = 1.0;
        }
        else if(aux.r < 4.0/255.0){
            aux.g = 0.7;
        }
        else if(aux.r < 5.0/255.0){
            aux.b = 1.0;
        }
        out_Color = aux*(1-color_scalar);
    }
    
    if (is_visible[0] == true) {
        vec4 ortho = texture(ortho_tex,intpTexCoord);
        out_Color += ortho*color_scalar;
    }
    else if (is_visible[1] == true) {
        vec4 dhm = texture(dhm_tex,intpTexCoord);
        out_Color += dhm*color_scalar;
    }
    else if (is_visible[2] == true) {
        vec4 dsm = texture(dsm_tex,intpTexCoord);
        out_Color += dsm*color_scalar;
    }
    else if (is_visible[3] == true) {
        vec4 dtm = texture(dtm_tex,intpTexCoord);
        out_Color += dtm*color_scalar;
    }
    
}
