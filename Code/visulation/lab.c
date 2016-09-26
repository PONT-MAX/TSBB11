

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "common/stb_image.h"
#ifdef __APPLE__
	#include <OpenGL/gl3.h>
	#include "common/MicroGlut.h"
	// Linking hint for Lightweight IDE
	// uses framework Cocoa
#else
    #include <OpenGL/gl3.h> // Denna ska ändras för win
    #include "common/MicroGlut.h" //Samma?
#endif

#include "common/GL_utilities.h"
#include "common/VectorUtils3.h"
#include "common/loadobj.h"
#include <math.h>
#include <stdio.h>




Model* m;

mat4 trans;
mat4 tot;
mat4 rot;
float k = 1;
float move_step = 0.05;
GLint is_visible[] = {1,0,0,0,0};
GLfloat mask_transparency = 0.5;

void keyUpdate(){
    
    if (glutKeyIsDown('a')) { // Zoom in
        trans = S(1.1,1.1,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown('s')) { // Zoom out
        k = k - 0.1;
        trans = S(0.9,0.9,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_LEFT)) { // Move map left
        k = k + 0.1;
        trans = T(move_step,0.0,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_RIGHT)) { // Move map right
        k = k + 0.1;
        trans = T(-move_step,0.0,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_UP)) { // Move map up
        k = k + 0.1;
        trans = T(0.0,-move_step,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_DOWN)) { // Move map down
        k = k + 0.1;
        trans = T(0.0,move_step,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown('q')) { // Show Ortho-map
        is_visible[0] = 1;
        is_visible[1] = 0;
        is_visible[2] = 0;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('w')) { // Show DHM-map
        is_visible[0] = 0;
        is_visible[1] = 1;
        is_visible[2] = 0;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('r')) { // Show DSM-map
        is_visible[0] = 0;
        is_visible[1] = 0;
        is_visible[2] = 1;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('t')) { // Show DTM-map
        is_visible[0] = 0;
        is_visible[1] = 0;
        is_visible[2] = 0;
        is_visible[3] = 1;
    }
    else if (glutKeyIsDown('o')) { // Overlay class mask
        is_visible[4] = 1;
    }
    else if (glutKeyIsDown('p')) { // Hide class mask
        is_visible[4] = 0;
    }
    else if (glutKeyIsDown('0')) { // Lower opacity
        if(mask_transparency < 0.96){
        mask_transparency += 0.02;
        }
    }
    else if (glutKeyIsDown('+')) { // HIger opacity
        if(mask_transparency > 0.04){
            mask_transparency -= 0.02;
        }
    }

}

void OnTimer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(10, &OnTimer, value);
}

void texture_setting(int mipmap){

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    
    if(mipmap == 1){
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    }
    else if(mipmap == 2){
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    else{
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }

    
}

void add_texture(const GLchar name[], const GLchar location[], GLuint texture_id, GLuint* program,int texture_setting_var){
    
    glUniform1i(glGetUniformLocation(*program, name), texture_id); // Texture unit 0
    
    //Load iamge
    int width, hight, comp;
    unsigned char* tex_image = stbi_load(location, &width, &hight, &comp, 0);
    
    if (!tex_image){
        printf("stbi_load failed \n");
    }
    
    GLuint tex_id = texture_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, hight, 0, GL_RED, GL_UNSIGNED_BYTE, tex_image);
    
    
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    texture_setting(texture_setting_var);
    stbi_image_free(tex_image);
    
    
}

void add_texture_rgb(const GLchar name[], const GLchar location[], GLuint texture_id, GLuint* program,int texture_setting_var){
    
    glUniform1i(glGetUniformLocation(*program, name), texture_id); // Texture unit 0
    
    //Load iamge
    int width, hight, comp;
    unsigned char* tex_image = stbi_load(location, &width, &hight, &comp, STBI_rgb);
    
    if (!tex_image){
        printf("stbi_load failed \n");
    }
    
    GLuint tex_id = texture_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, hight, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_image);
    
    
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    texture_setting(texture_setting_var);
    stbi_image_free(tex_image);
    
    
}


// vertex array object
unsigned int VertexArrayObjID;
unsigned int VertexBufferObjID;
unsigned int IndexBufferObjID;
unsigned int NormalBufferObjID;
unsigned int TexCoordBufferObjID;



    GLuint program;
// Reference to texture
GLuint tex;

void init(void)
{
	// vertex buffer object, used for uploading the geometry

	// Reference to shader program
    m = LoadModel("cubeplus1.obj");
    trans = S(2.0,2.0,0.0);
    tot = T(0.0,0.0,0.0);
    rot = Ry(M_PI);
    tot = Mult(trans,tot);
    tot = Mult(rot,tot);
    rot = Rz(M_PI/2.00);
    tot = Mult(rot,tot);

	dumpInfo();
    glutTimerFunc(1, &OnTimer, 0);

	// GL inits
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glClearColor(0.1,0.5,0.5,0);
	glEnable(GL_DEPTH_TEST);
    
    glEnable(GL_CULL_FACE); // Ome en bild inte syns kan det bero på detta
    glCullFace(GL_FRONT);
    
	printError("GL inits");

	// Load and compile shader
	program = loadShaders("lab2-2.vert", "lab2-2.frag");
	printError("init shader");
	
	// Upload geometry to the GPU:
    
    glGenVertexArrays(1, &VertexArrayObjID);
    glGenBuffers(1, &VertexBufferObjID);
    glGenBuffers(1, &IndexBufferObjID);
    glGenBuffers(1, &NormalBufferObjID);
    glGenBuffers(1, &TexCoordBufferObjID);
    
    glBindVertexArray(VertexArrayObjID);
    
    // VBO for vertex data
    glBindBuffer(GL_ARRAY_BUFFER, VertexBufferObjID);
    glBufferData(GL_ARRAY_BUFFER, m->numVertices*3*sizeof(GLfloat), m->vertexArray, GL_STATIC_DRAW);
    glVertexAttribPointer(glGetAttribLocation(program, "in_Position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(glGetAttribLocation(program, "in_Position"));
    printError("load to vertex shader: inPosition");
    
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferObjID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m->numIndices*sizeof(GLuint), m->indexArray, GL_STATIC_DRAW);
    printError("load to vertex shader: indices");
    
    if (m->texCoordArray != 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, TexCoordBufferObjID);
        glBufferData(GL_ARRAY_BUFFER, m->numVertices*2*sizeof(GLfloat), m->texCoordArray, GL_STATIC_DRAW);
        glVertexAttribPointer(glGetAttribLocation(program, "inTexCoord"), 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(glGetAttribLocation(program, "inTexCoord"));
        printError("load to vertex shader: inTexCoord");
    }
    
    
    add_texture_rgb("ortho_tex", "./images/ortho.png",0,&program,1);
    add_texture_rgb("cls_tex"  , "./images/aux.png",1,&program,0);
    add_texture("dhm_tex"  , "./images/dhm_n.png",2,&program,0);
    //add_texture("dsm_tex"  , "./images/dsm_n.png",3,&program,0);
    //add_texture("dtm_tex"  , "./images/dtm_n.png",4,&program,0);
    

	
	printError("init Textures");
}

GLfloat t = 0.0;



void display(void)
{

    t = t+0.01;
    
    keyUpdate();

    // Update Variables i Fragment Shader
    glUniform1iv(glGetUniformLocation(program, "is_visible"), 5, is_visible);
    glUniform1f(glGetUniformLocation(program, "mask_transparency"),mask_transparency);
    printError("update shader");
    // Matrixes

	// clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    glUniformMatrix4fv(glGetUniformLocation(program, "tot"), 1, GL_TRUE, tot.m);


    
    glBindVertexArray(VertexArrayObjID);    // Select VAO
    glDrawElements(GL_TRIANGLES, m->numIndices, GL_UNSIGNED_INT, 0L);
	
	printError("display");
	
	glutSwapBuffers();                  // Swap
}

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitContextVersion(3, 2);
    glutInitWindowSize (1000, 1000);
    glutCreateWindow ("3DVricon");
	glutDisplayFunc(display); 
	init ();
	glutMainLoop();
    printf("The end \n");
}
