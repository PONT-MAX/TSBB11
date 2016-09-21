

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "common/stb_image.h"
#ifdef __APPLE__
	#include <OpenGL/gl3.h>
	#include "common/MicroGlut.h"
	// Linking hint for Lightweight IDE
	// uses framework Cocoa
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
    
    if (glutKeyIsDown('a')) {
        trans = S(1.1,1.1,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown('s')) {
        k = k - 0.1;
        trans = S(0.9,0.9,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_LEFT)) {
        k = k + 0.1;
        trans = T(move_step,0.0,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_RIGHT)) {
        k = k + 0.1;
        trans = T(-move_step,0.0,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_UP)) {
        k = k + 0.1;
        trans = T(0.0,-move_step,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown(GLUT_KEY_DOWN)) {
        k = k + 0.1;
        trans = T(0.0,move_step,0.0);
        tot = Mult(trans,tot);
    }
    else if (glutKeyIsDown('q')) {
        is_visible[0] = 1;
        is_visible[1] = 0;
        is_visible[2] = 0;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('w')) {
        is_visible[0] = 0;
        is_visible[1] = 1;
        is_visible[2] = 0;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('r')) {
        is_visible[0] = 0;
        is_visible[1] = 0;
        is_visible[2] = 1;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('t')) {
        is_visible[0] = 0;
        is_visible[1] = 0;
        is_visible[2] = 0;
        is_visible[3] = 1;
    }
    else if (glutKeyIsDown('o')) {
        is_visible[4] = 1;
    }
    else if (glutKeyIsDown('p')) {
        is_visible[4] = 0;
    }
    else if (glutKeyIsDown('i')) {
        is_visible[0] = 0;
        is_visible[1] = 0;
        is_visible[2] = 0;
        is_visible[3] = 0;
    }
    else if (glutKeyIsDown('0')) {
        if(mask_transparency < 0.96){
        mask_transparency += 0.02;
        }
    }
    else if (glutKeyIsDown('+')) {
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

void add_texture(const GLchar name[], const GLchar location[], GLuint texture_id, GLuint* program){
    
    glUniform1i(glGetUniformLocation(*program, name), texture_id); // Texture unit 0
    
    //Load iamge
    int width, hight, comp;
    unsigned char* tex_image = stbi_load(location, &width, &hight, &comp, STBI_rgb_alpha);
    
    if (!tex_image){
        printf("stbi_load failed \n");
    }
    
    GLuint tex_id = texture_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, hight, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_image);
    
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    
    glActiveTexture(GL_TEXTURE0 + tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    
    stbi_image_free(tex_image);
    
    
}


// vertex array object
unsigned int vertexArrayObjID;

unsigned int bunnyVertexArrayObjID;
unsigned int bunnyVertexBufferObjID;
unsigned int bunnyIndexBufferObjID;
unsigned int bunnyNormalBufferObjID;
unsigned int bunnyTexCoordBufferObjID;



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
    
    glGenVertexArrays(1, &bunnyVertexArrayObjID);
    glGenBuffers(1, &bunnyVertexBufferObjID);
    glGenBuffers(1, &bunnyIndexBufferObjID);
    glGenBuffers(1, &bunnyNormalBufferObjID);
    glGenBuffers(1, &bunnyTexCoordBufferObjID);
    
    glBindVertexArray(bunnyVertexArrayObjID);
    
    // VBO for vertex data
    glBindBuffer(GL_ARRAY_BUFFER, bunnyVertexBufferObjID);
    glBufferData(GL_ARRAY_BUFFER, m->numVertices*3*sizeof(GLfloat), m->vertexArray, GL_STATIC_DRAW);
    glVertexAttribPointer(glGetAttribLocation(program, "in_Position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(glGetAttribLocation(program, "in_Position"));
    
    // VBO for normal data
//    glBindBuffer(GL_ARRAY_BUFFER, bunnyNormalBufferObjID);
//    glBufferData(GL_ARRAY_BUFFER, m->numVertices*3*sizeof(GLfloat), m->normalArray, GL_STATIC_DRAW);
//    glVertexAttribPointer(glGetAttribLocation(program, "in_Normal"), 3, GL_FLOAT, GL_FALSE, 0, 0);
//    glEnableVertexAttribArray(glGetAttribLocation(program, "in_Normal"));
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bunnyIndexBufferObjID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m->numIndices*sizeof(GLuint), m->indexArray, GL_STATIC_DRAW);
    
    if (m->texCoordArray != 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, bunnyTexCoordBufferObjID);
        glBufferData(GL_ARRAY_BUFFER, m->numVertices*2*sizeof(GLfloat), m->texCoordArray, GL_STATIC_DRAW);
        glVertexAttribPointer(glGetAttribLocation(program, "inTexCoord"), 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(glGetAttribLocation(program, "inTexCoord"));
    }
    
    
    
    add_texture("ortho_tex", "./images/ortho.bmp",0,&program);
    add_texture("dhm_tex"  , "./images/dhm_n.bmp",1,&program);
    add_texture("dsm_tex"  , "./images/dsm.bmp",2,&program);
    add_texture("dtm_tex"  , "./images/dtm.bmp",3,&program);
    add_texture("cls_tex"  , "./images/aux.bmp",4,&program);
    
	
	printError("init arrays");
}

GLfloat t = 0.0;



void display(void)
{

    t = t+0.01;
    
    keyUpdate();

    
    glUniform1iv(glGetUniformLocation(program, "is_visible"), 5, is_visible);
    glUniform1f(glGetUniformLocation(program, "mask_transparency"),mask_transparency);

    printError("init shader");
	printError("pre display");
    // Matrixes

	// clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    glUniformMatrix4fv(glGetUniformLocation(program, "tot"), 1, GL_TRUE, tot.m);


    
    glBindVertexArray(bunnyVertexArrayObjID);    // Select VAO
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
