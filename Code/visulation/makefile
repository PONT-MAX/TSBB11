all : lab

init=  -Wall -std=c99

fw= -DGL_GLEXT_PROTOTYPES -lm -framework OpenGL -framework Cocoa -I/opt/X11/include common/MicroGlut.m common/GL_utilities.c common/loadobj.c common/VectorUtils3.c




lab: lab.c common/GL_utilities.c 
	gcc $(init) -o lab $(fw) lab.c  



clean :
	rm lab

