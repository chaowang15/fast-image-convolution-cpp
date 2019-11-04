// main.cpp
// ========
// It convolves 2D image with Gaussian smoothing kernel,
// and compare the performance between normal 2D convolution and separable
// convolution.
// The result images will be displayed on the screen using OpenGL.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2005-08-31
// UPDATED: 2018-06-28
///////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "Timer.h"
#include "convolution.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// image data for openGL
struct ImageData
{
    GLint x;        // resolution X
    GLint y;        // resolution Y
    GLenum format;  // data format (RGB or INDEX..)
    GLenum type;    // data type (8bit, 16bit or 32bit..)
    GLvoid *buf;    // image pixel bits
};

// GLUT CALLBACK functions
void displayCB();
void displaySubWin1CB();
void displaySubWin2CB();
void displaySubWin3CB();
void reshapeCB(int w, int h);
void reshapeSubWin1CB(int w, int h);
void reshapeSubWin2CB(int w, int h);
void reshapeSubWin3CB(int w, int h);
void keyboardHandlerCB(unsigned char key, int x, int y);

void initGL();
int initGLUT(int argc, char **argv);
bool initSharedMem();
void clearSharedMem();
bool loadRawImage(char *fileName, int x, int y, unsigned char *data);
void drawString(const char *str, int x, int y, void *font);

// constants ////////////////////////
const char *FILE_NAME = "lena.raw";
const int IMAGE_X = 256;
const int IMAGE_Y = 256;
const int MAX_NAME = 1024;

// global variables ////////////////
ImageData *image;
unsigned char *inBuf;
unsigned char *outBuf1;
unsigned char *outBuf2;
char fileName[MAX_NAME];
int imageX;
int imageY;
void *font = GLUT_BITMAP_8_BY_13;
int fontWidth = 8;
int fontHeight = 13;
int mainWin, subWin1, subWin2, subWin3, subWin4;
double time1, time2;

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // use default image file if not specified
    if (argc == 4)
    {
        strcpy(fileName, argv[1]);
        imageX = atoi(argv[2]);
        imageY = atoi(argv[3]);
    }
    else
    {
        printf("Usage: %s <image-file> <width> <height>\n", argv[0]);
        strcpy(fileName, FILE_NAME);
        imageX = IMAGE_X;
        imageY = IMAGE_Y;
        printf("\nUse default image \"%s\", (%d,%d)\n", fileName, imageX, imageY);
    }

    // allocate memory for global variables
    if (!initSharedMem())
        return 0;

    // open raw image file
    if (!loadRawImage(fileName, imageX, imageY, inBuf))
    {
        clearSharedMem();  // exit program if failed to load image
        return 0;
    }

    // define 5x5 Gaussian kernel
    float kernel[25] = {1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f, 4 / 256.0f, 16 / 256.0f,
        24 / 256.0f, 16 / 256.0f, 4 / 256.0f, 6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f, 4 / 256.0f,
        16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f, 1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f};

    // Separable kernel
    float kernelX[5] = {1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f};
    float kernelY[5] = {1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f};

    // integer kernel
    float kernelFactor = 1 / 256.0f;
    int kernelInt[25] = {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1};

    // testing 3x3 ============================================================
    printf("\n===== Testing 3x3 input =====\n");
    // float x[] = {1,2,3,4,5,6,7,8,9};
    // float y[9];
    unsigned char x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    unsigned char y[9];
    float k[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    convolve2DSlow(x, y, 3, 3, k, 3, 3);
    // convolve2D(x, y, 3, 3, k, 3, 3);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
            printf("%7.2f", (float)y[i * 3 + j]);
        printf("\n");
    }
    printf("\n");
    //=========================================================================

    // perform convolution
    Timer t;
    t.start();
    convolve2D(inBuf, outBuf1, imageX, imageY, kernel, 5, 5);
    t.stop();
    time1 = t.getElapsedTimeInMilliSec();
    printf("Normal Convolution: %f ms\n", time1);

    t.start();
    convolve2DSeparableReadable(inBuf, outBuf2, imageX, imageY, kernelX, 5, kernelY, 5);
    t.stop();
    time2 = t.getElapsedTimeInMilliSec();
    printf("Separable Convolution: %f ms\n", time2);

    /*
        t.start();
        convolve2DFast(inBuf, outBuf2, imageX, imageY, kernel, 5, 5);
        //convolve2DSlow(inBuf, outBuf2, imageX, imageY, kernel, 5, 5);
        t.stop();
        time2 = t.getElapsedTimeInMilliSec();
        printf("Fast 2D Convolution: %f ms\n", time2);
    */

    // compare the results of separable convolution with normal convolution, they
    // should be equal.
    for (int i = 0; i < imageX * imageY; ++i)
    {
        if (outBuf1[i] != outBuf2[i])
            printf("different at %d (%d, %d), out1:%d, out2:%d\n", i, i % imageX, i / imageX, outBuf1[i], outBuf2[i]);
    }

    // drawing graphics from here /////////////////////////////////////////////
    // init GLUT
    mainWin = initGLUT(argc, argv);

    // register GLUT callback functions
    glutDisplayFunc(displayCB);
    glutReshapeFunc(reshapeCB);  // subwindows do not need reshape call
    glutKeyboardFunc(keyboardHandlerCB);

    // 3 sub-windows
    // each sub-windows has its own openGL context, callbacks
    subWin1 = glutCreateSubWindow(mainWin, 0, 0, imageX, imageY);
    glutDisplayFunc(displaySubWin1CB);
    glutKeyboardFunc(keyboardHandlerCB);

    subWin2 = glutCreateSubWindow(mainWin, imageX, 0, imageX, imageY);
    glutDisplayFunc(displaySubWin2CB);
    glutKeyboardFunc(keyboardHandlerCB);

    subWin3 = glutCreateSubWindow(mainWin, 0, imageX * 2, imageX, imageY);
    glutDisplayFunc(displaySubWin3CB);
    glutKeyboardFunc(keyboardHandlerCB);

    // turn off unused features
    initGL();

    // the last GLUT call (LOOP)
    // window will be shown and display callback is triggered by events
    // NOTE: this call never return main().
    glutMainLoop(); /* Start GLUT event-processing loop */

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// initialize global variables
///////////////////////////////////////////////////////////////////////////////
bool initSharedMem()
{
    image = new ImageData;
    if (!image)
    {
        printf("ERROR: Memory Allocation Failed.\n");
        return false;
    }

    // allocate input/output buffer
    inBuf = new unsigned char[imageX * imageY];
    outBuf1 = new unsigned char[imageX * imageY];
    outBuf2 = new unsigned char[imageX * imageY];

    if (!inBuf || !outBuf1 || !outBuf2)
    {
        printf("ERROR: Memory Allocation Failed.\n");
        return false;
    }

    // set image data
    image->x = imageX;
    image->y = imageY;
    image->format = GL_LUMINANCE;
    image->type = GL_UNSIGNED_BYTE;
    image->buf = (GLvoid *)inBuf;

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// clean up shared memory
///////////////////////////////////////////////////////////////////////////////
void clearSharedMem()
{
    delete image;
    delete[] inBuf;
    delete[] outBuf1;
    delete[] outBuf2;
}

///////////////////////////////////////////////////////////////////////////////
// load 8-bit RAW image
///////////////////////////////////////////////////////////////////////////////
bool loadRawImage(char *fileName, int x, int y, unsigned char *data)
{
    // check params
    if (!fileName || !data)
        return false;

    FILE *fp;
    if ((fp = fopen(fileName, "r")) == NULL)
    {
        printf("Cannot open %s.\n", fileName);
        return false;
    }

    // read pixel data
    fread(data, 1, x * y, fp);
    fclose(fp);

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// initialize GLUT for windowing
///////////////////////////////////////////////////////////////////////////////
int initGLUT(int argc, char **argv)
{
    // GLUT stuff for windowing
    // initialization openGL window.
    // it is called before any other GLUT routine
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);  // display mode

    glutInitWindowSize(3 * imageX, imageY);  // window size

    glutInitWindowPosition(100, 100);  // window location

    // finally, create a window with openGL context
    // Window will not displayed until glutMainLoop() is called
    // it returns a unique ID
    int handle = glutCreateWindow(argv[0]);  // param is the title of window

    return handle;
}

///////////////////////////////////////////////////////////////////////////////
// initialize OpenGL
// disable unused features
///////////////////////////////////////////////////////////////////////////////
void initGL()
{
    glClearColor(0, 0, 0, 0);
    glShadeModel(GL_FLAT);  // shading mathod: GL_SMOOTH or GL_FLAT
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // TUNNING IMAGING PERFORMANCE
    // turn off  all pixel path and per-fragment operation
    // which slow down OpenGL imaging operation (glDrawPixels glReadPixels...).
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_FOG);
    glDisable(GL_LIGHTING);
    glDisable(GL_LOGIC_OP);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);
}

///////////////////////////////////////////////////////////////////////////////
// write 2d text using GLUT
///////////////////////////////////////////////////////////////////////////////
void drawString(const char *str, int x, int y, void *font)
{
    glRasterPos2i(x, y);

    // loop all characters in the string
    while (*str)
    {
        glutBitmapCharacter(font, *str);
        ++str;
    }
}

//=============================================================================
// CALLBACKS
//=============================================================================

void displayCB()
{
    glutSetWindow(mainWin);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

void displaySubWin1CB(void)
{
    glutSetWindow(subWin1);
    glClear(GL_COLOR_BUFFER_BIT);  // clear canvas

    // specify current raster position(coordinate)
    glRasterPos2i(0, imageY);  // upper-left corner in openGL coordinates
    glPixelZoom(1.0, -1.0);

    glDrawPixels(image->x, image->y, image->format, image->type, image->buf);
    glPixelZoom(1.0, 1.0);

    glColor3f(0, 0, 1);
    drawString("Original Image", fontWidth, imageY - fontHeight, font);

    glutSwapBuffers();
}

void displaySubWin2CB()
{
    glutSetWindow(subWin2);
    glClear(GL_COLOR_BUFFER_BIT);  // clear canvas

    // specify current raster position(coordinate)
    glRasterPos2i(0, imageY);
    glPixelZoom(1.0, -1.0);

    glDrawPixels(image->x, image->y, image->format, image->type, outBuf1);
    glPixelZoom(1.0, 1.0);

    glColor3f(0, 0, 1);
    char str[64];
    sprintf(str, "Normal Convolution: %5.3f ms", time1);
    drawString(str, fontWidth, imageY - fontHeight, font);

    glutSwapBuffers();
}

void displaySubWin3CB(void)
{
    glutSetWindow(subWin3);
    glClear(GL_COLOR_BUFFER_BIT);  // clear canvas

    // specify current raster position(coordinate)
    glRasterPos2i(0, imageY);
    glPixelZoom(1.0, -1.0);

    glDrawPixels(image->x, image->y, image->format, image->type, outBuf2);
    glPixelZoom(1.0, 1.0);

    glColor3f(0, 0, 1);
    char str[64];
    sprintf(str, "Separable Convolution: %5.3f ms", time2);
    drawString(str, fontWidth, imageY - fontHeight, font);

    glutSwapBuffers();
}

void reshapeCB(int w, int h)
{
    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);

    // left sub-window
    glutSetWindow(subWin1);
    glutPositionWindow(0, 0);
    glutReshapeWindow(imageX, imageY);
    glViewport(0, 0, imageX, imageY);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, imageX, 0, imageY);

    // middle sub-window
    glutSetWindow(subWin2);
    glutPositionWindow(imageX, 0);
    glutReshapeWindow(imageX, imageY);
    glViewport(0, 0, imageX, imageY);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, imageX, 0, imageY);

    // right sub-window
    glutSetWindow(subWin3);
    glutPositionWindow(imageX * 2, 0);
    glutReshapeWindow(imageX, imageY);
    glViewport(0, 0, imageX, imageY);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, imageX, 0, imageY);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboardHandlerCB(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:  // ESCAPE
            clearSharedMem();
            exit(0);
            break;

        case 'r':
        case 'R':
            break;

        default:;
    }
    glutPostRedisplay();
}
