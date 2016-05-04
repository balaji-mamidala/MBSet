/* 
 * Created on June 24, 2012
 *
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 *
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"

#include <GL/freeglut.h>

#include <vector>
//#include <math.h>

#define THREADS_PER_BLOCK     32

using namespace std;

const uint16_t maxIt = 2000; // Maximum Iterations

// Size of window in pixels, both width and height
uint16_t WINDOW_DIM_X = 512;
uint16_t WINDOW_DIM_Y = 512;

uint16_t SIZE_X = WINDOW_DIM_X;
uint16_t SIZE_Y = WINDOW_DIM_Y;
uint32_t SIZE = SIZE_X*SIZE_Y;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);

vector<Complex*> vec_c; //Vector that stores zoomed c values
Complex *c;
Complex *d_c;
Complex *d_minC;
Complex *d_maxC;


vector<int*> vec_MBSet; //Vector that stores zoomed sets
int *MBSet; //MBSet for current frame
int *d_MBSet;

uint16_t x1draw, y1draw, x2draw, y2draw;

bool draw_box = false;
bool left_click = false;

enum color_base {red, red_orange, metallic_gold, gold};
color_base use_color = red;

int windowID;


// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};


RGB* colors = new RGB[maxIt + 1];

void InitializeColors()
{
  //Use histogram approach for color
  uint16_t histogram[maxIt+1];

  for(uint16_t i=0; i<maxIt+1; i++)
  {
    histogram[i] = 0;
  }

  for(uint32_t i=0; i<SIZE; i++)
  {
    histogram[MBSet[i]]++;
  }

  uint64_t total_sum = 0;
  for(uint16_t i=0; i<maxIt+1; i++)
  {
    total_sum += histogram[i];
  }


  float sred = 0.0;
  float ered;
  float sgreen = 0.0;
  float egreen;
  float sblue = 0.0;
  float eblue;

  //Red
  if(red == use_color)
  {
    ered = 0.9;
    egreen = 0.0;
    eblue = 0.0;
  }
  else if(red_orange == use_color)
  {
    ered = 1.0;
    egreen = 0.4;
    eblue = 0.0;
  }
  else if(metallic_gold == use_color)
  {
    ered = 0.828;
    egreen = 0.6853;
    eblue = 0.214;
  }
  else if(gold == use_color)
  {
    ered = 1.0;
    egreen = 0.84;
    eblue = 0.0;
  }

  uint64_t running_total = 0;
  for (int i = 0; i < maxIt; ++i)
  {
    running_total += histogram[i];

    float percent = running_total/(float)total_sum;

    float r = sred + ((ered - sred) * percent);
    float g = sgreen + ((egreen - sgreen) * percent);
    float b = sblue + ((eblue - sblue) * percent);

    colors[i] = RGB(r, g, b);
  }

  colors[maxIt] = RGB(); // black
}


/*void window_data_init()
{
  for(uint16_t row=0; row < WINDOW_DIM_X; row++)
  {
    for(uint16_t col=0; col< WINDOW_DIM_Y; col++)
    {
      c[(row*SIZE_X) + col].r =  minC.r + (rstep * row);
      c[(row*SIZE_X) + col].i =  minC.i + (istep * col);
    }
  }
}*/


__global__ void calcMBS_limit(Complex* minC, Complex* maxC, Complex* c, int* MBSet)
{
  uint32_t index = threadIdx.x + (blockIdx.x * blockDim.x);

  int row = index/512;
  int col = index - (row*512);

  float rstep = (maxC[0].r - minC[0].r)/(float)512; 
  float istep = (maxC[0].i - minC[0].i)/(float)512;

  c[index].r = minC[0].r + (rstep * row);
  c[index].i = minC[0].i + (istep * col);

  Complex Z(c[index].r, c[index].i);

  for(MBSet[index]=0; MBSet[index] < maxIt ; MBSet[index]++)
  {
    Z = (Z*Z) + c[index];

    if(Z.magnitude2() > 4.0)
    {
      break;
    }
  }
}


///Opengl functions begin
void drawModel()
{ 
  //Display MBset
  glBegin(GL_POINTS);
  uint32_t index;
  for(uint16_t row=0; row<WINDOW_DIM_X; row++)
  {
    for(uint16_t col=0; col<WINDOW_DIM_Y; col++)
    {
      index = row * SIZE_X + col;
      glColor3d(colors[MBSet[index]].r, colors[MBSet[index]].g, colors[MBSet[index]].b);
      glVertex2i(row, WINDOW_DIM_Y - col -1 );
    }
  }
  glEnd();

  if(true == draw_box)
  {
    glColor3f(1.0, 0.0, 0.0);
    glScalef(1.0, 1.0, 0);
    glLineWidth(2.0);

    glBegin(GL_LINE_LOOP);

    glVertex2i(x1draw, WINDOW_DIM_Y - y1draw); // Top left corner
    glVertex2i(x2draw, WINDOW_DIM_Y - y1draw); // Top right corner
    glVertex2i(x2draw, WINDOW_DIM_Y - y2draw); // Bottom right corner
    glVertex2i(x1draw, WINDOW_DIM_Y - y2draw); // Bottom left corner

    glEnd();
  } 
}


void display(void)
{
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();

  // Draw the model
  drawModel();
  // Swap the double buffers
  glutSwapBuffers();
}

void reshape(int w, int h)
{
  glViewport(0, 0, (GLsizei)WINDOW_DIM_X, (GLsizei)WINDOW_DIM_Y);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, (GLdouble)WINDOW_DIM_X, (GLdouble)0.0, (GLdouble)WINDOW_DIM_X, ((GLdouble)-1), (GLdouble)1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glutReshapeWindow(WINDOW_DIM_X, WINDOW_DIM_Y);

  glutPostRedisplay();   // repaint the window
}


void keyboard(unsigned char key, int x, int y)
{
  switch(key)
  {
    case 'q':
      glutDestroyWindow(windowID);
      exit(0);

    case 'b':
      if( (0!= vec_c.size()) && (0!= vec_MBSet.size()) )
      {
        free(c);
        free(MBSet);

        c = vec_c.back();
        MBSet = vec_MBSet.back();

        vec_c.pop_back();
        vec_MBSet.pop_back();
      
        InitializeColors();
        glutPostRedisplay();  
      }

      break;
    
    case 'c':
      if(red == use_color)
      {
        use_color = red_orange;
      }
      else if(red_orange == use_color)
      {
        use_color = metallic_gold;
      }
      else if(metallic_gold == use_color)
      {
        use_color = gold;
      }
      else if(gold == use_color)
      {
        use_color = red;
      }

      InitializeColors();
      glutPostRedisplay();   // repaint the window
      break;   

    default:
      break;
  }
}

void order_co_ordinate(void)
{

  //cout << "Before re-order :: " << "X1: " << x1draw << " Y1: " << y1draw << " X2: " << x2draw << " Y2: " << y2draw << endl;

  if(x1draw > x2draw)
  {
    uint16_t temp = x1draw;
    x1draw = x2draw;
    x2draw = temp;
  }

  if(y1draw > y2draw)
  {
    uint16_t temp = y1draw;
    y1draw = y2draw;
    y2draw = temp;
  }

  //cout << "After re-order :: " << "X1: " << x1draw << " Y1: " << y1draw << " X2: " << x2draw << " Y2: " << y2draw << endl;
}


void re_calc_draw_MBSet(void)
{
  order_co_ordinate();

  uint32_t min_index = (x1draw * SIZE_X) + y1draw;
  uint32_t max_index = (x2draw * SIZE_X) + y2draw;

  vec_c.push_back(c);
  vec_MBSet.push_back(MBSet);

  minC.r = c[min_index].r;
  minC.i = c[min_index].i;

  maxC.r = c[max_index].r;
  maxC.i = c[max_index].i;

  //Allocate more memory for new arrays
  c = (Complex*) malloc(SIZE*sizeof(Complex));
  MBSet = (int*) malloc(SIZE*sizeof(int));

  //Copy from Host to device
  cudaMemcpy(d_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_MBSet, MBSet, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  // Calculate the interation counts
  //Launch MBS calculation on GPU
  calcMBS_limit<<<SIZE/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_minC, d_maxC, d_c, d_MBSet);

  //Copy from Device to Host
  cudaMemcpy(MBSet, d_MBSet, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, SIZE*sizeof(Complex), cudaMemcpyDeviceToHost);  

  InitializeColors();
  glutPostRedisplay();   // repaint the window
}


void mouse_motion(int x, int y)
{
  if( (x<0) || (y<0) || (x>glutGet(GLUT_WINDOW_WIDTH)) || (y>glutGet(GLUT_WINDOW_HEIGHT)) )
  {
    return;
  }

  if(true == left_click)
  {
    draw_box = true;

    x2draw = x;
    y2draw = y;

    int length;

    if( ((x1draw-x2draw)*(x1draw-x2draw)) < ((y1draw-y2draw)*(y1draw-y2draw)) )
    {
      if(x1draw > x2draw)
      {
        length = x1draw - x2draw;
      }
      else
      {
        length = x2draw - x1draw;
      }

      if(y2draw > y1draw)
      {
        y2draw = y1draw + length;
      }
      else
      {
        y2draw = y1draw - length;
      }

    }
    else
    {
      if(y1draw > y2draw)
      {
        length = y1draw - y2draw;
      }
      else
      {
        length = y2draw - y1draw;
      }

      if(x2draw > x1draw)
      {
        x2draw = x1draw + length;
      }
      else
      {
        x2draw = x1draw - length;
      }
    }
  }

  glutPostRedisplay();   // repaint the window
  //cout << "X1: " << x1draw << " Y1: " << y1draw << " X2: " << x2draw << " Y2: " << y2draw << endl;
}


void mouse_click(int button, int state, int x, int y )
{

  if( (x<0) || (y<0) || (x>glutGet(GLUT_WINDOW_WIDTH)) || (y>glutGet(GLUT_WINDOW_HEIGHT)) )
  {
    //Stop drawing box if Left button is release outside window
    if( (GLUT_LEFT_BUTTON == button) && (GLUT_UP == state) )
    {
      draw_box = false;
      left_click = false;

      re_calc_draw_MBSet();
    }

    return;
  }

  if( (GLUT_LEFT_BUTTON == button) && (GLUT_DOWN == state) )
  {
    left_click = true;
    x1draw = x;
    y1draw = y;
  }
  else if( (GLUT_LEFT_BUTTON == button) && (GLUT_UP == state) )
  {
    draw_box = false;
    left_click = false;

    re_calc_draw_MBSet();
  }
}

//opengl functions end



int main(int argc, char** argv)
{
  // Initialize OPENGL here
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(WINDOW_DIM_X, WINDOW_DIM_Y);

  GLsizei windowX = (glutGet(GLUT_SCREEN_WIDTH)-WINDOW_DIM_X)/2;
  GLsizei windowY = (glutGet(GLUT_SCREEN_HEIGHT)-WINDOW_DIM_Y)/2;
  glutInitWindowPosition(windowX, windowY);

  windowID = glutCreateWindow("Mandelbrot Set");
  glDisable(GL_DEPTH_TEST);
  glShadeModel(GL_FLAT);

  // Set up necessary host and device buffers
  c = (Complex*) malloc(SIZE*sizeof(Complex));
  MBSet = (int*) malloc(SIZE*sizeof(int));

  cudaMalloc((void**)&d_c, SIZE*sizeof(Complex));
  cudaMalloc((void**)&d_MBSet, SIZE*sizeof(int));
  cudaMalloc((void**)&d_minC, sizeof(Complex));
  cudaMalloc((void**)&d_maxC, sizeof(Complex));  

  //Copy from Host to device
  cudaMemcpy(d_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_MBSet, MBSet, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  // Calculate the interation counts
  //Launch MBS calculation on GPU
  calcMBS_limit<<<SIZE/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_minC, d_maxC, d_c, d_MBSet);

  //Copy from Device to Host
  cudaMemcpy(MBSet, d_MBSet, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, SIZE*sizeof(Complex), cudaMemcpyDeviceToHost);


  glutDisplayFunc(display);
  glutIdleFunc(display);
  glutKeyboardFunc (keyboard);
  glutMouseFunc(mouse_click);
  glutMotionFunc(mouse_motion);
  glutReshapeFunc(reshape);

  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels

  InitializeColors();
  glutMainLoop(); // THis will callback the display, keyboard and mouse

  free(c);
  free(MBSet);
  delete(colors);

  vector<Complex*>::iterator it_c = vec_c.begin();
  while(it_c != vec_c.end())
  {
    free(*it_c);
    it_c++;
  }

  vector<int*>::iterator it_MBSet = vec_MBSet.begin();
  while(it_MBSet != vec_MBSet.end())
  {
    free(*it_MBSet);
    it_MBSet++;
  }

  cudaFree(d_c);
  cudaFree(d_MBSet);
  cudaFree(d_minC);
  cudaFree(d_maxC);

  return 0;
}

