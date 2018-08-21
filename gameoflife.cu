#include "common/book.h"
#include "common/cpu_bitmap.h"

#include "cuda.h"
#include "cuda_gl_interop.h"

int *field_read_d;
int *field_writ_d;
int *field_rdwr_h;

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

GLuint  bufferObj;
cudaGraphicsResource *resource;

uchar4 *pixMap;

int ROWS = 1024;
int COLS = 1024;

float zoom = 0.0;
float dx   = 0.0;
float dy   = 0.0;

/* Utilitary fucntion to count neighbourds around a cell.*/
__device__ int
count_alive( int *field, int i, int j, int ROWS, int COLS)
{
   int x, y, a=0;
   for(x=i-1; x <= (i+1) ; x++)
   {
      for(y=j-1; y <= (j+1) ; y++)
      {
         if ( (x==i) && (y==j) ) continue;
         if ( (y<ROWS) && (x<COLS) && (x>=0)   && (y>=0) )
         {
              a += field[ROWS*y+x];
         }
      }
   }
   
   return a;
	
}

/* Evolve step applies the Game Of Life rules to each pixel and the 
 * ones around to check if the cell keeps living or dies.
 */
__global__ void
evolveStep ( int *field_read, int *field_write, int ROWS, int COLS )
{
	
	int  x = threadIdx.x + blockIdx.x*blockDim.x;
	int  y = threadIdx.y + blockIdx.y*blockDim.y;
	
	int neigh = count_alive( field_read, x, y, ROWS, COLS );
	
	if( field_read[y*ROWS + x] ){
		if( (neigh > 3) || (neigh<2) )
			field_write[ y*ROWS + x ] = 0;
		else
			field_write[ y*ROWS + x ] = 1;
	}
	else{
		if( neigh == 3  )
			field_write[ y*ROWS + x ] = 1;
		else
			field_write[ y*ROWS + x ] = 0;
	}
	
}

/* This function transforms the unidimensional array into a bitmap buffer
 * using the GPU parallelism to speed up the process
 */
__global__ void 
toPixelMap ( uchar4 *ptr, int *field ) 
{
   
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

	if( field[offset] ){
		ptr[offset].x = 255;
		ptr[offset].y = 255;
		ptr[offset].z = 255;
		ptr[offset].w = 255;
	}
	else{
		ptr[offset].x = 0;
		ptr[offset].y = 0;
		ptr[offset].z = 0;
		ptr[offset].w = 0;
	}
}

void generate_frame( uchar4 *pixels ) 
{
    dim3    grids(ROWS/16,COLS/16);
    dim3    threads(16,16);
    
    /*Evolve step is a kernel*/
    evolveStep <<< grids, threads >>>( field_read_d, field_writ_d, ROWS, COLS );
    /*toPixelMap is another kernel!*/     
    toPixelMap <<< grids, threads >>>( pixels, field_writ_d                   );
    /* Swap buffer; copy the field_write to field_read.
     * This is neccesary since parallel execution and pixel interdepenceny
     * does not guarantee we can operate in "serial mode" without messing 
     * things up badly!
     */	
    cudaMemcpy( field_read_d, field_writ_d, sizeof(int)*ROWS*COLS, cudaMemcpyDeviceToDevice );
	
}
	
static void key_func( unsigned char key, int x, int y ) 
{
    /*ESC key terminates program*/
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
            exit(0);
    }
    
    /*Just to zoom around*/
    if( key == 'a' )
		dx +=    0.5;
	if( key == 'd' )
		dx +=   -0.5;
	if( key == 'w' )
		dy +=   -0.5;
	if( key == 's' )
		dy +=    0.5;
	if( key == 'i' )
		zoom +=  0.5;
	if( key == 'j' )
		zoom += -0.5;
}

static void draw_func( void ) 
{
   	
    HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
    
    size_t  size;
    
    HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&pixMap, &size, resource) );
	generate_frame( pixMap );
    
    HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
    
    glDrawPixels( ROWS, COLS, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    
    glutSwapBuffers();
    glutPostRedisplay();
}


int main( int argc, char **argv ) 
{
    cudaDeviceProp  prop;
    int dev;

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( ROWS, COLS );
    glutCreateWindow( "bitmap" );

    /*Bitmap buffer bullshit*/
    /*************************************************************************************/
    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)   GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)   GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)   GET_PROC_ADDRESS("glBufferData");
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, ROWS * COLS * 4, NULL, GL_DYNAMIC_DRAW_ARB );
    HANDLE_ERROR(    cudaGraphicsGLRegisterBuffer( &resource , cudaGraphicsMapFlagsNone ) );
    /*************************************************************************************/
    
    /*Array to hold cell status*/
    field_rdwr_h = (int *)malloc( sizeof(int)*ROWS*COLS );
    /*Initialize with 0s and 1s*/
    for( int i=0; i<ROWS*COLS; i++ ) 
      field_rdwr_h[i] = rand()%2;
    
    /* Allocate memory on device: note we have 2 buffers, to read an write!*/
    cudaMalloc( (void **)&field_read_d, sizeof(int)*ROWS*COLS );
    cudaMalloc( (void **)&field_writ_d, sizeof(int)*ROWS*COLS );
    /*Copy stuff around*/ 
    cudaMemcpy( field_read_d, field_rdwr_h, sizeof(int)*ROWS*COLS, cudaMemcpyHostToDevice );
    
    /*Attach keyboard function*/
    glutKeyboardFunc( key_func );
    /*Attach display function, yes, a pointer to a function LoL*/
    glutDisplayFunc( draw_func );
    /*Start loop*/
    glutMainLoop();
}
