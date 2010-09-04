#ifndef _CONVOLUTION_KERNEL_CU
#define _CONVOLUTION_KERNEL_CU

using namespace std;

#include <stdio.h>

#define IDX_1D(col, row, stride) ((col * stride) + row)
#define COL_2D(index, stride) (index / stride)
#define ROW_2D(index, stride) (index % stride)
#define ROUNDUP32(integer) ( ((integer-1)/32 + 1) * 32 )

#define SHMEM 8192
#define FLOAT_SZ sizeof(float)

texture<float, 3, cudaReadModeElementType> texPsf;
textureReference* texPsfRef;
// Use tex3D(texPsf, x, y, z) to access texture data


////////////////////////////////////////////////////////////////////////////////
//
// This macros is defined because EVERY convolution-like function has the same
// variables.  Mainly, the pixel identifiers for this thread based on block
// size, and the size of the padded rectangle that each block will work with
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad) \
\
   const int cornerCol = blockDim.x*blockIdx.x;   \
   const int cornerRow = blockDim.y*blockIdx.y;   \
   const int globalCol = cornerCol + threadIdx.x;   \
   const int globalRow = cornerRow + threadIdx.y;   \
   const int globalIdx = IDX_1D(globalCol, globalRow, imgRows);   \
\
   const int localCol    = threadIdx.x;   \
   const int localRow    = threadIdx.y;   \
   const int localIdx    = IDX_1D(localCol, localRow, blockDim.y);   \
   const int localPixels = blockDim.x*blockDim.y;   \
\
   const int padRectStride = blockDim.y + 2*psfRowRad;   \
   const int padRectCol    = localCol + psfColRad;   \
   const int padRectRow    = localRow + psfRowRad;   \
   /*const int padRectIdx    = IDX_1D(padRectCol, padRectRow, padRectStride); */ \
   const int padRectPixels = padRectStride * (blockDim.x + 2*psfColRad);   \
\
   __shared__ char sharedMem[SHMEM];   \
   float* shmPadRect  = (float*)sharedMem;   \
   float* shmOutput   = (float*)&shmPadRect[ROUNDUP32(padRectPixels)];   \
   int nLoop;



////////////////////////////////////////////////////////////////////////////////
//
// Every block will need a buffered copy of the input data in its shared memory,
// so it doesn't do multiple global memory reads to find the energy contributing
// to it's own value.
//
// This copy is very much like COPY_LIN_ARRAY_TO_SHMEM except that this isn't 
// a linear array, and this needs to accommodate pixels that fall out of bounds
// from the image.  Threads are temporarily reassigned to execute this copy in
// parallel.
//
////////////////////////////////////////////////////////////////////////////////
#define PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad) \
\
   nLoop = (padRectPixels/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < padRectPixels)   \
      {   \
         int prCol = COL_2D(prIndex, padRectStride);   \
         int prRow = ROW_2D(prIndex, padRectStride);   \
         int glCol = cornerCol + prCol - psfColRad;   \
         int glRow = cornerRow + prRow - psfRowRad;   \
         int glIdx = IDX_1D(glCol, glRow, imgRows);   \
         if(glRow >= 0        &&  \
            glRow <  imgRows  &&  \
            glCol >= 0        &&  \
            glCol <  imgCols)   \
            shmPadRect[prIndex] = imgInPtr[glIdx];   \
         else   \
            shmPadRect[prIndex] = 0.0f;   \
      }   \
   }   \

////////////////////////////////////////////////////////////////////////////////
//
// Same as above, except for binary images, using -1 as "OFF" and +1 as "ON"
// The user is not expected to do this him/herself, and it's easy enough to 
// manipulate the data on the way in and out (just don't forget to convert back
// before copying out the result
//
////////////////////////////////////////////////////////////////////////////////
#define PREPARE_PADDED_RECTANGLE_BINARY(psfColRad, psfRowRad) \
\
   nLoop = (padRectPixels/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < padRectPixels)   \
      {   \
         int prCol = COL_2D(prIndex, padRectStride);   \
         int prRow = ROW_2D(prIndex, padRectStride);   \
         int glCol = cornerCol + prCol - psfColRad;   \
         int glRow = cornerRow + prRow - psfRowRad;   \
         int glIdx = IDX_1D(glCol, glRow, imgRows);   \
         if(glRow >= 0        &&  \
            glRow <  imgRows  &&  \
            glCol >= 0        &&  \
            glCol <  imgCols)   \
            shmPadRect[prIndex] = imgInPtr[glIdx]*2 - 1;   \
         else   \
            shmPadRect[prIndex] = -1; \
      }   \
   }   \

////////////////////////////////////////////////////////////////////////////////
//
// Frequently, we want to pull some linear arrays into shared memory (usually 
// PSFs) which will be queried often, and we want them close to the threads.
//
// This macro temporarily reassigns all the threads to do the memory copy from
// global memory to shared memory in parallel.  Since the array may be bigger
// than the blocksize, some threads may be doing multiple mem copies
//
////////////////////////////////////////////////////////////////////////////////
#define COPY_LIN_ARRAY_TO_SHMEM(srcPtr, dstPtr, nValues) \
   nLoop = (nValues/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < nValues)   \
      {   \
         dstPtr[prIndex] = srcPtr[prIndex]; \
      } \
   } 




__global__ void   convolveBasic( 
               float* imgInPtr,    
               float* imgOutPtr,    
               float* imgPsfPtr,    
               int    imgCols,    
               int    imgRows,    
               int    psfColRad,
               int    psfRowRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad); 
   shmOutput[localIdx] = 0.0f;

   const int psfStride = psfRowRad*2+1;   
   const int psfPixels = psfStride*(psfColRad*2+1);   
   float* shmPsf = (float*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(imgPsfPtr, shmPsf, psfPixels); 

   PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad); 


   __syncthreads();   


   float accumFloat = 0.0f; 
   for(int coff=-psfColRad; coff<=psfColRad; coff++)   
   {   
      for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
      {   
         int psfCol = psfColRad - coff;   
         int psfRow = psfRowRad - roff;   
         int psfIdx = IDX_1D(psfCol, psfRow, psfStride);   
         float psfVal = shmPsf[psfIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
         accumFloat += psfVal * shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accumFloat;  
   __syncthreads();   

   imgOutPtr[globalIdx] = shmOutput[localIdx];  
}


/*
// TODO: Still need to debug this function
__global__ void   convolveBilateral( 
               float* imgInPtr,    
               float* imgOutPtr,    
               float* imgPsfPtr,    
               float* intPsfPtr,    
               int    imgCols,    
               int    imgRows,    
               int    psfColRad,
               int    psfRowRad,
               int    intPsfRad)
{  

   CREATE_CONVOLUTION_VARIABLES(psfColRad, psfRowRad); 
   shmOutput[localIdx] = 0.0f;

   const int psfStride = psfRowRad*2+1;   
   const int psfPixels = psfStride*(psfColRad*2+1);   
   float* shmPsf  = (float*)&shmOutput[ROUNDUP32(localPixels)];   
   float* shmPsfI = (float*)&shmPsf[ROUNDUP32(psfPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM(imgPsfPtr, shmPsf,  psfPixels); 
   COPY_LIN_ARRAY_TO_SHMEM(intPsfPtr, shmPsfI, 2*intPsfRad+1);

   PREPARE_PADDED_RECTANGLE(psfColRad, psfRowRad); 


   __syncthreads();   


   float accumFloat = 0.0f; 
   float myVal = shmPadRect[padRectIdx];
   for(int coff=-psfColRad; coff<=psfColRad; coff++)   
   {   
      for(int roff=-psfRowRad; roff<=psfRowRad; roff++)   
      {   
         int psfCol = psfColRad - coff;   
         int psfRow = psfRowRad - roff;   
         int psfIdx = IDX_1D(psfCol, psfRow, psfStride);   
         float psfVal = shmPsf[psfIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
         float thatVal = shmPadRect[shmPRIdx];

         float intVal = shmPsfI[(int)(thatVal-myVal+intPsfRad)];

         accumFloat += psfVal * intVal *shmPadRect[shmPRIdx];   
      }   
   }   
   shmOutput[localIdx] = accumFloat;  
   __syncthreads();   

   imgOutPtr[globalIdx] = shmOutput[localIdx];  
}
*/

#endif

