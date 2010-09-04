////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// GPU-Accelerated Morphology Toolbox
//
// Author:  Alan Reiner
// Email:   etotheipi@gmail.com
// Date:    03 September, 2010
//
// Description:  This header provides a complete set of tools to do any sequence
//               of arbitrary morphological operations on a binary mask.  The 
//               implementation is a little non-intuitive, because it uses the
//               values {-1, 0, 1} internally to represent the mask, instead of 
//               the usual {0, 1}.  This so that we can simultaneously represent
//               "Don't Care" values in a structuring element, and we use only
//               integer multiplications to do our Hit-Or-Miss (HoM) operations. 
//               Therefore, there is no need for conditionals in the inner loop 
//               (which are generally very slow on the GPU).
//
//               A user of this library does not really need to understand the 
//               implementation, only if he intends to expand the library, and
//               write extra morphological operations.
//          
//               The key to understanding the implementation is in the variable
//               SE_NON_ZERO:  
//
//
// SE_NON_ZERO:
//
//    This variable is a side-effect of the highly efficient morphology ops.
//    For a given pixel, when you compare the 3x3, you multiply the SE elements
//    to the image elements.  For each pixel compared, if the SE element is:
//       HIT:       result  1
//       MISS:      result -1
//       DONTCARE:  result  0
//
//    We sum up the results for each pixel in the NxN neighborhood.  In a hit-
//    or-miss operation, we need all the +1 and -1 to match exactly (each
//    pixel returns +1), so SE_NON_ZERO should be total number of non-zero
//    elements.  If we are looking for 
//    calculated on the fly, but it's constant for each SE, and would slow down
//    the kernel significantly
//
// EXAMPLE:
//  
//    STRUCTURING ELEMENT:    0  1  0
//                            1  1 -1
//                            0 -1 -1
// 
//    This SE has 6 non-zero elements.  Also important is the fact that ALL 
//    elements much hit in order to "pass", so we pass in 6 for SE_NON_ZERO
//
//    IMAGE CHUNK1:   1  1 -1      Dilate Result:  6
//                    1  1 -1
//                   -1 -1 -1
//
//
//    IMAGE CHUNK2:  -1  1 -1      Dilate Result:  6
//                    1  1 -1
//                   -1 -1 -1
//
//
//    IMAGE CHUNK3:  -1 -1  1      Dilate Result:  -6
//                   -1 -1  1
//                    1  1  1
//             
//
//    IMAGE CHUNK4:  -1 -1 -1      Dilate Result:  2
//                    1  1 -1
//                    1  1 -1
//           
//          
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#ifndef _GPU_MORPHOLOGY_CU_
#define _GPU_MORPHOLOGY_CU_


using namespace std;

#include <stdio.h>
#include <vector>

#define IDX_1D(col, row, stride) ((col * stride) + row)
#define COL_2D(index, stride) (index / stride)
#define ROW_2D(index, stride) (index % stride)
#define ROUNDUP32(integer) ( ((integer-1)/32 + 1) * 32 )

#define SHMEM 8192
#define FLOAT_SZ sizeof(float)
#define INT_SZ   sizeof(int)


////////////////////////////////////////////////////////////////////////////////
//
// This macros is defined because EVERY convolution-like function has the same
// variables.  Mainly, the pixel identifiers for this thread based on block
// size, and the size of the padded rectangle that each block will work with
//
// ***This is actually the same as the CONVOLVE version
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_CONVOLUTION_VARIABLES_MORPH(psfColRad, psfRowRad) \
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
   __shared__ int sharedMem[SHMEM];   \
   int* shmPadRect  = (int*)sharedMem;   \
   int* shmOutput   = (int*)&shmPadRect[ROUNDUP32(padRectPixels)];   \
   int nLoop;



////////////////////////////////////////////////////////////////////////////////
//
// We are using -1 as "OFF" and +1 as "ON" and 0 as "DONTCARE"
// The user is not expected to do this him/herself, and it's easy enough to 
// manipulate the data on the way in and out (just don't forget to convert back
// before copying out the result
//
////////////////////////////////////////////////////////////////////////////////
#define PREPARE_PADDED_RECTANGLE_MORPH(psfColRad, psfRowRad) \
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
            shmPadRect[prIndex] = devInPtr[glIdx]*2 - 1;   \
         else   \
            shmPadRect[prIndex] = -1; \
      }   \
   }   

////////////////////////////////////////////////////////////////////////////////
//
// Frequently, we want to pull some linear arrays into shared memory (usually 
// PSFs) which will be queried often, and we want them close to the threads.
//
// This macro temporarily reassigns all the threads to do the memory copy from
// global memory to shared memory in parallel.  Since the array may be bigger
// than the blocksize, some threads may be doing multiple mem copies
//
// ***This is actually the same as the FLOAT version
//
////////////////////////////////////////////////////////////////////////////////
#define COPY_LIN_ARRAY_TO_SHMEM_MORPH(srcPtr, dstPtr, nValues) \
   nLoop = (nValues/localPixels)+1;   \
   for(int loopIdx=0; loopIdx<nLoop; loopIdx++)   \
   {   \
      int prIndex = loopIdx*localPixels + localIdx;   \
      if(prIndex < nValues)   \
      {   \
         dstPtr[prIndex] = srcPtr[prIndex]; \
      } \
   } 




////////////////////////////////////////////////////////////////////////////////
//
// This macro creates optimized, unrolled versions of the generic
// morphological operation kernel for 3x3 structuring elements.
//
// Since it has no loops, and only one if-statement per thread, it should
// extremely fast.  The generic kernel is fast too, but slowed down slightly
// by the doubly-nested for-loops.
//
// TODO:  We should create 3x1 and 1x3 functions (and possibly Nx1 & 1xN)
//        so that we can further optimize morph ops for separable SEs
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_3X3_MORPH_KERNEL( name, seTargSum, \
                                            a00, a10, a20, \
                                            a01, a11, a21, \
                                            a02, a12, a22) \
__global__ void  Morph3x3_##name##_Kernel(       \
               int*   devInPtr,          \
               int*   devOutPtr,          \
               int    imgCols,          \
               int    imgRows)  \
{        \
   CREATE_CONVOLUTION_VARIABLES_MORPH(1, 1); \
\
   PREPARE_PADDED_RECTANGLE_MORPH(1, 1); \
\
   shmOutput[localIdx] = -1;\
\
   __syncthreads();   \
\
   int accum = 0;\
\
   accum += a00 * shmPadRect[IDX_1D(padRectCol-1, padRectRow-1, padRectStride)];  \
   accum += a01 * shmPadRect[IDX_1D(padRectCol-1, padRectRow  , padRectStride)];  \
   accum += a02 * shmPadRect[IDX_1D(padRectCol-1, padRectRow+1, padRectStride)];  \
   accum += a10 * shmPadRect[IDX_1D(padRectCol  , padRectRow-1, padRectStride)];  \
   accum += a11 * shmPadRect[IDX_1D(padRectCol  , padRectRow  , padRectStride)];  \
   accum += a12 * shmPadRect[IDX_1D(padRectCol  , padRectRow+1, padRectStride)];  \
   accum += a20 * shmPadRect[IDX_1D(padRectCol+1, padRectRow-1, padRectStride)];  \
   accum += a21 * shmPadRect[IDX_1D(padRectCol+1, padRectRow  , padRectStride)];  \
   accum += a22 * shmPadRect[IDX_1D(padRectCol+1, padRectRow+1, padRectStride)];  \
\
   if(accum >= seTargSum) \
      shmOutput[localIdx] = 1; \
\
   __syncthreads();   \
\
   devOutPtr[globalIdx] = (shmOutput[localIdx] + 1) / 2; \
}



////////////////////////////////////////////////////////////////////////////////
// This macro simply creates the declarations for the above functions, to be
// used in the header file
////////////////////////////////////////////////////////////////////////////////
#define DECLARE_3X3_MORPH_KERNEL( name ) \
__global__ void  Morph3x3_##name##_Kernel(       \
               int*   devInPtr,          \
               int*   devOutPtr,          \
               int    imgCols,          \
               int    imgRows); 




////////////////////////////////////////////////////////////////////////////////
// This macro creates member method wrappers for each of the kernels created
// with the CREATE_3X3_MORPH_KERNEL macro.
//
// NOTE:  CREATE_3X3_MORPH_KERNEL macro creates KERNEL functions, this macro
//        creates member methods in MorphWorkbench that wrap those kernel
//        functions.  When calling these, you don't need to include the  
//        <<<GRID,BLOCK>>> as you would with a kernel function
//
////////////////////////////////////////////////////////////////////////////////
#define CREATE_MWB_3X3_FUNCTION( name ) \
   void name(void) \
   {  \
      Morph3x3_##name##_Kernel<<<GRID_,BLOCK_>>>( \
                        *devBufferPtrA_, \
                        *devBufferPtrB_, \
                        imageCols_,  \
                        imageRows_);  \
      flipBuffers(); \
   } \
\
   void Z##name(int* src, int* dst) \
   {  \
      Morph3x3_##name##_Kernel<<<GRID_,BLOCK_>>>( \
                        src, \
                        dst, \
                        imageCols_,  \
                        imageRows_);  \
   } 



////////////////////////////////////////////////////////////////////////////////
// Standard 3x3 erosions, dilations and median filtering
DECLARE_3X3_MORPH_KERNEL( Dilate )
DECLARE_3X3_MORPH_KERNEL( Erode )
DECLARE_3X3_MORPH_KERNEL( DilateCross )
DECLARE_3X3_MORPH_KERNEL( ErodeCross )
DECLARE_3X3_MORPH_KERNEL( Median )
DECLARE_3X3_MORPH_KERNEL( MedianCross )

////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for THINNING
DECLARE_3X3_MORPH_KERNEL( Thin1 );
DECLARE_3X3_MORPH_KERNEL( Thin2 );
DECLARE_3X3_MORPH_KERNEL( Thin3 );
DECLARE_3X3_MORPH_KERNEL( Thin4 );
DECLARE_3X3_MORPH_KERNEL( Thin5 );
DECLARE_3X3_MORPH_KERNEL( Thin6 );
DECLARE_3X3_MORPH_KERNEL( Thin7 );
DECLARE_3X3_MORPH_KERNEL( Thin8 );
        
////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for PRUNING
DECLARE_3X3_MORPH_KERNEL( Prune1 );
DECLARE_3X3_MORPH_KERNEL( Prune2 );
DECLARE_3X3_MORPH_KERNEL( Prune3 );
DECLARE_3X3_MORPH_KERNEL( Prune4 );
DECLARE_3X3_MORPH_KERNEL( Prune5 );
DECLARE_3X3_MORPH_KERNEL( Prune6 );
DECLARE_3X3_MORPH_KERNEL( Prune7 );
DECLARE_3X3_MORPH_KERNEL( Prune8 );



////////////////////////////////////////////////////////////////////////////////
// BASIC UNARY & BINARY *MASK* OPERATORS
// 
// Could create LUTs, but I'm not sure the extra implementation complexity
// actually provides much benefit.  These ops already run on the order of
// microseconds.
//
// NOTE:  These operators are for images with {0,1}, only the MORPHOLOGICAL
//        operators will operate with {-1,0,1}
//
////////////////////////////////////////////////////////////////////////////////
__global__ void  MaskUnion_Kernel( int* A, int* B, int* devOut);
__global__ void  MaskIntersect_Kernel( int* A, int* B, int* devOut);
__global__ void  MaskSubtract_Kernel( int* A, int* B, int* devOut);
__global__ void  MaskInvert_Kernel( int* A, int* devOut);
__global__ void  MaskCopy_Kernel( int* A, int* devOut);
__global__ void  MaskCountDiff_Kernel( int* A, int* B, int* globalMemCount);
__global__ void  MaskSum_Kernel( int* A, int* globalMemSum);



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// ***Generic Morphologoical Operation Kernel Function***
// 
//    This is the basis for *ALL* other morpohological operations.  Every 
//    morphological operation in this library can be traced back to this
//    (the optimized 3x3 ops are hardcoded/unrolled versions of this function)
//
//    For all morph operations, we use {-1, 0, +1} ~ {OFF, DONTCARE, ON}.
//    This mapping allows us to use direct integer multiplication and 
//    summing of SE and image components.  Integer multiplication is 
//    much faster than using lots of if-statements.
//
//    Erosion, dilation, median, and a variety of weird and unique 
//    morphological operations are created solely by adjusting the 
//    target sum argument (seTargSum).
// 
////////////////////////////////////////////////////////////////////////////////
//
// Target Sum Values:
//
// The following describes under what conditions the SE is considered to "hit"
// a chunk of the image, based on how many indvidual pixels it "hits":
//
//
//    Erosion:  Hit every non-zero pixel
//
//          If we hit every pixel, we get a +1 for every non-zero elt
//          Therefore, our target should be [seNonZero]
//
//    Dilation:  Hit at least one non-zero pixel
//
//          If we miss every single pixel:  sum == -seNonZero
//          If we hit one pixel:            sum == -seNonZero+2;
//          If we hit two pixels:           sum == -seNonZero+4;
//          ...
//          Therefore, our target should be [-seNonZero+1] or greater
//
//
//    Median:   More pixels hit than not hit
//       
//          Since each pixel-hit is a +1, and each pixel-miss is a -1,
//          the median is 1 if and only if there are more +1s than -1s.
//          Therefore, our target should be [0] or greater
//
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void  Morph_Generic_Kernel( 
               int*  devInPtr,    
               int*  devOutPtr,    
               int   imgCols,    
               int   imgRows,    
               int*  sePtr,    
               int   seColRad,
               int   seRowRad,
               int   seTargSum);

////////////////////////////////////////////////////////////////////////////////
// 
// Structuring Element
//
// Structuring elements (SE) are the Point-Spread Functions (PSF) of image 
// morphology.  We use {-1, 0, +1} for {OFF, DONTCARE, ON}
//
// NOTE:  A structuring element object is directly linked to the device memory
//        where the SE data resides.  This class allocates the device memory
//        on construction and frees it on destruction
// 
////////////////////////////////////////////////////////////////////////////////
class StructElt
{
private:
   int* devPtr_;
   int  seCols_;
   int  seRows_;
   int  seElts_;
   int  seBytes_;
   int  seNonZero_;

public:
   void init(int* hostSE, int nc, int nr)
   {
      int numNonZero = 0;
      for(int i=0; i<seElts_; i++)
         if(hostSE[i] == -1 || hostSE[i] == 1)
           numNonZero++;

      init(hostSE, nc, nr, numNonZero);
   }

   void init(int* hostSE, int nc, int nr, int senz)
   {
      seCols_ = nc;
      seRows_ = nr;
      seElts_ = seCols_ * seRows_;
      seBytes_ = seElts_ * INT_SZ;
      seNonZero_ = senz;
      cudaMalloc((void**)&devPtr_, seBytes_);
      cudaMemcpy(devPtr_, hostSE, seBytes_, cudaMemcpyHostToDevice);
   }

   StructElt() :
      devPtr_(NULL),
      seCols_(-1),
      seRows_(-1),
      seElts_(-1),
      seBytes_(-1),
      seNonZero_(0) {}

   StructElt(int* hostSE, int nc, int nr) { init(hostSE, nc, nr); }

   ~StructElt() { cudaFree(devPtr_ ); }

   int* getDevPtr(void)  const  {return devPtr_;}
   int  getCols(void)    const  {return seCols_;}
   int  getRows(void)    const  {return seRows_;}
   int  getElts(void)    const  {return seElts_;}
   int  getBytes(void)   const  {return seBytes_;}
   int  getNonZero(void) const  {return seNonZero_;}
};




////////////////////////////////////////////////////////////////////////////////
// 
// MorphWorkbench
// 
// A morphology workbench is used when you have a single image to which you want
// to apply a sequence of dozens, hundreds or thousands of mophology operations.
//
// The workbench copies the input data to the device once at construction, 
// and then applies all the operations, only extracting the result from the
// device when "fetchBuffer" is called.
//
// The workbench uses two primary image buffers, which are used to as input and
// output buffers, flipping back and forth every operation.  This is so that
// we don't need to keep copying the output back to the input buffer after each
// operation.
// 
// There's also on-demand temporary buffers, which may be needed for more
// advanced morphological operations.  For instance, the pruning and thinning
// kernels only *locate* pixels that need to be removed.  So we have to apply
// the pruning/thinning SEs into a temp buffer, and then subtract that buffer
// from the input.  This is why we have devExtraBuffers_.
//
// Static Data:
//
//    masterSEList_:
//
//    This class keeps a master list of all structuring elements and all
//    workbenches.  The static list of structuring elements ensures that we
//    don't have to keep copying them into device memory every time we want 
//    to use them, and so that the numNonZero values can be stored and kept
//    with them.  Otherwise, we would need to recalculate it every time.
//
//    masterMwbList_:
//
//    Additionally, we keep a running list of pointers to every MorphWorkbench
//    ever created (set to null when destructor is called).  The only real
//    benefit of this is so that we can query how much device memory we are
//    using at any given time.  See the method, calculateDeviceMemUsage();
//
////////////////////////////////////////////////////////////////////////////////
class MorphWorkbench
{
private:

   // The locations of device memory that contain all of our stuff
   int* devBuffer1_;
   int* devBuffer2_;
   vector<int*> devExtraBuffers_;

   // We want to keep track of every MWB and structuring element created
   // so we can calculate the total memory usage of all workbenches, which 
   // would include all buffers and SEs
   static vector<MorphWorkbench*> masterMwbList_;
   static vector<StructElt>       masterSEList_;

   // This workbench should know where it is in the master MWB list
   int mwbID_;


   // These two pointers will switch after every operation
   int** devBufferPtrA_;
   int** devBufferPtrB_;

   // Keep pointers to the host memory, so we know where to get input
   // and where to put the result
   int* hostImageIn_;
   bool imageCopied_;
   
   // All buffers in a workbench are the same size:  the size of the image
   unsigned int imageCols_;
   unsigned int imageRows_;
   unsigned int imagePixels_;
   unsigned int imageBytes_;

   // All kernel functions will be called with the same geometry
   dim3  GRID_;
   dim3  BLOCK_;

   // We need temp buffers for operations like thinning, pruning
   void createExtraBuffer(void);
   void deleteExtraBuffer(void);
   int* getExtraBufferPtr(int bufIdx);

   // This gets called after every operation to switch Input/Output buffers ptrs
   void flipBuffers(void);

public:

   dim3 getGridSize(void)  const {return GRID_;}
   dim3 getBlockSize(void) const {return BLOCK_;}
   void setBlockSize(dim3 newSize);

   // Calculate the device mem used by all MWBs and SEs
   static int calculateDeviceMemUsage(bool printToStdout=true);
   
   // Forking is the really just the same as copying
   // TODO:  not implemented yet
   void forkWorkbench(MorphWorkbench & mwb) const;

   static int addStructElt(int* hostSE, int ncols, int nrows);

   // Default Constructor
   MorphWorkbench();

   // Constructor
   MorphWorkbench(int* imageStart, int cols, int rows, bool COPY=false);

   // Copy host data to device, and prepare kernel parameters
   void Initialize(int* imageStart, int cols, int rows, bool COPY=false);

   // Destructor
   ~MorphWorkbench();

   // Copy the current state of the buffer to the host
   void fetchResult(int* hostTarget) const;
   
   // The basic morphological operations (CPU wrappers for GPU kernels)
   // NOTE: all batch functions, such as open, close, thinsweep, etc
   // are written so that when the user calls them, buffers A and B are 
   // distinctly before-and-after versions of the operation.  The
   // alternative is that A and B only contain the states before and
   // after the last SUB-operation, and then the user has no clean
   // way to determine if the image changed
   void GenericMorphOp(int seIndex, int targSum);
   void HitOrMiss(int seIndex); 
   void Erode(int seIndex);
   void Dilate(int seIndex);
   void Median(int seIndex);
   void Open(int seIndex);
   void Close(int seIndex);
   void FindAndRemove(int seIndex);

   // CPU wrappers for the mask op kernel functions which we need frequently
   void Union(int* mask2);
   void Intersect(int* mask2);
   void Subtract(int* mask2);
   void Invert(void);
   //int  NumPixelsChanged(void);
   //int  SumMask(void);

   void CopyBuffer(int* dst);
   static void CopyBuffer(int* src, int* dst, int bytes);

   /////////////////////////////////////////////////////////////////////////////
   // Thinning is a sequence of 8 hit-or-miss operations which each find
   // pixels contributing to the blob width, and then removes them from
   // the original image.  Very similar to skeletonization
   void ThinningSweep(void);

   /////////////////////////////////////////////////////////////////////////////
   // Pruning uses a sequence of 8 hit-or-miss operations to remove "loose ends"
   // from a thinned/skeletonized image.  
   void PruningSweep(void);



   // The macro calls below create wrappers for the optimized 3x3 kernel fns
   //
   //    void NAME(void)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(&debBufA, &devBufB, ...);
   //       flipBuffers();
   //    }
   //    void ZNAME(int* src, int* dst)
   //    {
   //       Morph3x3_NAME_Kernel<<GRID,BLOCK>>>(src, dst, ...);
   //    }
   //
   CREATE_MWB_3X3_FUNCTION( Dilate );
   CREATE_MWB_3X3_FUNCTION( DilateCross );
   CREATE_MWB_3X3_FUNCTION( Erode );
   CREATE_MWB_3X3_FUNCTION( ErodeCross );
   CREATE_MWB_3X3_FUNCTION( Median );
   CREATE_MWB_3X3_FUNCTION( MedianCross );
   CREATE_MWB_3X3_FUNCTION( Thin1 );
   CREATE_MWB_3X3_FUNCTION( Thin2 );
   CREATE_MWB_3X3_FUNCTION( Thin3 );
   CREATE_MWB_3X3_FUNCTION( Thin4 );
   CREATE_MWB_3X3_FUNCTION( Thin5 );
   CREATE_MWB_3X3_FUNCTION( Thin6 );
   CREATE_MWB_3X3_FUNCTION( Thin7 );
   CREATE_MWB_3X3_FUNCTION( Thin8 );
   CREATE_MWB_3X3_FUNCTION( Prune1 );
   CREATE_MWB_3X3_FUNCTION( Prune2 );
   CREATE_MWB_3X3_FUNCTION( Prune3 );
   CREATE_MWB_3X3_FUNCTION( Prune4 );
   CREATE_MWB_3X3_FUNCTION( Prune5 );
   CREATE_MWB_3X3_FUNCTION( Prune6 );
   CREATE_MWB_3X3_FUNCTION( Prune7 );
   CREATE_MWB_3X3_FUNCTION( Prune8 );

private:
   // These operations are the same as above, but with custom src-dst
   // and they don't flip the buffers.  These are "unsafe" for the
   // user to use, since he can destroy the current buffer, but the
   // developer can use them in MWB to ensure that batch operations
   // leave buffers A and B in a compare-able state
   void ZGenericMorphOp(int seIndex, int targSum, int* src, int* dst);
   void ZHitOrMiss(int seIndex, int* src, int* dst);
   void ZErode(int seIndex, int* src, int* dst);
   void ZDilate(int seIndex, int* src, int* dst);
   void ZMedian(int seIndex, int* src, int* dst);
   void ZOpen(int seIndex, int* src, int* dst, int useTempBuf=0);
   void ZClose(int seIndex, int* src, int* dst, int useTempBuf=0);
   void ZFindAndRemove(int seIndex, int* src, int* dst, int useTempBuf=0);

   // CPU wrappers for the mask op kernel functions which we need frequently
   void ZUnion(int* mask2, int* src, int* dst);
   void ZIntersect(int* mask2, int* src, int* dst);
   void ZSubtract(int* mask2, int* src, int* dst);
   void ZInvert(int* src, int* dst);

};

#endif
