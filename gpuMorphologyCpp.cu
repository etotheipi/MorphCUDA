#include "gpuMorphology.cu"



////////////////////////////////////////////////////////////////////////////////
// Standard 3x3 erosions and dilations and median filtering
CREATE_3X3_MORPH_KERNEL( Dilate, -8,  
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( Erode, 9,
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( DilateCross, -4,  
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);

CREATE_3X3_MORPH_KERNEL( ErodeCross, 5,
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);

CREATE_3X3_MORPH_KERNEL( Median, 0,
                                             1,  1,  1,
                                             1,  1,  1,
                                             1,  1,  1);

CREATE_3X3_MORPH_KERNEL( MedianCross, 0,
                                             0,  1,  0,
                                             1,  1,  1,
                                             0,  1,  0);

////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for THINNING
CREATE_3X3_MORPH_KERNEL( Thin1, 7,
                                             1,  1,  1,
                                             0,  1,  0,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Thin2, 7,
                                            -1,  0,  1,
                                            -1,  1,  1,
                                            -1,  0,  1);
CREATE_3X3_MORPH_KERNEL( Thin3, 7,
                                            -1, -1, -1,
                                             0,  1,  0,
                                             1,  1,  1);
CREATE_3X3_MORPH_KERNEL( Thin4, 7,
                                             1,  0, -1,
                                             1,  1, -1,
                                             1,  0, -1);

CREATE_3X3_MORPH_KERNEL( Thin5, 6,
                                             0, -1, -1,
                                             1,  1, -1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Thin6, 6,
                                             0,  1,  0,
                                             1,  1, -1,
                                             0, -1, -1);
CREATE_3X3_MORPH_KERNEL( Thin7, 6,
                                             0,  1,  0,
                                            -1,  1,  1,
                                            -1, -1,  0);
CREATE_3X3_MORPH_KERNEL( Thin8, 6,
                                            -1, -1,  0,
                                            -1,  1,  1,
                                             0,  1,  0);
        
////////////////////////////////////////////////////////////////////////////////
// There are 8 standard structuring elements for PRUNING
CREATE_3X3_MORPH_KERNEL( Prune1, 7,
                                             0,  1,  0,
                                            -1,  1, -1,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Prune2, 7,
                                            -1, -1,  0,
                                            -1,  1,  1,
                                            -1, -1,  0);
CREATE_3X3_MORPH_KERNEL( Prune3, 7,
                                            -1, -1, -1,
                                            -1,  1, -1,
                                             0,  1,  0);
CREATE_3X3_MORPH_KERNEL( Prune4, 7,
                                             0, -1, -1,
                                             1,  1, -1,
                                             0, -1, -1);

CREATE_3X3_MORPH_KERNEL( Prune5, 9,
                                            -1, -1, -1,
                                            -1,  1, -1,
                                             1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Prune6, 9,
                                            -1, -1, -1,
                                            -1,  1, -1,
                                            -1, -1,  1);
CREATE_3X3_MORPH_KERNEL( Prune7, 9,
                                            -1, -1,  1,
                                            -1,  1, -1,
                                            -1, -1, -1);
CREATE_3X3_MORPH_KERNEL( Prune8, 9,
                                             1, -1, -1,
                                            -1,  1, -1,
                                            -1, -1, -1);


////////////////////////////////////////////////////////////////////////////////
// Dilation and Erosion are just simple cases of Hit-or-Miss
// We expect the structuring element to consist only of 1s and 0s, no -1s
////////////////////////////////////////////////////////////////////////////////



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

////////////////////////////////////////////////////////////////////////////////
__global__ void  MaskUnion_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;

   if( A[idx] + B[idx] > 0)
      devOut[idx] = 1;
   else
      devOut[idx] = 0;
}

////////////////////////////////////////////////////////////////////////////////
__global__ void  MaskIntersect_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   devOut[idx] = A[idx] * B[idx];
}

////////////////////////////////////////////////////////////////////////////////
__global__ void  MaskSubtract_Kernel( int* A, int* B, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   if( B[idx] == 0)
      devOut[idx] = 0;
   else 
      devOut[idx] = A[idx];
}

////////////////////////////////////////////////////////////////////////////////
__global__ void  MaskInvert_Kernel( int* A, int* devOut)
{  
   const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   devOut[idx] = 1 - A[idx];
}

////////////////////////////////////////////////////////////////////////////////
// REMOVED:  this is what cudaMemcpy(..., cudaMemcpyDeviceToDevice) is for
//__global__ void  MaskCopy_Kernel( int* A, int* devOut)
//{  
   //const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   //devOut[idx] = A[idx];
//}

////////////////////////////////////////////////////////////////////////////////
// TODO: This is a very dumb/slow equal operator, actually won't even work
//       Perhaps have the threads atomicAdd to a globalMem location if !=
//__global__ void  MaskCountDiff_Kernel( int* A, int* B, int* globalMemCount)
//{  
   //const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   //if(A[idx] != B[idx])
      //atomicAdd(numNotEqual, 1);
//}


////////////////////////////////////////////////////////////////////////////////
// TODO: Need to use reduction for this, but that can be kind of complicated
//__global__ void  MaskSum_Kernel( int* A, int* globalMemSum)
//{  
   //const int idx = blockDim.x*blockIdx.x + threadIdx.x;
   //if(A[idx] != B[idx])
      //atomicAdd(numNotEqual, 1);
//}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// The heart of the library!
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
               int   seTargSum)
{  

   CREATE_CONVOLUTION_VARIABLES_MORPH(seColRad, seRowRad); 

   const int seStride = seRowRad*2+1;   
   const int sePixels = seStride*(seColRad*2+1);   
   int* shmSE = (int*)&shmOutput[ROUNDUP32(localPixels)];   

   COPY_LIN_ARRAY_TO_SHMEM_MORPH(sePtr, shmSE, sePixels); 

   PREPARE_PADDED_RECTANGLE_MORPH(seColRad, seRowRad); 

   shmOutput[localIdx] = -1;

   __syncthreads();   

   int accumInt = 0;
   for(int coff=-seColRad; coff<=seColRad; coff++)   
   {   
      for(int roff=-seRowRad; roff<=seRowRad; roff++)   
      {   
         int seCol = seColRad + coff;   
         int seRow = seRowRad + roff;   
         int seIdx = IDX_1D(seCol, seRow, seStride);   
         int seVal = shmSE[seIdx];   

         int shmPRCol = padRectCol + coff;   
         int shmPRRow = padRectRow + roff;   
         int shmPRIdx = IDX_1D(shmPRCol, shmPRRow, padRectStride);   
         accumInt += seVal * shmPadRect[shmPRIdx];
      }   
   }   
   // If every pixel was identical as expected, accumInt==seTargSum
   if(accumInt >= seTargSum)
      shmOutput[localIdx] = 1;

   __syncthreads();   

   devOutPtr[globalIdx] = (shmOutput[localIdx] + 1) / 2.0f;
}

vector<MorphWorkbench*> MorphWorkbench::masterMwbList_ = 
                                    vector<MorphWorkbench*>(0);
vector<StructElt>       MorphWorkbench::masterSEList_  = 
                                    vector<StructElt>(0);

////////////////////////////////////////////////////////////////////////////////
// 
// MorphWorkbench  (MWB)
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

////////////////////////////////////////////////////////////////////////////////
//  
//  AddStructElt takes only the host memory location and size information.
//  It creates a StructElt object, allocates the device memory, and returns
//  and its index in the master SE list
//
////////////////////////////////////////////////////////////////////////////////

int MorphWorkbench::addStructElt(int* hostSE, 
                                 int  seCols, 
                                 int  seRows)
{
   int newIndex = (int)masterSEList_.size();
   masterSEList_.push_back( StructElt() );
   masterSEList_[newIndex].init(hostSE, seCols, seRows);
   return newIndex;
}


/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::setBlockSize(dim3 newSize)
{
   BLOCK_ = newSize;
   GRID_ = dim3(imageCols_/BLOCK_.x, imageRows_/BLOCK_.y, 1);
}


/////////////////////////////////////////////////////////////////////////////
MorphWorkbench::MorphWorkbench() : 
   devBuffer1_(NULL),
   devBuffer2_(NULL),
   devBufferPtrA_(NULL),
   devBufferPtrB_(NULL),
   hostImageIn_(NULL),
   mwbID_(-1)
{ 
   // No code needed here
}



/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Initialize(int* imageStart, 
                                int  ncols, 
                                int  nrows,    
                                bool COPY)
{
   imageCols_     = ncols;
   imageRows_     = nrows;
   imagePixels_   = ncols*nrows;
   imageBytes_    = ncols*nrows*INT_SZ;

   // 8x32 dramatically reduces bank conflicts, compared to 16x16
   int bx = 8;
   int by = 32;
   int gx = ncols/bx;
   int gy = nrows/by;
   BLOCK_ = dim3(bx, by, 1);
   GRID_  = dim3(gx, gy, 1);

   devExtraBuffers_ = vector<int*>(0);

   // The COPY flag determines whether we pass-through the pointer
   // or do a memory copy
   if( !COPY )
   {
      imageCopied_ = false;
      hostImageIn_ = imageStart;
   }
   else
   {
      imageCopied_ = true;
      hostImageIn_ = (int*)malloc(imageBytes_);
      memcpy(hostImageIn_, imageStart, imageBytes_);
   }

   cudaMalloc((void**)&devBuffer1_, imageBytes_);
   cudaMalloc((void**)&devBuffer2_, imageBytes_);

   // Copy the starting image to the device
   cudaMemcpy(devBuffer1_,  
              hostImageIn_,   
              imageBytes_,
              cudaMemcpyHostToDevice);

   // BufferA is input for a morph op, BufferB is the target, then switch
   devBufferPtrA_ = &devBuffer1_;
   devBufferPtrB_ = &devBuffer2_;

   // Initialize static lists if this is the first constructed MWB
   static bool firstMwbInit = true;
   if(firstMwbInit)
   {
      masterMwbList_ = vector<MorphWorkbench*>(0);
      masterSEList_  = vector<StructElt>(0);
      firstMwbInit = false;
   }
   
   // Assign an ID and store pointer, so we can later calculate memory usage
   mwbID_ = (int)masterMwbList_.size(); 
   masterMwbList_.push_back(this);
}


/////////////////////////////////////////////////////////////////////////////
MorphWorkbench::MorphWorkbench(int* imageStart, 
                               int  ncols, 
                               int  nrows,    
                               bool COPY)
{
   Initialize(imageStart, ncols, nrows, COPY);
}


/////////////////////////////////////////////////////////////////////////////
MorphWorkbench::~MorphWorkbench()
{
   if(imageCopied_)
      free(hostImageIn_);

   cudaFree(devBuffer1_);
   cudaFree(devBuffer2_);

   // Probably not necessary since this is the destructor, but it's habit
   hostImageIn_  = NULL;
   devBuffer1_   = NULL;
   devBuffer2_   = NULL;

   for(int i=0; i<(int)devExtraBuffers_.size(); i++)
   {
      cudaFree(devExtraBuffers_[i]);
      devExtraBuffers_[i] = NULL;
   }
      
   if(mwbID_ != -1)
      masterMwbList_[mwbID_] = NULL;
}

/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::createExtraBuffer(void)
{
   int* newBuf;
   devExtraBuffers_.push_back(newBuf);
   cudaMalloc((void**)&newBuf, imageBytes_);
}

/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::deleteExtraBuffer(void)
{
   int nBuf = (int)devExtraBuffers_.size();
   int* topPtr = devExtraBuffers_[nBuf-1];
   cudaFree(topPtr);
   devExtraBuffers_.pop_back();
}

/////////////////////////////////////////////////////////////////////////////
int* MorphWorkbench::getExtraBufferPtr(int bufIdx)
{
   int numBufAvail = (int)devExtraBuffers_.size();

   if(numBufAvail < bufIdx+1)
      for(int i=0; i<(bufIdx+1-numBufAvail); i++)
         createExtraBuffer();
   
   return devExtraBuffers_[bufIdx];
}

/////////////////////////////////////////////////////////////////////////////
// Copy the current state of the buffer to the host
void MorphWorkbench::fetchResult(int* hostTarget) const
{
   cudaThreadSynchronize();
   cudaMemcpy(hostTarget,
              *devBufferPtrA_,
              imageBytes_, 
              cudaMemcpyDeviceToHost);
}



/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::flipBuffers(void)
{
   static bool A_is_first = true;
   if(A_is_first)
   {
      devBufferPtrA_ = &devBuffer1_;
      devBufferPtrB_ = &devBuffer2_;
      A_is_first = false;
   }
   else
   {
      devBufferPtrA_ = &devBuffer2_;
      devBufferPtrB_ = &devBuffer1_;
      A_is_first = true;
   }
}


////////////////////////////////////////////////////////////////////////////////
//
// Finally, we get to define all the morphological operators!
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::GenericMorphOp(int seIndex, int targSum)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *devBufferPtrA_,
                  *devBufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  targSum);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::HitOrMiss(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *devBufferPtrA_,
                  *devBufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  se.getNonZero());
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void MorphWorkbench::Erode(int seIndex)
{
   HitOrMiss(seIndex);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Dilate(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *devBufferPtrA_,
                  *devBufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  -se.getNonZero()+1);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Median(int seIndex)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  *devBufferPtrA_,
                  *devBufferPtrB_,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  0);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Open(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZErode(seIndex, *devBufferPtrA_, tbuf);
   ZDilate(seIndex, tbuf, *devBufferPtrB_);
   flipBuffers();
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Close(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZDilate(seIndex, *devBufferPtrA_, tbuf);
   ZErode(seIndex, tbuf, *devBufferPtrB_);
   flipBuffers();
}

void MorphWorkbench::FindAndRemove(int seIndex)
{
   int* tbuf = getExtraBufferPtr(0);
   ZHitOrMiss(seIndex, *devBufferPtrA_, tbuf);
   ZSubtract(tbuf, *devBufferPtrA_, *devBufferPtrB_);
   flipBuffers();
}


////////////////////////////////////////////////////////////////////////////////
//
// These are the basic binary mask operations
//
// These are CPU methods which wrap the GPU kernel functions
//
////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::Union(int* devMask2)
{
   MaskUnion_Kernel<<<GRID_,BLOCK_>>>(
                              *devBufferPtrA_, 
                               devMask2, 
                              *devBufferPtrB_);
   flipBuffers();
}

void MorphWorkbench::Intersect(int* devMask2)
{
   MaskIntersect_Kernel<<<GRID_,BLOCK_>>>(
                              *devBufferPtrA_,
                               devMask2,
                              *devBufferPtrB_);
   flipBuffers();
}

void MorphWorkbench::Subtract(int* devMask2)
{
   MaskSubtract_Kernel<<<GRID_,BLOCK_>>>(
                              *devBufferPtrA_,
                               devMask2,
                              *devBufferPtrB_);
   flipBuffers();
}

void MorphWorkbench::Invert()
{
   MaskInvert_Kernel<<<GRID_,BLOCK_>>>(
                              *devBufferPtrA_, 
                              *devBufferPtrB_);
   flipBuffers();
}

void MorphWorkbench::CopyBuffer(int* dst)
{
   cudaMemcpy(dst,   
              *devBufferPtrA_,  
              imageBytes_, 
              cudaMemcpyDeviceToDevice);
}

// Since this is static, 
void MorphWorkbench::CopyBuffer(int* src, int* dst, int bytes)
{
   cudaMemcpy(dst,   
              src,
              bytes, 
              cudaMemcpyDeviceToDevice);
}


////////////////////////////////////////////////////////////////////////////////
// Z FUNCTIONS (PRIVATE)
////////////////////////////////////////////////////////////////////////////////
// These operations are the same as above, but with custom src-dst
// and they don't flip the buffers.  These are "unsafe" for the
// user to use, since he can lose the current buffer, but the
// developer can use them in MWB to ensure that batch operations
// leave buffers A and B in a compare-able state
// 
// Here's what happens if you use regular methods for batch methods
// (THE WRONG WAY)
//       void ThinningSweep(idx)
//       {
//          Thin1();
//          Thin2();
//          Thin3();
//          Thin4();
//          ...
//          Thin8();
   //    }
//
// The user wants to know whether the mask has reached equilibrium and
// calls NumChanged(), expecting to see 0 if it is at equilibrium.  The 
// problem is that since we've been flipping buffers constantly, the 
// NumChanged() function only gives us the num changed from the Thin8() 
// operation.  In fact, doing it this way, it is impossible for the user
// to check with whether Thin1, Thin2, ..., etc changed anything.
//
// Remember that SRC and DST are both device memory pointers
// which is another reason these are private
////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZGenericMorphOp(int seIndex, int targSum, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  targSum);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZHitOrMiss(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  se.getNonZero());
}

////////////////////////////////////////////////////////////////////////////////
// In out implementation, HitOrMiss is identical to erosion.  Theoretically,
// the difference is that the erode operator is expecting an SE that consists
// only of 1s (ON) and 0s (DONTCARE), while the HitOrMiss operation takes
// SEs that also have -1s (OFF).  However, this implementation allows -1s in 
// any SE, so they are interchangeable.
void MorphWorkbench::ZErode(int seIndex, int* src, int* dst)
{
   //ZHitOrMiss(seIndex, int* src, int* dst);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZDilate(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  -se.getNonZero()+1);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZMedian(int seIndex, int* src, int* dst)
{
   StructElt const & se = masterSEList_[seIndex];

   Morph_Generic_Kernel<<<GRID_,BLOCK_>>>(
                  src,
                  dst,
                  imageCols_,
                  imageRows_,
                  se.getDevPtr(),
                  se.getCols()/2,
                  se.getRows()/2,
                  0);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZOpen(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZErode(seIndex, src, tbuf);
   ZDilate(seIndex, tbuf, dst);
}

////////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::ZClose(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZErode(seIndex, src, tbuf);
   ZDilate(seIndex, tbuf, dst);
}


////////////////////////////////////////////////////////////////////////////////
// Can't remember what this is supposed to be called, but it's the process by 
// which you do a Hit-or-Miss operation which is expected to return a sparse
// mask that fully intersects the original image, and then subtract. 
void MorphWorkbench::ZFindAndRemove(int seIndex, int* src, int* dst, int useTempBuf)
{
   int* tbuf = getExtraBufferPtr(useTempBuf);
   ZHitOrMiss(seIndex, src, tbuf);
   ZSubtract(tbuf, src, dst);
}

//int MorphWorkbench::NumPixelsChanged()
//{
   //MaskCountDiff_Kernel<<<GRID_,BLOCK_>>>(
                               //*devBufferPtrA_, 
                               //*devBufferPtrB_, 
                               //&nChanged);
   // No flip
//}


//int MorphWorkbench::SumMask()
//{
   //MaskSum_Kernel<<<GRID_,BLOCK_>>>(
                               //*devBufferPtrA_, 
                               //*devBufferPtrB_, 
                               //&nChanged);
   // No flip
//}

void MorphWorkbench::ZUnion(int* devMask2, int* src, int* dst)
{
   MaskUnion_Kernel<<<GRID_,BLOCK_>>>(
                               src, 
                               devMask2, 
                               dst);
}

void MorphWorkbench::ZIntersect(int* devMask2, int* src, int* dst)
{
   MaskIntersect_Kernel<<<GRID_,BLOCK_>>>(
                               src,
                               devMask2,
                               dst);
}

void MorphWorkbench::ZSubtract(int* devMask2, int* src, int* dst)
{
   MaskSubtract_Kernel<<<GRID_,BLOCK_>>>(
                               src,
                               devMask2,
                               dst);
}

void MorphWorkbench::ZInvert(int* src, int* dst)
{
   MaskInvert_Kernel<<<GRID_,BLOCK_>>>( src, dst);
}

/////////////////////////////////////////////////////////////////////////////
// With all ZOperations implemented, I can finally implement complex batch
// operations like Thinning
void MorphWorkbench::ThinningSweep(void)
{
   int* tbuf0 = getExtraBufferPtr(0);

   // 1  (A->B)
   ZThin1(*devBufferPtrA_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrA_, *devBufferPtrB_);

   // 2  (B->B)
   ZThin2(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 3  (B->B)
   ZThin3(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 4  (B->B)
   ZThin4(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 5  (B->B)
   ZThin5(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 6  (B->B)
   ZThin6(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 7  (B->B)
   ZThin7(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 8  (B->B)
   ZThin8(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // And we're done
   flipBuffers();
}

/////////////////////////////////////////////////////////////////////////////
void MorphWorkbench::PruningSweep(void)
{
   int* tbuf0 = getExtraBufferPtr(0);

   // 1  (A->B)
   ZPrune1(*devBufferPtrA_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrA_, *devBufferPtrB_);

   // 2  (B->B)
   ZPrune2(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 3  (B->B)
   ZPrune3(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 4  (B->B)
   ZPrune4(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 5  (B->B)
   ZPrune5(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 6  (B->B)
   ZPrune6(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 7  (B->B)
   ZPrune7(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // 8  (B->B)
   ZPrune8(*devBufferPtrB_, tbuf0);
   ZSubtract(tbuf0, *devBufferPtrB_, *devBufferPtrB_);

   // And we're done
   flipBuffers();
}



////////////////////////////////////////////////////////////////////////////////
// We can calculate the memory usage of a workbench by adding up the bytes
// in the primary buffers, the other buffers, and the masterSEList
int MorphWorkbench::calculateDeviceMemUsage(bool printToStdout)
{
   if(printToStdout)
      printf("Counting total device memory used by all workbenches...\n");

   int sizekB = 1024;
   int sizeMB = sizekB*sizekB;
   int sizeHalfkB = sizekB/2;
   int sizeHalfMB = sizeMB/2;

   int totalBytesMwb = 0;
   for(int i=0; i<(int)masterMwbList_.size(); i++)
   {
      MorphWorkbench * & thisMwb = masterMwbList_[i];
      if(thisMwb != NULL)
      {
         int totalBuffers = 2 + (int)(thisMwb->devExtraBuffers_.size());
         int totalBytesThisMwb = totalBuffers * thisMwb->imageBytes_;
         if(printToStdout)
            if(totalBytesThisMwb < sizeMB)
               printf("\tWorkbench %d:    %d kB\n", i, 
                        (totalBytesThisMwb+sizeHalfkB)/sizekB);
            else
               printf("\tWorkbench %d:    %d MB\n", i, 
                        (totalBytesThisMwb+sizeHalfMB)/sizeMB);
         totalBytesMwb += totalBytesThisMwb;
      }
      else
         if(printToStdout)
            printf("\tWorkbench %d:   <self-destructed>\n", i);
   }

   // Now we add up the structuring elements, which we don't usually expect
   // to be a lot, but surprises can happen
   int totalBytesSE = 0;
   for(int i=0; i<(int)masterSEList_.size(); i++)
      totalBytesSE += masterSEList_[i].getBytes();  

   if(printToStdout)
      if(totalBytesSE < sizekB)
         printf("\tStructuring Elts:   %d bytes\n", totalBytesSE);
      else if(totalBytesSE < sizeMB)
         printf("\tStructuring Elts:   %d kB\n", (totalBytesSE+sizeHalfkB)/sizekB);
      else
         printf("\tStructuring Elts:   %d MB\n", (totalBytesSE+sizeHalfMB)/sizeMB);

   if(printToStdout)
   {
      printf("-----------------------------------------");
      printf("\tTotal Device Mem:   %d kB\n\n", (totalBytesSE+totalBytesMwb+512)/1024);
   }

   return totalBytesMwb + totalBytesSE;
}




