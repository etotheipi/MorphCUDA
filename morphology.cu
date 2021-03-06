/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>
#include <stopwatch.h>
#include <cmath>
#include "convolutionUtilities.h"
#include "gpuMorphology.cu"

using namespace std;

unsigned int timer;

////////////////////////////////////////////////////////////////////////////////
// Simple Timing Calls
void startTimer(void)
{
   // GPU Timer Functions
   timer = 0;
   cutilCheckError( cutCreateTimer( &timer));
   cutilCheckError( cutStartTimer( timer));
}

////////////////////////////////////////////////////////////////////////////////
// Stopping also resets the timer
float stopTimer(void)
{
   cutilCheckError( cutStopTimer( timer));
   float gpuTime = cutGetTimerValue(timer);
   cutilCheckError( cutDeleteTimer( timer));
   return gpuTime;
}


////////////////////////////////////////////////////////////////////////////////
// Read/Write images from/to files
void ReadFile(string fn, int* targPtr, int nCols, int nRows)
{
   ifstream in(fn.c_str(), ios::in);
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
      for(int c=0; c<nCols; c++)
         in >> targPtr[c*nCols+r];
   in.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing file in space-separated format
void WriteFile(string fn, int* srcPtr, int nCols, int nRows)
{
   ofstream out(fn.c_str(), ios::out);
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
   {
      for(int c=0; c<nCols; c++)
      {
         out << srcPtr[c*nRows+r] << " ";
      }
      out << endl;
   }
   out.close();
}

////////////////////////////////////////////////////////////////////////////////
// Writing image to stdout
void PrintArray(int* srcPtr, int nCols, int nRows)
{
   // We work with col-row format, but files written in row-col, so switch loop
   for(int r=0; r<nRows; r++)
   {
      for(int c=0; c<nCols; c++)
      {
         cout << srcPtr[c*nRows+r] << " ";
      }
      cout << endl;
   }
}




////////////////////////////////////////////////////////////////////////////////
// Copy a 3D texture from a host (float*) array to a device cudaArray
// The extent should be specified with all dimensions in units of *elements*
void prepareCudaTexture(float* h_src, 
                        cudaArray *d_dst,
                        cudaExtent const texExtent)
{
   cudaMemcpy3DParms copyParams = {0};
   cudaPitchedPtr cppImgPsf = make_cudaPitchedPtr( (void*)h_src, 
                                                   texExtent.width*FLOAT_SZ,
                                                   texExtent.width,  
                                                   texExtent.height);
   copyParams.srcPtr   = cppImgPsf;
   copyParams.dstArray = d_dst;
   copyParams.extent   = texExtent;
   copyParams.kind     = cudaMemcpyHostToDevice;
   cutilSafeCall( cudaMemcpy3D(&copyParams) );
}
////////////////////////////////////////////////////////////////////////////////


void runMorphologyUnitTests();

////////////////////////////////////////////////////////////////////////////////
//
// Program main
//
// TODO:  Remove the CUTIL calls so libcutil is not required to compile/run
//
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

   cout << endl << "Executing GPU-accelerated convolution..." << endl;

   /////////////////////////////////////////////////////////////////////////////
   // Query the devices on the system and select the fastest
   int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
   {
		cout << "cudaGetDeviceCount() FAILED." << endl;
      cout << "CUDA Driver and Runtime version may be mismatched.\n";
      return -1;
	}

   // Check to make sure we have at least on CUDA-capable device
   if( deviceCount == 0)
   {
      cout << "No CUDA devices available.  Exiting." << endl;
      return -1;
	}

   // Fastest device automatically selected.  Can override below
   int fastestDeviceID = cutGetMaxGflopsDeviceId() ;
   //fastestDeviceID = 0;
   cudaSetDevice(fastestDeviceID);

   cudaDeviceProp gpuProp;
   cout << "CUDA-enabled devices on this system:  " << deviceCount <<  endl;
   for(int dev=0; dev<deviceCount; dev++)
   {
      cudaGetDeviceProperties(&gpuProp, dev); 
      char* devName = gpuProp.name;
      int mjr = gpuProp.major;
      int mnr = gpuProp.minor;
      int memMB = gpuProp.totalGlobalMem / (1024*1024);
      if( dev==fastestDeviceID )
         cout << "\t* ";
      else
         cout << "\t  ";

      printf("(%d) %20s (%d MB): \tCUDA Capability %d.%d \n", dev, devName, memMB, mjr, mnr);
   }

   /////////////////////////////////////////////////////////////////////////////
   runMorphologyUnitTests();

   /////////////////////////////////////////////////////////////////////////////
   cudaThreadExit();

   //cutilExit(argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
void runMorphologyUnitTests()
{
   /////////////////////////////////////////////////////////////////////////////
   // Allocate host memory and read in the test image from file
   /////////////////////////////////////////////////////////////////////////////
   unsigned int imgW  = 256;
   unsigned int imgH  = 256;
   unsigned int nPix  = imgH*imgW;
   unsigned int imgBytes = nPix*INT_SZ;
   int* imgIn  = (int*)malloc(imgBytes);
   int* imgOut = (int*)malloc(imgBytes);
   string fn("salt256.txt");

   cout << endl;
   printf("\nTesting morphology operations on %dx%d mask.\n", imgW,imgH);
   cout << "Reading mask from " << fn.c_str() << endl;
   ReadFile(fn, imgIn, imgW, imgH);



   /////////////////////////////////////////////////////////////////////////////
   // Create a bunch of structuring elements to test
   //
   // All the important 3x3 SEs are "hardcoded" into dedicated functions 
   // See all the CREATE_3X3_MORPH_KERNEL/CREATE_MWB_3X3_FUNCTION calls in .cu
   /////////////////////////////////////////////////////////////////////////////
   // A very unique 17x17 to test coord systems
   int  se17W = 17;
   int  se17H = 17;
   int  se17Pixels = se17W*se17H;
   int  se17Bytes = se17Pixels * INT_SZ;
   int* se17 = (int*)malloc(se17Bytes);
   ReadFile("asymmPSF_17x17.txt",   se17,  se17W,  se17H);

   // Test a rectangular SE
   int  seRectW = 9;
   int  seRectH = 5;
   int  seRectPixels = seRectW*seRectH;
   int  seRectBytes  = seRectPixels * INT_SZ;
   int* seRect = (int*)malloc(seRectBytes);
   for(int i=0; i<seRectPixels; i++)
      seRect[i] = 1;

   // Also test a circle
   int  seCircD = 11; // D~Diameter
   int  seCircPixels = seCircD*seCircD;
   int  seCircBytes = seCircPixels*INT_SZ;
   int* seCirc = (int*)malloc(seCircBytes);
   int  seCircNZ = createBinaryCircle(seCirc, seCircD); // return #non-zero

   // Add the structuring elements to the master SE list, which copies 
   // them into device memory.  Note that you need separate SEs for
   // erosion and dilation, even if they are the same img-data (target
   // sum is different)
   int seIdxUnique17x17  = MorphWorkbench::addStructElt(se17,   se17W,   se17H  );
   int seIdxRect9x5      = MorphWorkbench::addStructElt(seRect, seRectW, seRectH);
   int seIdxCircle11x11  = MorphWorkbench::addStructElt(seCirc, seCircD, seCircD);
   

   /////////////////////////////////////////////////////////////////////////////
   // Let's start testing MorphWorkbench
   /////////////////////////////////////////////////////////////////////////////

   // Create the workbench, which copies the image into device memory
   MorphWorkbench theMWB(imgIn, imgW, imgH);

   dim3 bsize = theMWB.getBlockSize();
   dim3 gsize = theMWB.getGridSize();
   printf("Using the following kernel geometry for morphology operations:\n");
   printf("\tBlock Size = (%d, %d, %d) threads\n", bsize.x, bsize.y, bsize.z);
   printf("\tGrid Size  = (%d, %d, %d) blocks\n ", gsize.x, gsize.y, gsize.z);

   // Start by simply fetching the unmodified image (sanity check)
   theMWB.fetchResult(imgOut);
   WriteFile("Image1_Orig.txt", imgOut, imgW, imgH);
   
   // Dilate by the 17x17
   theMWB.Dilate(seIdxUnique17x17);
   theMWB.fetchResult(imgOut);
   WriteFile("Image2_Dilate17.txt", imgOut, imgW, imgH);

   // We Erode the image now, which means it's actually been "closed"
   theMWB.Erode(seIdxUnique17x17);
   theMWB.fetchResult(imgOut);
   WriteFile("Image3_Close17.txt", imgOut, imgW, imgH);

   // Dilate with rectangle
   theMWB.Dilate(seIdxRect9x5);
   theMWB.fetchResult(imgOut);
   WriteFile("Image4_DilateRect.txt", imgOut, imgW, imgH);

   // Try a thinning sweep on the dilated image (8 findandremove ops) 
   theMWB.ThinningSweep();
   theMWB.fetchResult(imgOut);
   WriteFile("Image6_ThinSw1.txt", imgOut, imgW, imgH);

   // Again...
   theMWB.ThinningSweep();
   theMWB.fetchResult(imgOut);
   WriteFile("Image7_ThinSw2.txt", imgOut, imgW, imgH);

   // And again...
   theMWB.ThinningSweep();
   theMWB.fetchResult(imgOut);
   WriteFile("Image8_ThinSw3.txt", imgOut, imgW, imgH);

   // Check to see how much device memory we're using right now
   MorphWorkbench::calculateDeviceMemUsage(true);  // printToStdOut==true

   free(imgIn);
   free(imgOut);
   free(se17);
   free(seRect);
   free(seCirc);
   /////////////////////////////////////////////////////////////////////////////
}
