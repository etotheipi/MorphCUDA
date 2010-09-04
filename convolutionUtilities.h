#ifndef _CONVOLUTION_UTILITIES_
#define _CONVOLUTION_UTILITIES_

#include <iostream>
using namespace std;

// Assume target memory has already been allocated, nPixels is odd
void createGaussian1D(float* targPtr, 
                      int    nPixels, 
                      float  sigma, 
                      float  ctr)
{
   if(nPixels%2 != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtr = (float)(nPixels/2 + ctr);   
   float sigmaSq = sigma*sigma;
   float denom = sqrt(2*M_PI*sigmaSq);
   float dist;
   for(int i=0; i<nPixels; i++)
   {
      dist = (float)i - pxCtr;
      targPtr[i] = exp(-0.5 * dist * dist / sigmaSq) / denom;
   }
}

// Assume target memory has already been allocate, nPixels is odd
// Use col-row (D00_UL_ES)
void createGaussian2D(float* targPtr, 
                      int    nPixelsCol,
                      int    nPixelsRow,
                      float  sigmaCol,
                      float  sigmaRow,
                      float  ctrCol,
                      float  ctrRow)
{
   if(nPixelsCol%2 != 1 || nPixelsRow != 1)
   {
      cout << "***Warning: createGaussian(...) only defined for odd pixel"  << endl;
      cout << "            dimensions.  Undefined behavior for even sizes." << endl;
   }

   float pxCtrCol = (float)(nPixelsCol/2 + ctrCol);   
   float pxCtrRow = (float)(nPixelsRow/2 + ctrRow);   
   float distCol, distRow, distColSqNorm, distRowSqNorm;
   float denom = 2*M_PI*sigmaCol*sigmaRow;
   for(int c=0; c<nPixelsCol; c++)
   {
      distCol = (float)c - pxCtrCol;
      distColSqNorm = distCol*distCol / (sigmaCol*sigmaCol);
      for(int r=0; r<nPixelsRow; r++)
      {
         distRow = (float)r - pxCtrRow;
         distRowSqNorm = distRow*distRow / (sigmaRow*sigmaRow);
         
         targPtr[c*nPixelsRow+r] = exp(-0.5*(distColSqNorm + distRowSqNorm)) / denom;
      }
   }
}


// Assume diameter^2 target memory has already been allocated
// This filter is used for edge detection.  Convolve with the
// kernel created by this function, and then look for the 
// zero-crossings
// As always, we expect an odd diameter
// For LoG kernels, we always assume square and symmetric,
// which is why there are no options for different dimensions
void createLaplacianOfGaussianKernel(float* targPtr,
                                     int    diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float dc, dr, dcSq, drSq;
   float sigma = diameter/10.0f;
   float sigmaSq = sigma*sigma;
   for(int c=0; c<diameter; c++)
   {
      dc = (float)c - pxCtr;
      dcSq = dc*dc;
      for(int r=0; r<diameter; r++)
      {
         dr = (float)r - pxCtr;
         drSq = dr*dr;
   
         float firstTerm  = (dcSq + drSq - 2*sigmaSq) / (sigmaSq * sigmaSq);
         float secondTerm = exp(-0.5 * (dcSq + drSq) / sigmaSq);
         targPtr[c*diameter+r] = firstTerm * secondTerm;
      }
   }
}

// Assume diameter^2 target memory has already been allocated
int createBinaryCircle(float* targPtr,
                       int    diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   int seNonZero = 0;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
         {
            targPtr[c*diameter+r] = 1.0f;
            seNonZero++;
         }
         else
         {
            targPtr[c*diameter+r] = 0.0f;
         }
      }
   }
   return seNonZero;
}

// Assume diameter^2 target memory has already been allocated
int createBinaryCircle(int*   targPtr,
                       int    diameter)
{
   float pxCtr = (float)(diameter-1) / 2.0f;
   float rad;
   int seNonZero = 0;
   for(int c=0; c<diameter; c++)
   {
      for(int r=0; r<diameter; r++)
      {
         rad = sqrt((c-pxCtr)*(c-pxCtr) + (r-pxCtr)*(r-pxCtr));
         if(rad <= pxCtr+0.5)
         {
            targPtr[c*diameter+r] = 1.0f;
            seNonZero++;
         }
         else
         {
            targPtr[c*diameter+r] = 0.0f;
         }
      }
   }
   return seNonZero;
}


#endif
