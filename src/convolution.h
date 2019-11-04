///////////////////////////////////////////////////////////////////////////////
// convolution.h
// =============
// convolution 1D and 2D
//
//  AUTHOR: Song Ho Ahn
// CREATED: 2005-07-18
// UPDATED: 2018-06-28
//
// Copyright (c) 2005 Song Ho Ahn
///////////////////////////////////////////////////////////////////////////////

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// 1D convolution /////////////////////////////////////////////////////////////
// We assume input and kernel signal start from t=0. (The first element of
// kernel and input signal is at t=0)
// it returns false if parameters are not valid.
bool convolve1D(float* in, float* out, int size, float* kernel, int kernelSize);


// 2D convolution (No Optimization) ///////////////////////////////////////////
// Simplest 2D convolution routine. It is easy to understand how convolution
// works, but is very slow, because of no optimization.
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSlow(unsigned char* in, unsigned char* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2DSlow(float* in, float* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);



// 2D convolution /////////////////////////////////////////////////////////////
// 2D data are usually stored as contiguous 1D array in computer memory.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
// It returns false if parameters are not valid.
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(unsigned char* in, unsigned char* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2D(unsigned short* in, unsigned short* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2D(int* in, int* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2D(float* in, float* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2D(double* in, double* out, int sizeX, int sizeY, double* kernel, int kSizeX, int kSizeY);



// 2D separable convolution ///////////////////////////////////////////////////
// If the MxN kernel can be separable to (Mx1) and (1xN) matrices, the
// multiplication can be reduced to M+N comapred to MxN in normal convolution.
// It does not check the output is excceded max for performance reason. And we
// assume the kernel contains good(valid) data, therefore, the result cannot be
// larger than max.
// It returns false if parameters are not valid.
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(unsigned char* in, unsigned char* out, int sizeX, int sizeY, float* xKernel, int kSizeX, float* yKernel, int kSizeY);
bool convolve2DSeparable(unsigned short* in, unsigned short* out, int sizeX, int sizeY, float* xKernel, int kSizeX, float* yKernel, int kSizeY);
bool convolve2DSeparable(int* in, int* out, int sizeX, int sizeY, float* xKernel, int kSizeX, float* yKernel, int kSizeY);
bool convolve2DSeparable(float* in, float* out, int sizeX, int sizeY, float* xKernel, int kSizeX, float* yKernel, int kSizeY);
bool convolve2DSeparable(double* in, double* out, int sizeX, int sizeY, double* xKernel, int kSizeX, double* yKernel, int kSizeY);

bool convolve2DSeparableReadable(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernelX, int kSizeX, float* kernelY, int kSizeY);

// 2D convolution Fast ////////////////////////////////////////////////////////
// In order to improve the performance, this function uses multple cursors of
// input signal. It avoids indexing input array during convolution. And, the
// input signal is partitioned to 9 different sections, so we don't need to
// check the boundary for every samples.
///////////////////////////////////////////////////////////////////////////////
bool convolve2DFast(unsigned char* in, unsigned char* out, int sizeX, int sizeY, float* kernel, int kSizeX, int kSizeY);
bool convolve2DFast2(unsigned char* in, unsigned char* out, int sizeX, int sizeY, int* kernel, float factor, int kSizeX, int kSizeY);



#endif
