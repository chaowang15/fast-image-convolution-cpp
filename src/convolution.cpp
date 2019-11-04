///////////////////////////////////////////////////////////////////////////////
// convolution.cpp
// ===============
// convolution 1D and 2D
//
//  AUTHOR: Song Ho Ahn
// CREATED: 2005-07-18
// UPDATED: 2018-06-28
//
// Copyright (c) 2005 Song Ho Ahn
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include "convolution.h"
#include <stdio.h>

///////////////////////////////////////////////////////////////////////////////
// 1D convolution
// We assume input and kernel signal start from t=0.
///////////////////////////////////////////////////////////////////////////////
bool convolve1D(float* in, float* out, int dataSize, float* kernel, int kernelSize)
{
    int i, j, k;

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSize <= 0 || kernelSize <= 0)
        return false;

    // start convolution from out[kernelSize-1] to out[dataSize-1] (last)
    for (i = kernelSize - 1; i < dataSize; ++i)
    {
        out[i] = 0;  // init to 0 before accumulate

        for (j = i, k = 0; k < kernelSize; --j, ++k)
            out[i] += in[j] * kernel[k];
    }

    // convolution from out[0] to out[kernelSize-2]
    for (i = 0; i < kernelSize - 1; ++i)
    {
        out[i] = 0;  // init to 0 before sum

        for (j = i, k = 0; j >= 0; --j, ++k)
            out[i] += in[j] * kernel[k];
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Simplest 2D convolution routine. It is easy to understand how convolution
// works, but is very slow, because of no optimization.
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSlow(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX,
    int kernelSizeY)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;  // center index of kernel
    float sum;               // temp accumulation buffer
    int rowIndex, colIndex;

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    for (i = 0; i < dataSizeY; ++i)  // rows
    {
        for (j = 0; j < dataSizeX; ++j)  // columns
        {
            sum = 0;                           // init to 0 before sum
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                mm = kernelSizeY - 1 - m;  // row index of flipped kernel

                for (n = 0; n < kernelSizeX; ++n)  // kernel columns
                {
                    nn = kernelSizeX - 1 - n;  // column index of flipped kernel

                    // index of input signal, used for checking boundary
                    rowIndex = i + (kCenterY - mm);
                    colIndex = j + (kCenterX - nn);

                    // ignore input samples which are out of bound
                    if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                        sum += in[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
                }
            }
            out[dataSizeX * i + j] = (unsigned char)((float)fabs(sum) + 0.5f);
        }
    }

    return true;
}

bool convolve2DSlow(
    float* in, float* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;  // center index of kernel
    float sum;               // temp accumulation buffer
    int rowIndex, colIndex;

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    for (i = 0; i < dataSizeY; ++i)  // rows
    {
        for (j = 0; j < dataSizeX; ++j)  // columns
        {
            sum = 0;                           // init to 0 before sum
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                mm = kernelSizeY - 1 - m;  // row index of flipped kernel

                for (n = 0; n < kernelSizeX; ++n)  // kernel columns
                {
                    nn = kernelSizeX - 1 - n;  // column index of flipped kernel

                    // index of input signal, used for checking boundary
                    rowIndex = i + (kCenterY - mm);
                    colIndex = j + (kCenterX - nn);

                    // ignore input samples which are out of bound
                    if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                        sum += in[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
                }
            }
            out[dataSizeX * i + j] = sum;
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// 2D convolution
// 2D data are usually stored in computer memory as contiguous 1D array.
// So, we are using 1D array for 2D data.
// 2D convolution assumes the kernel is center originated, which means, if
// kernel size 3 then, k[-1], k[0], k[1]. The middle of index is always 0.
// The following programming logics are somewhat complicated because of using
// pointer indexing in order to minimize the number of multiplications.
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// unsigned char version (8bit): Note that the output is always positive number
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX,
    int kernelSizeY)
{
    int i, j, m, n;
    unsigned char *inPtr, *inPtr2, *outPtr;
    float* kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;  // to check boundary of input array
    int colMin, colMax;  //
    float sum;           // temp accumulation buffer

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for (j = 0; j < dataSizeX; ++j)  // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            sum = 0;  // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                // check if the index is out of bound of input array
                if (m <= rowMax && m > rowMin)
                {
                    for (n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if (n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;

                        ++kPtr;  // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;  // out of bound, move to next row of kernel

                inPtr -= dataSizeX;  // move input data 1 raw up
            }

            // convert negative number to positive
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);

            kPtr = kernel;     // reset kernel to (0,0)
            inPtr = ++inPtr2;  // next input
            ++outPtr;          // next output
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// unsigned short (16bit)
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(unsigned short* in, unsigned short* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX,
    int kernelSizeY)
{
    int i, j, m, n;
    unsigned short *inPtr, *inPtr2, *outPtr;
    float* kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;  // to check boundary of input array
    int colMin, colMax;  //
    float sum;           // temp accumulation buffer

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for (j = 0; j < dataSizeX; ++j)  // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            sum = 0;  // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                // check if the index is out of bound of input array
                if (m <= rowMax && m > rowMin)
                {
                    for (n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if (n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;

                        ++kPtr;  // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;  // out of bound, move to next row of kernel

                inPtr -= dataSizeX;  // move input data 1 raw up
            }

            // convert negative number to positive
            *outPtr = (unsigned short)((float)fabs(sum) + 0.5f);

            kPtr = kernel;     // reset kernel to (0,0)
            inPtr = ++inPtr2;  // next input
            ++outPtr;          // next output
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// signed integer (32bit) version:
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(int* in, int* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    int *inPtr, *inPtr2, *outPtr;
    float* kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;  // to check boundary of input array
    int colMin, colMax;  //
    float sum;           // temp accumulation buffer

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for (j = 0; j < dataSizeX; ++j)  // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            sum = 0;  // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                // check if the index is out of bound of input array
                if (m <= rowMax && m > rowMin)
                {
                    for (n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if (n <= colMax && n > colMin)
                            sum += *(inPtr - n) * *kPtr;

                        ++kPtr;  // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;  // out of bound, move to next row of kernel

                inPtr -= dataSizeX;  // move input data 1 raw up
            }

            // convert integer number
            if (sum >= 0)
                *outPtr = (int)(sum + 0.5f);
            else
                *outPtr = (int)(sum - 0.5f);

            kPtr = kernel;     // reset kernel to (0,0)
            inPtr = ++inPtr2;  // next input
            ++outPtr;          // next output
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// single float precision version:
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(float* in, float* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    float *inPtr, *inPtr2, *outPtr, *kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;  // to check boundary of input array
    int colMin, colMax;  //

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for (j = 0; j < dataSizeX; ++j)  // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            *outPtr = 0;  // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                // check if the index is out of bound of input array
                if (m <= rowMax && m > rowMin)
                {
                    for (n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if (n <= colMax && n > colMin)
                            *outPtr += *(inPtr - n) * *kPtr;
                        ++kPtr;  // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;  // out of bound, move to next row of kernel

                inPtr -= dataSizeX;  // move input data 1 raw up
            }

            kPtr = kernel;     // reset kernel to (0,0)
            inPtr = ++inPtr2;  // next input
            ++outPtr;          // next output
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// double float precision version:
///////////////////////////////////////////////////////////////////////////////
bool convolve2D(double* in, double* out, int dataSizeX, int dataSizeY, double* kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n;
    double *inPtr, *inPtr2, *outPtr, *kPtr;
    int kCenterX, kCenterY;
    int rowMin, rowMax;  // to check boundary of input array
    int colMin, colMax;  //

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;

    // init working  pointers
    inPtr = inPtr2 = &in[dataSizeX * kCenterY + kCenterX];  // note that  it is shifted (kCenterX, kCenterY),
    outPtr = out;
    kPtr = kernel;

    // start convolution
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        // compute the range of convolution, the current row of kernel should be between these
        rowMax = i + kCenterY;
        rowMin = i - dataSizeY + kCenterY;

        for (j = 0; j < dataSizeX; ++j)  // number of columns
        {
            // compute the range of convolution, the current column of kernel should be between these
            colMax = j + kCenterX;
            colMin = j - dataSizeX + kCenterX;

            *outPtr = 0;  // set to 0 before accumulate

            // flip the kernel and traverse all the kernel values
            // multiply each kernel value with underlying input data
            for (m = 0; m < kernelSizeY; ++m)  // kernel rows
            {
                // check if the index is out of bound of input array
                if (m <= rowMax && m > rowMin)
                {
                    for (n = 0; n < kernelSizeX; ++n)
                    {
                        // check the boundary of array
                        if (n <= colMax && n > colMin)
                            *outPtr += *(inPtr - n) * *kPtr;
                        ++kPtr;  // next kernel
                    }
                }
                else
                    kPtr += kernelSizeX;  // out of bound, move to next row of kernel

                inPtr -= dataSizeX;  // move input data 1 raw up
            }

            kPtr = kernel;     // reset kernel to (0,0)
            inPtr = ++inPtr2;  // next input
            ++outPtr;          // next output
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Separable 2D Convolution
// If the MxN kernel can be separable to (Mx1) and (1xN) matrices, the
// multiplication can be reduced to M+N comapred to MxN in normal convolution.
// It does not check the output is excceded max for performance reason. And we
// assume the kernel contains good(valid) data, therefore, the result cannot be
// larger than max.
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// unsigned char (8-bit) version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernelX,
    int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                // intermediate data buffer
    unsigned char *inPtr, *outPtr;   // working pointers
    float *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned char)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset to zero for next summing
            ++outPtr;    // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned char)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset for next summing
            ++outPtr;    // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned char)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset for next summing
            ++outPtr;    // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// unsigned short (16-bit) version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(unsigned short* in, unsigned short* out, int dataSizeX, int dataSizeY, float* kernelX,
    int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                // intermediate data buffer
    unsigned short *inPtr, *outPtr;  // working pointers
    float *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned short)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset to zero for next summing
            ++outPtr;    // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned short)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset before next summing
            ++outPtr;    // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            // covert negative to positive
            *outPtr = (unsigned short)((float)fabs(sum[n]) + 0.5f);
            sum[n] = 0;  // reset before next summing
            ++outPtr;    // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// integer (32-bit) version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(
    int* in, int* out, int dataSizeX, int dataSizeY, float* kernelX, int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                // intermediate data buffer
    int *inPtr, *outPtr;             // working pointers
    float *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            if (sum[n] >= 0)
                *outPtr = (int)(sum[n] + 0.5f);  // store final result to output array
            else
                *outPtr = (int)(sum[n] - 0.5f);  // store final result to output array

            sum[n] = 0;  // reset to zero for next summing
            ++outPtr;    // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            if (sum[n] >= 0)
                *outPtr = (int)(sum[n] + 0.5f);  // store final result to output array
            else
                *outPtr = (int)(sum[n] - 0.5f);  // store final result to output array
            sum[n] = 0;                          // reset to 0 before next summing
            ++outPtr;                            // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            if (sum[n] >= 0)
                *outPtr = (int)(sum[n] + 0.5f);  // store final result to output array
            else
                *outPtr = (int)(sum[n] - 0.5f);  // store final result to output array
            sum[n] = 0;                          // reset before next summing
            ++outPtr;                            // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// single precision float version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(
    float* in, float* out, int dataSizeX, int dataSizeY, float* kernelX, int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    float *tmp, *sum;                // intermediate data buffer
    float *inPtr, *outPtr;           // working pointers
    float *tmpPtr, *tmpPtr2;         // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new float[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new float[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output buffer
            sum[n] = 0;        // reset before next summing
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to 0 for next sum
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// double precision float version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DSeparable(
    double* in, double* out, int dataSizeX, int dataSizeY, double* kernelX, int kSizeX, float* kernelY, int kSizeY)
{
    int i, j, k, m, n;
    double *tmp, *sum;               // intermediate data buffer
    double *inPtr, *outPtr;          // working pointers
    double *tmpPtr, *tmpPtr2;        // working pointers
    int kCenter, kOffset, endIndex;  // kernel indice

    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // allocate temp storage to keep intermediate result
    tmp = new double[dataSizeX * dataSizeY];
    if (!tmp)
        return false;  // memory allocation error

    // store accumulated sum
    sum = new double[dataSizeX];
    if (!sum)
        return false;  // memory allocation error

    // covolve horizontal direction ///////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeX >> 1;           // center index of kernel array
    endIndex = dataSizeX - kCenter;  // index for full kernel convolution

    // init working pointers
    inPtr = in;
    tmpPtr = tmp;  // store intermediate results from 1D horizontal convolution

    // start horizontal convolution (x-direction)
    for (i = 0; i < dataSizeY; ++i)  // number of rows
    {
        kOffset = 0;  // starting index of partial kernel varies for each sample

        // COLUMN FROM index=0 TO index=kCenter-1
        for (j = 0; j < kCenter; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kCenter + kOffset, m = 0; k >= 0; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++tmpPtr;   // next output
            ++kOffset;  // increase starting index of kernel
        }

        // COLUMN FROM index=kCenter TO index=(dataSizeX-kCenter-1)
        for (j = kCenter; j < endIndex; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulate

            for (k = kSizeX - 1, m = 0; k >= 0; --k, ++m)  // full kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;   // next input
            ++tmpPtr;  // next output
        }

        kOffset = 1;  // ending index of partial kernel varies for each sample

        // COLUMN FROM index=(dataSizeX-kCenter) TO index=(dataSizeX-1)
        for (j = endIndex; j < dataSizeX; ++j)
        {
            *tmpPtr = 0;  // init to 0 before accumulation

            for (k = kSizeX - 1, m = 0; k >= kOffset; --k, ++m)  // convolve with partial of kernel
            {
                *tmpPtr += *(inPtr + m) * kernelX[k];
            }
            ++inPtr;    // next input
            ++tmpPtr;   // next output
            ++kOffset;  // increase ending index of partial kernel
        }

        inPtr += kCenter;  // next row
    }
    // END OF HORIZONTAL CONVOLUTION //////////////////////

    // start vertical direction ///////////////////////////

    // find center position of kernel (half of kernel size)
    kCenter = kSizeY >> 1;           // center index of vertical kernel
    endIndex = dataSizeY - kCenter;  // index where full kernel convolution should stop

    // set working pointers
    tmpPtr = tmpPtr2 = tmp;
    outPtr = out;

    // clear out array before accumulation
    for (i = 0; i < dataSizeX; ++i)
        sum[i] = 0;

    // start to convolve vertical direction (y-direction)

    // ROW FROM index=0 TO index=(kCenter-1)
    kOffset = 0;  // starting index of partial kernel varies for each sample
    for (i = 0; i < kCenter; ++i)
    {
        for (k = kCenter + kOffset; k >= 0; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // next element of output
        }

        tmpPtr = tmpPtr2;  // reset input pointer
        ++kOffset;         // increase starting index of kernel
    }

    // ROW FROM index=kCenter TO index=(dataSizeY-kCenter-1)
    for (i = kCenter; i < endIndex; ++i)
    {
        for (k = kSizeY - 1; k >= 0; --k)  // convolve with full kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // next output
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;
    }

    // ROW FROM index=(dataSizeY-kCenter) TO index=(dataSizeY-1)
    kOffset = 1;  // ending index of partial kernel varies for each sample
    for (i = endIndex; i < dataSizeY; ++i)
    {
        for (k = kSizeY - 1; k >= kOffset; --k)  // convolve with partial kernel
        {
            for (j = 0; j < dataSizeX; ++j)
            {
                sum[j] += *tmpPtr * kernelY[k];
                ++tmpPtr;
            }
        }

        for (n = 0; n < dataSizeX; ++n)  // convert and copy from sum to out
        {
            *outPtr = sum[n];  // store final result to output array
            sum[n] = 0;        // reset to zero for next summing
            ++outPtr;          // increase ending index of partial kernel
        }

        // move to next row
        tmpPtr2 += dataSizeX;
        tmpPtr = tmpPtr2;  // next input
        ++kOffset;         // increase ending index of kernel
    }
    // END OF VERTICAL CONVOLUTION ////////////////////////

    // deallocate temp buffers
    delete[] tmp;
    delete[] sum;
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// 2D Convolution Fast
// In order to improve the performance, this function uses multple cursors of
// input signal. It avoids indexing input array during convolution. And, the
// input signal is partitioned to 9 different sections, so we don't need to
// check the boundary for every samples.
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// unsigned char (8-bit) version
///////////////////////////////////////////////////////////////////////////////
bool convolve2DFast(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernel, int kernelSizeX,
    int kernelSizeY)
{
    int i, j, m, n, x, y, t;
    unsigned char **inPtr, *outPtr, *ptr;
    int kCenterX, kCenterY;
    int rowEnd, colEnd;  // ending indice for section divider
    float sum;           // temp accumulation buffer
    int k, kSize;

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;
    kSize = kernelSizeX * kernelSizeY;  // total kernel size

    // allocate memeory for multi-cursor
    inPtr = new unsigned char*[kSize];
    if (!inPtr)
        return false;  // allocation error

    // set initial position of multi-cursor, NOTE: it is swapped instead of kernel
    ptr = in + (dataSizeX * kCenterY + kCenterX);  // the first cursor is shifted (kCenterX, kCenterY)
    for (m = 0, t = 0; m < kernelSizeY; ++m)
    {
        for (n = 0; n < kernelSizeX; ++n, ++t)
        {
            inPtr[t] = ptr - n;
        }
        ptr -= dataSizeX;
    }

    // init working  pointers
    outPtr = out;

    rowEnd = dataSizeY - kCenterY;  // bottom row partition divider
    colEnd = dataSizeX - kCenterX;  // right column partition divider

    // convolve rows from index=0 to index=kCenterY-1
    y = kCenterY;
    for (i = 0; i < kCenterY; ++i)
    {
        // partition #1 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = 0;
            for (m = 0; m <= y; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);  // jump to next row
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #2 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = 0;
            for (m = 0; m <= y; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #3 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = x;
            for (m = 0; m <= y; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;  // jump to next row
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        ++y;  // add one more row to convolve for next run
    }

    // convolve rows from index=kCenterY to index=(dataSizeY-kCenterY-1)
    for (i = kCenterY; i < rowEnd; ++i)  // number of rows
    {
        // partition #4 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = 0;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #5 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = 0;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++inPtr[t];  // in this partition, all cursors are used to convolve. moving cursors to next is safe
                                 // here
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
        }

        // partition #6 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = x;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }
    }

    // convolve rows from index=(dataSizeY-kCenterY) to index=(dataSizeY-1)
    y = 1;
    for (i = rowEnd; i < dataSizeY; ++i)  // number of rows
    {
        // partition #7 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = kernelSizeX * y;

            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #8 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = kernelSizeX * y;
            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];
        }

        // partition #9 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = kernelSizeX * y + x;
            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;
            }

            // store output
            *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        ++y;  // the starting row index is increased
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Fast 2D Convolution using integer multiplication instead of float.
// Multiply coefficient(factor) to accumulated sum at last.
// NOTE: IT IS NOT FASTER THAN FLOAT MULTIPLICATION, TRY YOURSELF!!!
///////////////////////////////////////////////////////////////////////////////
bool convolve2DFast2(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, int* kernel, float factor,
    int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n, x, y, t;
    unsigned char **inPtr, *outPtr, *ptr;
    int kCenterX, kCenterY;
    int rowEnd, colEnd;  // ending indice for section divider
    int sum;             // temp accumulation buffer
    int k, kSize;

    // check validity of params
    if (!in || !out || !kernel)
        return false;
    if (dataSizeX <= 0 || kernelSizeX <= 0)
        return false;

    // find center position of kernel (half of kernel size)
    kCenterX = kernelSizeX >> 1;
    kCenterY = kernelSizeY >> 1;
    kSize = kernelSizeX * kernelSizeY;  // total kernel size

    // allocate memeory for multi-cursor
    inPtr = new unsigned char*[kSize];
    if (!inPtr)
        return false;  // allocation error

    // set initial position of multi-cursor, NOTE: it is swapped instead of kernel
    ptr = in + (dataSizeX * kCenterY + kCenterX);  // the first cursor is shifted (kCenterX, kCenterY)
    for (m = 0, t = 0; m < kernelSizeY; ++m)
    {
        for (n = 0; n < kernelSizeX; ++n, ++t)
        {
            inPtr[t] = ptr - n;
        }
        ptr -= dataSizeX;
    }

    // init working  pointers
    outPtr = out;

    rowEnd = dataSizeY - kCenterY;  // bottom row partition divider
    colEnd = dataSizeX - kCenterX;  // right column partition divider

    // convolve rows from index=0 to index=kCenterY-1
    y = kCenterY;
    for (i = 0; i < kCenterY; ++i)
    {
        // partition #1 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = 0;
            for (m = 0; m <= y; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);  // jump to next row
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #2 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = 0;
            for (m = 0; m <= y; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #3 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = x;
            for (m = 0; m <= y; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;  // jump to next row
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        ++y;  // add one more row to convolve for next run
    }

    // convolve rows from index=kCenterY to index=(dataSizeY-kCenterY-1)
    for (i = kCenterY; i < rowEnd; ++i)  // number of rows
    {
        // partition #4 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = 0;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #5 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = 0;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++inPtr[t];  // in this partition, all cursors are used to convolve. moving cursors to next is safe
                                 // here
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
        }

        // partition #6 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = x;
            for (m = 0; m < kernelSizeY; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }
    }

    // convolve rows from index=(dataSizeY-kCenterY) to index=(dataSizeY-1)
    y = 1;
    for (i = rowEnd; i < dataSizeY; ++i)  // number of rows
    {
        // partition #7 ***********************************
        x = kCenterX;
        for (j = 0; j < kCenterX; ++j)  // column from index=0 to index=kCenterX-1
        {
            sum = 0;
            t = kernelSizeX * y;

            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = 0; n <= x; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += (kernelSizeX - x - 1);
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        // partition #8 ***********************************
        for (j = kCenterX; j < colEnd; ++j)  // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
        {
            sum = 0;
            t = kernelSizeX * y;
            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = 0; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];
        }

        // partition #9 ***********************************
        x = 1;
        for (j = colEnd; j < dataSizeX; ++j)  // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
        {
            sum = 0;
            t = kernelSizeX * y + x;
            for (m = y; m < kernelSizeY; ++m)
            {
                for (n = x; n < kernelSizeX; ++n)
                {
                    sum += *inPtr[t] * kernel[t];
                    ++t;
                }
                t += x;
            }

            // store output
            *outPtr = (unsigned char)(fabs(sum * factor) + 0.5f);
            ++outPtr;
            ++x;
            for (k = 0; k < kSize; ++k)
                ++inPtr[k];  // move all cursors to next
        }

        ++y;  // the starting row index is increased
    }

    return true;
}

//! Use 2D indices instead of array pointer. Better for understanding.
bool convolve2DSeparableReadable(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY, float* kernelX,
    int kSizeX, float* kernelY, int kSizeY)
{
    // check validity of params
    if (!in || !out || !kernelX || !kernelY)
        return false;
    if (dataSizeX <= 0 || kSizeX <= 0)
        return false;

    // Save temporary horizontal convolution for the entire image
    int N = dataSizeX * dataSizeY;
    float* tmpx = new float[N];
    if (!tmpx)
        return false;  // memory allocation error
    for (int i = 0; i < N; ++i)
        tmpx[i] = 0;

    // Save temporary vertical convolution for one row
    float* tmpsum = new float[dataSizeX];
    if (!tmpsum)
        return false;  // memory allocation error
    for (int i = 0; i < dataSizeX; ++i)
        tmpsum[i] = 0;

    // find center position of kernel (half of kernel size)
    int kCenter = kSizeX >> 1;           // center index of kernel array
    int endIndex = dataSizeX - kCenter;  // index for full kernel convolution
    int right_half = kSizeX - kCenter - 1; // size of right half right to index 'kCenter' in the kernel

    /* Convolution in horizontal direction.
        We split [0, dataSizeX-1] into three parts: [0, kCenter - 1], [kCenter, endIndex - 1] and
        [endIndex, dataSizeX - 1]. The middle part is general case, while the first and last part
        are image border cases.
    */
    // [0, kCenter - 1]
    for (int j = 0; j < dataSizeY; ++j)
    {
        int offset = 0;
        for (int i = 0; i < kCenter; ++i)
        {
            int idx = j * dataSizeX + i;
            for (int k = kCenter + offset, m = 0; k >= 0; k--, m++)
                tmpx[idx] += in[idx - offset + m] * kernelX[k];
            offset++;
        }
    }
    // [kCenter, endIndex - 1]
    for (int j = 0; j < dataSizeY; ++j)
    {
        for (int i = kCenter; i < endIndex; ++i)
        {
            int idx = j * dataSizeX + i;
            for (int k = kSizeX - 1, m = 0; k >= 0; k--, m++)
                tmpx[idx] += in[idx - right_half + m] * kernelX[k];
        }
    }
    // [endIndex, dataSizeX - 1]
    for (int j = 0; j < dataSizeY; ++j)
    {
        int offset = 1;
        for (int i = endIndex; i < dataSizeX; ++i)
        {
            int idx = j * dataSizeX + i;
            for (int k = kSizeX - 1, m = 0; k >= offset; k--, m++)
                tmpx[idx] += in[idx - right_half + m] * kernelX[k];
            offset++;
        }
    }

    /* Convolution in vertical direction. Similiar to horizontal ones. */

    kCenter = kSizeY >> 1;
    endIndex = dataSizeY - kCenter;
    right_half = kSizeY - kCenter - 1;

    // [0, kCenter - 1]
    int offset = 0;
    for (int j = 0; j < kCenter; ++j)
    {
        for (int k = kCenter + offset, row = 0; k >= 0; k--, row++)
        {
            for (int i = 0; i < dataSizeX; ++i)
            {
                int idx = row * dataSizeX + i;
                tmpsum[i] += tmpx[idx] * kernelY[k];  // tmpsum is 1D row vector
            }
        }
        offset++;

        // Copy tmpSum result to final output image. One 1D row vector 'tmpsum'
        // is enough and this can save storage.
        for (int i = 0; i < dataSizeX; ++i)
        {
            int idx = j * dataSizeX + i;
            out[idx] = (unsigned char)(float(tmpsum[i]) + 0.5f);
            tmpsum[i] = 0;
        }
    }
    // [kCenter, endIndex - 1]
    for (int j = kCenter; j < endIndex; ++j)
    {
        for (int k = kSizeY - 1, m = 0; k >= 0; k--, m++)
        {
            int row = j - right_half + m;
            for (int i = 0; i < dataSizeX; ++i)
            {
                int idx = row * dataSizeX + i;
                tmpsum[i] += tmpx[idx] * kernelY[k];
            }
        }
        for (int i = 0; i < dataSizeX; ++i)
        {
            int idx = j * dataSizeX + i;
            out[idx] = (unsigned char)(float(tmpsum[i]) + 0.5f);
            tmpsum[i] = 0;
        }
    }

    // [endIndex, dataSizeY - 1]
    offset = 1;
    for (int j = endIndex; j < dataSizeY; ++j)
    {
        for (int k = kSizeY - 1, m = 0; k >= offset; k--, m++)
        {
            int row = j - right_half + m;
            for (int i = 0; i < dataSizeX; ++i)
            {
                int idx = row * dataSizeX + i;
                tmpsum[i] += tmpx[idx] * kernelY[k];  // tmpsum is 1D row vector
            }
        }
        offset++;
        for (int i = 0; i < dataSizeX; ++i)
        {
            int idx = j * dataSizeX + i;
            out[idx] = (unsigned char)(float(tmpsum[i]) + 0.5f);
            tmpsum[i] = 0;
        }
    }

    delete[] tmpx;
    delete[] tmpsum;
    return true;
}
