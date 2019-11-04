# fast-image-convolution-cpp

Fast image convolution only in CPU, no parallel computation but still very fast.

Most of this code is from the source code provided in this tutorial: http://www.songho.ca/dsp/convolution/convolution.html. I just add one function convolve2DSeparableReadable() to implement the separable 2D convolution but using indices of input image 1D array instead of pointers in other existing functions.

