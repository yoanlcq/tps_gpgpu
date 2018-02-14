#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

typedef unsigned char uchar;
typedef unsigned int uint;

static void HandleError(cudaError_t err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
    	std::stringstream ss;
    	ss << line;
        std::string errMsg(cudaGetErrorString(err));
        errMsg += " (file: " + std::string(file);
        errMsg += " at line: " + ss.str() + ")";
        throw std::runtime_error(errMsg);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void verifyDimGridBlock(const uint dimGrid, const uint dimBlock, const uint N) 
{
	cudaDeviceProp prop;
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

	unsigned long maxGridSize			= prop.maxGridSize[0];
	unsigned long maxThreadsPerBlock	= prop.maxThreadsPerBlock;

	if ( dimBlock > maxThreadsPerBlock ) 
    {   
        throw std::runtime_error("Maximum threads per block exceeded");
	}

	if  ( dimGrid > maxGridSize ) 
    {
        throw std::runtime_error("Maximum grid size exceeded");
	}
}

static uint nextPow2(uint x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#endif



