/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"
#include "scoped_chrono.hpp"
#include <assert.h>

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}


    // ==================================================

    __device__ int imaxCUDA(int a, int b) { return a > b ? a : b; }
    __device__ int iminCUDA(int a, int b) { return a < b ? a : b; }
    __device__ int clampiCUDA(int x, int a, int b) {
        return iminCUDA(imaxCUDA(x, a), b);
    }
    __device__ float clampfCUDA(float x, float a, float b) {
        return fminf(fmaxf(x, a), b);
    }


    // Ex1
    __global__ void convCUDAEx1(
        const uchar4* src, const uint w, const uint h,
        const float* mat, const uint mat_size,
        uchar4* dst)
    {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = i / w;
        const uint x = i - y*w;
        float3 sum = make_float3(0.f, 0.f, 0.f);

        // Convolution
        for(uint j=0 ; j<mat_size ; ++j) {
            for(uint i=0 ; i<mat_size ; ++i) {
                const int dX = clampiCUDA(x + i - mat_size / 2, 0, w-1);
                const int dY = clampiCUDA(y + j - mat_size / 2, 0, h-1);
                const uint iMat	  = j * mat_size + i;
                const uint iPixel = dY * w + dX;
                sum.x += float(src[iPixel].x) * mat[iMat];
                sum.y += float(src[iPixel].y) * mat[iMat];
                sum.z += float(src[iPixel].z) * mat[iMat];
            }
        }

        const uint iOut = y * w + x;
        dst[iOut] = make_uchar4(
            clampfCUDA(sum.x, 0, 255),
            clampfCUDA(sum.y, 0, 255),
            clampfCUDA(sum.z, 0, 255),
            255
        );
    }


    // Ex2

#define MAX_MAT_ELEMENT_COUNT (16*16)
    __constant__ float g_cst_dev_mat[MAX_MAT_ELEMENT_COUNT];

    __global__ void convCUDAEx2(const uchar4* src, const uint w, const uint h, const uint mat_size, uchar4* dst) {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = i / w;
        const uint x = i - y*w;
        float3 sum = make_float3(0.f, 0.f, 0.f);

        // Convolution
        for(uint j=0 ; j<mat_size ; ++j) {
            for(uint i=0 ; i<mat_size ; ++i) {
                const int dX = clampiCUDA(x + i - mat_size / 2, 0, w-1);
                const int dY = clampiCUDA(y + j - mat_size / 2, 0, h-1);
                const uint iMat	  = j * mat_size + i;
                const uint iPixel = dY * w + dX;
                sum.x += float(src[iPixel].x) * g_cst_dev_mat[iMat];
                sum.y += float(src[iPixel].y) * g_cst_dev_mat[iMat];
                sum.z += float(src[iPixel].z) * g_cst_dev_mat[iMat];
            }
        }

        const uint iOut = y * w + x;
        dst[iOut] = make_uchar4(
            clampfCUDA(sum.x, 0, 255),
            clampfCUDA(sum.y, 0, 255),
            clampfCUDA(sum.z, 0, 255),
            255
        );
    }

    // Ex3

    texture<uchar4, 1, cudaReadModeElementType> g_dev_src_tex_1d;

    __global__ void convCUDAEx3(const uint w, const uint h, const uint mat_size, uchar4* dst) {
        const uint i = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = i / w;
        const uint x = i - y*w;
        float3 sum = make_float3(0.f, 0.f, 0.f);

        // Convolution
        for(uint j=0 ; j<mat_size ; ++j) {
            for(uint i=0 ; i<mat_size ; ++i) {
                const int dX = clampiCUDA(x + i - mat_size / 2, 0, w-1);
                const int dY = clampiCUDA(y + j - mat_size / 2, 0, h-1);
                const uint iMat	  = j * mat_size + i;
                const uint iPixel = dY * w + dX;
                uchar4 texel = tex1Dfetch(g_dev_src_tex_1d, iPixel);
                sum.x += float(texel.x) * g_cst_dev_mat[iMat];
                sum.y += float(texel.y) * g_cst_dev_mat[iMat];
                sum.z += float(texel.z) * g_cst_dev_mat[iMat];
            }
        }

        const uint iOut = y * w + x;
        dst[iOut] = make_uchar4(
            clampfCUDA(sum.x, 0, 255),
            clampfCUDA(sum.y, 0, 255),
            clampfCUDA(sum.z, 0, 255),
            255
        );
    }

    // Ex4

    texture<uchar4, 2, cudaReadModeElementType> g_dev_src_tex_2d;

    __global__ void convCUDAEx4(const uint w, const uint h, const uint mat_size, uchar4* dst) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;
        float3 sum = make_float3(0.f, 0.f, 0.f);

        // Convolution
        for(uint j=0 ; j<mat_size ; ++j) {
            for(uint i=0 ; i<mat_size ; ++i) {
                const int dX = clampiCUDA(x + i - mat_size / 2, 0, w-1);
                const int dY = clampiCUDA(y + j - mat_size / 2, 0, h-1);
                const uint iMat = j * mat_size + i;
                // FIXME: Doesn't work!
                const float tu = (dX+0.5f) / float(w);
                const float tv = (dY+0.5f) / float(h);
                const uchar4 texel = tex2D(g_dev_src_tex_2d, tu, tv);
                sum.x += float(texel.x) * g_cst_dev_mat[iMat];
                sum.y += float(texel.y) * g_cst_dev_mat[iMat];
                sum.z += float(texel.z) * g_cst_dev_mat[iMat];
            }
        }

        const uint iOut = y * w + x;
        dst[iOut] = make_uchar4(
            clampfCUDA(sum.x, 0, 255),
            clampfCUDA(sum.y, 0, 255),
            clampfCUDA(sum.z, 0, 255),
            255
        );
    }


    // ==================================================


    void studentJobEx1(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
        typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

        assert(inputImg.size() == output.size());
        assert(matConv.size() == matSize*matSize);

        uchar4* dev_src = NULL;
        float*  dev_mat = NULL;
        uchar4 *dev_dst = NULL;

        {
            ScopedChronoGPU chr("Allocating GPU memory (3 arrays)");
            cudaMalloc((void**) &dev_src, inputImg.size() * sizeof inputImg[0]);
            cudaMalloc((void**) &dev_mat, matConv.size() * sizeof matConv[0]);
            cudaMalloc((void**) &dev_dst, output.size() * sizeof output[0]);
        }

        {
            ScopedChronoGPU chr("Uploading data to GPU memory (2 arrays)");
            cudaMemcpy(dev_src, inputImg.data(), inputImg.size() * sizeof inputImg[0], cudaMemcpyHostToDevice);
            cudaMemcpy(dev_mat, matConv.data(), matConv.size() * sizeof matConv[0], cudaMemcpyHostToDevice);
        }

        const uint n_threads = 512;
        const uint n_blocks = (inputImg.size()+n_threads-1) / n_threads;

        {
            ScopedChronoGPU chr("Process on GPU (parallel)");
            convCUDAEx1<<<n_blocks, n_threads>>>(dev_src, imgWidth, imgHeight, dev_mat, matSize, dev_dst);
        }

        {
            ScopedChronoGPU chr("Downloading output from GPU");
            cudaMemcpy(output.data(), dev_dst, output.size() * sizeof output[0], cudaMemcpyDeviceToHost);
        }

		cudaFree(dev_src);
		cudaFree(dev_mat);
		cudaFree(dev_dst);

        compareImages(output, resultCPU);
	}

    void studentJobEx2(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
        typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

        assert(inputImg.size() == output.size());
        assert(matConv.size() == matSize*matSize);

        uchar4* dev_src = NULL;
        uchar4* dev_dst = NULL;

        {
            ScopedChronoGPU chr("Allocating GPU memory (2 arrays)");
            cudaMalloc((void**) &dev_src, inputImg.size() * sizeof inputImg[0]);
            cudaMalloc((void**) &dev_dst, output.size() * sizeof output[0]);
        }

        {
            ScopedChronoGPU chr("Uploading data to GPU memory (2 arrays)");
            cudaMemcpy(dev_src, inputImg.data(), inputImg.size() * sizeof inputImg[0], cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_cst_dev_mat, matConv.data(), matConv.size() * sizeof matConv[0]);
        }

        const uint n_threads = 512;
        const uint n_blocks = (inputImg.size()+n_threads-1) / n_threads;

        {
            ScopedChronoGPU chr("Process on GPU (parallel)");
            convCUDAEx2<<<n_blocks, n_threads>>>(dev_src, imgWidth, imgHeight, matSize, dev_dst);
        }

        {
            ScopedChronoGPU chr("Downloading output from GPU");
            cudaMemcpy(output.data(), dev_dst, output.size() * sizeof output[0], cudaMemcpyDeviceToHost);
        }

		cudaFree(dev_src);
		cudaFree(dev_dst);

        compareImages(output, resultCPU);
	}

    void studentJobEx3(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
        typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

        assert(inputImg.size() == output.size());
        assert(matConv.size() == matSize*matSize);

        uchar4* dev_dst = NULL;
        uchar4* dev_src = NULL;

        {
            ScopedChronoGPU chr("Allocating GPU memory (2 arrays)");
            cudaMalloc((void**) &dev_src, inputImg.size() * sizeof inputImg[0]);
            cudaMalloc((void**) &dev_dst, output.size() * sizeof output[0]);
        }

        {
            ScopedChronoGPU chr("Uploading data to GPU memory (2 arrays)");
            cudaMemcpy(dev_src, inputImg.data(), inputImg.size() * sizeof inputImg[0], cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_cst_dev_mat, matConv.data(), matConv.size() * sizeof matConv[0]);
        }

        cudaError_t status = cudaBindTexture(NULL, g_dev_src_tex_1d, dev_src, inputImg.size() * sizeof inputImg[0]);
        assert(status == cudaSuccess);

        const uint n_threads = 512;
        const uint n_blocks = (inputImg.size()+n_threads-1) / n_threads;

        {
            ScopedChronoGPU chr("Process on GPU (parallel)");
            convCUDAEx3<<<n_blocks, n_threads>>>(imgWidth, imgHeight, matSize, dev_dst);
        }

        {
            ScopedChronoGPU chr("Downloading output from GPU");
            cudaMemcpy(output.data(), dev_dst, output.size() * sizeof output[0], cudaMemcpyDeviceToHost);
        }

        cudaUnbindTexture(g_dev_src_tex_1d);
		cudaFree(dev_src);
		cudaFree(dev_dst);

        compareImages(output, resultCPU);
	}

    void studentJobEx4(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
        typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

        assert(inputImg.size() == output.size());
        assert(matConv.size() == matSize*matSize);

        uchar4* dev_dst = NULL;
        uchar4* dev_src = NULL;

        {
            ScopedChronoGPU chr("Allocating GPU memory (2 arrays)");
            cudaMalloc((void**) &dev_src, inputImg.size() * sizeof inputImg[0]);
            cudaMalloc((void**) &dev_dst, output.size() * sizeof output[0]);
        }

        {
            ScopedChronoGPU chr("Uploading data to GPU memory (2 arrays)");
            cudaMemcpy(dev_src, inputImg.data(), inputImg.size() * sizeof inputImg[0], cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(g_cst_dev_mat, matConv.data(), matConv.size() * sizeof matConv[0]);
        }

        cudaError_t status = cudaBindTexture(NULL, g_dev_src_tex_2d, dev_src, inputImg.size() * sizeof inputImg[0]);
        assert(status == cudaSuccess);

        // 16*16 = 256 threads/tile
        // 32*32 = 1024 threads/tile
        const dim3 n_threads(32, 32);
        const dim3 n_blocks(
            (imgWidth +n_threads.x-1) / n_threads.x,
            (imgHeight+n_threads.y-1) / n_threads.y
        );

        {
            ScopedChronoGPU chr("Process on GPU (parallel)");
            convCUDAEx4<<<n_blocks, n_threads>>>(imgWidth, imgHeight, matSize, dev_dst);
        }

        {
            ScopedChronoGPU chr("Downloading output from GPU");
            cudaMemcpy(output.data(), dev_dst, output.size() * sizeof output[0], cudaMemcpyDeviceToHost);
        }

        cudaUnbindTexture(g_dev_src_tex_2d);
		cudaFree(dev_src);
		cudaFree(dev_dst);

        compareImages(output, resultCPU);
	}

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					) {
        std::cout << std::endl << "--- Student job Ex1 ---"  << std::endl;
        studentJobEx1(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
        std::cout << std::endl << "--- Student job Ex2 ---"  << std::endl;
        studentJobEx2(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
        std::cout << std::endl << "--- Student job Ex3 ---"  << std::endl;
        studentJobEx3(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
        std::cout << std::endl << "--- Student job Ex4 ---"  << std::endl;
        studentJobEx4(inputImg, imgWidth, imgHeight, matConv, matSize, resultCPU, output);
        std::cout << std::endl;
    }
}
