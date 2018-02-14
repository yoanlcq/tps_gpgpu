/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include <assert.h>

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
    __global__ void sepiaCUDA(const uchar* in, uint width, uint height, uchar* out)
    {
        for(uint y = blockIdx.y * blockDim.y + threadIdx.y ; y < height ; y += gridDim.y * blockDim.y) {
            for(uint x = blockIdx.x * blockDim.x + threadIdx.x ; x < width ; x += gridDim.x * blockDim.x) {
                const uint i = (y * width + x) * 3;
                const uchar r = in[i+0];
                const uchar g = in[i+1];
                const uchar b = in[i+2];
                out[i+0] = fminf(255.f, r * .393f + g * .769f + b * .189f);
                out[i+1] = fminf(255.f, r * .349f + g * .686f + b * .168f);
                out[i+2] = fminf(255.f, r * .272f + g * .534f + b * .131f);
            }
        }
    }

    void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
    {
        assert(input.size() == output.size());

        ChronoGPU chrGPU;

        uchar *dev_in = NULL;
        uchar *dev_out = NULL;

        const size_t nbytes = input.size() * sizeof input[0];
        std::cout << "Allocating input (2 arrays): " 
                  << ((2 * nbytes) >> 20) << " MB on Device" << std::endl;
        chrGPU.start();
        
        cudaMalloc((void**) &dev_in, nbytes);
        cudaMalloc((void**) &dev_out, nbytes);

        chrGPU.stop();
        std::cout   << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

        cudaMemcpy(dev_in, input.data(), nbytes, cudaMemcpyHostToDevice);

        // 16*16 = 256 threads/tile
        // 32*32 = 1024 threads/tile
        dim3 tile_size(32, 32);
        dim3 n_tiles(
            min(65535, (width  + tile_size.x - 1) / tile_size.x),
            min(65535, (height + tile_size.y - 1) / tile_size.y)
        );

        std::cout << "Launching kernel..." << std::endl;
        chrGPU.start();
        sepiaCUDA<<<n_tiles, tile_size>>>(dev_in, width, height, dev_out);
        chrGPU.stop();
        std::cout   << "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

        cudaMemcpy(output.data(), dev_out, nbytes, cudaMemcpyDeviceToHost);

        cudaFree(dev_in);
        cudaFree(dev_out);
    }
}
