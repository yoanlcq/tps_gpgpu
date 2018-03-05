/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== Ex 0
    // Q2: Ca se foire parce qu'il n'y a pas assez de blocks et threads pour couvrir tout le tableau.
    // Spécifiquement, dans notre cas, on était à 1024*65535 = 67107840. Or, 2^26 = 67108864.
    __global__ void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax) {
        extern __shared__ uint sharedMem[];

        const uint dev_i = blockIdx.x * blockDim.x + threadIdx.x;
        if(dev_i >= size)
            return;
        sharedMem[threadIdx.x] = dev_array[dev_i];
        __syncthreads();

        for(uint stride=1 ;  ; stride *= 2) {
            const uint i = threadIdx.x * 2 * stride;
            const uint j = i + stride;
            if(j >= blockDim.x)
                break;
            sharedMem[i] = umax(sharedMem[i], sharedMem[j]);
            __syncthreads();
        }

        if(threadIdx.x == 0)
            dev_partialMax[blockIdx.x] = sharedMem[0];
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		
		std::cout << "========== Ex 2 " << std::endl;
		/// TODO

		std::cout << "========== Ex 3 " << std::endl;
		/// TODO
		
		std::cout << "========== Ex 4 " << std::endl;
		/// TODO
		
		std::cout << "========== Ex 5 " << std::endl;
		/// TODO
		

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
