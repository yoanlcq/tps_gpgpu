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
        extern __shared__ uint shm[];

        const uint dev_i = blockIdx.x * blockDim.x + threadIdx.x;
        shm[threadIdx.x] = (dev_i < size) * dev_array[dev_i];

        for(uint stride=1 ; ; stride *= 2) {
            const uint i = threadIdx.x * 2 * stride;
            const uint j = i + stride;
            if(j >= blockDim.x)
                break;
            __syncthreads();
            shm[i] = umax(shm[i], shm[j]);
        }

        __syncthreads();
        if(threadIdx.x == 0)
            dev_partialMax[blockIdx.x] = shm[0];
	}

    __global__ void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax) {
        extern __shared__ uint shm[];

        const uint dev_i = blockIdx.x * blockDim.x + threadIdx.x;
        shm[threadIdx.x] = (dev_i < size) * dev_array[dev_i];

        const uint i = threadIdx.x;
        for(uint stride = blockDim.x/2 ; stride > 0 ; stride /= 2) {
            if(i >= stride)
                return; // NOTE: return, not break.
            __syncthreads();
            shm[i] = umax(shm[i], shm[i + stride]);
        }

        __syncthreads();
        // If we're here, threadIdx.x == 0.
        dev_partialMax[blockIdx.x] = shm[0];
	}

    __global__ void maxReduce_ex3(const uint *const dev_array, const uint size, uint *const dev_partialMax) {
        extern __shared__ uint shm[];

        // If 1 block = 1024 threads, then shm has 2048 elements here.
        // Each block processes 2048 elements, and each thread has 2 load/stores to perform: Tidx and Tidx+1024.
        // At step 0, elements from 0 to 1024 are stored in shm.
        // At step 1, elements from 1024 to 2048 are stored in shm.
        for(uint step=0 ; step<=1 ; ++step) {
            const uint shm_i = threadIdx.x + step * blockDim.x;
            const uint dev_i = blockIdx.x * 2 * blockDim.x + shm_i;
            shm[shm_i] = (dev_i < size) * dev_array[dev_i];
        }

        const uint i = threadIdx.x;
        for(uint stride = blockDim.x ; stride > 1 ; stride /= 2) {
            if(i >= stride)
                return; // NOTE: return, not break.
            __syncthreads();
            shm[i] = umax(shm[i], shm[i + stride]);
        }
        __syncthreads();
        dev_partialMax[blockIdx.x] = umax(shm[0], shm[1]);
	}

    __global__ void maxReduce_ex4(const uint *const dev_array, const uint size, uint *const dev_partialMax) {
        extern __shared__ uint shm[];

        // See maxReduce_ex3
        for(uint step=0 ; step<=1 ; ++step) {
            const uint shm_i = threadIdx.x + step * blockDim.x;
            const uint dev_i = blockIdx.x * 2 * blockDim.x + shm_i;
            shm[shm_i] = (dev_i < size) * dev_array[dev_i];
        }

        const uint i = threadIdx.x;
        for(uint stride = blockDim.x ; stride > 32 ; stride /= 2) {
            if(i >= stride)
                return; // NOTE: return, not break.
            __syncthreads();
            shm[i] = umax(shm[i], shm[i + stride]);
        }
        __syncthreads();
        volatile uint* const vshm = shm;
        // XXX: Does it need __threadfence_block() ?
        // See slide 20 of http://www.irisa.fr/alf/downloads/collange/cours/gpuprog_ufmg_2015/gpu_ufmg_2015_5.pdf
        vshm[i] = umax(vshm[i], vshm[i+32]);
        vshm[i] = umax(vshm[i], vshm[i+16]);
        vshm[i] = umax(vshm[i], vshm[i+8]); 
        vshm[i] = umax(vshm[i], vshm[i+4]); 
        vshm[i] = umax(vshm[i], vshm[i+2]); 
        if(i == 0)
            dev_partialMax[blockIdx.x] = umax(vshm[0], vshm[1]);
	}

    __global__ void maxReduce_ex5(const uint *const dev_array, const uint size, uint *const dev_partialMax) {
        extern __shared__ uint shm[];

        const uint shm_i = threadIdx.x * 2;
        const uint dev_i = blockIdx.x * 2048 + shm_i;
        shm[shm_i  ] = (dev_i   < size) * dev_array[dev_i  ];
        shm[shm_i+1] = (dev_i+1 < size) * dev_array[dev_i+1];
        __syncthreads();

        const uint i = threadIdx.x;
        shm[i] = umax(shm[i], shm[i+1024]); __syncthreads();
        shm[i] = umax(shm[i], shm[i+ 512]); __syncthreads();
        shm[i] = umax(shm[i], shm[i+ 256]); __syncthreads();
        shm[i] = umax(shm[i], shm[i+ 128]); __syncthreads();
        shm[i] = umax(shm[i], shm[i+  64]); __syncthreads();

        volatile uint* const vshm = shm;
        // XXX: Does it need __threadfence_block() ?
        // See slide 20 of http://www.irisa.fr/alf/downloads/collange/cours/gpuprog_ufmg_2015/gpu_ufmg_2015_5.pdf
        vshm[i] = umax(vshm[i], vshm[i+32]);
        vshm[i] = umax(vshm[i], vshm[i+16]);
        vshm[i] = umax(vshm[i], vshm[i+8]); 
        vshm[i] = umax(vshm[i], vshm[i+4]); 
        vshm[i] = umax(vshm[i], vshm[i+2]); 
        if(i == 0)
            dev_partialMax[blockIdx.x] = umax(vshm[0], vshm[1]);
	}

    template<uint kernelType>
    static void studentJob_testKernel(uint* const dev_array, uint const arraySize, const uint resCPU /* Just for comparison */) {
        uint res = 0;
        std::cout << "========== Ex " << (kernelType+1) << " " << std::endl;
        float2 timing = reduce<kernelType>(dev_array, arraySize, res);
        std::cout << " -> Done: ";
        printTiming(timing);
        compare(res, resCPU); // Compare results
    }

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

        studentJob_testKernel<KERNEL_EX1>(dev_array, array.size(), resCPU);
        studentJob_testKernel<KERNEL_EX2>(dev_array, array.size(), resCPU);
        studentJob_testKernel<KERNEL_EX3>(dev_array, array.size(), resCPU);
        studentJob_testKernel<KERNEL_EX4>(dev_array, array.size(), resCPU);
        studentJob_testKernel<KERNEL_EX5>(dev_array, array.size(), resCPU);

		cudaFree(dev_array);
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
