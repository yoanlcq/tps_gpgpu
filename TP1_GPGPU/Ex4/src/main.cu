#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include <vector>
#include "chronoCPU.hpp"
#include "chronoGPU.hpp"

__global__ void addMatricesCUDA(float* a, const float* b, size_t w, size_t h) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h)
        return;
    const size_t i = y * w + x;
    a[i] += b[i];
}

void addMatricesCPU(float* a, const float* b, size_t w, size_t h) {
    for(size_t y=0 ; y<h ; ++y) {
        for(size_t x=0 ; x<w ; ++x) {
            const size_t i = y * w + x;
            a[i] += b[i];
        }
    }
}

bool eqMatrices(const float* cpu_m, const float* gpu_m, size_t w, size_t h) {
    for(size_t y=0 ; y<h ; ++y) {
        for(size_t x=0 ; x<w ; ++x) {
            size_t i = y*w + x;
            if(fabsf(cpu_m[i] - gpu_m[i]) > 0.0001f) {
                printf(
                    "Results do not match (at x=%zu, y=%zu): %f (CPU) vs. %f (GPU)\n",
                    x, y, cpu_m[i], gpu_m[i]
                );
                return false;
            }
        }
    }
    return true;
}

// Juste pratique pour aligner horizontalement les mesures
void printTimeMs(const char *s, float t) {
    printf("%-24s : %f ms\n", s, t);
}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        printf("Usage: %s <width> <height> [<nthreads_x> <nthreads_y>]\n", argv[0]);
        return EXIT_FAILURE;
    }
    long long llw = strtoll(argv[1], NULL, 0);
    long long llh = strtoll(argv[2], NULL, 0);
    assert(llw >= 0 && llh >= 0); // Ca ne devrait pas vraiment être un assert() ici mais bon
    size_t w = llw;
    size_t h = llh;

    const size_t nbytes = w*h*sizeof(float);

    // 16*16 = 256 threads/tile
    // 32*32 = 1024 threads/tile
    dim3 tile_size(16, 16);
    if(argc >= 5) {
        long long tx = strtoll(argv[3], NULL, 0);
        long long ty = strtoll(argv[4], NULL, 0);
        assert(tx >= 0 && ty >= 0);
        tile_size = dim3(tx, ty);
    }
    dim3 n_tiles(1 + w/tile_size.x, 1 + h/tile_size.y);

    printf(
        "Matrix size: %zux%zu, Threads: %ux%u, Blocks: %ux%u\n", 
        w, h, tile_size.x, tile_size.y, n_tiles.x, n_tiles.y
    );

    ChronoGPU chrGPU;
    ChronoCPU chrCPU;

    std::vector<float> downloaded_dev_a(w*h);
    chrCPU.start();
    std::vector<float> a(w*h);
    std::vector<float> b(w*h);
    chrCPU.stop();
    printTimeMs("CPU alloc (2 matrices)", chrCPU.elapsedTime());

    float* dev_a = NULL;
    float* dev_b = NULL;
    chrGPU.start();
    cudaError status_a = cudaMalloc((void**) &dev_a, nbytes);
    cudaError status_b = cudaMalloc((void**) &dev_b, nbytes);
    chrGPU.stop();
    printTimeMs("GPU alloc (2 matrices)", chrGPU.elapsedTime());
    assert(status_a == cudaSuccess);
    assert(status_b == cudaSuccess);

    srand(time(NULL));

    for(size_t y=0 ; y<h ; ++y) {
        for(size_t x=0 ; x<w ; ++x) {
            size_t i = y*w + x;
            a[i] = (rand()%100) / 100.f;
            b[i] = (rand()%100) / 100.f;
        }
    }

    chrGPU.start();
    cudaMemcpy(dev_a, a.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), nbytes, cudaMemcpyHostToDevice);
    chrGPU.stop();
    printTimeMs("GPU upload (2 matrices)", chrGPU.elapsedTime());

    chrCPU.start();
    addMatricesCPU(a.data(), b.data(), w, h);
    chrCPU.stop();
    printTimeMs("CPU addMatrices", chrCPU.elapsedTime());

    chrGPU.start();
    addMatricesCUDA<<<n_tiles, tile_size>>>(dev_a, dev_b, w, h);
    chrGPU.stop();
    printTimeMs("GPU addMatrices", chrGPU.elapsedTime());

    chrGPU.start();
    cudaMemcpy(downloaded_dev_a.data(), dev_a, nbytes, cudaMemcpyDeviceToHost);
    chrGPU.stop();
    printTimeMs("GPU download (1 matrix)", chrGPU.elapsedTime());

    cudaFree(dev_a);
    cudaFree(dev_b);

    if(!eqMatrices(a.data(), downloaded_dev_a.data(), w, h)) {
        puts("Failure: matrices don't match!");
        return EXIT_FAILURE;
    }
    
    puts("Success: matrices do match!");
    return EXIT_SUCCESS;
}