#include <tone_map.hpp>
#include <rgbhsv.hpp>
#include <ChronoGPU.hpp>
#include <ScopedChrono.hpp>

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <math_functions.h>


#define L TONEMAP_LEVELS

__global__ static void tone_map_gpu(
          float* __restrict__ dev_dst, uint32_t dev_dst_pitch,
    const float* __restrict__ dev_src, uint32_t dev_src_pitch,
    uint32_t w, uint32_t h
) {
    for(uint y = blockIdx.y * blockDim.y + threadIdx.y ; y < h ; y += gridDim.y * blockDim.y) {
        for(uint x = blockIdx.x * blockDim.x + threadIdx.x ; x < w ; x += gridDim.x * blockDim.x) {
            // TODO
        }
    }
}

static texture<uchar3, cudaTextureType2D, cudaReadModeElementType> dev_src_tex;

// NOTE: The input parameter is actually dev_src_tex
__global__ static void rgb_to_hsv_gpu(
    float* __restrict__ dev_hue, uint32_t dev_hue_pitch,
    float* __restrict__ dev_sat, uint32_t dev_sat_pitch,
    float* __restrict__ dev_val, uint32_t dev_val_pitch,
    uint32_t w, uint32_t h
) {
    for(uint y = blockIdx.y * blockDim.y + threadIdx.y ; y < h ; y += gridDim.y * blockDim.y) {
        for(uint x = blockIdx.x * blockDim.x + threadIdx.x ; x < w ; x += gridDim.x * blockDim.x) {
            // TODO: fetch from dev_src_tex
        }
    }
}

__global__ static void hsv_to_rgb_gpu(
         uchar3* __restrict__ dev_rgb, uint32_t dev_rgb_pitch,
    const float* __restrict__ dev_hue, uint32_t dev_hue_pitch,
    const float* __restrict__ dev_sat, uint32_t dev_sat_pitch,
    const float* __restrict__ dev_val, uint32_t dev_val_pitch,
    uint32_t w, uint32_t h
) {
    for(uint y = blockIdx.y * blockDim.y + threadIdx.y ; y < h ; y += gridDim.y * blockDim.y) {
        for(uint x = blockIdx.x * blockDim.x + threadIdx.x ; x < w ; x += gridDim.x * blockDim.x) {
            // TODO
        }
    }
}

typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

void tone_map_gpu_rgb(Rgb24* __restrict__ host_dst, const Rgb24* __restrict__ host_src, uint32_t w, uint32_t h) {

    assert(sizeof(Rgb24) == sizeof(uchar3));

    // TODO: Spare some memory!

    uchar3* dev_src = NULL;
    uchar3* dev_dst = NULL;
    float* dev_hue = NULL;
    float* dev_sat = NULL;
    float* dev_src_val = NULL;
    float* dev_dst_val = NULL;

    size_t dev_src_pitch = 0;
    size_t dev_dst_pitch = 0;
    size_t dev_hue_pitch = 0;
    size_t dev_sat_pitch = 0;
    size_t dev_src_val_pitch = 0;
    size_t dev_dst_val_pitch = 0;

    cudaMallocPitch(&dev_src, &dev_src_pitch, w * sizeof dev_src[0], h);
    cudaMallocPitch(&dev_dst, &dev_dst_pitch, w * sizeof dev_dst[0], h);
    cudaMallocPitch(&dev_hue, &dev_hue_pitch, w * sizeof dev_hue[0], h);
    cudaMallocPitch(&dev_sat, &dev_sat_pitch, w * sizeof dev_sat[0], h);
    cudaMallocPitch(&dev_src_val, &dev_src_val_pitch, w * sizeof dev_src_val[0], h);
    cudaMallocPitch(&dev_dst_val, &dev_dst_val_pitch, w * sizeof dev_dst_val[0], h);

    dev_src_tex.normalized = false;
    dev_src_tex.filterMode = cudaFilterModePoint;
    dev_src_tex.addressMode[0] = cudaAddressModeClamp;
    dev_src_tex.addressMode[1] = cudaAddressModeClamp;
    dev_src_tex.addressMode[2] = cudaAddressModeClamp;
    cudaMemcpy2D(dev_src, dev_src_pitch, host_src, w * sizeof dev_src[0], w * sizeof dev_src[0], h, cudaMemcpyHostToDevice);
    cudaBindTexture2D(NULL, dev_src_tex, dev_src, w, h, dev_src_pitch);

    // 16*16 = 256 threads/tile
    // 32*32 = 1024 threads/tile
    const dim3 n_threads(32, 32);
    const dim3 n_blocks(
        (w + n_threads.x - 1) / n_threads.x,
        (h + n_threads.y - 1) / n_threads.y
    );

    // STEPS:
    // malloc-memset hist;
    //
    // per-pixel: RGB -> HSV
    // per-pixel: atomicInc(&hist[pixel])
    // inclusive scan: for l in 0..L: cdf[l] = ...
    // per-pixel: dst_val[i] = tone_map(src_val[i]);
    // per-pixel: HSV -> RGB

    // tex: src_rgb
    // buf: dev_hue
    // buf: dev_sat
    // buf: dev_src_val
    // buf: dev_dst_val

    // i: src_tex_rgb
    // o: dev_hue
    // o: dev_sat
    // o: dev_src_val
    rgb_to_hsv_gpu<<<n_blocks, n_threads>>>(
        dev_hue, dev_hue_pitch,
        dev_sat, dev_sat_pitch,
        dev_src_val, dev_src_val_pitch,
        w, h
    );
    // i: dev_src_val
    // o: dev_dst_val
    tone_map_gpu<<<n_blocks, n_threads>>>(
        dev_dst_val, dev_dst_val_pitch,
        dev_src_val, dev_src_val_pitch,
        w, h
    );
    // i: dev_hue
    // i: dev_sat
    // i: dev_dst_val
    // o: dev_dst_rgb
    hsv_to_rgb_gpu<<<n_blocks, n_threads>>>(
        dev_dst, dev_dst_pitch,
        dev_hue, dev_hue_pitch,
        dev_sat, dev_sat_pitch,
        dev_dst_val, dev_dst_val_pitch,
        w, h
    );

    cudaMemcpy2D(host_dst, w * sizeof host_dst[0], dev_dst, dev_dst_pitch, w * sizeof dev_dst[0], h, cudaMemcpyDeviceToHost);

    cudaUnbindTexture(dev_src_tex);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(dev_hue);
    cudaFree(dev_sat);
    cudaFree(dev_src_val);
    cudaFree(dev_dst_val);

    // TODO compare images (host_dst and dst_cpu)
}
