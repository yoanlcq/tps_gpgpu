#include <tone_map.hpp>
#include <rgbhsv.hpp>
#include <handle_cuda_error.hpp>
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

// TODO 
// - GPU: Profiler et sauver les résultats pour le rapport !
// - GPU: Once it works:
//   - Use a texture;
//   - Use constant or shared memory;
//   - Minimize atomicAdd()s;

#define L TONEMAP_LEVELS

__global__ static void rgb_to_hsv_then_put_in_histogram(
          uint32_t* const __restrict__ dev_hist,
          float*    const __restrict__ dev_hue,
          float*    const __restrict__ dev_sat,
          float*    const __restrict__ dev_val,
    const uchar3*   const __restrict__ dev_rgb,
    const uint32_t w, const uint32_t h
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t i = y * w + x;

    if(x >= w || y >= h)
        return;

    const uchar3 rgb = dev_rgb[i];
    const float r = rgb.x / 255.f;
    const float g = rgb.y / 255.f;
    const float b = rgb.z / 255.f;
    const float cmax = fmaxf(fmaxf(r, g), b);
    const float cmin = fminf(fminf(r, g), b);
    const float delta = cmax - cmin;
    const float EPSILON = 0.0001f;

    if(delta <= EPSILON || cmax <= EPSILON) {
        dev_hue[i] = 0.f;
        dev_sat[i] = 0.f;
    } else {
        float hue = 0.f;

             if(r >= cmax) hue = 0 + (g-b) / delta;
        else if(g >= cmax) hue = 2 + (b-r) / delta;
        else               hue = 4 + (r-g) / delta;

        if(hue < 0)
            hue += 6;

        hue *= 60;
        dev_hue[i] = hue;
        dev_sat[i] = delta / cmax;
    }

    const float val = cmax;
    dev_val[i] = val;

    const uint32_t l = val * 255;
    atomicAdd(&dev_hist[l], 1);
}

// CDF = Cumulative Distribution Function
__global__ static void generate_cdf_via_inclusive_scan_histogram(
          uint32_t* const __restrict__ dev_cdf, 
    const uint32_t* const __restrict__ dev_hist
) {
    // Slide 20 of
    // http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf

    // Assume L/2 threads, and only 1 block
    __shared__ uint32_t shared_cdf[L];
    shared_cdf[threadIdx.x*2 + 0] = dev_hist[threadIdx.x*2 + 0];
    shared_cdf[threadIdx.x*2 + 1] = dev_hist[threadIdx.x*2 + 1];
    __syncthreads();

    // Reduction step
    for(uint32_t stride=1 ; stride <= L/2 ; stride *= 2) {
        const uint32_t i = (threadIdx.x+1) * stride * 2 - 1;
        if(i < L) {
            shared_cdf[i] += shared_cdf[i - stride];
        }
        __syncthreads();
    }
    // Post scan step
    for(int32_t stride=L/4 ; stride > 0 ; stride /= 2) {
        const uint32_t i = (threadIdx.x+1) * stride * 2 - 1;
        if(i + stride < L) {
            shared_cdf[i + stride] += shared_cdf[i];
        }
        __syncthreads();
    }
    dev_cdf[threadIdx.x*2 + 0] = shared_cdf[threadIdx.x*2 + 0];
    dev_cdf[threadIdx.x*2 + 1] = shared_cdf[threadIdx.x*2 + 1];
}

__global__ static void tone_map_then_hsv_to_rgb(
         uchar3*    const __restrict__ dev_rgb,
    const float*    const __restrict__ dev_hue,
    const float*    const __restrict__ dev_sat,
    const float*    const __restrict__ dev_val, 
    const uint32_t* const __restrict__ dev_cdf,
    const uint32_t w, const uint32_t h
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t i = y * w + x;

    if(x >= w || y >= h)
        return;

    const uint32_t l = dev_val[i] * (L-1);
    const float val = (dev_cdf[l] - dev_cdf[0]) / float(w*h);
    const float hue = dev_hue[i];
    const float sat = dev_sat[i];

    const float hp = hue / 60;
    const float c = val * sat; // chroma
    const float X = c * (1 - fabsf(fmodf(hp, 2) - 1));
    float r, g, b;
    switch((int)hp) {
    case 0: r = c, g = X, b = 0; break;
    case 1: r = X, g = c, b = 0; break;
    case 2: r = 0, g = c, b = X; break;
    case 3: r = 0, g = X, b = c; break;
    case 4: r = X, g = 0, b = c; break;
    case 5: r = c, g = 0, b = X; break;
    default: r = g = b = 0; break;
    }
    const float m = val - c;
    r += m, g += m, b += m;
    dev_rgb[i] = make_uchar3(r * 255, g * 255, b * 255);
}

__global__ static void sanity_check_kernel() {}

typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

void tone_map_gpu_rgb(Rgb24* __restrict__ host_dst, const Rgb24* __restrict__ host_src, uint32_t w, uint32_t h) {

    // Vérifier que l'architecture du code est compatible avec ce PC. Ca m'avait silencieusement trahi.
    sanity_check_kernel<<<1,1>>>();
    handle_cuda_error(cudaGetLastError());

    uchar3*   dev_rgb = NULL;
    float*    dev_hue = NULL;
    float*    dev_sat = NULL;
    float*    dev_val = NULL;
    uint32_t* dev_hist = NULL;
    uint32_t* dev_cdf = NULL;

    {
        char txt[128];
        uint32_t total_bytes = w * h * (3+4+4+4) + L * (4+4);
        snprintf(txt, sizeof txt, "GPU: Allocating %u bytes (~%u MiB)",
            total_bytes, total_bytes / (1024 * 1024)
        );
        ScopedChronoGPU chr(txt);
        handle_cuda_error(cudaMalloc(&dev_rgb, w * h * sizeof dev_rgb[0]));
        handle_cuda_error(cudaMalloc(&dev_hue, w * h * sizeof dev_hue[0]));
        handle_cuda_error(cudaMalloc(&dev_sat, w * h * sizeof dev_sat[0]));
        handle_cuda_error(cudaMalloc(&dev_val, w * h * sizeof dev_val[0]));
        handle_cuda_error(cudaMalloc(&dev_hist, L * sizeof dev_hist[0]));
        handle_cuda_error(cudaMalloc(&dev_cdf, L * sizeof dev_cdf[0]));

        handle_cuda_error(cudaMemset(dev_hist, 0, L * sizeof dev_hist[0]));
    }

    {
        ScopedChronoGPU chr("GPU: Uploading RGB data");
        assert(sizeof(host_src[0]) == sizeof(dev_rgb[0]));
        handle_cuda_error(cudaMemcpy(dev_rgb, host_src, w * h * sizeof dev_rgb[0], cudaMemcpyHostToDevice));
    }

    // 16*16 = 256 threads/tile
    // 32*32 = 1024 threads/tile
    const dim3 img_threads(32, 32);
    const dim3 img_blocks(
        (w + img_threads.x - 1) / img_threads.x,
        (h + img_threads.y - 1) / img_threads.y
    );

#ifdef NDEBUG
#define check_kernel_error()
#else
#define check_kernel_error() handle_cuda_error(cudaGetLastError())
#endif

    {
        ScopedChronoGPU chr("GPU: RGB to HSV, then incrementing histogram slots");
        rgb_to_hsv_then_put_in_histogram<<<img_blocks, img_threads>>>(
            dev_hist, dev_hue, dev_sat, dev_val, dev_rgb, w, h
        );
    }
    check_kernel_error();

    {
        ScopedChronoGPU chr("GPU: Generating CDF via inclusive scan of histogram");
        generate_cdf_via_inclusive_scan_histogram<<<1, L/2>>>(
            dev_cdf, dev_hist
        );
    }
    check_kernel_error();

    {
        ScopedChronoGPU chr("GPU: Tone mapping, then HSV to RGB");
        tone_map_then_hsv_to_rgb<<<img_blocks, img_threads>>>(
            dev_rgb, dev_hue, dev_sat, dev_val, dev_cdf, w, h
        );
    }
    check_kernel_error();

    {
        ScopedChronoGPU chr("GPU: Downloading RGB data");
        assert(sizeof(host_dst[0]) == sizeof(dev_rgb[0]));
        handle_cuda_error(cudaMemcpy(host_dst, dev_rgb, w * h * sizeof dev_rgb[0], cudaMemcpyDeviceToHost));
    }

    {
        ScopedChronoGPU chr("GPU: Freeing memory");
        handle_cuda_error(cudaFree(dev_rgb));
        handle_cuda_error(cudaFree(dev_hue));
        handle_cuda_error(cudaFree(dev_sat));
        handle_cuda_error(cudaFree(dev_val));
        handle_cuda_error(cudaFree(dev_hist));
        handle_cuda_error(cudaFree(dev_cdf));
    }
}

#if 0 // code de test
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
#endif
