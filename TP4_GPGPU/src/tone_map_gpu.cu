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
//   - Use a 2D texture for input dev_rgb (gain spatial locality?);
//   - Minimize atomicAdd()s by accumulating in local and shared memory;
//   - Once filled, put dev_cdf into constant memory.

// NOTE: scores pour image/Chateau.png:
//
// GeForce GT 635M (laptop, Fermi architecture, arch=compute_20, code=sm_20):
// GPU: Allocating 73327298 bytes (~69 MiB): 0.658176 ms
// GPU: Uploading RGB data: 5.834688 ms
// GPU: RGB to HSV: 11.987400 ms (average of 100 invocations)
// GPU: Histogram with per-pixel global atomicAdd(): 14.994180 ms (average of 100 invocations)
// GPU: Histogram with shared mem atomicAdd(): 20.790686 ms (average of 100 invocations)
// GPU: Generate CDF via inclusive scan of histogram: 0.009258 ms (average of 100 invocations)
// GPU: Tone map, then HSV to RGB: 11.447376 ms (average of 100 invocations)
// GPU: Downloading RGB data: 4.783200 ms
// GPU: Freeing memory: 0.439808 ms
//
// Quadro K620 (salle 0B002, Maxwell architecture, arch=compute_50, code=sm_50):
// GPU: Allocating 73327298 bytes (~69 MiB): 0.441376 ms
// GPU: Uploading RGB data: 2.450944 ms
// GPU: RGB to HSV: 4.085881 ms (average of 100 invocations)
// GPU: Histogram with per-pixel global atomicAdd(): 5.294244 ms (average of 100 invocations)
// GPU: Histogram with shared mem atomicAdd(): 1.468803 ms (average of 100 invocations)
// GPU: Generate CDF via inclusive scan of histogram: 0.007121 ms (average of 100 invocations)
// GPU: Tone map, then HSV to RGB: 4.288925 ms (average of 100 invocations)
// GPU: Downloading RGB data: 2.559008 ms
// GPU: Freeing memory: 0.358400 ms


#define L TONEMAP_LEVELS

__global__ static void rgb_to_hsv(
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
    dev_val[i] = cmax;
}

__global__ static void generate_histogram_simple(
       uint32_t* const __restrict__ dev_hist, 
    const float* const __restrict__ dev_val,
    const uint32_t w, const uint32_t h
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t i = y * w + x;

    if(x >= w || y >= h)
        return;

    const uint32_t l = dev_val[i] * 255;
    atomicAdd(&dev_hist[l], 1);
}


__global__ static void generate_histogram_smarter(
       uint32_t* const __restrict__ dev_hist, 
    const float* const __restrict__ dev_val,
    const uint32_t w, const uint32_t h
) {
    __shared__ uint32_t shared_hist[L];
    for(uint32_t l = threadIdx.x ; l < L ; l += blockDim.x) {
        shared_hist[l] = 0;
    }
    __syncthreads();
    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x ; i < w*h ; i += blockDim.x * gridDim.x) {
        const uint32_t l = dev_val[i] * 255;
        atomicAdd(&shared_hist[l], 1);
    }
    __syncthreads();
    for(uint32_t l = threadIdx.x ; l < L ; l += blockDim.x) {
        atomicAdd(&dev_hist[l], shared_hist[l]);
    }
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

typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;


struct ToneMapGpu {
    const uint32_t w, h;
    const dim3 img_threads;
    const dim3 img_blocks;

    uchar3*   dev_rgb;
    float*    dev_hue;
    float*    dev_sat;
    float*    dev_val;
    uint32_t* dev_hist;
    uint32_t* dev_cdf;

    enum Kernel {
        KERNEL_RGB_TO_HSV = 0,
        KERNEL_HISTOGRAM_SIMPLE,
        KERNEL_HISTOGRAM_SMARTER,
        KERNEL_CDF_VIA_INCLUSIVE_SCAN,
        KERNEL_TONE_MAP_THEN_HSV_TO_RGB,
        KERNEL_COUNT,
    };

    ToneMapGpu(uint32_t w, uint32_t h);
    ~ToneMapGpu();
    void upload_rgb(const Rgb24* host_src);
    void invoke_kernels(uint32_t nb_loops);
    void download_rgb(Rgb24* host_dst) const;

private:
    template<Kernel> const char* get_kernel_name();
    template<Kernel> void run_kernel_prerequisites();
    template<Kernel> void do_invoke_kernel();
    template<Kernel> void invoke_kernel(uint32_t nb_loops);

    ToneMapGpu(const ToneMapGpu&);
    ToneMapGpu& operator=(const ToneMapGpu&);
};


ToneMapGpu::ToneMapGpu(uint32_t w, uint32_t h):
    w(w), 
    h(h),
    // 16*16 = 256 threads/tile
    // 32*32 = 1024 threads/tile
    img_threads(32, 32),
    img_blocks(
        (w + img_threads.x - 1) / img_threads.x,
        (h + img_threads.y - 1) / img_threads.y
    )
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
}

void ToneMapGpu::upload_rgb(const Rgb24* host_src) {
    ScopedChronoGPU chr("GPU: Uploading RGB data");
    assert(sizeof(host_src[0]) == sizeof(dev_rgb[0]));
    handle_cuda_error(cudaMemcpy(dev_rgb, host_src, w * h * sizeof dev_rgb[0], cudaMemcpyHostToDevice));
}

void ToneMapGpu::download_rgb(Rgb24* host_dst) const {
    ScopedChronoGPU chr("GPU: Downloading RGB data");
    assert(sizeof(host_dst[0]) == sizeof(dev_rgb[0]));
    handle_cuda_error(cudaMemcpy(host_dst, dev_rgb, w * h * sizeof dev_rgb[0], cudaMemcpyDeviceToHost));
}

ToneMapGpu::~ToneMapGpu() {
    ScopedChronoGPU chr("GPU: Freeing memory");
    handle_cuda_error(cudaFree(dev_rgb));
    handle_cuda_error(cudaFree(dev_hue));
    handle_cuda_error(cudaFree(dev_sat));
    handle_cuda_error(cudaFree(dev_val));
    handle_cuda_error(cudaFree(dev_hist));
    handle_cuda_error(cudaFree(dev_cdf));
}

template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::run_kernel_prerequisites() {
    switch(kernel) {
    case KERNEL_HISTOGRAM_SIMPLE:
    case KERNEL_HISTOGRAM_SMARTER:
        handle_cuda_error(cudaMemset(dev_hist, 0, L * sizeof dev_hist[0]));
        break;
    // Explicitly handle all other cases to remove warnings and be future-proof
    case KERNEL_RGB_TO_HSV:
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN:
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB:
        break;
    }
}

template<ToneMapGpu::Kernel kernel>
const char* ToneMapGpu::get_kernel_name() {
    switch(kernel) {
    case KERNEL_RGB_TO_HSV: return "RGB to HSV";
    case KERNEL_HISTOGRAM_SIMPLE: return "Histogram with per-pixel global atomicAdd()";
    case KERNEL_HISTOGRAM_SMARTER: return "Histogram with shared mem atomicAdd()";
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN: return "Generate CDF via inclusive scan of histogram";
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB: return "Tone map, then HSV to RGB";
    }
    return NULL;
}

template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::do_invoke_kernel() {
    switch(kernel) {
    case KERNEL_RGB_TO_HSV:
        rgb_to_hsv<<<img_blocks, img_threads>>>(dev_hue, dev_sat, dev_val, dev_rgb, w, h);
        break;
    case KERNEL_HISTOGRAM_SIMPLE:
        generate_histogram_simple<<<img_blocks, img_threads>>>(dev_hist, dev_val, w, h);
        break;
    case KERNEL_HISTOGRAM_SMARTER:
        generate_histogram_smarter<<<(w*h + 1024 - 1) / 1024, 1024>>>(dev_hist, dev_val, w, h);
        break;
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN:
        generate_cdf_via_inclusive_scan_histogram<<<1, L/2>>>(dev_cdf, dev_hist);
        break;
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB:
        tone_map_then_hsv_to_rgb<<<img_blocks, img_threads>>>(dev_rgb, dev_hue, dev_sat, dev_val, dev_cdf, w, h);
        break;
    }
}

template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::invoke_kernel(uint32_t nb_loops) {
    ChronoGPU chr;
    double time_accum = 0;
    for(uint32_t i=0 ; i<nb_loops ; ++i) {
        run_kernel_prerequisites<kernel>();
        chr.start();
        do_invoke_kernel<kernel>();
        chr.stop();
        handle_cuda_error(cudaGetLastError());
        time_accum += chr.elapsedTime();
    }
    printf("GPU: %s: %lf ms (average of %u invocations)\n",
        get_kernel_name<kernel>(), time_accum / nb_loops, nb_loops
    );
}

void ToneMapGpu::invoke_kernels(uint32_t nb_loops) {
    invoke_kernel<KERNEL_RGB_TO_HSV>(nb_loops);
    invoke_kernel<KERNEL_HISTOGRAM_SIMPLE>(nb_loops);
    invoke_kernel<KERNEL_HISTOGRAM_SMARTER>(nb_loops);
    invoke_kernel<KERNEL_CDF_VIA_INCLUSIVE_SCAN>(nb_loops);
    invoke_kernel<KERNEL_TONE_MAP_THEN_HSV_TO_RGB>(nb_loops);
}


__global__ static void sanity_check_kernel() {}

void tone_map_gpu_rgb(Rgb24* __restrict__ host_dst, const Rgb24* __restrict__ host_src, uint32_t w, uint32_t h) {

    // Vérifier que l'architecture du code est compatible avec ce PC. Ca m'avait silencieusement trahi.
    sanity_check_kernel<<<1,1>>>();
    handle_cuda_error(cudaGetLastError());

    ToneMapGpu gpu(w, h);
    gpu.upload_rgb(host_src);
    gpu.invoke_kernels(100);
    gpu.download_rgb(host_dst);
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
