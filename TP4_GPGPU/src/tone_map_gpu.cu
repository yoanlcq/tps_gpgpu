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

#define L TONEMAP_LEVELS

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> tex_rgb4;

// The HSV to RGB kernel, which either fetches from tex_rgb4 or dev_rgb
// depending on the value of uses_texture.
// This one runs at one thread per pixel.
template<bool uses_texture>
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

    float r, g, b;
    if(uses_texture) { // template, compile-time
        const uchar4 rgb4 = tex2D(tex_rgb4, x + 0.5f, y + 0.5f); // Maybe + 0.5f isn't needed here
        r = rgb4.x / 255.f;
        g = rgb4.y / 255.f;
        b = rgb4.z / 255.f;
    } else {
        const uchar3 rgb = dev_rgb[i];
        r = rgb.x / 255.f;
        g = rgb.y / 255.f;
        b = rgb.z / 255.f;
    }
    
    // Doing fmaxf(..., EPSILON) is better than branching to avoid division by zero.
    // We gain 0.1 ms on Chateau.png, yay !
    const float EPSILON = 0.0001f;
    const float cmax = fmaxf(fmaxf(fmaxf(r, g), b), EPSILON);
    const float cmin = fminf(fminf(r, g), b);
    const float delta = fmaxf(cmax - cmin, EPSILON);

    dev_sat[i] = delta / cmax;
    dev_val[i] = cmax;

    float hue = 0.f;

         if(r >= cmax) hue = 0 + (g-b) / delta;
    else if(g >= cmax) hue = 2 + (b-r) / delta;
    else               hue = 4 + (r-g) / delta;

#if 1 // Actually faster
    if(hue < 0)
        hue += 6;
#else
    hue = fmodf(hue + 6, 6);
#endif

    dev_hue[i] = hue;
}

__global__ static void generate_histogram_per_pixel_global_atomicadd(
       uint32_t* const __restrict__ dev_hist, 
    const float* const __restrict__ dev_val,
    const uint32_t w, const uint32_t h
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t i = y * w + x;

    if(x >= w || y >= h)
        return;

    const uint32_t l = dev_val[i] * (L-1);
    atomicAdd(&dev_hist[l], 1);
}


// Runs with any number of threads and blocks, but in 1D.
__global__ static void generate_histogram_shared_mem_atomicadd(
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
        const uint32_t l = dev_val[i] * (L-1);
        atomicAdd(&shared_hist[l], 1);
    }
    __syncthreads();
    for(uint32_t l = threadIdx.x ; l < L ; l += blockDim.x) {
        atomicAdd(&dev_hist[l], shared_hist[l]);
    }
}


// CDF = Cumulative Distribution Function
// This kernel assumes L/2 threads, and only 1 block.
__global__ static void generate_cdf_via_inclusive_scan_histogram(
          uint32_t* const __restrict__ dev_cdf, 
    const uint32_t* const __restrict__ dev_hist
) {
    // Slide 20 of
    // http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf

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


__constant__ uint32_t cst_cdf[L];

// This kernel reads the CDF either from dev_cdf or cst_cdf depending on the value of uses_constant_mem_cdf.
template<bool uses_constant_mem_cdf>
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

    const uint32_t* const cdf = uses_constant_mem_cdf ? cst_cdf : dev_cdf; // template, compile-time
    const uint32_t l = dev_val[i] * (L-1);
    const float val = (cdf[l] - cdf[0]) / float(w*h);
    const float hue = dev_hue[i];
    const float sat = dev_sat[i];

    const float c = val * sat; // chroma
    const float X = c * (1 - fabsf(fmodf(hue, 2) - 1));

#if 1 // This one performs better
    float r, g, b;
    switch((int)hue) {
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
#else
    const float3 parts[6] = {
        make_float3(c, X, 0),
        make_float3(X, c, 0),
        make_float3(0, c, X),
        make_float3(0, X, c),
        make_float3(X, 0, c),
        make_float3(c, 0, X)
    };
    const float m = val - c;
    float3 rgb = parts[(int)hue];
    rgb.x += m;
    rgb.y += m;
    rgb.z += m;
    dev_rgb[i] = make_uchar3(rgb.x * 255, rgb.y * 255, rgb.z * 255);
#endif
}


typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;


struct KernelLaunchSettings {
    dim3 n_blocks, n_threads;
    KernelLaunchSettings(dim3 n_blocks, dim3 n_threads):
        n_blocks(n_blocks),
        n_threads(n_threads)
        {}
};

struct ToneMapGpu {
    cudaDeviceProp device_properties;
    const uint32_t w, h;

    uchar4*   dev_rgb4;
    size_t    dev_rgb4_pitch;
    uchar3*   dev_rgb;
    float*    dev_hue;
    float*    dev_sat;
    float*    dev_val;
    uint32_t* dev_hist;
    uint32_t* dev_cdf;

    enum Kernel {
        KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM = 0,
        KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D,
        KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD,
        KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD,
        KERNEL_CDF_VIA_INCLUSIVE_SCAN,
        KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM,
        KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM,
        // KERNEL_COUNT, // Not needed anymore
    };

    ToneMapGpu(uint32_t w, uint32_t h);
    ~ToneMapGpu();
    void upload_rgb(const Rgb24* host_src);
    void perform_tone_map(uint32_t nb_loops);
    void download_rgb(Rgb24* host_dst) const;

private:
    template<Kernel> KernelLaunchSettings get_kernel_settings() const;
    template<Kernel> static const char* get_kernel_name();
    template<Kernel> void reset_kernel_inputs_if_needed();
    template<Kernel> void invoke_kernel();
    template<Kernel> void invoke_kernel_n_times(uint32_t nb_loops);

    ToneMapGpu(const ToneMapGpu&);
    ToneMapGpu& operator=(const ToneMapGpu&);
};


ToneMapGpu::ToneMapGpu(uint32_t w, uint32_t h): w(w), h(h) {

    handle_cuda_error(cudaGetDeviceProperties(&device_properties, 0));

    tex_rgb4.normalized = false;
    tex_rgb4.filterMode = cudaFilterModePoint;
    tex_rgb4.addressMode[0] = cudaAddressModeClamp;
    tex_rgb4.addressMode[1] = cudaAddressModeClamp;
    tex_rgb4.addressMode[2] = cudaAddressModeClamp;

    char txt[128];
    uint32_t total_bytes = w * h * (4+3+4+4+4) + L * (4+4);
    snprintf(txt, sizeof txt, "GPU: Allocating %u bytes (~%u MiB)",
        total_bytes, total_bytes / (1024 * 1024)
    );

    ScopedChronoGPU chr(txt);
    handle_cuda_error(cudaMallocPitch(&dev_rgb4, &dev_rgb4_pitch, w * sizeof dev_rgb4[0], h));
    handle_cuda_error(cudaMalloc(&dev_rgb, w * h * sizeof dev_rgb[0]));
    handle_cuda_error(cudaMalloc(&dev_hue, w * h * sizeof dev_hue[0]));
    handle_cuda_error(cudaMalloc(&dev_sat, w * h * sizeof dev_sat[0]));
    handle_cuda_error(cudaMalloc(&dev_val, w * h * sizeof dev_val[0]));
    handle_cuda_error(cudaMalloc(&dev_hist, L * sizeof dev_hist[0]));
    handle_cuda_error(cudaMalloc(&dev_cdf, L * sizeof dev_cdf[0]));
}

void ToneMapGpu::upload_rgb(const Rgb24* host_src) {
    {
        ScopedChronoGPU chr("GPU: Uploading RGB data (1D, 3 components)");
        assert(sizeof(host_src[0]) == sizeof(dev_rgb[0]));
        handle_cuda_error(cudaMemcpy(dev_rgb, host_src, w * h * sizeof dev_rgb[0], cudaMemcpyHostToDevice));
    }

    // Let's pretend this was done as part of loading the image from the file.
    std::vector<uchar4> rgb4(w*h);
    for(uint32_t i = 0 ; i<w*h ; ++i) {
        const Rgb24& c = host_src[i];
        rgb4[i] = make_uchar4(c.r, c.g, c.b, 0);
    }

    ScopedChronoGPU chr("GPU: Uploading RGB data (2D, 3 components extended to 4)");
    handle_cuda_error(cudaMemcpy2D(dev_rgb4, dev_rgb4_pitch, rgb4.data(), w * sizeof rgb4[0], w * sizeof rgb4[0], h, cudaMemcpyHostToDevice));
}

void ToneMapGpu::download_rgb(Rgb24* host_dst) const {
    ScopedChronoGPU chr("GPU: Downloading RGB data");
    assert(sizeof(host_dst[0]) == sizeof(dev_rgb[0]));
    handle_cuda_error(cudaMemcpy(host_dst, dev_rgb, w * h * sizeof dev_rgb[0], cudaMemcpyDeviceToHost));
}

ToneMapGpu::~ToneMapGpu() {
    ScopedChronoGPU chr("GPU: Freeing memory");
    handle_cuda_error(cudaFree(dev_rgb4));
    handle_cuda_error(cudaFree(dev_rgb));
    handle_cuda_error(cudaFree(dev_hue));
    handle_cuda_error(cudaFree(dev_sat));
    handle_cuda_error(cudaFree(dev_val));
    handle_cuda_error(cudaFree(dev_hist));
    handle_cuda_error(cudaFree(dev_cdf));
}

template<ToneMapGpu::Kernel kernel>
const char* ToneMapGpu::get_kernel_name() {
    switch(kernel) {
    case KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM: return "RGB to HSV (RGB in global mem)";
    case KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D: return "RGB to HSV (RGB in 2D texture)";
    case KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD: return "Histogram via per-pixel global atomicAdd()";
    case KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD: return "Histogram using shared mem atomicAdd()";
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN: return "CDF via inclusive scan of histogram";
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM: return "Tone mapping, then HSV to RGB (CDF in global mem)";
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM: return "Tone mapping, then HSV to RGB (CDF in constant mem)";
    }
    return NULL;
}

template<ToneMapGpu::Kernel kernel>
KernelLaunchSettings ToneMapGpu::get_kernel_settings() const {

    const uint32_t nt = device_properties.maxThreadsPerBlock;
    const dim3 nt_2d(nt >> 5, nt >> 5);

    switch(kernel) {
    case KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM:
    case KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D:
    case KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD:
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM:
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM:
        // All kernels that use one thread per pixel
        return KernelLaunchSettings(dim3(
            (w + nt_2d.x - 1) / nt_2d.x,
            (h + nt_2d.y - 1) / nt_2d.y
        ), nt_2d);
    case KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD:
        return KernelLaunchSettings((w*h + nt - 1) / nt, nt);
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN:
        assert(L/2 <= nt); // XXX: If this triggers, we're in for a refactoring!
        return KernelLaunchSettings(1, L/2);
    }
    // Supposedly unreachable
    return KernelLaunchSettings(0,0);
}

template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::invoke_kernel() {
    const KernelLaunchSettings settings = get_kernel_settings<kernel>();
    const dim3 nb = settings.n_blocks;
    const dim3 nt = settings.n_threads;

    switch(kernel) {
    case KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM:
        rgb_to_hsv<false><<<nb, nt>>>(dev_hue, dev_sat, dev_val, dev_rgb, w, h);
        break;
    case KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D:
        rgb_to_hsv<true><<<nb, nt>>>(dev_hue, dev_sat, dev_val, dev_rgb, w, h);
        break;
    case KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD:
        generate_histogram_per_pixel_global_atomicadd<<<nb, nt>>>(dev_hist, dev_val, w, h);
        break;
    case KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD:
        generate_histogram_shared_mem_atomicadd<<<nb, nt>>>(dev_hist, dev_val, w, h);
        break;
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN:
        generate_cdf_via_inclusive_scan_histogram<<<nb, nt>>>(dev_cdf, dev_hist);
        break;
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM:
        tone_map_then_hsv_to_rgb<false><<<nb, nt>>>(dev_rgb, dev_hue, dev_sat, dev_val, dev_cdf, w, h);
        break;
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM:
        tone_map_then_hsv_to_rgb<true><<<nb, nt>>>(dev_rgb, dev_hue, dev_sat, dev_val, dev_cdf, w, h);
        break;
    }
}

// The histogram kernels give only correct results if the histogram is zeroed beforehand,
// but since we run kernel multiple times for profiling, we need to clear the
// histogram every time BUT without taking it into account when measuring.
template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::reset_kernel_inputs_if_needed() {
    switch(kernel) {
    case KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD:
    case KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD:
        handle_cuda_error(cudaMemset(dev_hist, 0, L * sizeof dev_hist[0]));
        break;
    // Explicitly handle all other cases to remove warnings and be future-proof
    case KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM:
    case KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D:
    case KERNEL_CDF_VIA_INCLUSIVE_SCAN:
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM:
    case KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM:
        break;
    }
}


template<ToneMapGpu::Kernel kernel>
void ToneMapGpu::invoke_kernel_n_times(uint32_t nb_loops) {
    ChronoGPU chr;
    double time_accum = 0;
    for(uint32_t i=0 ; i<nb_loops ; ++i) {
        reset_kernel_inputs_if_needed<kernel>();
        chr.start();
        invoke_kernel<kernel>();
        chr.stop();
        handle_cuda_error(cudaGetLastError());
        time_accum += chr.elapsedTime();
    }
    const KernelLaunchSettings settings = get_kernel_settings<kernel>();
    const dim3 nb = settings.n_blocks;
    const dim3 nt = settings.n_threads;
    printf("GPU: %s: %lf ms (%u times, %ux%u blocks of %ux%u threads)\n",
        get_kernel_name<kernel>(), time_accum / nb_loops, nb_loops,
        nb.x, nb.y,
        nt.x, nt.y
    );
}

void ToneMapGpu::perform_tone_map(uint32_t nb_loops) {
    // Most kernels overwrite results of previous ones, so if one of them
    // were to be incorrect, that could be shadowed by a correct one that runs next.
    // I'm leaving all of them here only so we can see differences in performance.

    // XXX Putting NULL as 1st argument isn't supposed to work, because memory was obtained by cudaMallocPitch().
    // Handling this would require too much refactoring for what it's worth.
    handle_cuda_error(cudaBindTexture2D(NULL, tex_rgb4, dev_rgb4, w, h, dev_rgb4_pitch));
    invoke_kernel_n_times<KERNEL_RGB_TO_HSV__RGB_IN_TEXTURE_2D>(nb_loops);
    handle_cuda_error(cudaUnbindTexture(tex_rgb4));

    invoke_kernel_n_times<KERNEL_RGB_TO_HSV__RGB_IN_GLOBAL_MEM>(nb_loops);

    invoke_kernel_n_times<KERNEL_HISTOGRAM_PER_PIXEL_GLOBAL_ATOMICADD>(nb_loops);
    invoke_kernel_n_times<KERNEL_HISTOGRAM_SHARED_MEM_ATOMICADD>(nb_loops);

    invoke_kernel_n_times<KERNEL_CDF_VIA_INCLUSIVE_SCAN>(nb_loops);
    handle_cuda_error(cudaMemcpyToSymbol(cst_cdf, dev_cdf, L * sizeof cst_cdf[0], 0, cudaMemcpyDeviceToDevice));

    invoke_kernel_n_times<KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_GLOBAL_MEM>(nb_loops);
    invoke_kernel_n_times<KERNEL_TONE_MAP_THEN_HSV_TO_RGB__CDF_IN_CONSTANT_MEM>(nb_loops);
}


__global__ static void sanity_check_kernel() {}

void tone_map_gpu_rgb(Rgb24* __restrict__ host_dst, const Rgb24* __restrict__ host_src, uint32_t w, uint32_t h) {
    // Ensure that kernels can run at all (happened to me on my laptop with the wrong arch/code settings).
    sanity_check_kernel<<<1,1>>>();
    handle_cuda_error(cudaGetLastError());

    ToneMapGpu gpu(w, h);
    gpu.upload_rgb(host_src);
    gpu.perform_tone_map(200);
    gpu.download_rgb(host_dst);
}

