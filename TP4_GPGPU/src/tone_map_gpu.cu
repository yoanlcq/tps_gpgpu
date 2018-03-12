#include <tone_map.hpp>
#include <rgbhsv.hpp>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include <ChronoGPU.hpp>
#include <ScopedChrono.hpp>

typedef ScopedChrono<ChronoGPU> ScopedChronoGPU;

void tone_map_gpu_rgb(Rgb24* __restrict__ dst, const Rgb24* __restrict__ src, uint32_t w, uint32_t h) {
}
