#include <tone_map.hpp>
#include <ChronoCPU.hpp>
#include <ScopedChrono.hpp>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <rgbhsv.hpp>

static const uint32_t L = 256;

static void tone_map_cpu(
    float* __restrict__ dst,
    const float* __restrict__ src,
    uint32_t w, uint32_t h
) {
    uint32_t hist[L] = {};
    for(uint32_t i = 0 ; i < w*h ; ++i) {
        hist[(uint32_t) (src[i] * (L-1))] += 1;
    }

    uint32_t cdf[L]; // Cumulative Distribution Function
    uint32_t sum = 0;
    for(uint32_t l = 0 ; l < L ; ++l) {
        sum += hist[l];
        cdf[l] = sum;
        // printf("cdf[%u] = %u\n", l, cdf[l]);
    }

    for(uint32_t i = 0 ; i < w*h ; ++i) {
        dst[i] = (cdf[(uint32_t) (src[i] * (L-1))] - cdf[0]) / float(w*h-1);
        // printf("dst[%u] = %f\n", i, dst[i]);
    }
}

static void rgb_to_hsv_cpu(
    float* __restrict__ hue,
    float* __restrict__ sat,
    float* __restrict__ val,
    const Rgb24* __restrict__ rgb, 
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        Hsv hsv = Hsv(rgb[i]);
        hue[i] = hsv.h;
        sat[i] = hsv.s;
        val[i] = hsv.v;
    }
}

static void hsv_to_rgb_cpu(
    Rgb24* __restrict__ rgb,
    const float* __restrict__ hue,
    const float* __restrict__ sat,
    const float* __restrict__ val,
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        rgb[i] = Rgb24(Hsv(hue[i], sat[i], val[i]));
    }
}

typedef ScopedChrono<ChronoCPU> ScopedChronoCPU;

void tone_map_cpu_rgb(Rgb24* __restrict__ dst, const Rgb24* __restrict__ src, uint32_t w, uint32_t h) {
    unsigned total_bytes = 4*w*h*sizeof(float);
    printf("CPU: Allocating %u bytes (~%u MiB)\n", total_bytes, total_bytes / (1024 * 1024));
    std::vector<float> hue(w*h), sat(w*h), src_val(w*h), dst_val(w*h);
    {
        ScopedChronoCPU chr("CPU: RGB to HSV");
        rgb_to_hsv_cpu(hue.data(), sat.data(), src_val.data(), src, w, h);
    }
    {
        ScopedChronoCPU chr("CPU: Tone mapping");
        tone_map_cpu(dst_val.data(), src_val.data(), w, h);
    }
    {
        ScopedChronoCPU chr("CPU: HSV to RGB");
        hsv_to_rgb_cpu(dst, hue.data(), sat.data(), dst_val.data(), w, h);
    }
}
