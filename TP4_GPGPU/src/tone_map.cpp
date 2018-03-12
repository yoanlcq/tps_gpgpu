#include <tone_map.hpp>
#include <ChronoCPU.hpp>
#include <ScopedChrono.hpp>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <rgbhsv.hpp>

typedef ScopedChrono<ChronoCPU> ScopedChronoCPU;

static void HSVtoRGB(float& r, float& g, float& b, const float& h, const float& s, const float& v) {
    Rgb3f rgb(Hsv(h, s, v));
    r = rgb.r, g = rgb.g, b = rgb.b;
}
static void RGBtoHSV(const float& r, const float& g, const float& b, float& h, float& s, float& v) {
    Hsv hsv(Rgb3f(r, g, b));
    h = hsv.h, s = hsv.s, v = hsv.v;
}

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
    const uint8_t* __restrict__ rgb, 
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        float r, g, b, h, s, v;
        r = rgb[3*i+0] / 255.f;
        g = rgb[3*i+1] / 255.f;
        b = rgb[3*i+2] / 255.f;
        RGBtoHSV(r, g, b, h, s, v);
        hue[i] = h;
        sat[i] = s;
        val[i] = v;
    }
}

static void hsv_to_rgb_cpu(
    uint8_t* __restrict__ rgb,
    const float* __restrict__ hue,
    const float* __restrict__ sat,
    const float* __restrict__ val,
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        float r, g, b, h, s, v;
        h = hue[i];
        s = sat[i];
        v = val[i];
        HSVtoRGB(r, g, b, h, s, v);
        rgb[3*i+0] = r * 255.f;
        rgb[3*i+1] = g * 255.f;
        rgb[3*i+2] = b * 255.f;
    }
}

void tone_map_cpu_rgb(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, uint32_t w, uint32_t h) {
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
