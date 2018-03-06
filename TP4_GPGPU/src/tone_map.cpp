#include <tone_map.hpp>
#include <ChronoCPU.hpp>
#include <ScopedChrono.hpp>
#include <hsvrgb.hpp>
#include <stdint.h>
#include <vector>

typedef ScopedChrono<ChronoCPU> ScopedChronoCPU;

static const uint32_t L = 256;

// Here, each byte is the V in HSV.
// H and S are kept aside by the caller.
static void tone_map_cpu_hsv_v(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    uint32_t w, uint32_t h
) {
    // Histogramme
    uint32_t hist[L];
    for(uint32_t i = 0 ; i < w*h ; ++i) {
        hist[src[i]] += 1;
    }

    // Répartition
    uint32_t r[L];
    uint32_t sum = 0;
    for(uint32_t l = 0 ; l < L ; ++l) {
        sum += hist[l];
        r[l] = sum;
    }

    // Egalisation
    // TODO: Transformation (valeurs): T(x) = r(x) * (L-1)/L
    for(uint32_t i = 0 ; i < w*h ; ++i) {
        dst[i] = r[src[i]] * (L-1) / float(L);
    }
}

static void rgb_to_hsv_cpu(
    uint8_t* __restrict__ hue,
    uint8_t* __restrict__ sat,
    uint8_t* __restrict__ val,
    const uint8_t* __restrict__ rgb, 
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        float r, g, b, h, s, v;
        r = rgb[3*i+0] / 255.f;
        g = rgb[3*i+1] / 255.f;
        b = rgb[3*i+2] / 255.f;
        RGBtoHSV(r, g, b, h, s, v);
        hue[i] = (h / 360.f) * 255;
        sat[i] = s * 255;
        val[i] = v * 255;
    }
}

static void hsv_to_rgb_cpu(
    uint8_t* __restrict__ rgb,
    const uint8_t* __restrict__ hue,
    const uint8_t* __restrict__ sat,
    const uint8_t* __restrict__ val,
    uint32_t w, uint32_t h
) {
    for(uint32_t i=0 ; i<w*h ; ++i) {
        float r, g, b, h, s, v;
        h = (hue[i] / 255.f) * 360.f;
        s = sat[i] / 255.f;
        v = val[i] / 255.f;
        HSVtoRGB(r, g, b, h, s, v);
        rgb[3*i+0] = r * 255.f;
        rgb[3*i+1] = g * 255.f;
        rgb[3*i+2] = b * 255.f;
    }
}

void tone_map_cpu_rgb(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, uint32_t w, uint32_t h) {
    std::vector<uint8_t> hue(w*h), sat(w*h), src_v(w*h), dst_v(w*h);
    rgb_to_hsv_cpu(hue.data(), sat.data(), src_v.data(), src, w, h);
    tone_map_cpu_hsv_v(dst_v.data(), src_v.data(), w, h);
    hsv_to_rgb_cpu(dst, hue.data(), sat.data(), dst_v.data(), w, h);
}
