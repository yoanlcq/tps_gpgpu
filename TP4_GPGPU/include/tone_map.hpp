#pragma once

#include <stdint.h>
#include <rgbhsv.hpp>

extern const uint32_t TONEMAP_LEVELS;

void tone_map_cpu_rgb(Rgb24* __restrict__ dst, const Rgb24* __restrict__ src, uint32_t w, uint32_t h);
void tone_map_gpu_rgb(Rgb24* __restrict__ dst, const Rgb24* __restrict__ src, uint32_t w, uint32_t h);
