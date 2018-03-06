#pragma once

#include <stdint.h>

void tone_map_cpu_rgb(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, uint32_t w, uint32_t h);
void tone_map_gpu_rgb(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, uint32_t w, uint32_t h);
