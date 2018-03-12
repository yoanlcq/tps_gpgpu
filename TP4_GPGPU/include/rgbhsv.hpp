#pragma once

#include <stdint.h>

struct Hsv;

struct Rgb24 {
    uint8_t r, g, b;
    Rgb24(uint8_t r, uint8_t g, uint8_t b);
};

struct Rgb3f {
    float r, g, b;
    Rgb3f(float r, float g, float b);
    Rgb3f(const Hsv& hsv);
};

struct Hsv {
    float h, s, v;
    Hsv(float h, float s, float v);
    Hsv(const Rgb3f& rgb);
};

