#include <rgbhsv.hpp>
#include <algorithm>
#include <utility>
#include <math.h>

Rgb24::Rgb24(uint8_t r, uint8_t g, uint8_t b): r(r), g(g), b(b) {}
Rgb3f::Rgb3f(float r, float g, float b): r(r), g(g), b(b) {}
Hsv::Hsv(float h, float s, float v): h(h), s(s), v(v) {}

Rgb24::Rgb24(const Rgb3f& c):
    r(c.r * 255),
    g(c.g * 255),
    b(c.b * 255)
{}
Rgb3f::Rgb3f(const Rgb24& c):
    r(c.r / 255.f),
    g(c.g / 255.f),
    b(c.b / 255.f)
{}

Hsv::Hsv(const Rgb24& rgb): Hsv(Rgb3f(rgb)) {}
Rgb24::Rgb24(const Hsv& hsv): Rgb24(Rgb3f(hsv)) {}


Rgb3f::Rgb3f(const Hsv& hsv) {
    const float h = hsv.h, s = hsv.s, v = hsv.v;
    const float hp = h / 60;
    const float c = v * s; // chroma
    const float x = c * (1 - fabsf(fmodf(hp, 2) - 1));
    switch((int)hp) {
    case 0: r = c, g = x, b = 0; break;
    case 1: r = x, g = c, b = 0; break;
    case 2: r = 0, g = c, b = x; break;
    case 3: r = 0, g = x, b = c; break;
    case 4: r = x, g = 0, b = c; break;
    case 5: r = c, g = 0, b = x; break;
    default: r = g = b = 0; break;
    }
    const float m = v - c;
    r += m, g += m, b += m;
}

Hsv::Hsv(const Rgb3f& rgb) {
    using namespace std;
    const float r = rgb.r, g = rgb.g, b = rgb.b;
    const float cmax = max(max(r, g), b);
    const float cmin = min(min(r, g), b);
    const float delta = cmax - cmin;
    v = cmax;

    static const float EPSILON = 0.0001f;
    
    if(delta <= EPSILON || cmax <= EPSILON) {
        h = 0, s = 0;
        return;
    }

    s = delta / cmax;

         if(r >= cmax) h = 0 + (g-b) / delta;
    else if(g >= cmax) h = 2 + (b-r) / delta;
    else               h = 4 + (r-g) / delta;

    if(h < 0)
        h += 6;

    h *= 60;
}

