#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <lodepng.h>
#include <tone_map.hpp>
#include <handle_cuda_error.hpp>
#include <iostream> // sync_with_stdio()

static void usage(FILE* f, const char* exe) {
    fprintf(f, "Usage: %s <image>\n", exe);
}

int main(int argc, char *argv[]) {

    std::ios::sync_with_stdio(true);

    if(argc < 2) {
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    const LodePNGColorType lct = LCT_RGB;
    const char* lct_str = "RGB";

    const char* path = argv[1];
    std::vector<uint8_t> img;
    uint32_t w, h;

    unsigned err = lodepng::decode(img, w, h, path, lct);
    if(err) {
        fprintf(stderr, "Error: lodepng::decode: %s\n", lodepng_error_text(err));
        return EXIT_FAILURE;
    }

    printf("Compiled with arch=%s, code=%s\n", GPU_ARCH, GPU_CODE);
    {
        cudaDeviceProp props;
        handle_cuda_error(cudaGetDeviceProperties(&props, 0));
        printf("Device: %s, CUDA %u.%u\n", props.name, props.major, props.minor);
    }
    printf("Using image `%s`, %ux%u %s\n", path, w, h, lct_str);

    std::vector<uint8_t> img_cpu(3*w*h);
    std::vector<uint8_t> img_gpu(3*w*h);
    tone_map_cpu_rgb((Rgb24*) img_cpu.data(), (Rgb24*) img.data(), w, h);
    tone_map_gpu_rgb((Rgb24*) img_gpu.data(), (Rgb24*) img.data(), w, h);

    const char *extension = 1 + strchr(path, '.');
    char* name = strndup(path, extension - 1 - path);
    char* path_cpu;
    char* path_gpu;
    asprintf(&path_cpu, "%s_CPU.%s", name, extension);
    asprintf(&path_gpu, "%s_GPU.%s", name, extension);
    free(name);

    int exit_status = EXIT_SUCCESS;

    printf("Saving CPU tone-mapped image as: %s\n", path_cpu);
	err = lodepng::encode(path_cpu, img_cpu.data(), w, h, lct);
    if(err) {
        fprintf(stderr, "Error: lodepng::encode: %s\n", lodepng_error_text(err));
        exit_status = EXIT_FAILURE;
    }

    printf("Saving GPU tone-mapped image as: %s\n", path_gpu);
	err = lodepng::encode(path_gpu, img_gpu.data(), w, h, lct);
    if(err) {
        fprintf(stderr, "Error: lodepng::encode: %s\n", lodepng_error_text(err));
        exit_status = EXIT_FAILURE;
    }

    free(path_cpu);
    free(path_gpu);

    return exit_status;
}
