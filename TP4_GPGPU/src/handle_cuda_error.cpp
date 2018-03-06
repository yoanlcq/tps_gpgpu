#include <handle_cuda_error.hpp>
#include <stdio.h>
#include <stdlib.h>

void handle_cuda_error_f(cudaError_t err, const char *file, unsigned line) {
    if(err == cudaSuccess)
        return;

    fprintf(stderr, "%s:%u: CUDA error : %s\n", file, line, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


