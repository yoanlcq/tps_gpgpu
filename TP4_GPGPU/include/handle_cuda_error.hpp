#pragma once

#include <cuda_runtime.h>

void handle_cuda_error_f(cudaError_t err, const char *file, unsigned line);
#define handle_cuda_error(err) do { handle_cuda_error_f(err, __FILE__, __LINE__ ); } while(0)
