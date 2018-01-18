#include <stdlib.h>
#include <stdio.h>

__global__ void square(float *d_out, float *d_in) {
    int i = threadIdx.x;
    float f = d_in[i];
    d_out[i] = f*f;
}

int main(int argc, char *argv[]) {
    const int ARRAY_COUNT = 64;

    float h_in[ARRAY_COUNT], h_out[ARRAY_COUNT];
    for(int i=0 ; i<ARRAY_COUNT ; ++i) {
        h_in[i] = i;
    }

    const int ARRAY_SIZE = ARRAY_COUNT * sizeof *h_in;

    float* d_in = NULL;
    float* d_out = NULL;

    cudaMalloc((void**) &d_in, ARRAY_SIZE);
    cudaMalloc((void**) &d_out, ARRAY_SIZE);
    cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice);
    square<<<1, ARRAY_COUNT>>>(d_out, d_in);
    cudaMemcpy(h_out, d_out, ARRAY_SIZE, cudaMemcpyDeviceToHost);

    for(int i=0 ; i<ARRAY_COUNT ; ++i) {
        printf("%f%s", h_out[i], (i+1)%4 ? "\t" : "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return EXIT_SUCCESS;
}
