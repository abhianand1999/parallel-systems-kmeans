#include <iostream> 

#include "cuda.hpp"
#include "cuda_runtime.h"

_global__ void helloFromGPU(void) {
    printf(“Hello World from GPU!\n”);
}

void cuda_kmeans(KMeansOptions* opts, double* time_per_iteration_ms, int* iterations) {
    helloFromGPU();
}
