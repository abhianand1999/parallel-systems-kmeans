#include <iostream> 

#include "cuda.hpp"
#include "cuda_runtime.h"

void cuda_kmeans(KMeansOptions* opts, double* time_per_iteration_ms, int* iterations) {
    cout << opts->dims; 
}