#include <iostream> 
#include <vector> 
#include <cstddef> 

#include "cuda.hpp"
#include "cuda_runtime.h"


void cuda_kmeans(
    KMeansOptions* opts, 
    std::vector<double>* centroids, 
    std::vector<int>* labels,
    double* time_per_iteration_ms, 
    int* iterations
) {
    /*
    Copy data from CPU memory to GPU memory.
    Invoke kernels to operate on the data stored in GPU memory.
    Copy data back from GPU memory to CPU memory.
    */
    size_t point_bytes = opts->input_data_size * sizeof(double);
    size_t centroid_bytes = opts->num_clusters * sizeof(double); 
    size_t labels_bytes = opts->num_points * sizeof(int); 
    size_t cluster_count_bytes = opts->num_clusters * sizeof(int); 

    size_t float_threshold = sizeof(float); 
    size_t max_iterations = sizeof(int); 

    // 1. Caclulate closest centroid for each point (labels) 

    cudaDeviceSynchronize();
}
