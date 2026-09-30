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

    double* device_points = nullptr;
    cudaMalloc((void**) &device_points, point_bytes);

    double* device_centroids = nullptr; 
    cudaMalloc((void**) &device_centroids, centroid_bytes);

    double* device_old_centroids = nullptr; 
    cudaMalloc((void**) &device_old_centroids, centroid_bytes);

    int* device_labels = nullptr;
    cudaMalloc((void**) &device_labels, labels_bytes);

    int* device_cluster_counts = nullptr;
    cudaMalloc((void**) &device_cluster_counts, cluster_count_bytes);

    cudaMemcpy(device_points, opts->input_data.data(), point_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids->data(), centroid_bytes, cudaMemcpyHostToDevice);
    cudaMemset(device_cluster_counts, 0, cluster_count_bytes); 
    
    // 1. Caclulate closest centroid for each point (labels) 

    cudaDeviceSynchronize();
}
