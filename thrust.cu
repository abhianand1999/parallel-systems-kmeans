#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/copy.h> 

#include "thrust.hpp"
#include "cuda_runtime.h"

void thrust_kmeans(
    KMeansOptions* opts,
    std::vector<double>* centroids, 
    std::vector<int>* labels,
    double* time_per_iteration_ms, 
    int* iterations
) {
    thrust::device_vector<double> device_points(opts->input_data.begin(), opts->input_data.end());
    thrust::device_vector<double> device_centroids(centroids->begin(), centroids->end());
    thrust::device_vector<double> device_old_centroids(opts->num_clusters * opts->dims);
    thrust::device_vector<int> device_labels(opts->num_points);
    thrust::device_vector<double> device_cluster_counts(opts->num_clusters);

    int not_converged = 1;

    // Timer
    cudaEvent_t start_time;
    cudaEvent_t end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    cudaEventRecord(start_time);

    while ((*iterations < opts->max_num_iter) and (not_converged)) {

        ++(*iterations);
    }


    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    float iteration_time_ms = 0.0f;
    cudaEventElapsedTime(&iteration_time_ms, start_time, end_time);

    if (!*iterations) {
        *time_per_iteration_ms = 0.0;
    } else {
        *time_per_iteration_ms = iteration_time_ms / *iterations;
    }


    thrust::copy(device_centroids.begin(), device_centroids.end(), centroids->begin());
    thrust::copy(device_labels.begin(), device_labels.end(), labels->begin());
}