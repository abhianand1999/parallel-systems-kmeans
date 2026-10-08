#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>
#include <thrust/device_vector.h>

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
}