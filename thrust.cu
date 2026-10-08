#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h> 
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "thrust.hpp"
#include "cuda_runtime.h"

// One functor invocation handles one point, like one CUDA thread in
// assign_closest_centroid.
struct AssignClosestCentroid {
    thrust::device_vector<double>::iterator points;
    thrust::device_vector<double>::iterator centroids;
    thrust::device_vector<int>::iterator labels;
    int dims;
    int num_clusters;

    __host__ __device__ void operator()(int point) const {
        double min_distance = DBL_MAX;
        int closest_cluster = 0;

        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            double distance = 0.0;
            for (int dim = 0; dim < dims; ++dim) {
                double difference = centroids[cluster * dims + dim] - points[point * dims + dim];
                distance += difference * difference;
            }
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = cluster;
            }
        }
        labels[point] = closest_cluster;
    }
};

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
    thrust::device_vector<int> device_cluster_counts(opts->num_clusters);
    thrust::device_vector<int> device_not_converged(1);


    int not_converged = 1;

    // Timer
    cudaEvent_t start_time;
    cudaEvent_t end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    cudaEventRecord(start_time);

    while ((*iterations < opts->max_num_iter) and (not_converged)) {
        // Clear contents
        thrust::fill(device_cluster_counts.begin(), device_cluster_counts.end(), 0);
        device_not_converged[0] = 0;

        // Assign each point to its nearest centroid. The counting iterator
        // supplies point indices 0, 1, ..., num_points - 1.
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(opts->num_points),
            AssignClosestCentroid{
                device_points.begin(),
                device_centroids.begin(),
                device_labels.begin(),
                opts->dims,
                opts->num_clusters
            }
        );

        // Copy out convergence
        thrust::copy(device_not_converged.begin(), device_not_converged.end(), &not_converged);
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
