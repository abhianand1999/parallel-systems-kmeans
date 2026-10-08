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
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h> 
#include <thrust/transform.h>
#include <thrust/logical.h>

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


struct GetCoordinate {
    thrust::device_vector<double>::iterator points;
    int dims;
    int coordinate;

    __host__ __device__
    double operator()(int point) const {
        return points[point * dims + coordinate];
    }
};

struct CalculateCentroidSum {
    thrust::device_vector<int>::iterator cluster_ids;
    thrust::device_vector<double>::iterator reduced_sums;
    thrust::device_vector<double>::iterator centroid_sums;
    int dims;
    int coordinate;

    __host__ __device__ void operator()(int cluster_index) const {
        centroid_sums[cluster_ids[cluster_index] * dims + coordinate] = reduced_sums[cluster_index];
    }
};


struct CalculateClusterCounts { 
    thrust::device_vector<int>::iterator cluster_ids;
    thrust::device_vector<int>::iterator reduced_cluster_counts;
    thrust::device_vector<int>::iterator cluster_counts;

    __host__ __device__ void operator()(int cluster_index) const {
        cluster_counts[cluster_ids[cluster_index]] = reduced_cluster_counts[cluster_index];
    }
};


struct CalculateNewCentroids {
    thrust::device_vector<double>::iterator centroid_sums;
    thrust::device_vector<int>::iterator cluster_counts; 
    thrust::device_vector<double>::iterator centroids;
    thrust::device_vector<double>::iterator old_centroids;
    int dims;

    __host__ __device__ void operator()(int i) const {
        int cluster = i / dims;
        int count = cluster_counts[cluster];

        if (!count) {
            centroids[i] = old_centroids[i];
        } else {
            centroids[i] = centroid_sums[i] / count;
        }
    }
};


struct CheckConvergence {
    thrust::device_vector<double>::iterator centroids;
    thrust::device_vector<double>::iterator old_centroids;
    int dims;
    double threshold;
    
    __host__ __device__ int operator()(int cluster) const {
        double distance_squared = 0.0;

        for (int dim = 0; dim < dims; ++dim) {
            int index = cluster * dims + dim;
            double difference = centroids[index] - old_centroids[index];
            distance_squared += difference * difference;
        }

        return distance_squared <= threshold * threshold;
        // if (distance_squared > threshold * threshold) {
        //     return false;
        // } else {
        //     return true;
        // }
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
    
    // additional ds for reduce by key - sums
    thrust::device_vector<int> sorted_labels(opts->num_points); 
    thrust::device_vector<int> point_ids(opts->num_points);

    thrust::device_vector<int> reduced_cluster_ids(opts->num_points);
    thrust::device_vector<double> reduced_sums(opts->num_points);
    thrust::device_vector<double> centroid_sums(opts->num_clusters * opts->dims);
    thrust::device_vector<double> coordinate_values(opts->num_points);

    // additional ds for normalization
    thrust::device_vector<int> partial_cluster_counts(opts->num_points, 1);
    thrust::device_vector<int> reduced_cluster_counts(opts->num_points);

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
        not_converged = 1;

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

        std::swap(device_old_centroids, device_centroids);
        thrust::fill(device_centroids.begin(), device_centroids.end(), 0.0);

        // Sort labels for groupby agg 
        thrust::copy(device_labels.begin(), device_labels.end(), sorted_labels.begin());
        thrust::sequence(point_ids.begin(), point_ids.end());
        thrust::stable_sort_by_key(sorted_labels.begin(), sorted_labels.end(), point_ids.begin());

        // calculate cluster counts
        auto count_result = thrust::reduce_by_key(
            sorted_labels.begin(),
            sorted_labels.end(),
            partial_cluster_counts.begin(),
            reduced_cluster_ids.begin(),
            reduced_cluster_counts.begin()
        );

        int num_count_groups = static_cast<int>(
            count_result.first - reduced_cluster_ids.begin()
        );

        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_count_groups),
            CalculateClusterCounts{
                reduced_cluster_ids.begin(),
                reduced_cluster_counts.begin(),
                device_cluster_counts.begin()
            }
        );
        
        // calculate sums
        thrust::fill(centroid_sums.begin(), centroid_sums.end(), 0.0);

        for (int dim = 0; dim < opts->dims; ++dim) {
            // Get this coordinate for each point, in sorted-label order.
            thrust::transform(
                point_ids.begin(),
                point_ids.end(),
                coordinate_values.begin(),
                GetCoordinate{device_points.begin(), opts->dims, dim}
            );

            auto result = thrust::reduce_by_key(
                sorted_labels.begin(),
                sorted_labels.end(),
                coordinate_values.begin(),
                reduced_cluster_ids.begin(),
                reduced_sums.begin()
            );

            int num_reduced = static_cast<int>(result.first - reduced_cluster_ids.begin());

            thrust::for_each(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_reduced),
                CalculateCentroidSum{
                    reduced_cluster_ids.begin(),
                    reduced_sums.begin(),
                    centroid_sums.begin(),
                    opts->dims,
                    dim
                }
            );
        }


        // calculate new centroids 
        thrust::for_each(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(opts->num_clusters * opts->dims),
            CalculateNewCentroids{
                centroid_sums.begin(),
                device_cluster_counts.begin(),
                device_centroids.begin(),
                device_old_centroids.begin(),
                opts->dims
            }
        );

        // check convergence
        not_converged = thrust::none_of(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(opts->num_clusters),
            CheckConvergence{
                device_centroids.begin(),
                device_old_centroids.begin(),
                opts->dims,
                opts->threshold
            }
        );
        // Copy out convergence
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
