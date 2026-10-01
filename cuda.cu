#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>

#include "cuda.hpp"
#include "cuda_runtime.h"

__global__ void assign_closest_centroid(
    const double* points,
    const double* old_centroids,
    const double* centroids,
    int* labels,
    int* cluster_counts, 
    int num_points,
    int dims, 
    int num_clusters
) {
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
     
    if (point_index >= num_points) {
        return; 
    }

    double min_distance = DBL_MAX; 
    int closest_cluster = -1;
    for (int cluster=0; cluster < num_clusters; ++cluster) {
        double distance = 0; 
        for (int offset=0; offset < dims; ++offset) {
            double difference = centroids[cluster * dims + offset] - points[point_index * dims + offset];
            distance += difference * difference;
        }
        if (distance < min_distance) {
            min_distance = distance;
            closest_cluster = cluster;
        }
    }
    labels[point_index] = closest_cluster; 
    atomicAdd(&cluster_counts[closest_cluster], 1); 
}

__global__ void calculate_centroids(
    const double* points,
    const double* old_centroids,
    double* centroids,
    int* labels,
    int* cluster_counts, 
    int num_points,
    int dims, 
    int num_clusters
) {
    int cluster_index = blockIdx.x * blockDim.x + threadIdx.x;
     
    if (cluster_index >= num_clusters) {
        return; 
    }

    if (cluster_counts[cluster_index]) {
        for (int p = 0; p < num_points; ++p) {
            if (labels[p] == cluster_index) {
                for (int d = 0; d < dims; ++d) {
                    centroids[cluster_index * dims + d] += points[p * dims + d];
                }
            }
        }
        
        // Normalization
        for (int d = 0; d < dims; ++d) {
            centroids[cluster_index * dims + d] /= cluster_counts[cluster_index]; 
        }

    } else {
        // Fallback
        for (int offset=0; offset<dims; ++offset) {
            centroids[cluster_index * dims + offset] = old_centroids[cluster_index * dims + offset];
        }
    }
}

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
    size_t centroid_bytes = (opts->num_clusters * opts->dims) * sizeof(double); 
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
    
    while (*iterations < opts->max_num_iter) {
        // Clear states
        cudaMemset(device_cluster_counts, 0, cluster_count_bytes); 

        // 1. Caclulate closest centroid for each point (labels) 
        int blocks = (opts -> num_points + 255) / 256;
        assign_closest_centroid<<<blocks, 256>>>(
            device_points,
            device_old_centroids,
            device_centroids,
            device_labels,
            device_cluster_counts,
            opts->num_points,
            opts->dims,
            opts->num_clusters
        );

        std::swap(device_old_centroids, device_centroids);
        cudaMemset(device_centroids, 0.0, centroid_bytes); 

        // 2. Calculate new centroid 
        blocks = (opts -> num_clusters + 255) / 256; 
        calculate_centroids<<<blocks, 256>>>(
            device_points,
            device_old_centroids, 
            device_centroids,
            device_labels,
            device_cluster_counts,
            opts->num_points,
            opts->dims,
            opts->num_clusters
        );

        // 3. Check for convergence
        ++(*iterations);
    }

    // Copy out data
    cudaMemcpy(centroids->data(), device_centroids, centroid_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels->data(), device_labels, labels_bytes, cudaMemcpyDeviceToHost);

    // Cleanup 
    cudaFree(device_cluster_counts);
    cudaFree(device_labels);
    cudaFree(device_old_centroids);
    cudaFree(device_centroids);
    cudaFree(device_points);
    cudaDeviceSynchronize();
}
