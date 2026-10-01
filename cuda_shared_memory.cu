#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>

#include "cuda_shared_memory.hpp"
#include "cuda_runtime.h"

/*
load data from gmem to shmem 
__syncthreads();
update shmem
__syncthreads();
store data from shmem to gmem
It is very easy to make mistakes when you do not need all threads to load and store the data but need them all to update the data in shmem. This happens when the total number of threads does not match the number of tasks (such as points). Be careful when you try to mask some threads when using shared memory.
 */


__global__ void find_closest_centroid(
    const double* points,
    const double* old_centroids,
    const double* centroids,
    int* labels,
    int* cluster_counts, 
    int num_points,
    int dims, 
    int num_clusters
) {
    // set up shared mem
    extern __shared__ int local_cluster_counts[]; // size dynamically injected on startup
    for (int i=threadIdx.x; i < num_clusters; i+=blockDim.x) {
        local_cluster_counts[i] = 0;
    }
    __syncthreads();

    int point = blockIdx.x * blockDim.x + threadIdx.x;
     
    if (point < num_points) {
        double min_distance = DBL_MAX; 
        int closest_cluster = -1;
        for (int cluster=0; cluster < num_clusters; ++cluster) {
            double distance = 0; 
            for (int offset=0; offset < dims; ++offset) {
                double difference = centroids[cluster * dims + offset] - points[point * dims + offset];
                distance += difference * difference;
            }
            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = cluster;
            }
        }
        labels[point] = closest_cluster; 
        atomicAdd(&local_cluster_counts[closest_cluster], 1);
        // atomicAdd(&cluster_counts[closest_cluster], 1); 
    }
    __syncthreads();

    // update global counts
    for (int i=threadIdx.x; i < num_clusters; i+=blockDim.x) {
        if (local_cluster_counts[i] > 0) {
            atomicAdd(&cluster_counts[i], local_cluster_counts[i]);
        }
    }
}

__global__ void centroid_sum(
    double* points, 
    double* old_centroids,
    double* centroids, 
    int* labels,
    int* cluster_counts, 
    int num_points,
    int dims,
    int num_clusters
) {
    // set up shared mem 
    extern __shared__ double local_centroid_sums[]; 
    for (int i=threadIdx.x; i < num_clusters * dims; i+=blockDim.x) {
        local_centroid_sums[i] = 0;
    }
    __syncthreads(); 

    int point = blockIdx.x * blockDim.x + threadIdx.x; 
    if (point >= num_points) {
        return;
    }
    int cluster = labels[point]; 
    for (int offset=0; offset < dims; ++offset) {
        atomicAdd(&local_centroid_sums[cluster * dims + offset], points[point * dims + offset]);
    }

    __syncthreads();
    for (int i=threadIdx.x; i < num_clusters * dims; i+=blockDim.x) {
        if (local_centroid_sums[i] >0) {
            atomicAdd(&centroids[i], local_centroid_sums[i]);
        }
    }
}

__global__ void normalize_centroids(
    double* centroids,
    const double* old_centroids,
    const int* cluster_counts,
    int num_clusters,
    int dims
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster = index / dims; 
    int offset = index % dims; 

    if (index >= dims * num_clusters) {
        return;
    }

    int cluster_count = cluster_counts[cluster]; 

    if (!cluster_count) {
        centroids[cluster * dims + offset] = old_centroids[cluster * dims + offset];
    } else {
        centroids[cluster * dims + offset] /= cluster_count;
    }
}

__global__ void check_convergence(
    const double* centroids,
    const double* old_centroids, 
    int num_centroids, 
    int dims,
    float threshold, 
    int* not_converged
) {
    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= num_centroids) {
        return;
    }
    double distance = 0.0; 

    for (int offset=0; offset < dims; ++offset) {
        double difference = centroids[cluster * dims + offset] - old_centroids[cluster * dims + offset];
        distance += difference * difference;
        if (distance > threshold * threshold) {
            atomicExch(not_converged, 1); 
            break;
        }
    }
}

void cuda_shared_memory_kmeans(
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
    int nthreads = 256; 
    
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

    int* device_not_converged = nullptr; 
    cudaMalloc((void**) &device_not_converged, (size_t) sizeof(int)); 

    cudaMemcpy(device_points, opts->input_data.data(), point_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, centroids->data(), centroid_bytes, cudaMemcpyHostToDevice);
    
    int not_converged = 1;

    // Timer
    cudaEvent_t start_time;
    cudaEvent_t end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    cudaEventRecord(start_time);

    while ((*iterations < opts->max_num_iter) and (not_converged)) {
        // Clear states
        cudaMemset(device_cluster_counts, 0, cluster_count_bytes); 
        cudaMemset(device_not_converged, 0, sizeof(int));

        // 1. Caclulate closest centroid for each point (labels) 
        int blocks = (opts -> num_points + nthreads-1) / nthreads;
        size_t local_count_bytes = opts->num_clusters * sizeof(int);

        find_closest_centroid<<<blocks, nthreads, local_count_bytes>>>(
            device_points,
            device_old_centroids,
            device_centroids,
            device_labels,
            device_cluster_counts,
            opts->num_points,
            opts->dims,
            opts->num_clusters
        );
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "cuda sync failure";
        }

        std::swap(device_old_centroids, device_centroids);
        cudaMemset(device_centroids, 0.0, centroid_bytes); 

        size_t local_centroid_bytes = opts->num_clusters * opts -> dims * sizeof(double);
        centroid_sum<<<blocks, nthreads, local_centroid_bytes>>>(
            device_points,
            device_old_centroids,
            device_centroids,
            device_labels,
            device_cluster_counts,
            opts->num_points,
            opts->dims,
            opts->num_clusters
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "cuda sync failure";
        }

        int normalization_blocks = (opts->dims * opts->num_clusters + nthreads-1)/ nthreads;
        normalize_centroids<<<normalization_blocks, nthreads>>>(
            device_centroids,
            device_old_centroids,
            device_cluster_counts,
            opts->num_clusters,
            opts->dims
        );
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "cuda sync failure";
        }
        
        // 3. Check for convergence
        int convergence_blocks = (opts->num_clusters + nthreads - 1) / nthreads;
        check_convergence<<<convergence_blocks, nthreads>>>(
            device_centroids,
            device_old_centroids,
            opts->num_clusters,
            opts->dims,
            opts->threshold,
            device_not_converged
        );  
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "cuda sync failure";
        }

        cudaMemcpy(&not_converged, device_not_converged, sizeof(int), cudaMemcpyDeviceToHost);
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

    // Copy out data
    cudaMemcpy(centroids->data(), device_centroids, centroid_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels->data(), device_labels, labels_bytes, cudaMemcpyDeviceToHost);

    // Cleanup 
    cudaFree(device_not_converged);
    cudaFree(device_cluster_counts);
    cudaFree(device_labels);
    cudaFree(device_old_centroids);
    cudaFree(device_centroids);
    cudaFree(device_points);
    cudaDeviceSynchronize();
}
