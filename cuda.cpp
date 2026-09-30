#include <iostream>
#include <vector>
#include "kmeans_utilities.hpp"
#include "cuda.hpp"

void kmeans(KMeansOptions& opts) {
    // Set up required inputs (centroids and labels)
    kmeans_srand(opts.seed);
    std::vector<double> centroids(opts.num_clusters * opts.dims, 0.0);
    std::vector<int> labels(opts.num_points);

    for (int cluster=0; cluster < opts.num_clusters; ++cluster) {
        int point_index = kmeans_rand() % opts.num_points;
        for (int offset=0; offset < opts.dims; ++offset) {
            centroids[cluster * opts.dims + offset] = opts.input_data[point_index * opts.dims + offset];
        }
    }

    double time_per_iteration_ms = 0.0;
    int iterations = 0;

    cuda_kmeans(&opts, &time_per_iteration_ms, &iterations);

    // display outputs 
    display_outputs(&opts, centroids, labels, iterations, time_per_iteration_ms);
}