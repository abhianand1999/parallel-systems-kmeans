#include "kmeans_utilities.hpp"
#include "kmeans.hpp"
#include <limits> 
#include <iostream> 
#include <cmath> 
#include <chrono> 

void kmeans(KMeansOptions& opts) {
    kmeans_srand(opts.seed);
    std::vector<double> centroids(opts.num_clusters * opts.dims, 0.0);
    std::vector<int> labels(opts.num_points); 

    for (int cluster=0; cluster < opts.num_clusters; ++cluster) {
        int point_index = kmeans_rand() % opts.num_points;
        for (int offset=0; offset < opts.dims; ++offset) {
            centroids[cluster * opts.dims + offset] = opts.input_data[point_index * opts.dims + offset];
        }
    }

    int iterations = 0; 
    bool converged = false; 

    std::vector<double> old_centroids;

    auto start_time = std::chrono::steady_clock::now(); 

    while (iterations < opts.max_num_iter) {
        // assign labels 
        old_centroids = centroids;
        for (int point=0; point < opts.num_points; ++point) {
            double min_distance = std::numeric_limits<double>::infinity(); 
            for (int cluster=0; cluster < opts.num_clusters; ++cluster) {
                double distance = 0; 
                for (int offset=0; offset < opts.dims; ++offset) {
                    double difference = centroids[cluster * opts.dims + offset] - opts.input_data[point * opts.dims + offset];
                    distance += difference * difference;
                }
                if (distance < min_distance) {
                    min_distance = distance;
                    labels[point] = cluster; 
                }
            }
        }

        // re-calculate centroids
        // average coordinates across all points w/ label c 
        // clear centroids 
        std::fill(centroids.begin(), centroids.end(), 0.0);

        std::vector<int> centroid_point_count(opts.num_clusters, 0);
        for (int p = 0; p < opts.num_points; ++p) {
            // update means using incremental formula
            ++centroid_point_count[labels[p]];
            for (int offset=0; offset<opts.dims; ++offset) {
                centroids[labels[p] * opts.dims + offset] += (
                    opts.input_data[p * opts.dims + offset] - centroids[labels[p] * opts.dims + offset]
                ) / centroid_point_count[labels[p]];
            }
        }
        
        // fallback - take old centroid if nothing got mapped to this cluster
        for (int cluster = 0; cluster < opts.num_clusters; ++cluster) {
            if (centroid_point_count[cluster] == 0) {
                for (int offset=0; offset<opts.dims; ++offset) {
                    centroids[cluster * opts.dims + offset] = old_centroids[cluster * opts.dims + offset];
                }
            }
        }

        // check convergence
        converged = true;
        for (int cluster = 0; cluster < opts.num_clusters; ++cluster) {
            double distance = 0.0; 
            for (int offset=0; offset < opts.dims; ++offset) {
                double difference = centroids[cluster * opts.dims + offset] - old_centroids[cluster * opts.dims + offset];
                distance += difference * difference;
            }
            if (distance > opts.threshold * opts.threshold) {
                converged = false; 
                break;
            }
        }
        ++iterations; 
        if (converged) { 
            break;
        } 
    }

    auto end_time = std::chrono::steady_clock::now(); 
    auto elapsed_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double time_per_iteration_ms;
    if (!iterations) {
        time_per_iteration_ms = 0; 
    } else {
        time_per_iteration_ms = elapsed_time_ms / iterations; 
    }

    display_outputs(&opts, centroids, labels, iterations, time_per_iteration_ms); 
}
