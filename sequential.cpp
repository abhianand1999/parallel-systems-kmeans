#include "kmeans_utilities.hpp"
#include "kmeans.hpp"
#include <limits> 
#include <iostream> 
#include <cmath> 
#include <chrono> 

void kmeans(KMeansOptions& opts) {
    kmeans_srand(opts.seed);
    std::vector<std::vector<double>> centroids(opts.num_clusters, std::vector<double>(opts.dims, 0.0));
    std::vector<int> labels(opts.num_points); 

    for (int i=0; i < opts.num_clusters; ++i) {
        int index = kmeans_rand() % opts.num_points;
        centroids[i] = opts.input_data[index]; 
    }

    int iterations = 0; 
    bool converged = false; 

    std::vector<std::vector<double>> old_centroids;

    auto start_time = std::chrono::steady_clock::now(); 

    while (iterations < opts.max_num_iter) {
        // assign labels 
        old_centroids = centroids;
        for (int i = 0; i < opts.num_points; ++i) {
            double min_distance = std::numeric_limits<double>::infinity(); 
            std::vector<double>& point = opts.input_data[i]; 
            for (int c = 0; c < opts.num_clusters; ++c) {
                std::vector<double>& centroid = centroids[c];
                double distance = 0; 
                for (int elem = 0; elem < opts.dims; ++elem) {
                    distance += (
                        (point[elem] - centroid[elem]) 
                        * (point[elem] - centroid[elem])
                    ); 
                }
                if (distance < min_distance) {
                    min_distance = distance;
                    labels[i] = c; 
                }
            }
        }

        // re-calculate centroids
        // average coordinates across all points w/ label c 
        // clear centroids 
        for (auto& row: centroids) {
            std::fill(row.begin(), row.end(), 0.0);
        }
        std::vector<int> centroid_point_count(opts.num_clusters, 0);
        for (int p = 0; p < opts.num_points; ++p) {
            // update means using incremental formula
            std::vector<double>& mapped_centroid = centroids[labels[p]];
            ++centroid_point_count[labels[p]]; 
    
            for (int d = 0; d < opts.dims; ++d) {
                mapped_centroid[d] += (
                    opts.input_data[p][d] - mapped_centroid[d]
                ) / centroid_point_count[labels[p]];
            }
        }
        
        for (int cluster = 0; cluster < opts.num_clusters; ++cluster) {
            if (centroid_point_count[cluster] == 0) {
                centroids[cluster] = old_centroids[cluster];
            }
        }

        // check convergence
        converged = true;
        for (int cluster = 0; cluster < opts.num_clusters; ++cluster) {
            double distance = 0.0; 
            for (int dim = 0; dim < opts.dims; ++dim) {
                double difference = centroids[cluster][dim] - old_centroids[cluster][dim];
                distance += difference * difference;
            }
            if (distance > opts.threshold) {
                converged = false; 
                break;
            }
        }

        if (converged) { 
            break;
        } 
        ++iterations; 
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
