#pragma once 
#include <string>
#include <vector>
#include <unordered_set> 

// Options and Parsing 
struct KMeansOptions {
    int num_clusters = 0;
    int dims = 0;
    std::string input_file;

    int num_points = 0; 
    int input_data_size;
    std::vector<std::vector<double>> input_data;

    int max_num_iter = 150;
    double threshold = 10e-5; 
    bool output_centroids = false; 
    int seed = 8675309;
};
bool parse_arguments(int argc, char* argv[], KMeansOptions* opts);
bool read_data(KMeansOptions* opts); 

// Randomization 
void kmeans_srand(unsigned int seed);
int kmeans_rand();

// stdout 
void display_outputs(
    KMeansOptions* opts,
    std::vector<std::vector<double>>& centroids, 
    std::vector<int>& labels,
    int iterations,
    double time_per_iteration_ms
);