#include <string>
#include <unordered_set> 
#include <iostream>
#include <fstream>

#include "kmeans_utilities.hpp"

std::unordered_set<std::string> VALID_FLAGS{"-k", "-d", "-i", "-m", "-t", "-s", "-c"};
std::unordered_set<std::string> REQUIRED_FLAGS{}; // todo: check to see whether we have a set of required flags to pass through

// Argument Parsing
bool parse_arguments(int argc, char* argv[], KMeansOptions* opts) {
    std::unordered_set<std::string> seen_flags{};

    int i = 1; 

    while (i < argc) {
        if (!VALID_FLAGS.count(argv[i])) {
            std::cerr << "Expected flag, received " << argv[i] << "\n";
            return false;
        }

        seen_flags.emplace(argv[i]);
        std::string_view flag = std::string_view(argv[i]);

        if (!flag.compare("-c")) {
            opts -> output_centroids = true;
            ++i; 
            continue;
        } else {
            if (i + 1 == argc) {
                std::cerr << "Flag " << argv[i] << " requires a corresponding input value.\n";
                return false;
            }
            
            if (!flag.compare("-k")) {
                opts -> num_clusters = std::atoi(argv[i+1]);
            } else if (!flag.compare("-d")) {
                opts -> dims = std::atoi(argv[i+1]);
            } else if (!flag.compare("-i")) {
                opts -> input_file = argv[i+1];
            } else if (!flag.compare("-m")) {
                opts -> max_num_iter = std::atoi(argv[i+1]);
            } else if (!flag.compare("-t")) {
                opts -> threshold = std::atof(argv[i+1]);
            } else if (!flag.compare("-s")) {
                opts -> seed = std::atoi(argv[i+1]);
            }
            i += 2; 
        }
    }
    return true;
}

bool read_data(KMeansOptions* opts) {
    std::ifstream input_file(opts->input_file);
    if (!input_file) {
        std::cerr << "Error reading data.\n";
        return false;
    }
    int num_points;
    if (!(input_file >> num_points)) {
        std::cerr << "Could not read number of data points"; 
        return false;
    }
    opts->num_points = num_points; 

    int input_data_size = opts->dims * opts->num_points;
    if (!input_data_size) {
        std::cerr << "Either dims or num points is zero.\n";
        return false;
    }
    opts->input_data_size = input_data_size; 

    std::vector<double> data(opts->num_points * opts->dims, 0.0);
    for (int point=0; point < opts->num_points; ++point) {
        int data_index; 
        if (!(input_file >> data_index)) {
            std::cerr << "Missing data index in input file.\n";
            return false;
        }
        for (int dim=0; dim < opts->dims; ++dim) {
            if (!(input_file >> data[point * opts->dims + dim])) {
                std::cerr << "Missing data for index " << data_index << " and position " << dim << "\n";
            }
        }
    }

    opts->input_data = std::move(data); 
    return true;
}

// Randomization Utilities
namespace {
    static unsigned long int next = 1;
    static unsigned long kmeans_rmax = 32767;
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}

// stdout
void display_outputs(
    KMeansOptions* opts, 
    std::vector<double>& centroids, 
    std::vector<int>& labels,
    int iterations,
    double time_per_iteration_ms
) {
    printf("%d,%lf\n", iterations, time_per_iteration_ms);
    if (opts->output_centroids) {
        for (int clusterId = 0; clusterId < opts->num_clusters; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < opts->dims; d++) {
                printf("%lf ", centroids[clusterId * opts->dims + d]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int p=0; p < opts->num_points; p++) {
            printf(" %d", labels[p]);
        }
    }
}
