#pragma once 

#include <vector> 
#include "kmeans_utilities.hpp"

void cuda_kmeans(
    KMeansOptions* opts, 
    std::vector<double>* centroids, 
    std::vector<int>* labels, 
    double* time_per_iteration_ms, 
    int* iterations
); 
