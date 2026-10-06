#include <iostream> 
#include <vector> 
#include <cstddef> 
#include <cfloat>

#include "thrust.hpp"
#include "cuda_runtime.h"

void thrust_kmeans(
    KMeansOptions* opts,
    std::vector<double>* centroids, 
    std::vector<int>* labels,
    double* time_per_iteration_ms, 
    int* iterations
) {

}