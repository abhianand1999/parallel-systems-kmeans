#pragma once 

#include "kmeans_utilities.hpp"

void cuda_kmeans(KMeansOptions* opts, double* time_per_iteration_ms, int* iterations); 
