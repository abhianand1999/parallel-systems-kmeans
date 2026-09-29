#include <iostream>
#include <unordered_set>

#include "kmeans.hpp"
#include "kmeans_utilities.hpp"

int main(int argc, char* argv[]) {
    KMeansOptions opts; 
    if (!parse_arguments(argc, argv, &opts)) {
        return 1;
    }
    if (!read_data(&opts)) {
        return 1;
    }
    kmeans(opts);
}

