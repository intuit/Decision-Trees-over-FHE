#ifndef FHE_RANDOM_FOREST_UTILS_H
#define FHE_RANDOM_FOREST_UTILS_H

#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <limits>
#include <algorithm>
#include <numeric>
#include "seal/seal.h"
#include "data_types.h"

using namespace std;
using namespace seal;

/*
Helper function: Reads data from csv and translates it to a vector of samples.
*/
static vector<Sample<double>> read_data(const string& plain_file_full_path, size_t features_count = 2, size_t labels_count = 2) {
    vector<Sample<double>> data;
    ifstream data_file(plain_file_full_path);

    string data_line, label_line, x_line;
    if (data_file.is_open()) {
        getline(data_file, data_line);
        while (getline(data_file, data_line)) {
            stringstream ss(data_line);

            getline(ss,label_line,',');
            int label = std::stoi(label_line);

            Sample<double> xy;
            sample_init(xy, features_count, labels_count);

            size_t k = 0;
            while (ss.good() && k < features_count) {
                getline(ss,x_line,',');
                double x = std::stod(x_line);
                access_features(xy)[k] = x;
                k++;
            }

            for (size_t l = 0; l < labels_count; l++) {
                double y = 0.0;
                if (l == label) {
                    y = 1.0;
                }

                access_labels(xy)[l] = y;
            }


            data.push_back(xy);
        }

        data_file.close();
    }

    return data;
}

/*
Helper function: Prints a vector of floating-point values.
*/
template<typename T>
inline void print_vector(vector<T> vec, bool compact=false, size_t print_size = 4) {
    size_t size = vec.size();

    cout << "    [";
    if(!compact || size <= 2 * print_size)
    {
        for (size_t i = 0; i < size; i++)
        {
            cout << " " << to_string(vec[i]) << ((i != size - 1) ? "," : " ]\n");
        }
    }
    else
    {
        vec.resize(max(vec.size(), 2 * print_size));
        for (size_t i = 0; i < print_size; i++)
        {
            cout << " " << to_string(vec[i]) << ",";
        }
        if(vec.size() > 2 * print_size)
        {
            cout << " ...,";
        }
        for (size_t i = size - print_size; i < size; i++)
        {
            cout << " " << to_string(vec[i]) << ((i != size - 1) ? "," : " ]\n");
        }
    }
}

#endif //FHE_RANDOM_FOREST_UTILS_H
