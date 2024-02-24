#ifndef READCSV_HPP
#define READCSV_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

// Function to load data from a file into a 2D vector
std::vector<std::vector<double>> readcsv(const std::string& filename, char delimiter = ',') {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data; // Return an empty vector if the file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            row.push_back(std::stod(token));
        }

        data.push_back(row);
    }

    return data;
}

#endif // READCSV_HPP