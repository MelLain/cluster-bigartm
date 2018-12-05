#pragma once

#include <ctime>

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

typedef std::string ModelName;

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kBatchNameLength = 6;

const std::string kEscChar = "|";

typedef std::unordered_map<std::string, std::vector<double>> Normalizers;

inline std::string generate_command_key(const std::string& id) {
	return kEscChar + std::string("cmd-") + id;
}

inline std::string generate_data_key(const std::string& id) {
	return kEscChar + std::string("data-") + id;
}

inline void log(const std::string& message, const std::clock_t& start_time) {
    std::cout << message << ", elapsed time: " << (std::clock() - start_time) / CLOCKS_PER_SEC << std::endl;
}
