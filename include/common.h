#pragma once

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
