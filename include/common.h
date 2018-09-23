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
