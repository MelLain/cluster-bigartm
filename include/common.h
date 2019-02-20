#pragma once

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

typedef std::string ModelName;

const int UnknownId = -1;

const std::string kBatchExtension = ".batch";

const int kBatchNameLength = 6;

const std::string kEscChar = "|";

const float kEps = 1e-16f;

const int kDefaultTimeout = 100;

typedef std::unordered_map<std::string, std::vector<double>> Normalizers;

inline std::vector<std::string> generate_command_keys(int executor_id, int num_threads) {
  std::vector<std::string> retval;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    retval.push_back(kEscChar + std::string("cmd-") + std::to_string(executor_id) + "-" + std::to_string(thread_id));
  }
  return retval;
}

inline std::vector<std::string> generate_data_keys(int executor_id, int num_threads) {
  std::vector<std::string> retval;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    retval.push_back(kEscChar + std::string("dat-") + std::to_string(executor_id) + "-" + std::to_string(thread_id));
  }
  return retval;
}
