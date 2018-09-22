#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"

#include "common.h"
#include "messages.pb.h"

struct Token;

// 'class Helpers' is a utility with several static methods.
class Helpers {
 public:
  // Generates random vector using mersenne_twister_engine from boost library.
  // The goal is to ensure that this method is cross-platrofm, e.g. the resulting random vector
  // are the same on Linux, Mac OS and Windows. This is important because
  // the method is used to initialize entries in the phi matrix.
  // For unit-tests it is important that such initialization is deterministic
  // (depends only on the keyword and class_id of the token.
  static std::vector<float> GenerateRandomVector(int size, size_t seed);
  static std::vector<float> GenerateRandomVector(int size, const Token& token, int seed = -1);

  static void LoadBatch(const std::string& full_filename, artm::Batch* batch);
};
