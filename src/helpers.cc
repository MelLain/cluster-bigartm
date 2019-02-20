#include <cstdlib>

#include <sys/time.h>
#include <sys/resource.h>

#include <fstream>  // NOLINT
#include <sstream>

#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/variate_generator.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "common.h"
#include "helpers.h"
#include "protobuf_helpers.h"
#include "token.h"


long Helpers::get_peak_memory_kb() {
  rusage info;
  if (!getrusage(RUSAGE_SELF, &info)) {
    return info.ru_maxrss;
  }
  return 0;
}

std::vector<float> Helpers::generate_random_vector(int size, size_t seed) {
  std::vector<float> retval;
  retval.reserve(size);

  boost::mt19937 rng(seed);
  boost::uniform_real<float> u(0.0f, 1.0f);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > gen(rng, u);

  for (int i = 0; i < size; ++i) {
    retval.push_back(gen());
  }

  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    sum += retval[i];
  }
  if (sum > 0) {
    for (int i = 0; i < size; ++i) retval[i] /= sum;
  }

  return retval;
}

std::vector<float> Helpers::generate_random_vector(int size, const Token& token, int seed) {
  size_t h = 1125899906842597L;  // prime

  if (token.class_id != DefaultClass) {
    for (unsigned i = 0; i < token.class_id.size(); i++) {
      h = 31 * h + token.class_id[i];
    }
  }

  h = 31 * h + 255;  // separate class_id and token

  for (unsigned i = 0; i < token.keyword.size(); i++) {
    h = 31 * h + token.keyword[i];
  }

  if (seed > 0) {
    h = 31 * h + seed;
  }

  return generate_random_vector(size, h);
}

void Helpers::load_batch(const std::string& full_filename, artm::Batch* batch) {
  std::ifstream fin(full_filename.c_str(), std::ifstream::binary);
  if (!fin.is_open()) {
    throw std::runtime_error("Unable to open file " + full_filename);
  }

  batch->Clear();
  if (!batch->ParseFromIstream(&fin)) {
    throw std::runtime_error("Unable to parse protobuf message from " + full_filename);
  }

  fin.close();

  if ((batch != nullptr) && !batch->has_id()) {
    boost::uuids::uuid uuid;

    try {
      // Attempt to detect UUID based on batche's filename
      std::string filename_only = boost::filesystem::path(full_filename).stem().string();
      uuid = boost::lexical_cast<boost::uuids::uuid>(filename_only);
    } catch (...) { }

    if (uuid.is_nil()) {
      // Otherwise throw the exception
        throw std::runtime_error("Unable to detect batch.id or uuid filename in " + full_filename);
    }

    batch->set_id(boost::lexical_cast<std::string>(uuid));
  }
}
