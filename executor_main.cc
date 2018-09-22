#include <ctime>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options.hpp>

#include "messages.pb.h"

#include "blas.h"
#include "token.h"
#include "redis_phi_matrix.h"
#include "helpers.h"
#include "processor_helpers.h"
#include "redis_client.h"
#include "protocol.h"

namespace po = boost::program_options;
namespace bf = boost::filesystem;

typedef std::unordered_map<ClassId, std::vector<double>> Normalizers;

static const int kConnTimeout = 100;
static const int kNumRetries = 10;

struct Parameters {
    int num_topics;
    int num_inner_iters;
    std::string batches_dir_path;
    std::string vocab_path;
    std::string redis_ip;
    std::string redis_port;
    int continue_fitting;
    int token_begin_index;
    int token_end_index;
    int batch_begin_index;
    int batch_end_index;
    std::string command_key;
    std::string data_key;
    int debug_print;
};

void ParseAndPrintArgs(int argc, char* argv[], Parameters* p) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help", "Show help")
    ("num-topics", po::value(&p->num_topics)->default_value(1), "Input number of topics")
    ("num-inner-iter", po::value(&p->num_inner_iters)->default_value(1), "Input number of document passes")
    ("batches-dir-path", po::value(&p->batches_dir_path)->default_value("."), "Input path to files with documents")
    ("vocab-path", po::value(&p->vocab_path)->default_value("."), "Input path to files with documents")
    ("redis-ip", po::value(&p->redis_ip)->default_value(""), "ip of redis instance")
    ("redis-port", po::value(&p->redis_port)->default_value(""), "port of redis instance")
    ("continue-fitting", po::value(&p->continue_fitting)->default_value(0), "1 - continue fitting redis model, 0 - restart")
    ("token-begin-index", po::value(&p->token_begin_index)->default_value(0), "index of token to init/norm from")
    ("token-end-index", po::value(&p->token_end_index)->default_value(0), "index of token to init/norm to (excluding)")
    ("batch-begin-index", po::value(&p->batch_begin_index)->default_value(0), "index of batch to process from")
    ("batch-end-index", po::value(&p->batch_end_index)->default_value(0), "index of batch to process to (excluding)")
    ("command-key", po::value(&p->command_key)->default_value(""), "redis key to communicate with master")
    ("data-key", po::value(&p->data_key)->default_value(""), "redis key to exchange data with master")
    ("debug-print", po::value(&p->debug_print)->default_value(0), "1 - print debug info, 0 - not")
    ;

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
  notify(vm);

  if (p->debug_print == 1) {
    std::cout << std::endl << "======= Executor info, cmd key '" << p->command_key << "' =======" << std::endl;
    std::cout << "num-topics:        " << p->num_topics << std::endl;
    std::cout << "num-inner-iter:    " << p->num_inner_iters << std::endl;
    std::cout << "batches-dir-path:  " << p->batches_dir_path << std::endl;
    std::cout << "vocab-path:        " << p->vocab_path << std::endl;
    std::cout << "redis-ip:          " << p->redis_ip << std::endl;
    std::cout << "redis-port:        " << p->redis_port << std::endl;
    std::cout << "continue-fitting:  " << p->continue_fitting << std::endl;
    std::cout << "token-begin-index: " << p->token_begin_index << std::endl;
    std::cout << "token-end-index: " << p->token_end_index << std::endl;
    std::cout << "batch-begin-index: " << p->batch_begin_index << std::endl;
    std::cout << "batch-end-index: " << p->batch_end_index << std::endl;
    std::cout << "data-key:          " << p->data_key << std::endl << std::endl;
  }
}

/*
Normalizers FindNt(const PhiMatrix& n_wt) {
  Normalizers retval;
  std::vector<float> helper = std::vector<float>(n_wt.topic_size(), 0.0f);
  for (int token_id = 0; token_id < n_wt.token_size(); ++token_id) {
    const Token& token = n_wt.token(token_id);
    auto normalizer_key = token.class_id;

    auto iter = retval.find(normalizer_key);
    if (iter == retval.end()) {
      retval.insert(std::make_pair(normalizer_key, std::vector<double>(n_wt.topic_size(), 0)));
      iter = retval.find(normalizer_key);
    }

    n_wt.get(token_id, &helper);
    for (int topic_id = 0; topic_id < n_wt.topic_size(); ++topic_id) {
      const double sum = helper[topic_id];
      if (sum > 0) {
        iter->second[topic_id] += sum;
      }
    }
  }
  return retval;
}

void Normalize(std::shared_ptr<PhiMatrix> p_wt, const PhiMatrix& n_wt) {
  const int topic_size = n_wt.topic_size();
  const int token_size = n_wt.token_size();

  assert(p_wt->token_size() == n_wt.token_size() && p_wt->topic_size() == n_wt.topic_size());

  Normalizers n_t = FindNt(n_wt);
  for (int token_id = 0; token_id < token_size; ++token_id) {
    const Token& token = n_wt.token(token_id);
    assert(p_wt->token(token_id) == token);
    const std::vector<double>& nt = n_t[token.class_id];
    std::vector<float> helper = std::vector<float>(n_wt.topic_size(), 0.0f);
    std::vector<float> helper_n_wt = std::vector<float>(n_wt.topic_size(), 0.0f);
    n_wt.get(token_id, &helper_n_wt);
    for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
      float value = 0.0f;
      if (nt[topic_index] > 0) {
        value = std::max<double>(helper_n_wt[topic_index], 0.0f) / nt[topic_index];
        if (value < 1e-16) {
          value = 0.0f;
        }
      }
      helper[topic_index] = value;
    }
    p_wt->set(token_id, helper);
  }
}

void ProcessEStep(const artm::Batch& batch,
                  const PhiMatrix& p_wt,
                  std::shared_ptr<PhiMatrix> n_wt,
                  int num_inner_iters,
                  Blas* blas,
                  float* perplexity_value) {
  std::shared_ptr<CsrMatrix<float>> sparse_ndw;
  sparse_ndw = ProcessorHelpers::InitializeSparseNdw(batch);

  std::shared_ptr<LocalThetaMatrix<float>> theta_matrix;
  theta_matrix = ProcessorHelpers::InitializeTheta(p_wt.topic_size(), batch);

  std::shared_ptr<NwtWriteAdapter> nwt_writer = std::make_shared<NwtWriteAdapter>(n_wt.get());

  ProcessorHelpers::InferThetaAndUpdateNwtSparse(batch, *sparse_ndw, p_wt, theta_matrix.get(),
                                                 nwt_writer.get(), blas, num_inner_iters, perplexity_value);
}
*/

bool CheckNonTerminatedAndUpdate(const RedisClient& redis_client, const std::string& key, const std::string& flag) {
  auto reply = redis_client.get_value(key);
  if (reply == START_TERMINATION) {
    return false;
  }

  redis_client.set_value(key, flag);
  return true;
}

bool WaitForFlag(const RedisClient& redis_client, const std::string& key, const std::string& flag) {
  while (true) {
    auto reply = redis_client.get_value(key);
    if (reply == START_TERMINATION) {
      break;
    }

    if (reply == flag) {
      return true;
    }

    usleep(2000);
  }
  return false;
}


int main(int argc, char* argv[]) {
  const clock_t begin_time = clock();

  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);

  RedisClient redis_client = RedisClient(params.redis_ip, std::stoi(params.redis_port), kNumRetries, kConnTimeout);
  try {
    if (!CheckNonTerminatedAndUpdate(redis_client, params.command_key, FINISH_GLOBAL_START)) {
      throw std::runtime_error("Step 0, got termination command");
    };

    if (!WaitForFlag(redis_client, params.command_key, START_INITIALIZATION)) {
      throw std::runtime_error("Step 1 start, got termination command");
    };

    bool debug_print = (params.debug_print == 1);

    std::string current_cmd = redis_client.get_value(params.command_key);
    if (debug_print) {
      std::cout << "Read from key: " << params.command_key << ", value: " << current_cmd << std::endl;
    }

    std::vector<std::string> topics;
    for (int i = 0; i < params.num_topics; ++i) {
      topics.push_back("topic_" + std::to_string(i));
    }

    auto p_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("pwt"), topics, redis_client));
    auto n_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("nwt"), topics, redis_client));

    bool continue_fitting = (params.continue_fitting == 1);
    std::ifstream fin;
    fin.open(params.vocab_path);

    std::string line;
    int counter = 0;
    auto zero_vector = std::vector<float>(params.num_topics, 0.0f);
    while (std::getline(fin, line)) {
      auto token = Token(DefaultClass, line);
      bool add_token_to_redis = false;
      if (counter >= params.token_begin_index && counter < params.token_end_index) {
        add_token_to_redis = !continue_fitting;
      }
      p_wt->AddToken(token, add_token_to_redis, zero_vector);
      n_wt->AddToken(token, add_token_to_redis,
        add_token_to_redis ? Helpers::GenerateRandomVector(params.num_topics, token) : zero_vector);
    }
    fin.close();

    double n = 0.0f;
    counter = 0;
    for(const auto& entry : boost::make_iterator_range(bf::directory_iterator(params.batches_dir_path), { })) {
      if (counter >= params.batch_begin_index && counter < params.batch_end_index) {
        artm::Batch batch;
        Helpers::LoadBatch(entry.path().string(), &batch);
        for (const auto& item : batch.item()) {
          for (float val : item.token_weight()) {
            n += static_cast<double>(val);
          }
        }
      }
      ++counter;
    }

    redis_client.set_value(params.data_key, std::to_string(n));

    if (!CheckNonTerminatedAndUpdate(redis_client, params.command_key, FINISH_INITIALIZATION)) {
      throw std::runtime_error("Step 1 finish, got termination command");
    }

    

// УСЛОВИЕ НА ТО, ЧЕГО ЖДЁМ
    //if (!continue_fitting) {
   // if (!WaitForFlag(redis_client, params.command_key, START_INITIALIZATION)) { return -2; };


/*
  if (!continue_fitting) {
    Normalize(p_wt, *n_wt);
  }
  */


  /*
  Blas* blas = Blas::builtin();
  for (int iter = 0; iter < params.num_outer_iters; ++iter) {
    float perplexity_value = 0.0f;
    for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(params.batches_dir_path), { })) {
      artm::Batch batch;
      Helpers::LoadBatch(entry.path().string(), &batch);
      ProcessEStep(batch, *p_wt, n_wt, params.num_inner_iters, blas, &perplexity_value);
    }
    std::cout << "Perplexity: " << exp(-(1.0f / n) * perplexity_value) << std::endl;
    Normalize(p_wt, *n_wt);
  }
  */
  } catch (const std::exception& error) {
    redis_client.set_value(params.command_key, FINISH_TERMINATION);
    throw std::runtime_error(params.data_key + " " + error.what());
  } catch (...) {
    redis_client.set_value(params.command_key, FINISH_TERMINATION);
    throw;
  }
  
  // normal termination
  redis_client.set_value(params.command_key, FINISH_TERMINATION);

  std::cout << "Executor with cmd key '" << params.command_key
            << "' has finished! Elapsed time: " << float(clock() - begin_time) / CLOCKS_PER_SEC
            << " sec." << std::endl;
  return 0;
}
