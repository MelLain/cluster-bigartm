#include <unistd.h>
#include <signal.h>

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

#include "glog/logging.h"

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

volatile sig_atomic_t signal_flag = 0;

void signal_handler(int sig) {
  signal_flag = 1;
}

struct Parameters {
  int num_topics;
  int num_inner_iters;
  std::string batches_dir_path;
  std::string vocab_path;
  std::string redis_ip;
  std::string redis_port;
  int continue_fitting;
  int cache_phi;
  int token_begin_index;
  int token_end_index;
  int batch_begin_index;
  int batch_end_index;
  int executor_id;
};

void log_parameters(const Parameters& parameters) {
    LOG(INFO) << "num-topics: "        << parameters.num_topics        << "; "
              << "num-inner-iter: "    << parameters.num_inner_iters   << "; "
              << "batches-dir-path: "  << parameters.batches_dir_path  << "; "
              << "vocab-path: "        << parameters.vocab_path        << "; "
              << "redis-ip: "          << parameters.redis_ip          << "; "
              << "redis-port: "        << parameters.redis_port        << "; "
              << "continue-fitting: "  << parameters.continue_fitting  << "; "
              << "cache phi: "         << parameters.cache_phi         << "; "
              << "token-begin-index: " << parameters.token_begin_index << "; "
              << "token-end-index: "   << parameters.token_end_index   << "; "
              << "batch-begin-index: " << parameters.batch_begin_index << "; "
              << "batch-end-index: "   << parameters.batch_end_index;
}

void check_parameters(const Parameters& parameters) {
  if (parameters.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }

  if (parameters.num_inner_iters <= 0) {
    throw std::runtime_error("num_inner_iters should be a positive integer");
  }

  if (parameters.batches_dir_path == "") {
    throw std::runtime_error("batches_dir_path should be non-empty");
  }

  if (parameters.vocab_path == "") {
    throw std::runtime_error("vocab_path should be non-empty");
  }

  if (parameters.redis_ip == "") {
    throw std::runtime_error("redis_ip should be non-empty");
  }

  if (parameters.redis_port == "") {
    throw std::runtime_error("redis_port should be non-empty");
  }

  if (parameters.continue_fitting != 0 && parameters.continue_fitting != 1) {
    throw std::runtime_error("continue_fitting should be equal to 0 or 1");
  }

  if (parameters.cache_phi != 0 && parameters.cache_phi != 1) {
    throw std::runtime_error("cache_phi should be equal to 0 or 1");
  }

  if (parameters.redis_port == "") {
    throw std::runtime_error("redis_port should be non-empty");
  }

  if (parameters.token_begin_index < 0 || parameters.token_end_index < 0
      || parameters.token_end_index < parameters.token_begin_index)
  {
    throw std::runtime_error("token_begin_index should be > 0 and <= token_end_index");
  }

  if (parameters.batch_begin_index < 0 || parameters.batch_end_index < 0
      || parameters.batch_end_index < parameters.batch_begin_index)
  {
    throw std::runtime_error("batch_begin_index should be > 0 and <= batch_end_index");
  }

  if (parameters.executor_id <= 0) {
    throw std::runtime_error("executor_id should be a positive integer");
  }
}

bool parse_and_print_parameters(int argc, char* argv[], Parameters* parameters) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help", "Show help")
    ("num-topics",        po::value(&parameters->num_topics)->default_value(1),         "Number of topics")  // NOLINT
    ("num-inner-iter",    po::value(&parameters->num_inner_iters)->default_value(1),    "Number of document passes")  // NOLINT
    ("batches-dir-path",  po::value(&parameters->batches_dir_path)->default_value("."), "Path to files with documents")  // NOLINT
    ("vocab-path",        po::value(&parameters->vocab_path)->default_value("."),       "Path to files with documents")  // NOLINT
    ("redis-ip",          po::value(&parameters->redis_ip)->default_value(""),          "IP of redis instance")  // NOLINT
    ("redis-port",        po::value(&parameters->redis_port)->default_value(""),        "Port of redis instance")  // NOLINT
    ("continue-fitting",  po::value(&parameters->continue_fitting)->default_value(0),   "1 - continue fitting redis model, 0 - restart")  // NOLINT
    ("cache-phi",         po::value(&parameters->cache_phi)->default_value(0),          "1 - cache phi matrix per iter, 0 - go to redis")  // NOLINT
    ("token-begin-index", po::value(&parameters->token_begin_index)->default_value(0),  "Index of token to init/norm from")  // NOLINT
    ("token-end-index",   po::value(&parameters->token_end_index)->default_value(0),    "Index of token to init/norm to (excluding)")  // NOLINT
    ("batch-begin-index", po::value(&parameters->batch_begin_index)->default_value(0),  "Index of batch to process from")  // NOLINT
    ("batch-end-index",   po::value(&parameters->batch_end_index)->default_value(0),    "Index of batch to process to (excluding)")  // NOLINT
    ("executor-id",       po::value(&parameters->executor_id)->default_value(-1),       "Unique identifier of the process")  // NOLINT
    ;

  po::variables_map variables_map;
  store(po::command_line_parser(argc, argv).options(all_options).run(), variables_map);
  notify(variables_map);

  bool show_help = (variables_map.count("help") > 0);
  if (show_help) {
    std::cerr << all_options;
    return true;
  }

  return false;
}

bool check_non_terminated_and_update(const RedisClient& redis_client,
                                     const std::string& key,
                                     const std::string& flag,
                                     bool force = false)
{
  if (signal_flag) {
    LOG(ERROR) << "SIGINT has been caught, start terminating" << std::endl;
    return false;
  }

  if (!force) {
  auto reply = redis_client.get_value(key);
    if (reply == START_TERMINATION) {
      return false;
    }
  }

  redis_client.set_value(key, flag);
  return true;
}

bool wait_for_flag(const RedisClient& redis_client, const std::string& key, const std::string& flag) {
  while (true) {
    if (signal_flag) {
      LOG(ERROR) << "SIGINT has been caught, start terminating" << std::endl;
      return false;
    }

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


Normalizers find_nt(const PhiMatrix& n_wt, const std::pair<int, int> begin_end_indices) {
  LOG(INFO) << "FindNt: begin_index = " << begin_end_indices.first
                    << ", end_index = " << begin_end_indices.second;
  Normalizers retval;
  std::vector<float> helper = std::vector<float>(n_wt.topic_size(), 0.0f);
  for (int token_id = 0; token_id < n_wt.token_size(); ++token_id) {
    if (token_id < begin_end_indices.first || token_id >= begin_end_indices.second) {
      continue;
    }

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

// protocol:
// 1) wait for START_NORMALIZATION flag
// 2) after reaching it compute n_t on tokens from executor range
// 3) put results into data slot and set cmd slot to FINISH_NORMALIZATION
// 4) wait for new START_NORMALIZATION flag
// 5) read total n_t from data slot
// 6) proceed final normalization on tokens from executor range
// 7) set FINISH_NORMALIZATION flag and return
bool normalize_nwt(std::shared_ptr<PhiMatrix> p_wt,
                  std::shared_ptr<PhiMatrix> n_wt,
                  const std::pair<int, int> begin_end_indices,
                  const RedisClient& redis_client,
                  const std::string& command_key,
                  const std::string& data_key)
{
  if(!wait_for_flag(redis_client, command_key, START_NORMALIZATION)) {
    return false;
  }

  LOG(INFO) << "NormalizeNwt: begin_index = " << begin_end_indices.first
                          << ", end_index = " << begin_end_indices.second;

  const int num_topics = n_wt->topic_size();
  const int num_tokens = n_wt->token_size();
  const std::vector<float> zeros(num_topics, 0.0f);

  assert(p_wt->token_size() == n_wt->token_size() && p_wt->topic_size() == n_wt->topic_size());

  Normalizers n_t = find_nt(*n_wt, begin_end_indices);
  redis_client.set_hashmap(data_key, n_t);

  if (!check_non_terminated_and_update(redis_client, command_key, FINISH_NORMALIZATION)) {
    return false;
  }

  if(!wait_for_flag(redis_client, command_key, START_NORMALIZATION)) {
    return false;
  } 

  n_t = redis_client.get_hashmap(data_key, num_topics);

  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    if (token_id < begin_end_indices.first || token_id >= begin_end_indices.second) {
      continue;
    }

    const Token& token = n_wt->token(token_id);
    assert(p_wt->token(token_id) == token);
    const std::vector<double>& n_t_for_class_id = n_t[token.class_id];

    std::vector<float> helper = std::vector<float>(num_topics, 0.0f);
    std::vector<float> helper_n_wt = std::vector<float>(num_topics, 0.0f);

    n_wt->get_set(token_id, &helper_n_wt, zeros);
    for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
      float value = 0.0f;
      if (n_t_for_class_id[topic_index] > 0) {
        value = std::max<double>(helper_n_wt[topic_index], 0.0f) / n_t_for_class_id[topic_index];
        if (value < 1e-16f) {
          value = 0.0f;
        }
      }
      helper[topic_index] = value;
    }
    p_wt->set(token_id, helper);
  }

  if (!check_non_terminated_and_update(redis_client, command_key, FINISH_NORMALIZATION)) {
    return false;
  }

  return true;
}

void process_e_step(const artm::Batch& batch,
                    const PhiMatrix& p_wt,
                    std::shared_ptr<PhiMatrix> n_wt,
                    int num_inner_iters,
                    Blas* blas,
                    double* perplexity_value)
{
  std::shared_ptr<CsrMatrix<float>> sparse_ndw;
  sparse_ndw = ProcessorHelpers::initialize_sparse_ndw(batch);

  std::shared_ptr<LocalThetaMatrix<float>> theta_matrix;
  theta_matrix = ProcessorHelpers::initialize_theta(p_wt.topic_size(), batch);

  std::shared_ptr<NwtWriteAdapter> nwt_writer = std::make_shared<NwtWriteAdapter>(n_wt.get());

  ProcessorHelpers::infer_theta_and_update_nwt_sparse(batch, *sparse_ndw, p_wt, theta_matrix.get(),
                                                 nwt_writer.get(), blas, num_inner_iters, perplexity_value);
}

int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  Parameters parameters;
  bool is_help_call = parse_and_print_parameters(argc, argv, &parameters);
  if (is_help_call) {
    return 0;
  }

  const std::string command_key = generate_command_key(parameters.executor_id);
  const std::string data_key = generate_data_key(parameters.executor_id);

  FLAGS_minloglevel = 0;
  FLAGS_log_dir = ".";

  std::string log_file = std::string("cluster-bigartm-") + command_key;
  google::InitGoogleLogging(log_file.c_str());
  log_parameters(parameters);
  check_parameters(parameters);

  LOG(INFO) << "Processor " << command_key << ": has started";

  LOG(INFO) << "Processor " << command_key << ": start connecting redis at "
            << parameters.redis_ip << ":" << parameters.redis_port;

  RedisClient redis_client = RedisClient(parameters.redis_ip, std::stoi(parameters.redis_port), 100);
  LOG(INFO) << "Processor " << command_key << ": finish connecting to redis";

  try {
    LOG(INFO) << "Processor " << command_key << ": start connecting to master";

    if (!check_non_terminated_and_update(redis_client, command_key, FINISH_GLOBAL_START, true)) {
      throw std::runtime_error("Step 0, got termination command");
    };

    if (!wait_for_flag(redis_client, command_key, START_INITIALIZATION)) {
      throw std::runtime_error("Step 1 start, got termination command");
    };

    LOG(INFO) << "Processor " << command_key << ": finish connecting to master";

    std::vector<std::string> topics;
    for (int i = 0; i < parameters.num_topics; ++i) {
      topics.push_back("topic_" + std::to_string(i));
    }

    LOG(INFO) << "Processor " << command_key << ": start creating and initialization of matrices";

    bool use_cache = (parameters.cache_phi == 1);
    auto p_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("pwt"), topics, redis_client, use_cache));
    auto n_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("nwt"), topics, redis_client));

    bool continue_fitting = (parameters.continue_fitting == 1);

    std::pair<int, int> token_indices = std::make_pair(parameters.token_begin_index, parameters.token_end_index);

    int counter = 0;
    auto zero_vector = std::vector<float>(parameters.num_topics, 0.0f);

    LOG(INFO) << "Gather dictionary and create matrices";
    std::ifstream fin;
    std::string line;
    fin.open(parameters.vocab_path);
    while (std::getline(fin, line)) {
      auto token = Token(DefaultClass, line);
      bool add_token_to_redis = false;
      if (counter >= token_indices.first && counter < token_indices.second) {
        add_token_to_redis = !continue_fitting;
      }
      p_wt->add_token(token, add_token_to_redis, zero_vector);
      n_wt->add_token(token, add_token_to_redis,
        add_token_to_redis ? Helpers::generate_random_vector(parameters.num_topics, token) : zero_vector);
      ++counter;
    }
    fin.close();

    LOG(INFO) << "Number of tokens: " << n_wt->token_size()
              << "; redis matrices had been reset: " << !continue_fitting;

    LOG(INFO) << "Gather total number of token slots in batches from "
              << parameters.batch_begin_index << " to " << parameters.batch_end_index;

    double n = 0.0;
    counter = 0;
    for(const auto& entry : boost::make_iterator_range(bf::directory_iterator(parameters.batches_dir_path), { })) {
      if (counter >= parameters.batch_begin_index && counter < parameters.batch_end_index) {
        artm::Batch batch;
        Helpers::load_batch(entry.path().string(), &batch);
        for (const auto& item : batch.item()) {
          for (float val : item.token_weight()) {
            n += static_cast<double>(val);
          }
        }
      }
      ++counter;
    }

    redis_client.set_value(data_key, std::to_string(n));
    LOG(INFO) << "Processor " << command_key << ": finish initialization of matrices, total number of slots: " << n
              << " from " << counter << " batches";

    if (!check_non_terminated_and_update(redis_client, command_key, FINISH_INITIALIZATION)) {
      throw std::runtime_error("Step 1 finish, got termination command");
    }

    if (!continue_fitting) {
      LOG(INFO) << "Processor " << command_key << ": start normalization";

      if (!normalize_nwt(p_wt, n_wt, token_indices, redis_client, command_key, data_key)) {
        throw std::runtime_error("Step 2, got termination status");
      }

      LOG(INFO) << "Processor " << command_key << ": finish normalization";
    }

    Blas* blas = Blas::builtin();
    while (true) {
      LOG(INFO) << "Processor " << command_key << ": start new iteration";

      // false here and only here means valid termination
      if (!wait_for_flag(redis_client, command_key, START_ITERATION)) {
        break;
      };

      double perplexity_value = 0.0;
      counter = 0;
      LOG(INFO) << "Processor " << command_key << ": start processing of E-step";

      for(const auto& entry : boost::make_iterator_range(bf::directory_iterator(parameters.batches_dir_path), { })) {
        if (counter >= parameters.batch_begin_index && counter < parameters.batch_end_index) {
          artm::Batch batch;
          const std::string batch_name = entry.path().string();
          LOG(INFO) << "Start processing batch " << batch_name;

          Helpers::load_batch(batch_name, &batch);
          process_e_step(batch, *p_wt, n_wt, parameters.num_inner_iters, blas, &perplexity_value);

          LOG(INFO) << "Finish processing batch " << batch_name;
        }
        ++counter;
      }
      // ToDo(mel-lain): add option to clear per batch, not per iter
      p_wt->clear_cache();

      LOG(INFO) << "Local pre-perplexity value: " << perplexity_value;

      redis_client.set_value(data_key, std::to_string(perplexity_value));

      if (!check_non_terminated_and_update(redis_client, command_key, FINISH_ITERATION)) {
        throw std::runtime_error("Step 3 start, got termination command");
      }

      LOG(INFO) << "Processor " << command_key << ": finish processing of E-step, start M-step";

      if (!normalize_nwt(p_wt, n_wt, token_indices, redis_client, command_key, data_key)) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      LOG(INFO) << "Processor " << command_key << ": finish iteration";
      LOG(INFO) << "Processor " << command_key << ": maxrss= " << Helpers::get_peak_memory_kb() << " KB";
    }

    // normal termination
    redis_client.set_value(command_key, FINISH_TERMINATION);

  } catch (const std::exception& error) {
    redis_client.set_value(command_key, FINISH_TERMINATION);
    throw std::runtime_error(data_key + " " + error.what());
  } catch (...) {
    redis_client.set_value(command_key, FINISH_TERMINATION);
    throw;
  }

  LOG(INFO) << "Executor with cmd key '" << command_key << "' has finished!";
  LOG(INFO) << "Executor with cmd key '" << command_key << "' final maxrss= " << Helpers::get_peak_memory_kb() << " KB";

  return 0;
}
