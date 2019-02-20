#include <unistd.h>
#include <signal.h>

#include <cmath> 

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <boost/program_options.hpp>

#include "glog/logging.h"

#include "executor_thread.h"
#include "helpers.h"
#include "redis_phi_matrix.h"
#include "redis_client.h"
#include "protocol.h"
#include "token.h"

namespace po = boost::program_options;

// ToDo(MelLain): add correct signal handling and threads termination

//volatile sig_atomic_t signal_flag = 0;

//void signal_handler(int sig) {
//  signal_flag = 1;
//}

struct Parameters {
  int num_topics;
  int num_inner_iters;
  int num_threads;
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
              << "num-threads: "       << parameters.num_threads       << "; "
              << "batches-dir-path: "  << parameters.batches_dir_path  << "; "
              << "vocab-path: "        << parameters.vocab_path        << "; "
              << "redis-ip: "          << parameters.redis_ip          << "; "
              << "redis-port: "        << parameters.redis_port        << "; "
              << "continue-fitting: "  << parameters.continue_fitting  << "; "
              << "cache phi: "         << parameters.cache_phi         << "; "
              << "token-begin-index: " << parameters.token_begin_index << "; "
              << "token-end-index: "   << parameters.token_end_index   << "; "
              << "batch-begin-index: " << parameters.batch_begin_index << "; "
              << "batch-end-index: "   << parameters.batch_end_index   << "; "
              << "executor-id: "       << parameters.executor_id;
}

void check_parameters(const Parameters& parameters) {
  if (parameters.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }

  if (parameters.num_inner_iters <= 0) {
    throw std::runtime_error("num_inner_iters should be a positive integer");
  }

  if (parameters.num_threads <= 0) {
    throw std::runtime_error("num_threads should be a positive integer");
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

  if (parameters.executor_id < 0) {
    throw std::runtime_error("executor_id should be a non-negative integer");
  }
}

bool parse_and_print_parameters(int argc, char* argv[], Parameters* parameters) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help", "Show help")
    ("num-topics",        po::value(&parameters->num_topics)->default_value(1),         "Number of topics")  // NOLINT
    ("num-inner-iter",    po::value(&parameters->num_inner_iters)->default_value(1),    "Number of document passes")  // NOLINT
    ("num-threads",       po::value(&parameters->num_threads)->default_value(1),        "Number of executor processor threads")  // NOLINT
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

std::vector<std::pair<int, int>> get_indices(int num_threads, int begin_index, int end_index) {
  int step = std::ceil((end_index - begin_index) / static_cast<double>(num_threads));

  std::vector<std::pair<int, int>> retval;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    retval.push_back(std::make_pair(begin_index + step * thread_id,
                                    std::min(begin_index + step * (thread_id + 1), end_index)));
  }

  return retval;
}

//bool wait_for_termination() {
    //if (signal_flag) {
      //LOG(ERROR) << "SIGINT has been caught, start terminating" << std::endl;
      //return false;
    //}
//}

int main(int argc, char* argv[]) {
  //signal(SIGINT, signal_handler);

  Parameters parameters;
  bool is_help_call = parse_and_print_parameters(argc, argv, &parameters);
  if (is_help_call) {
    return 0;
  }

  FLAGS_minloglevel = 0;
  FLAGS_log_dir = ".";

  const int executor_id = parameters.executor_id;
  std::string log_file = std::string("cluster-bigartm-") + std::to_string(executor_id);
  google::InitGoogleLogging(log_file.c_str());
  log_parameters(parameters);
  check_parameters(parameters);

  LOG(INFO) << "Executor " << executor_id << ": has started";

  LOG(INFO) << "Executor " << executor_id << ": start connecting redis at "
            << parameters.redis_ip << ":" << parameters.redis_port;

  auto redis_client = std::shared_ptr<RedisClient>(new RedisClient(parameters.redis_ip,
                                                                   std::stoi(parameters.redis_port)));

  LOG(INFO) << "Executor " << executor_id << ": finish connecting to redis";
 
  LOG(INFO) << "Executor " << executor_id << ": start creating threads";

  std::vector<std::string> command_keys = generate_command_keys(parameters.executor_id, parameters.num_threads);
  std::vector<std::string> data_keys = generate_data_keys(parameters.executor_id, parameters.num_threads);

  try {
    std::vector<std::pair<int, int>> token_indices = get_indices(parameters.num_threads,
                                                                 parameters.token_begin_index,
                                                                 parameters.token_end_index);

    std::vector<std::pair<int, int>> batch_indices = get_indices(parameters.num_threads,
                                                                 parameters.batch_begin_index,
                                                                 parameters.batch_end_index);
    LOG(INFO) << "Executor " << executor_id
              << ": first token index is " << token_indices[0].first
              << ", last token index is " << token_indices[token_indices.size() - 1].second
              << "; first batch index is " << batch_indices[0].first
              << ", last batch index is " << batch_indices[batch_indices.size() - 1].second;

    std::vector<std::string> topics;
    for (int i = 0; i < parameters.num_topics; ++i) {
      topics.push_back("topic_" + std::to_string(i));
    }

    LOG(INFO) << "Executor " << executor_id << ": start creating of matrices";

    bool use_cache = (parameters.cache_phi == 1);
    auto p_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("pwt"), topics, use_cache));

    auto n_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("nwt"), topics));

    int counter = 0;
    auto zero_vector = std::vector<float>(p_wt->topic_size(), 0.0f);

    std::ifstream fin;
    std::string line;
    fin.open(parameters.vocab_path);

    bool continue_fitting = parameters.continue_fitting == 1;
    while (std::getline(fin, line)) {
      auto token = Token(DefaultClass, line);
      bool add_token_to_redis = false;

      if (counter >= parameters.token_begin_index && counter < parameters.token_end_index) {
        add_token_to_redis = !continue_fitting;
      }

      p_wt->add_token(redis_client, token, add_token_to_redis, zero_vector);
      n_wt->add_token(redis_client,
                      token,
                      add_token_to_redis,
                      add_token_to_redis ? Helpers::generate_random_vector(p_wt->topic_size(), token) : zero_vector);
      ++counter;
    }
    fin.close();

    LOG(INFO) << "Executor " << executor_id << ": " << "number of tokens: " << p_wt->token_size()
              << "; redis matrices had been reset: " << !continue_fitting;

    std::vector<std::shared_ptr<ExecutorThread>> threads;
    for (int thread_id = 0; thread_id < parameters.num_threads; ++thread_id) {
      auto thread_redis_client = std::shared_ptr<RedisClient>(
          new RedisClient(parameters.redis_ip, std::stoi(parameters.redis_port)));

      threads.push_back(std::shared_ptr<ExecutorThread>(
        new ExecutorThread(command_keys[thread_id],
                           data_keys[thread_id],
                           thread_redis_client,
                           continue_fitting,
                           parameters.batches_dir_path,
                           token_indices[thread_id].first,
                           token_indices[thread_id].second,
                           batch_indices[thread_id].first,
                           batch_indices[thread_id].second,
                           parameters.num_inner_iters,
                           std::make_shared<RedisPhiMatrixAdapter>(RedisPhiMatrixAdapter(p_wt, thread_redis_client)),
                           std::make_shared<RedisPhiMatrixAdapter>(RedisPhiMatrixAdapter(n_wt, thread_redis_client)))
      ));
    }

    while (true) {
      bool terminate = false;
      for (const auto& thread : threads) {
        if (thread->is_stopping()) {
          terminate = true;
          break;
        }
      }

      if (terminate) {
        break;
      }
      usleep(2000);
    }

    //if (!wait_for_termination()) {
    //}
  } catch (...) {
    // ToDo(MelLain): add correct handling
    throw;
  }

  return 0;
}
