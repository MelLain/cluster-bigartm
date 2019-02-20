#include <cstdio>
#include <unistd.h>
#include <signal.h>

#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "glog/logging.h"

#include "common.h"
#include "phi_matrix.h"
#include "protocol.h"
#include "redis_client.h"
#include "redis_phi_matrix.h"
#include "token.h"
#include "helpers.h"

namespace po = boost::program_options;

volatile sig_atomic_t signal_flag = 0;

void signal_handler(int sig) {
  signal_flag = 1;
}

struct Parameters {
  int num_topics;
  int num_outer_iters;
  int num_executors;
  std::string batches_dir_path;
  std::string vocab_path;
  std::string redis_ip;
  std::string redis_port;
  int show_top_tokens;
  int continue_fitting;
};

void log_parameters(const Parameters& parameters) {
  LOG(INFO) << "num-topics:       " << parameters.num_topics       << "; "
            << "num-outer-iter:   " << parameters.num_outer_iters  << "; "
            << "num_executors:    " << parameters.num_executors    << "; "
            << "batches-dir-path: " << parameters.batches_dir_path << "; "
            << "vocab-path:       " << parameters.vocab_path       << "; "
            << "redis-ip:         " << parameters.redis_ip         << "; "
            << "redis-port:       " << parameters.redis_port       << "; "
            << "show-top-tokens:  " << parameters.show_top_tokens  << "; "
            << "continue-fitting: " << parameters.continue_fitting;
}

void check_parameters(const Parameters& parameters) {
  if (parameters.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }

  if (parameters.num_outer_iters <= 0) {
    throw std::runtime_error("num_outer_iters should be a positive integer");
  }

  if (parameters.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }

  if (parameters.num_executors <= 0) {
    throw std::runtime_error("num_executors should be a positive integer");
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

  if (parameters.show_top_tokens != 0 && parameters.show_top_tokens != 1) {
    throw std::runtime_error("show_top_tokens should be equal to 0 or 1");
  }
}

bool parse_and_print_parameters(int argc, char* argv[], Parameters* parameters) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help", "Show help")
    ("num-topics",       po::value(&parameters->num_topics)->default_value(1),         "Number of topics")  // NOLINT
    ("num-outer-iter",   po::value(&parameters->num_outer_iters)->default_value(1),    "Number of collection passes")  // NOLINT
    ("num-executors",    po::value(&parameters->num_executors)->default_value(1),      "Number of working processes")  // NOLINT
    ("batches-dir-path", po::value(&parameters->batches_dir_path)->default_value("."), "Path to batches with documents")  // NOLINT
    ("vocab-path",       po::value(&parameters->vocab_path)->default_value("."),       "Path to file with vocabulary")  // NOLINT
    ("redis-ip",         po::value(&parameters->redis_ip)->default_value(""),          "IP of redis instance")  // NOLINT
    ("redis-port",       po::value(&parameters->redis_port)->default_value(""),        "Port of redis instance")  // NOLINT
    ("show-top-tokens",  po::value(&parameters->show_top_tokens)->default_value(0),    "1 - print top tokens, 0 - not")  // NOLINT
    ("continue-fitting", po::value(&parameters->continue_fitting)->default_value(0),   "1 - continue fitting redis model, 0 - restart")  // NOLINT
    ;

  po::variables_map variables_map;
  store(po::command_line_parser(argc, argv).options(all_options).run(), variables_map);
  notify(variables_map);

  bool show_help = (variables_map.count("help") > 0);
  if (show_help) {
    std::cerr << all_options;
    return true;
  }

  std::cout << "num-topics:       " << parameters->num_topics        << std::endl;
  std::cout << "num-outer-iter:   " << parameters->num_outer_iters   << std::endl;
  std::cout << "num-executors:    " << parameters->num_executors     << std::endl;
  std::cout << "batches-dir-path: " << parameters->batches_dir_path  << std::endl;
  std::cout << "vocab-path:       " << parameters->vocab_path        << std::endl;
  std::cout << "redis-ip:         " << parameters->redis_ip          << std::endl;
  std::cout << "redis-port:       " << parameters->redis_port        << std::endl;
  std::cout << "show-top-tokens:  " << parameters->show_top_tokens   << std::endl;
  std::cout << "continue-fitting: " << parameters->continue_fitting  << std::endl;

  return false;
}

bool check_finished_or_terminated(const RedisClient& redis_client,
                                  const std::vector<std::string>& command_keys,
                                  const std::string& old_flag,
                                  const std::string& new_flag,
                                  int timeout = -1)
{
  int time_passed = 0;
  bool terminated = false;
  while (true) {
    if (signal_flag) {
      LOG(ERROR) << "SIGINT has been caught, start terminating" << std::endl;
      return false;
    }

    int executors_finished = 0;
    for (const auto& key : command_keys) {
      auto reply = redis_client.get_value(key);
      if (reply == old_flag) {
        break;
      }

      if (reply == new_flag) {
        ++executors_finished;
        continue;
      }

      if (reply == FINISH_TERMINATION) {
        terminated = true;
        break;
      }
    }

    if (executors_finished == command_keys.size()) {
      return true;
    }

    if ((timeout > 0 && time_passed > timeout) || terminated) {
      break;
    }

    usleep(2000);
    time_passed += 2000;
  }
  return false;
}

// this function firstly check the availability of executor and then send him new command,
// it's not fully safe, as if the executor fails in between get and set, it will cause
// endless loop during the next syncronozation
bool check_non_terminated_and_update(const RedisClient& redis_client,
                                     const std::vector<std::string>& command_keys,
                                     const std::string& flag)
{
  if (signal_flag) {
    LOG(ERROR) << "SIGINT has been caught, start terminating" << std::endl;
    return false;
  }

  for (const auto& key : command_keys) {
    auto reply = redis_client.get_value(key);
    if (reply == FINISH_TERMINATION) {
      return false;
    }
  }

  for (const auto& key : command_keys) {
    redis_client.set_value(key, flag);
  }

  return true;
}

// protocol:
// 1) set everyone START_NORMALIZATION flag
// 2) wait for everyone to set FINISH_NORMALIZATION flag
// 3) read results from data slots
// 4) merge results and put final n_t into data slots
// 5) set everyone START_NORMALIZATION flag
// 6) wait for everyone to set FINISH_NORMALIZATION flag
bool normalize_nwt(const RedisClient& redis_client,
                  const std::vector<std::string>& command_keys,
                  const std::vector<std::string>& data_keys,
                  int num_topics)
{
  if (!check_non_terminated_and_update(redis_client, command_keys, START_NORMALIZATION)) {
    return false;
  }

  if (!check_finished_or_terminated(redis_client, command_keys, START_NORMALIZATION, FINISH_NORMALIZATION)) {
    return false;
  }

  Normalizers n_t;
  Normalizers helper;
  for (const auto& key : data_keys) {
    helper = redis_client.get_hashmap(key, num_topics);
    for (const auto& kv : helper) {
      auto iter = n_t.find(kv.first);
      if (iter == n_t.end()) {
        n_t.emplace(kv);
      } else {
        for (int i = 0; i < kv.second.size(); ++i) {
          iter->second[i] += kv.second[i];
        }
      }
    }
    helper.clear();
  }

  // ToDo(MelLain): maybe it'll be better to keep only one version of n_t for
  //                all executors, need to be checked with large number of topics
  for (const auto& key : data_keys) {
    redis_client.set_hashmap(key, n_t);
  }

  if (!check_non_terminated_and_update(redis_client, command_keys, START_NORMALIZATION)) {
    return false;
  }

  if (!check_finished_or_terminated(redis_client, command_keys, START_NORMALIZATION, FINISH_NORMALIZATION)) {
    return false;
  }

  return true;
}

// ToDo(MelLain): rewrite this function, as it is very inefficient and hacked now
void print_top_tokens(RedisClient& redis_client,
                      const std::string& vocab_path,
                      int num_topics,
                      int num_tokens = 10)
{
  std::vector<std::string> topics;
  for (int i = 0; i < num_topics; ++i) {
    topics.push_back("topic_" + std::to_string(i));
  }

  auto p_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("pwt"), topics, redis_client));
  auto zero_vector = std::vector<float>(num_topics, 0.0f);

  std::ifstream fin;
  std::string line;
  fin.open(vocab_path);
  while (std::getline(fin, line)) {
    p_wt->add_token(Token(DefaultClass, line), false, zero_vector);
  }
  fin.close();

  for (int i = 0; i < p_wt->topic_size(); ++i) {
    std::vector<std::pair<Token, float>> pairs;
    for (int j = 0; j < p_wt->token_size(); ++j) {
      pairs.push_back(std::make_pair(p_wt->token(j), p_wt->get(j, i)));
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<Token, float>& p1, const std::pair<Token, float>& p2) {
                return p1.second > p2.second;
              });
    std::cout << "\nTopic: " << p_wt->topic_name(i) << std::endl;
    for (int j = 0; j < num_tokens; ++j) {
      std::cout << pairs[j].first.keyword << " (" << pairs[j].second << ")\n";
    }
  }
}

// In case of fault of master without exceptions and SIGINT signal all sub-processes
// (executors) can be killed within a single node via command:
// ps -ef | grep './executor_main' | grep -v grep | awk '{print $2}' | xargs kill -9
int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  Parameters parameters;
  bool is_help_call = parse_and_print_parameters(argc, argv, &parameters);
  if (is_help_call) {
    return 0;
  }

  FLAGS_minloglevel = 0;
  FLAGS_log_dir = ".";

  std::string log_file = std::string("cluster-bigartm-master");
  google::InitGoogleLogging(log_file.c_str());  

  log_parameters(parameters);
  check_parameters(parameters);

  LOG(INFO) << "Master: start connecting redis at " << parameters.redis_ip << ":" << parameters.redis_port;

  RedisClient redis_client = RedisClient(parameters.redis_ip, std::stoi(parameters.redis_port), 100);

  LOG(INFO) << "Master: finish connecting to redis";

  std::vector<std::string> executor_command_keys;
  std::vector<std::string> executor_data_keys;

  LOG(INFO) << "Master: start creating ids";
  for (int executor_id = 0; executor_id < parameters.num_executors; ++executor_id) {
    executor_command_keys.push_back(generate_command_key(executor_id));
    executor_data_keys.push_back(generate_data_key(executor_id));
  }

  LOG(INFO) << "Master: finish creating ids";

  try {
    // we give 1.0 sec to all executors to start, if even one of them
    // didn't response, it means, that it had failed to start
    LOG(INFO) << "Master: start connecting to processors";

    bool ok = check_finished_or_terminated(redis_client, executor_command_keys,
                                           START_GLOBAL_START, FINISH_GLOBAL_START, 1000000);
    if (!ok) { throw std::runtime_error("Master: step 0, got termination status"); }

    LOG(INFO) << "Master: finish connecting to processors";

    LOG(INFO) << "Master: start initialization";

    ok = check_non_terminated_and_update(redis_client, executor_command_keys, START_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 start, got termination status"); }

    ok = check_finished_or_terminated(redis_client, executor_command_keys,
                                      START_INITIALIZATION, FINISH_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 finish, got termination status"); }

    LOG(INFO) << "Master: finish initialization";

    double n = 0.0;
    for (const auto& key : executor_data_keys) {
      n += std::stod(redis_client.get_value(key));
    }

    LOG(INFO) << "Master: all executors have started! Total number of token slots in collection: " << n;
    std::cout << "Master: all executors have started! Total number of token slots in collection: " << n << std::endl;

    if (!parameters.continue_fitting) {
      if (!normalize_nwt(redis_client, executor_command_keys, executor_data_keys, parameters.num_topics)) {
        throw std::runtime_error("Step 2, got termination status");
      }
    }

    // EM-iterations
    for (int iteration = 0; iteration < parameters.num_outer_iters; ++iteration) {
      LOG(INFO) << "Master: start iteration " << iteration;

      ok = check_non_terminated_and_update(redis_client, executor_command_keys, START_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 start, got termination status"); }

      ok = check_finished_or_terminated(redis_client, executor_command_keys, START_ITERATION, FINISH_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 intermediate, got termination status"); }

      double perplexity_value = 0.0;
      for (const auto& key : executor_data_keys) {
        perplexity_value += std::stod(redis_client.get_value(key));
      }

      LOG(INFO) << "Master: finish e-step, start m-step";

      if (!normalize_nwt(redis_client, executor_command_keys, executor_data_keys, parameters.num_topics)) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      perplexity_value = exp(-(1.0f / n) * perplexity_value);
      LOG(INFO) << "Iteration: " << iteration << ", perplexity: " << perplexity_value;
      std::cout << "Iteration: " << iteration << ", perplexity: " << perplexity_value << std::endl;
      LOG(INFO) << "Iteration: " << iteration << ", maxrss: " << Helpers::get_peak_memory_kb() << " KB";
    }

    // finalization (correct in any way)
    for (const auto& key : executor_command_keys) {
      redis_client.set_value(key, START_TERMINATION);
    }

  } catch (...) {
    for (const auto& key : executor_command_keys) {
      redis_client.set_value(key, START_TERMINATION);
    }
    throw;
  }
  check_finished_or_terminated(redis_client, executor_command_keys, START_TERMINATION, FINISH_TERMINATION);

  if (parameters.show_top_tokens) {
    print_top_tokens(redis_client, parameters.vocab_path, parameters.num_topics);
  }

  LOG(INFO) << "Model fitting is finished!";
  LOG(INFO) << "final maxrss= " << Helpers::get_peak_memory_kb() << " KB";

  return 0;
}
