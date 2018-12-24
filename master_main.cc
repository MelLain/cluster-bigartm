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
  int num_inner_iters;
  std::string executor_ids_path;
  std::string batches_dir_path;
  std::string vocab_path;
  std::string redis_ip;
  std::string redis_port;
  int show_top_tokens;
  int continue_fitting;
  int debug_print;
};

void LogParams(const Parameters& params) {
  LOG(INFO) << "num-topics: "        << params.num_topics << "; "
            << "num-outer-iter: "    << params.num_outer_iters << "; "
            << "executor-ids-path: " << params.executor_ids_path << "; "
            << "batches-dir-path: "  << params.batches_dir_path << "; "
            << "vocab-path: "        << params.vocab_path << "; "
            << "redis-ip: "          << params.redis_ip << "; "
            << "redis-port: "        << params.redis_port << "; "
            << "show-top-tokens: "   << params.show_top_tokens << "; "
            << "continue-fitting: "  << params.continue_fitting;
}

void CheckParams(const Parameters& params) {
  if (params.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }

  if (params.num_outer_iters <= 0) {
    throw std::runtime_error("num_outer_iters should be a positive integer");
  }

  if (params.num_topics <= 0) {
    throw std::runtime_error("num_topics should be a positive integer");
  }


  if (params.batches_dir_path == "") {
    throw std::runtime_error("batches_dir_path should be non-empty");
  }

  if (params.vocab_path == "") {
    throw std::runtime_error("vocab_path should be non-empty");
  }

  if (params.executor_ids_path == "") {
    throw std::runtime_error("executor_ids_path should be non-empty");
  }

  if (params.redis_ip == "") {
    throw std::runtime_error("redis_ip should be non-empty");
  }

  if (params.redis_port == "") {
    throw std::runtime_error("redis_port should be non-empty");
  }

  if (params.continue_fitting != 0 && params.continue_fitting != 1) {
    throw std::runtime_error("continue_fitting should be equal to 0 or 1");
  }

  if (params.show_top_tokens != 0 && params.show_top_tokens != 1) {
    throw std::runtime_error("show_top_tokens should be equal to 0 or 1");
  }
}

void ParseAndPrintArgs(int argc, char* argv[], Parameters* p) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help", "Show help")
    ("num-topics", po::value(&p->num_topics)->default_value(1), "number of topics")
    ("num-outer-iter", po::value(&p->num_outer_iters)->default_value(1), "number of collection passes")
    ("executor-ids-path", po::value(&p->executor_ids_path)->default_value("."), "path to ids of all working processes")
    ("batches-dir-path", po::value(&p->batches_dir_path)->default_value("."), "path to batches with documents")
    ("vocab-path", po::value(&p->vocab_path)->default_value("."), "path to file with vocabulary")
    ("redis-ip", po::value(&p->redis_ip)->default_value(""), "ip of redis instance")
    ("redis-port", po::value(&p->redis_port)->default_value(""), "port of redis instance")
    ("show-top-tokens", po::value(&p->show_top_tokens)->default_value(0), "1 - print top tokens, 0 - not")
    ("continue-fitting", po::value(&p->continue_fitting)->default_value(0), "1 - continue fitting redis model, 0 - restart")
    ;

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
  notify(vm);

  std::cout << "num-topics:        " << p->num_topics << std::endl;
  std::cout << "num-outer-iter:    " << p->num_outer_iters << std::endl;
  std::cout << "executor-ids-path: " << p->executor_ids_path << std::endl;
  std::cout << "batches-dir-path:  " << p->batches_dir_path << std::endl;
  std::cout << "vocab-path:        " << p->vocab_path << std::endl;
  std::cout << "redis-ip:          " << p->redis_ip << std::endl;
  std::cout << "redis-port:        " << p->redis_port << std::endl;
  std::cout << "show-top-tokens:   " << p->show_top_tokens << std::endl;
  std::cout << "continue-fitting:  " << p->continue_fitting << std::endl;
}

std::vector<std::string> GetExecutorIds(const std::string& executor_ids_path) {
  std::vector<std::string> retval;
  std::ifstream fin;
  std::string line;

  fin.open(executor_ids_path);
  while (std::getline(fin, line)) {
    retval.push_back(line);
  }
  fin.close();
  return retval;
}

bool CheckFinishedOrTerminated(const RedisClient& redis_client,
                               const std::vector<std::string>& command_keys,
                               const std::string& old_flag,
                               const std::string& new_flag,
                               int timeout = -1) {
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
bool CheckNonTerminatedAndUpdate(const RedisClient& redis_client,
                                 const std::vector<std::string>& command_keys,
                                 const std::string& flag) {
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
bool NormalizeNwt(const RedisClient& redis_client,
                  const std::vector<std::string>& command_keys,
                  const std::vector<std::string>& data_keys,
                  int num_topics) {
  if (!CheckNonTerminatedAndUpdate(redis_client, command_keys, START_NORMALIZATION)) {
    return false;
  }

  if (!CheckFinishedOrTerminated(redis_client, command_keys, START_NORMALIZATION, FINISH_NORMALIZATION)) {
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

  if (!CheckNonTerminatedAndUpdate(redis_client, command_keys, START_NORMALIZATION)) {
    return false;
  }

  if (!CheckFinishedOrTerminated(redis_client, command_keys, START_NORMALIZATION, FINISH_NORMALIZATION)) {
    return false;
  }

  return true;
}


// ToDo(MelLain): rewrite this function, as it is very inefficient and hacked now
void PrintTopTokens(RedisClient& redis_client,
                    const std::string& vocab_path,
                    int num_topics,
                    int num_tokens = 10) {
  std::vector<std::string> topics;
  for (int i = 0; i < num_topics; ++i) {
    topics.push_back("topic_" + std::to_string(i));
  }

  auto p_wt = RedisPhiMatrix(ModelName("pwt"), topics, redis_client);
  auto zero_vector = std::vector<float>(num_topics, 0.0f);

  std::ifstream fin;
  std::string line;
  fin.open(vocab_path);
  while (std::getline(fin, line)) {
    p_wt.AddToken(Token(DefaultClass, line), false, zero_vector);
  }
  fin.close();

  for (int i = 0; i < p_wt.topic_size(); ++i) {
    std::vector<std::pair<Token, float>> pairs;
    for (int j = 0; j < p_wt.token_size(); ++j) {
      pairs.push_back(std::make_pair(p_wt.token(j), p_wt.get(j, i)));
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<Token, float>& p1, const std::pair<Token, float>& p2) {
                return p1.second > p2.second;
              });
    std::cout << "\nTopic: " << p_wt.topic_name(i) << std::endl;
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

  FLAGS_minloglevel = 0;
  //FLAGS_alsologtostderr = 0;
  //FLAGS_logbuflevel = -1;
  //FLAGS_stderrthreshold = 0;
  FLAGS_log_dir = ".";
  std::string log_file = std::string("cluster-bigartm-master");
  google::InitGoogleLogging(log_file.c_str());  

  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);
  LogParams(params);
  CheckParams(params);

  LOG(INFO) << "Master: start connecting redis at " << params.redis_ip << ":" << params.redis_port;

  RedisClient redis_client = RedisClient(params.redis_ip, std::stoi(params.redis_port), 10, 100);

  LOG(INFO) << "Master: finish connecting to redis";

  std::vector<std::string> executor_command_keys;
  std::vector<std::string> executor_data_keys;

  LOG(INFO) << "Master: start creating ids";

  for (const std::string& id : GetExecutorIds(params.executor_ids_path)) {
    executor_command_keys.push_back(generate_command_key(id));
    executor_data_keys.push_back(generate_data_key(id));
  }

  LOG(INFO) << "Master: finish creating ids";

  try {
    // we give 1.0 sec to all executors to start, if even one of them
    // didn't response, it means, that it had failed to start
    LOG(INFO) << "Master: start connecting to processors";

    bool ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                        START_GLOBAL_START, FINISH_GLOBAL_START, 1000000);
    if (!ok) { throw std::runtime_error("Master: step 0, got termination status"); }

    LOG(INFO) << "Master: finish connecting to processors";

    LOG(INFO) << "Master: start initialization";

    ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 start, got termination status"); }

    ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                   START_INITIALIZATION, FINISH_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 finish, got termination status"); }

    LOG(INFO) << "Master: finish initialization";

    double n = 0.0;
    for (const auto& key : executor_data_keys) {
      n += std::stod(redis_client.get_value(key));
    }

    LOG(INFO) << "Master: all executors have started! Total number of token slots in collection: " << n;
    std::cout << "Master: all executors have started! Total number of token slots in collection: " << n << std::endl;

    if (!params.continue_fitting) {
      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 2, got termination status");
      }
    }

    // EM-iterations
    for (int iteration = 0; iteration < params.num_outer_iters; ++iteration) {
      LOG(INFO) << "Master: start iteration " << iteration;

      ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 start, got termination status"); }

      ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                     START_ITERATION, FINISH_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 intermediate, got termination status"); }

      double perplexity_value = 0.0;
      for (const auto& key : executor_data_keys) {
        perplexity_value += std::stod(redis_client.get_value(key));
      }

      LOG(INFO) << "Master: finish e-step, start m-step";

      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      LOG(INFO) << "Iteration: " << iteration << ", perplexity: " << exp(-(1.0f / n) * perplexity_value);
      std::cout << "Iteration: " << iteration << ", perplexity: " << exp(-(1.0f / n) * perplexity_value) << std::endl;
      LOG(INFO) << "Iteration: " << iteration << ", maxrss: " << Helpers::GetPeakMemoryKb() << " KB";
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
  CheckFinishedOrTerminated(redis_client, executor_command_keys, START_TERMINATION, FINISH_TERMINATION);

  if (params.show_top_tokens) {
    PrintTopTokens(redis_client, params.vocab_path, params.num_topics);
  }

  LOG(INFO) << "Model fitting is finished!";
  LOG(INFO) << "final maxrss= " << Helpers::GetPeakMemoryKb() << " KB";

  return 0;
}
