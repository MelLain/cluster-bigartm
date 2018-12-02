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

#include "common.h"
#include "phi_matrix.h"
#include "protocol.h"
#include "redis_client.h"
#include "redis_phi_matrix.h"
#include "token.h"

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
      std::cout << "SIGINT has been caught, start terminating" << std::endl;
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
    std::cout << "SIGINT has been caught, start terminating" << std::endl;
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


// In case of fault of master without exceptions all sub-processes
// (executors) can be killed on sinngle node via command:
// ps -ef | grep './executor_main' | grep -v grep | awk '{print $2}' | xargs kill -9
int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);

  std::cout << "Master: start connecting redis at " << params.redis_ip << ":" << params.redis_port << std::endl;
  RedisClient redis_client = RedisClient(params.redis_ip, std::stoi(params.redis_port), 10, 100);
  std::cout << "Master: finish conneting to redis" << std::endl;

  std::vector<std::string> executor_command_keys;
  std::vector<std::string> executor_data_keys;

  std::cout << "Master: start creating ids" << std::endl;
  for (const std::string& id : GetExecutorIds(params.executor_ids_path)) {
    executor_command_keys.push_back(generate_command_key(id));
    executor_data_keys.push_back(generate_data_key(id));
  }
  std::cout << "Master: finish creating ids" << std::endl;

  try {
    // we give 1.0 sec to all executors to start, if even one of them
    // didn't response, it means, that it had failed to start
    std::cout << "Master: start connecting to processors" << std::endl;

    bool ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                        START_GLOBAL_START, FINISH_GLOBAL_START, 1000000);
    if (!ok) { throw std::runtime_error("Master: step 0, got termination status"); }

    std::cout << "Master: finish connecting to processors" << std::endl;


    std::cout << "Master: start initialization" << std::endl;
    ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 start, got termination status"); }

    ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                   START_INITIALIZATION, FINISH_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Master: step 1 finish, got termination status"); }

    std::cout << "Master: finish initialization" << std::endl;


    double n = 0.0;
    for (const auto& key : executor_data_keys) {
      n += std::stod(redis_client.get_value(key));
    }

    std::cout << std::endl
              << "Master: all executors have started! Total number of token slots in collection: "
              << n << std::endl;

    if (!params.continue_fitting) {
      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 2, got termination status");
      }
    }

    // EM-iterations
    for (int iteration = 0; iteration < params.num_outer_iters; ++iteration) {

      std::cout << "Master: start iteration " << iteration << std::endl;
      ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 start, got termination status"); }

      ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                     START_ITERATION, FINISH_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 intermediate, got termination status"); }


      double perplexity_value = 0.0;
      for (const auto& key : executor_data_keys) {
        perplexity_value += std::stod(redis_client.get_value(key));
      }

      std::cout << "Master: finish e-step, start m-step" << std::endl;

      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      std::cout << "Iteration: " << iteration << ", perplexity: " << exp(-(1.0f / n) * perplexity_value) << std::endl;
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

  std::cout << "Model fitting is finished!" << std::endl;
  return 0;
}
