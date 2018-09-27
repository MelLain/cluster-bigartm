#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <array>
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

typedef std::vector<std::vector<std::string>> Keys;

struct Parameters {
  int num_topics;
  int num_outer_iters;
  int num_inner_iters;
  int num_executors;
  std::string batches_dir_path;
  std::string vocab_path;
  std::string redis_ip;
  std::string redis_port;
  std::string redis_instances_path;
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
    ("num-inner-iter", po::value(&p->num_inner_iters)->default_value(1), "number of document passes")
    ("num-executors", po::value(&p->num_executors)->default_value(1), "number of working processes")
    ("batches-dir-path", po::value(&p->batches_dir_path)->default_value("."), "path to batches with documents")
    ("vocab-path", po::value(&p->vocab_path)->default_value("."), "path to file with vocabulary")
    ("redis-ip", po::value(&p->redis_ip)->default_value(""), "ip of redis instance")
    ("redis-port", po::value(&p->redis_port)->default_value(""), "port of redis instance")
    ("redis-instances-path", po::value(&p->redis_instances_path)->default_value(""), "port of redis instances info file")
    ("show-top-tokens", po::value(&p->show_top_tokens)->default_value(0), "1 - print top tokens, 0 - not")
    ("continue-fitting", po::value(&p->continue_fitting)->default_value(0), "1 - continue fitting redis model, 0 - restart")
    ("debug-print", po::value(&p->debug_print)->default_value(0), "1 - print parameters of executors, 0 - not")
    ;

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
  notify(vm);

  std::cout << "num-topics:           " << p->num_topics << std::endl;
  std::cout << "num-outer-iter:       " << p->num_outer_iters << std::endl;
  std::cout << "num-inner-iter:       " << p->num_inner_iters << std::endl;
  std::cout << "num-executors:        " << p->num_executors << std::endl;
  std::cout << "batches-dir-path:     " << p->batches_dir_path << std::endl;
  std::cout << "vocab-path:           " << p->vocab_path << std::endl;
  std::cout << "redis-ip:             " << p->redis_ip << std::endl;
  std::cout << "redis-port:           " << p->redis_port << std::endl;
  std::cout << "redis-instances-path: " << p->redis_instances_path << std::endl;
  std::cout << "show-top-tokens:      " << p->show_top_tokens << std::endl;
  std::cout << "continue-fitting:     " << p->continue_fitting << std::endl;
  std::cout << "debug-print:          " << p->debug_print << std::endl;
}

std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr) {
      result += buffer.data();
    }
  }
  return result;
}


std::vector<std::string> ReadRedisAddresses(const std::string& redis_instances_path,
                                            const std::string& ip,
                                            const std::string& port) {
  std::unordered_set<std::string> temp;
  std::ifstream fin;
  std::string line;
  fin.open(redis_instances_path);
  while (std::getline(fin, line)) {
    temp.emplace(line);
  }
  fin.close();
  if (temp.find(ip + " " + port) == temp.end()) {
    throw std::runtime_error("Admin redis ip:port should be in the cluster");
  }

  std::vector<std::string> retval(temp.begin(), temp.end());
  return retval;
}


std::vector<std::vector<std::pair<int, int>>> GetIndices(int num_instances, int num_executors, int size) {
  std::vector<std::vector<std::pair<int, int>>> retval;
  const int total = num_executors * num_instances;
  const int step = size / total;

  for (int i = 0; i < num_instances; ++i) {
    retval.push_back(std::vector<std::pair<int, int>>());
    for (int e = 0; e < num_executors; ++e) {
      int end = (i == (num_instances - 1) && e == (num_executors - 1)) ? size : (e + 1 + (i * num_executors)) * step;
      retval[i].push_back(std::make_pair((e + (i * num_executors)) * step, end));
    }
  }
  return retval;
}


bool CheckFinishedOrTerminated(const RedisClient& redis_client,
                               const Keys& command_keys,
                               const std::string& old_flag,
                               const std::string& new_flag,
                               int timeout = -1) {
  int time_passed = 0;
  bool terminated = false;
  while (true) {
    int executors_finished = 0;
    for (const auto& keys : command_keys) {
      for (const auto& key : keys) {
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
    }

    if (executors_finished == command_keys.size() * command_keys[0].size()) {
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


// this function firstly check the availability of executer and then send him new command,
// it's not fully safe, as if the execiter fails in between get and set, it will cause
// endless loop during the next syncronozation
bool CheckNonTerminatedAndUpdate(const RedisClient& redis_client,
                                 const Keys& command_keys,
                                 const std::string& flag) {
  for (const auto& keys : command_keys) {
    for (const auto& key : keys) {
      auto reply = redis_client.get_value(key);
      if (reply == FINISH_TERMINATION) {
        return false;
      }
    }
  }

  for (const auto& keys : command_keys) {
    for (const auto& key : keys) {
      redis_client.set_value(key, flag);
    }
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
                  const Keys& command_keys,
                  const Keys& data_keys,
                  int num_topics) {
  if (!CheckNonTerminatedAndUpdate(redis_client, command_keys, START_NORMALIZATION)) {
    return false;
  }

  if (!CheckFinishedOrTerminated(redis_client, command_keys, START_NORMALIZATION, FINISH_NORMALIZATION)) {
    return false;
  }

  Normalizers n_t;
  Normalizers helper;
  for (const auto& keys : data_keys) {
    for (const auto& key : keys) {
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
  }

  // ToDo(MelLain): maybe it'll be better to keep only one version of n_t for
  //                all executers, need to be checked with large number of topics
  for (const auto& keys : data_keys) {
    for (const auto& key : keys) {
      redis_client.set_hashmap(key, n_t);
    }
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
// ps -ef | grep './executor_main' | grep -v grep | awk '{print $2}' | xargs  kill -9
int main(int argc, char* argv[]) {
  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);

  RedisClient redis_client = RedisClient(params.redis_ip, std::stoi(params.redis_port), 10, 100);

  // prepare parameters and keys
  std::string res = exec((std::string("wc -l ") + params.vocab_path).c_str());
  const int vocab_size = std::stoi(res.substr(0, res.find(" ", 5)));
  std::cout << "Total vocabulary size: " << vocab_size << std::endl;

  auto redis_addresses = ReadRedisAddresses(params.redis_instances_path, params.redis_ip, params.redis_port);

  const int num_instances = redis_addresses.size();
  std::cout << "Number of redis instances: " << num_instances << std::endl;

  auto token_begin_end_indices = GetIndices(num_instances, params.num_executors, vocab_size);

  res = exec((std::string("ls -lt ") + params.batches_dir_path + std::string(" | wc -l")).c_str());
  const int num_batches = std::stoi(res.substr(res.rfind(" "), res.size())) - 1;
  std::cout << "Total number of batches: " << num_batches << std::endl;

  auto batch_begin_end_indices = GetIndices(num_instances, params.num_executors, num_batches);

  Keys executor_command_keys;
  Keys executor_data_keys;

  std::cout << std::endl << "Executors start indices: " << std::endl;
  for (int i = 0; i < num_instances; ++i) {
    executor_command_keys.push_back(std::vector<std::string>());
    executor_data_keys.push_back(std::vector<std::string>());

    for (int e = 0; e < params.num_executors; ++e) {
      std::cout << "Executor " << e << " for instance " << i
                << ", token indices: (" << token_begin_end_indices[i][e].first
                << ", " << token_begin_end_indices[i][e].second << ")"
                << ", batch indices: " << batch_begin_end_indices[i][e].first
                << ", " << batch_begin_end_indices[i][e].second << ")"
                << std::endl;

      executor_command_keys[i].push_back(kEscChar + std::string("cmd-") + std::to_string(i) + ":" + std::to_string(e));
      executor_data_keys[i].push_back(kEscChar + std::string("dat-") + std::to_string(i) + ":" + std::to_string(e));
    }
  }
  std::cout << std::endl;

  auto split = [](const std::string& str) -> std::vector<std::string> {
    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                               std::istream_iterator<std::string>{}};
    return tokens;
  };

  try {
    // create communication slots, set and check start cmd flag, start executors and proceed init
    for (int i = 0; i < num_instances; ++i) {
      auto splitted = split(redis_addresses[i]);

      for (int e = 0; e < params.num_executors; ++e) {
        redis_client.set_value(executor_command_keys[i][e], START_GLOBAL_START);
        redis_client.set_value(executor_data_keys[i][e], "");

        // ToDo(mel-lain): start remote executors from here, need ssh access from redis_instances_path
        std::stringstream start_executor_cmd;
        start_executor_cmd << "./executor_main"
                           << " --batches-dir-path '" << params.batches_dir_path << "'"
                           << " --vocab-path '" << params.vocab_path << "'"
                           << " --num-topics " << params.num_topics
                           << " --num-inner-iter " << params.num_inner_iters
                           << " --redis-ip " << splitted[0]
                           << " --redis-port " << splitted[1]
                           << " --continue-fitting " << params.continue_fitting
                           << " --debug-print " << params.debug_print
                           << " --token-begin-index " << token_begin_end_indices[i][e].first
                           << " --token-end-index " << token_begin_end_indices[i][e].second
                           << " --batch-begin-index " << batch_begin_end_indices[i][e].first
                           << " --batch-end-index " << batch_begin_end_indices[i][e].second
                           << " --command-key '" << executor_command_keys[i][e] << "'"
                           << " --data-key '" << executor_data_keys[i][e] << "'"
                           << " &";

        std::system(start_executor_cmd.str().c_str());
    }
  }

    // we give 1.0 sec to all executers to start, if even one of them
    // didn't response, it means, that it had failed to start
    bool ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                        START_GLOBAL_START, FINISH_GLOBAL_START, 1000000);
    if (!ok) { throw std::runtime_error("Step 0, got termination status"); }

    ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Step 1 start, got termination status"); }

    ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                   START_INITIALIZATION, FINISH_INITIALIZATION);
    if (!ok) { throw std::runtime_error("Step 1 finish, got termination status"); }

    double n = 0.0;
    for (const auto& keys : executor_data_keys) {
      for (const auto& key : keys) {
        n += std::stod(redis_client.get_value(key));
      }
    }

    std::cout << std::endl
              << "All executors have started! Total number of token slots in collection: "
              << n << std::endl;

    if (!params.continue_fitting) {
      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 2, got termination status");
      }
    }

    // EM-iterations
    for (int iteration = 0; iteration < params.num_outer_iters; ++iteration) {
      ok = CheckNonTerminatedAndUpdate(redis_client, executor_command_keys, START_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 start, got termination status"); }

      ok = CheckFinishedOrTerminated(redis_client, executor_command_keys,
                                     START_ITERATION, FINISH_ITERATION);
      if (!ok) { throw std::runtime_error("Step 3 intermediate, got termination status"); }

      double perplexity_value = 0.0;
      for (const auto& keys : executor_data_keys) {
        for (const auto& key : keys) {
          perplexity_value += std::stod(redis_client.get_value(key));
        }
      }

      if (!NormalizeNwt(redis_client, executor_command_keys, executor_data_keys, params.num_topics)) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      std::cout << "Iteration: " << iteration << ", perplexity: " << exp(-(1.0f / n) * perplexity_value) << std::endl;
    }

    // finalization (correct in any way)
    for (const auto& keys : executor_command_keys) {
      for (const auto& key : keys) {
        redis_client.set_value(key, START_TERMINATION);
      }
    }

  } catch (...) {
    for (const auto& keys : executor_command_keys) {
      for (const auto& key : keys) {
        redis_client.set_value(key, START_TERMINATION);
      }
    }
    throw;
  }
  CheckFinishedOrTerminated(redis_client, executor_command_keys, START_TERMINATION, FINISH_TERMINATION);

  if (params.show_top_tokens) {
    PrintTopTokens(redis_client, params.vocab_path, params.num_topics);
  }

  std::cout << "Model fitteing is finished!" << std::endl;
  return 0;
}
