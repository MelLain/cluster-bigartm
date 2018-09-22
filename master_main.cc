#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "common.h"
#include "phi_matrix.h"
#include "protocol.h"
#include "redis_client.h"
#include "token.h"

namespace po = boost::program_options;

struct Parameters {
  int num_topics;
  int num_outer_iters;
  int num_inner_iters;
  int num_executors;
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
    ("num-inner-iter", po::value(&p->num_inner_iters)->default_value(1), "number of document passes")
    ("num-executors", po::value(&p->num_executors)->default_value(1), "number of working processes")
    ("batches-dir-path", po::value(&p->batches_dir_path)->default_value("."), "path to batches with documents")
    ("vocab-path", po::value(&p->vocab_path)->default_value("."), "path to file with vocabulary")
    ("redis-ip", po::value(&p->redis_ip)->default_value(""), "ip of redis instance")
    ("redis-port", po::value(&p->redis_port)->default_value(""), "port of redis instance")
    ("show-top-tokens", po::value(&p->show_top_tokens)->default_value(0), "1 - print top tokens, 0 - not")
    ("continue-fitting", po::value(&p->continue_fitting)->default_value(0), "1 - continue fitting redis model, 0 - restart")
    ("debug-print", po::value(&p->debug_print)->default_value(0), "1 - print parameters of executors, 0 - not")
    ;

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
  notify(vm);

  std::cout << "num-topics:       " << p->num_topics << std::endl;
  std::cout << "num-outer-iter:   " << p->num_outer_iters << std::endl;
  std::cout << "num-inner-iter:   " << p->num_inner_iters << std::endl;
  std::cout << "num-executors:    " << p->num_executors << std::endl;
  std::cout << "batches-dir-path: " << p->batches_dir_path << std::endl;
  std::cout << "vocab-path:       " << p->vocab_path << std::endl;
  std::cout << "redis-ip:         " << p->redis_ip << std::endl;
  std::cout << "redis-port:       " << p->redis_port << std::endl;
  std::cout << "show-top-tokens:  " << p->show_top_tokens << std::endl;
  std::cout << "continue-fitting: " << p->continue_fitting << std::endl;
  std::cout << "debug-print:      " << p->debug_print << std::endl;
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

std::vector<int> GetStartIndices(int num_executors, int size) {
  std::vector<int> retval;
  const int step = size / num_executors;
  for (int i = 0; i < num_executors; ++i) {
    retval.push_back(i * step);
  }
  return retval;
}


void PrintTopTokens(const PhiMatrix& p_wt, int num_tokens = 10) {
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


int main(int argc, char* argv[]) {
  const clock_t begin_time = clock();

  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);

  std::shared_ptr<RedisClient> redis_client = std::shared_ptr<RedisClient>(
    new RedisClient(params.redis_ip, std::stoi(params.redis_port), 10, 100));

  // step 0: prepare parameters
  std::string res = exec((std::string("wc -l ") + params.vocab_path).c_str());
  const int vocab_size = std::stoi(res.substr(0, res.find(" ", 5)));
  std::cout << "Total vocabulary size: " << vocab_size << std::endl;

  std::vector<int> token_start_indices = GetStartIndices(params.num_executors, vocab_size);

  res = exec((std::string("ls -lt ") + params.batches_dir_path + std::string(" | wc -l")).c_str());
  const int num_batches = std::stoi(res.substr(res.rfind(" "), res.size())) - 1;
  std::cout << "Total number of batches: " << num_batches << std::endl;

  std::vector<int> batch_start_indices = GetStartIndices(params.num_executors, num_batches);

  std::vector<std::string> exucutor_command_keys;
  std::vector<std::string> exucutor_data_keys;

  std::cout << "Executors start indices: " << std::endl;
  for (int i = 0; i < params.num_executors; ++i) {
    std::cout << "Executor " << i
              << ", token start index: " << token_start_indices[i]
              << ", batch start index: " << batch_start_indices[i]
              << std::endl;

    exucutor_command_keys.push_back(kEscChar + std::string("exec-cmd-") + std::to_string(i));
    exucutor_data_keys.push_back(kEscChar + std::string("exec-data-") + std::to_string(i));
  }
  std::cout << "===============" << std::endl << std::endl;

  // step 1: create communication slots, set initialization command and start executors
  for (int i = 0; i < params.num_executors; ++i) {
    redis_client->set_value(exucutor_command_keys[i], START_INITIALIZATION);
    redis_client->set_value(exucutor_data_keys[i], "");

    std::stringstream start_executor_cmd;
    start_executor_cmd << "./executor_main"
                       << " --batches-dir-path '" << params.batches_dir_path << "'"
                       << " --vocab-path '" << params.vocab_path << "'"
                       << " --num-topics " << params.num_topics
                       << " --num-inner-iter " << params.num_inner_iters
                       << " --redis-ip " << params.redis_ip
                       << " --redis-port " << params.redis_port
                       << " --continue-fitting " << params.continue_fitting
                       << " --debug-print " << params.debug_print
                       << " --token-start-index " << token_start_indices[i]
                       << " --batch-start-index " << batch_start_indices[i]
                       << " --command-key '" << exucutor_command_keys[i] << "'"
                       << " --data-key '" << exucutor_data_keys[i] << "'";

    std::system(start_executor_cmd.str().c_str());
  }


  // ещё каждый должен посчитать размер своей части и вернуть




  std::cout << "Finished! Elapsed time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << " sec." << std::endl;
  return 0;
}
