#include <cstdio>
#include <ctime>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "protocol.h"

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
  std::string show_top_tokens;
  std::string continue_fitting;
};

void ParseAndPrintArgs(int argc, char* argv[], Parameters* p) {
  po::options_description all_options("Options");
  all_options.add_options()
    ("help,H", "Show help")
    ("num-topics,T", po::value(&p->num_topics)->default_value(1), "number of topics")
    ("num-outer-iter,O", po::value(&p->num_outer_iters)->default_value(1), "number of collection passes")
    ("num-inner-iter,I", po::value(&p->num_inner_iters)->default_value(1), "number of document passes")
    ("num-executors,E", po::value(&p->num_executors)->default_value(1), "number of working processes")
    ("batches-dir-path,B", po::value(&p->batches_dir_path)->default_value("."), "path to batches with documents")
    ("vocab-path,V", po::value(&p->vocab_path)->default_value("."), "path to file with vocabulary")
    ("redis-ip,A", po::value(&p->redis_ip)->default_value(""), "ip of redis instance")
    ("redis-port,P", po::value(&p->redis_port)->default_value(""), "port of redis instance")
    ("show-top-tokens,S", po::value(&p->show_top_tokens)->default_value("0"), "1 - print top tokens, 0 - not")
    ("continue-fitting,F", po::value(&p->continue_fitting)->default_value("0"), "1 - continue fitting redis model, 0 - restart")
    ;

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
  notify(vm);

  std::cout << "num-topics:       " << p->num_topics << std::endl;
  std::cout << "num-outer-iter:   " << p->num_outer_iters << std::endl;
  std::cout << "num-inner-iter:   " << p->num_inner_iters << std::endl;
  std::cout << "num-executors:   " << p->num_executors << std::endl;
  std::cout << "batches-dir-path: " << p->batches_dir_path << std::endl;
  std::cout << "vocab-path:       " << p->vocab_path << std::endl;
  std::cout << "redis-ip:       " << p->redis_ip << std::endl;
  std::cout << "redis-port:       " << p->redis_port << std::endl;
  std::cout << "show-top-tokens:       " << p->show_top_tokens << std::endl;
  std::cout << "continue-fitting:       " << p->continue_fitting << std::endl;
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

std::vector<int> GetStartIndices(int num_executors, int vocab_size) {
  std::vector<int> retval;
  const int step = vocab_size / num_executors;
  for (int i = 0; i < num_executors; ++i) {
    retval.push_back(i * step);
  }
  return retval;
}


int main(int argc, char* argv[]) {
  const clock_t begin_time = clock();

  Parameters params;
  ParseAndPrintArgs(argc, argv, &params);

  bool use_redis = true;
  if (params.redis_ip.empty() or params.redis_port.empty()) {
    use_redis = false;
  }

  bool continue_fitting = (params.continue_fitting == "1");
  if (!use_redis && continue_fitting) {
    throw std::runtime_error("Unable to continue fitting of non-redis model!");
  }

  std::string res = exec((std::string("wc -l ") + params.vocab_path).c_str());
  const int vocab_size = std::stoi(res.substr(0, res.find(" ", 5)));
  std::cout << "Total vocabulary size: " << vocab_size << std::endl;

  std::vector<int> start_indices = GetStartIndices(params.num_executors, vocab_size);
  std::cout << "Executors start indices: " << std::endl;
  for (const int i : start_indices) {
    std::cout << i << std::endl;
  }





  std::cout << "Finished! Elapsed time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << " sec." << std::endl;
  return 0;
}
