#include <ctime>

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
#include "dense_phi_matrix.h"
#include "redis_phi_matrix.h"
#include "helpers.h"
#include "processor_helpers.h"
#include "redis_client.h"
#include "protocol.h"

namespace po = boost::program_options;

typedef std::unordered_map<ClassId, std::vector<double>> Normalizers;

static const int kConnectionTimeout = 100;
static const int kNumConnections = 1;
static const int kNumRetries = 10;

struct Parameters {
    int num_topics;
    int num_outer_iters;
    int num_inner_iters;
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
    ("num-topics,T", po::value(&p->num_topics)->default_value(1), "Input number of topics")
    ("num-outer-iter,O", po::value(&p->num_outer_iters)->default_value(1), "Input number of collection passes")
    ("num-inner-iter,I", po::value(&p->num_inner_iters)->default_value(1), "Input number of document passes")
    ("batches-dir-path,B", po::value(&p->batches_dir_path)->default_value("."), "Input path to files with documents")
    ("vocab-path,V", po::value(&p->vocab_path)->default_value("."), "Input path to files with documents")
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
  std::cout << "batches-dir-path: " << p->batches_dir_path << std::endl;
  std::cout << "vocab-path:       " << p->vocab_path << std::endl;
  std::cout << "redis-ip:       " << p->redis_ip << std::endl;
  std::cout << "redis-port:       " << p->redis_port << std::endl;
  std::cout << "show-top-tokens:       " << p->show_top_tokens << std::endl;
  std::cout << "continue-fitting:       " << p->continue_fitting << std::endl;
}


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

  bool use_redis = true;
  if (params.redis_ip.empty() or params.redis_port.empty()) {
    use_redis = false;
  }

  std::vector<std::string> topics;
  for (int i = 0; i < params.num_topics; ++i) {
    topics.push_back("topic_" + std::to_string(i));
  }

  std::vector<Token> tokens;
  std::ifstream fin;  

  fin.open(params.vocab_path);

  std::string line;
  while (std::getline(fin, line)) {
    tokens.push_back(Token(DefaultClass, line));
  }
  std::cout << "Number of tokens is " << tokens.size() << std::endl;

  fin.close();

  float n = 0.0f;
  for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(params.batches_dir_path), { })) {
    artm::Batch batch;
    Helpers::LoadBatch(entry.path().string(), &batch);
    for (const auto& item : batch.item()) {
      for (float val : item.token_weight()) {
        n += val;
      }
    }
  }
  std::cout << "Total number of token slots is: " << n << std::endl;

  std::shared_ptr<PhiMatrix> p_wt;
  std::shared_ptr<PhiMatrix> n_wt;

  std::shared_ptr<RedisClient> redis_client;
  if (use_redis) {
    redis_client = std::shared_ptr<RedisClient>(
      new RedisClient(params.redis_ip, std::stoi(params.redis_port), kNumRetries, kConnectionTimeout));
  }

  bool continue_fitting = (params.continue_fitting == "1");
  if (!use_redis && continue_fitting) {
    throw std::runtime_error("Unable to continue fitting of non-redis model!");
  }

  if (use_redis) {
    p_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("pwt"), topics, redis_client));
    n_wt = std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(ModelName("nwt"), topics, redis_client));
  } else {
    p_wt = std::shared_ptr<DensePhiMatrix>(new DensePhiMatrix(ModelName("pwt"), topics));
    n_wt = std::shared_ptr<DensePhiMatrix>(new DensePhiMatrix(ModelName("nwt"), topics));
  }

  for (const auto& token : tokens) {
    p_wt->AddToken(token, !continue_fitting);
    n_wt->AddToken(token, !continue_fitting);
  }

  if (!continue_fitting) {
    for (int i = 0; i < tokens.size(); ++i) {
      n_wt->increase(i, Helpers::GenerateRandomVector(params.num_topics, tokens[i]));
    }

    Normalize(p_wt, *n_wt);
  }

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

  if (params.show_top_tokens == "1") {
    PrintTopTokens(*p_wt);
  }

  std::cout << "Finished! Elapsed time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << " sec." << std::endl;
  return 0;
}
