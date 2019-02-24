#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include "glog/logging.h"

#include "token.h"
#include "redis_phi_matrix.h"
#include "helpers.h"
#include "processor_helpers.h"
#include "redis_client.h"

#include "executor_thread.h"

namespace bf = boost::filesystem;

bool ExecutorThread::check_non_terminated_and_update(const std::string& flag, bool force) {
  if (!force) {
    auto reply = redis_client_->get_value(command_key_);
    if (reply == START_TERMINATION) {
      return false;
    }
  }

  redis_client_->set_value(command_key_, flag);
  return true;
}

bool ExecutorThread::wait_for_flag(const std::string& flag) {
  while (true) {
    auto reply = redis_client_->get_value(command_key_);
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

Normalizers ExecutorThread::find_nt() {
  LOG(INFO) << "Executor thread " << command_key_ << ": start find_nt";

  Normalizers retval;
  std::vector<float> helper = std::vector<float>(n_wt_->topic_size(), 0.0f);
  for (int token_id = 0; token_id < n_wt_->token_size(); ++token_id) {
    if (token_id < token_begin_index_ || token_id >= token_end_index_) {
      continue;
    }

    const Token& token = n_wt_->token(token_id);
    auto normalizer_key = token.class_id;

    auto iter = retval.find(normalizer_key);
    if (iter == retval.end()) {
      retval.insert(std::make_pair(normalizer_key, std::vector<double>(n_wt_->topic_size(), 0)));
      iter = retval.find(normalizer_key);
    }

    n_wt_->get(token_id, &helper);
    for (int topic_id = 0; topic_id < n_wt_->topic_size(); ++topic_id) {
      iter->second[topic_id] += helper[topic_id];
    }
  }
  LOG(INFO) << "Executor thread " << command_key_ << ": finish find_nt";
  return retval;
}

bool ExecutorThread::normalize_nwt() {
  if(!wait_for_flag(START_NORMALIZATION)) {
    return false;
  }

  LOG(INFO) << "Executor thread " << command_key_ << ": start normalize_nwt";

  const int num_topics = n_wt_->topic_size();
  const int num_tokens = n_wt_->token_size();
  const std::vector<float> zeros(num_topics, 0.0f);

  assert(p_wt_->token_size() == num_tokens && p_wt_->topic_size() == num_topics);

  Normalizers n_t = find_nt();
  redis_client_->set_hashmap(data_key_, n_t);

  if (!check_non_terminated_and_update(FINISH_NORMALIZATION)) {
    return false;
  }

  if(!wait_for_flag(START_NORMALIZATION)) {
    return false;
  } 

  n_t = redis_client_->get_hashmap(data_key_, num_topics);

  for (int token_id = 0; token_id < num_tokens; ++token_id) {
    if (token_id < token_begin_index_ || token_id >= token_end_index_) {
      continue;
    }

    const Token& token = n_wt_->token(token_id);
    assert(p_wt_->token(token_id) == token);
    const std::vector<double>& n_t_for_class_id = n_t[token.class_id];

    std::vector<float> helper = std::vector<float>(num_topics, 0.0f);
    std::vector<float> helper_n_wt = std::vector<float>(num_topics, 0.0f);

    n_wt_->get_set(token_id, &helper_n_wt, zeros);
    for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
      float value = 0.0f;
      if (n_t_for_class_id[topic_index] > 0) {
        value = std::max<double>(helper_n_wt[topic_index], 0.0) / n_t_for_class_id[topic_index];
        if (value < kEps) {
          value = 0.0f;
        }
      }
      helper[topic_index] = value;
    }
    p_wt_->set(token_id, helper);
  }

  if (!check_non_terminated_and_update(FINISH_NORMALIZATION)) {
    return false;
  }

  LOG(INFO) << "Executor thread " << command_key_ << ": normalize_nwt - correct finish";
  return true;
}

void ExecutorThread::process_e_step(const artm::Batch& batch, Blas* blas, double* perplexity_value) {
  std::shared_ptr<CsrMatrix<float>> sparse_ndw;
  sparse_ndw = ProcessorHelpers::initialize_sparse_ndw(batch);

  std::shared_ptr<LocalThetaMatrix<float>> theta_matrix;
  theta_matrix = ProcessorHelpers::initialize_theta(p_wt_->topic_size(), batch);

  std::shared_ptr<NwtWriteAdapter> nwt_writer = std::make_shared<NwtWriteAdapter>(n_wt_);

  ProcessorHelpers::infer_theta_and_update_nwt_sparse(batch, *sparse_ndw, *p_wt_, theta_matrix.get(),
                                                      nwt_writer.get(), blas, num_inner_iters_, perplexity_value);
}

void ExecutorThread::thread_function() {
  LOG(INFO) << "Executor thread " << command_key_ << ": has started";

  try {
    LOG(INFO) << "Executor thread " << command_key_ << ": start connecting to master";

    if (!check_non_terminated_and_update(FINISH_GLOBAL_START, true)) {
      throw std::runtime_error("Step 0, got termination command");
    };

    if (!wait_for_flag(START_INITIALIZATION)) {
      throw std::runtime_error("Step 1 start, got termination command");
    };

    LOG(INFO) << "Executor thread " << command_key_ << ": finish connecting to master";

    LOG(INFO) << "Executor thread " << command_key_ << ": start preparations";
    double n = 0.0;
    int num_batches_counter = 0;
    int num_batches_processed = 0;
    for(const auto& entry : boost::make_iterator_range(bf::directory_iterator(batches_dir_path_), { })) {
      if (num_batches_counter >= batch_begin_index_ && num_batches_counter < batch_end_index_) {
        artm::Batch batch;
        Helpers::load_batch(entry.path().string(), &batch);
        for (const auto& item : batch.item()) {
          for (float val : item.token_weight()) {
            n += static_cast<double>(val);
          }
        }
        ++num_batches_processed;
      }
      ++num_batches_counter;
    }

    redis_client_->set_value(data_key_, std::to_string(n));
    LOG(INFO) << "Executor thread " << command_key_ << ": finish preparations, total number of slots: "
              << n << " from " << num_batches_processed << " batches";

    if (!check_non_terminated_and_update(FINISH_INITIALIZATION)) {
      throw std::runtime_error("Step 1 finish, got termination command");
    }

    if (!continue_fitting_) {
      LOG(INFO) << "Executor thread " << command_key_ << ": start normalization";

      if (!normalize_nwt()) {
        throw std::runtime_error("Step 2, got termination status");
      }

      LOG(INFO) << "Executor thread " << command_key_ << ": finish normalization";
    }

    Blas* blas = Blas::builtin();
    while (true) {
      LOG(INFO) << "Executor thread " << command_key_ << ": start new iteration";

      // false here and only here means valid termination
      if (!wait_for_flag(START_ITERATION)) {
        break;
      };

      double perplexity_value = 0.0;
      int counter = 0;
      LOG(INFO) << "Executor thread " << command_key_ << ": start processing of E-step";

      for(const auto& entry : boost::make_iterator_range(bf::directory_iterator(batches_dir_path_), { })) {
        if (counter >= batch_begin_index_ && counter < batch_end_index_) {
          artm::Batch batch;
          const std::string batch_name = entry.path().string();
          LOG(INFO) << "Executor thread " << command_key_ << ": start processing batch " << batch_name;

          Helpers::load_batch(batch_name, &batch);
          process_e_step(batch, blas, &perplexity_value);

          LOG(INFO) << "Executor thread " << command_key_ << ": finish processing batch " << batch_name;

          if (caching_phi_mode_ == CACHING_PHI_MODE_BATCH) {
            p_wt_->clear_cache();
          }
        }
        ++counter;
      }

      if (caching_phi_mode_ == CACHING_PHI_MODE_ITERATION) {
        p_wt_->clear_cache();
      }

      LOG(INFO) << "Executor thread " << command_key_ << ": local pre-perplexity value: " << perplexity_value;

      redis_client_->set_value(data_key_, std::to_string(perplexity_value));

      if (!check_non_terminated_and_update(FINISH_ITERATION)) {
        throw std::runtime_error("Step 3 start, got termination command");
      }

      LOG(INFO) << "Executor thread " << command_key_ << ": finish processing of E-step, start M-step";

      if (!normalize_nwt()) {
        throw std::runtime_error("Step 3 finish, got termination status");
      }

      LOG(INFO) << "Executor thread " << command_key_ << ": finish iteration";
      LOG(INFO) << "Executor thread " << command_key_ << ": maxrss= " << Helpers::get_peak_memory_kb() << " KB";
    }
  } catch (const std::exception& error) {
    LOG(FATAL) << "Error in thread " << command_key_ << ": " << error.what();
  } catch (...) {
    LOG(FATAL) << "Unknown error in thread " << command_key_;
  }

  is_stopping_ = true;

  LOG(INFO) << "Executor thread " << command_key_ << ": finish processing!";
  LOG(INFO) << "Executor thread " << command_key_ << ": final maxrss= " << Helpers::get_peak_memory_kb() << " KB";
}
