#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "boost/thread.hpp"
#include "boost/thread/mutex.hpp"
#include "boost/utility.hpp"

#include "messages.pb.h"

#include "blas.h"
#include "protocol.h"
#include "redis_phi_matrix.h"

class ExecutorThread : boost::noncopyable {
 public:
  explicit ExecutorThread(const std::string& command_key,
  	                      const std::string& data_key,
  	                      std::shared_ptr<RedisClient> redis_client,
  	                      bool continue_fitting,
  	                      const std::string& batches_dir_path,
  	                      int token_begin_index,
  	                      int token_end_index,
  	                      int batch_begin_index,
  	                      int batch_end_index,
  	                      int num_inner_iters,
  	                      std::shared_ptr<RedisPhiMatrixAdapter> p_wt,
  	                      std::shared_ptr<RedisPhiMatrixAdapter> n_wt)
    : command_key_(command_key)
    , data_key_(data_key)
    , redis_client_(redis_client)
    , continue_fitting_(continue_fitting)
    , batches_dir_path_(batches_dir_path)
    , token_begin_index_(token_begin_index)
    , token_end_index_(token_end_index)
    , batch_begin_index_(batch_begin_index)
    , batch_end_index_(batch_end_index)
    , num_inner_iters_(num_inner_iters)
    , p_wt_(p_wt)
    , n_wt_(n_wt)
    , is_stopping_(false)
    , thread_()
{
  boost::thread t(&ExecutorThread::thread_function, this);
  thread_.swap(t);
}

bool is_stopping() {
  return is_stopping_;
}

~ExecutorThread() {
  is_stopping_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }

  LOG(INFO) << "Executor thread " << command_key_ << ": stopping";

  // ToDo(MelLain): this set doesn't work, inspect it
  redis_client_->set_value(command_key_, FINISH_TERMINATION);
}

 private:
  std::string command_key_;
  std::string data_key_;
  std::shared_ptr<RedisClient> redis_client_;
  bool continue_fitting_;
  std::string batches_dir_path_;
  int token_begin_index_;
  int token_end_index_;
  int batch_begin_index_;
  int batch_end_index_;
  int num_inner_iters_;
  std::shared_ptr<RedisPhiMatrixAdapter> p_wt_;
  std::shared_ptr<RedisPhiMatrixAdapter> n_wt_;

  mutable std::atomic<bool> is_stopping_;
  boost::thread thread_;

  void thread_function();

  bool check_non_terminated_and_update(const std::string& flag, bool force = false);
  bool wait_for_flag(const std::string& flag);
  
  Normalizers find_nt();
  // protocol:
  // 1) wait for START_NORMALIZATION flag
  // 2) after reaching it compute n_t on tokens from executor range
  // 3) put results into data slot and set cmd slot to FINISH_NORMALIZATION
  // 4) wait for new START_NORMALIZATION flag
  // 5) read total n_t from data slot
  // 6) proceed final normalization on tokens from executor range
  // 7) set FINISH_NORMALIZATION flag and return
  bool normalize_nwt();

  void process_e_step(const artm::Batch& batch, Blas* blas, double* perplexity_value);
};
