#pragma once

#include <atomic>
#include <iterator>
#include <vector>
#include <memory>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "common.h"
#include "token.h"
#include "thread_safe_collection_holder.h"
#include "redis_client.h"

enum PhiMatrixCacheMode { NONE, READ, WRITE };

class SpinLock : boost::noncopyable {
 public:
  SpinLock() : state_(kUnlocked) { }
  void lock();
  void unlock();

 private:
  static const bool kLocked = true;
  static const bool kUnlocked = false;
  std::atomic<bool> state_;
};

class RedisPhiMatrix : boost::noncopyable {
 public:
  static const int kUndefIndex = -1;

  RedisPhiMatrix(const ModelName& model_name,
  	             const std::vector<std::string>& topic_name,
                 PhiMatrixCacheMode cache_mode = PhiMatrixCacheMode::NONE)
      : model_name_(model_name)
      , topic_name_(topic_name)
      , token_collection_()
      , cache_mode_(cache_mode)
      , cache_() { }

  int token_size() const;

  int topic_size() const { return topic_name_.size(); }
  std::vector<std::string> topic_name() const { return topic_name_; }

  const std::vector<std::string>& topic_name_ref() const { return topic_name_; }

  const std::string& topic_name(int topic_id) const { return topic_name_[topic_id]; }

  void set_topic_name(int topic_id, const std::string& topic_name) {
  	topic_name_[topic_id] = topic_name;
  }

  ModelName model_name() const { return model_name_; }

  const Token token(int token_id) const;
  bool has_token(const Token& token) const;
  int token_index(const Token& token) const;

  void set(std::shared_ptr<RedisClient> redis_client, int token_id, const std::vector<float>& buffer);

  float get(std::shared_ptr<RedisClient> redis_client, int token_id, int topic_id) const;
  void get(std::shared_ptr<RedisClient> redis_client, int token_id, std::vector<float>* buffer) const;

  void get_set(std::shared_ptr<RedisClient> redis_client, int token_id,
               std::vector<float>* buffer, const std::vector<float>& values);

  void increase(std::shared_ptr<RedisClient> redis_client, int token_id, const std::vector<float>& increment);

  int add_token(std::shared_ptr<RedisClient> redis_client, const Token& token, bool flag);
  int add_token(std::shared_ptr<RedisClient> redis_client,
                const Token& token, bool flag, const std::vector<float>& values);

  void clear_read_cache(std::shared_ptr<RedisClient> redis_client) {
    if (cache_mode_ == PhiMatrixCacheMode::READ) {
      cache_.clear();
    }
  }

  void dump_write_cache(std::shared_ptr<RedisClient> redis_client, int token_begin_index, int token_end_index);

  ~RedisPhiMatrix() {
    token_collection_.clear();
    spin_locks_.clear();
    cache_.clear();
  }

  PhiMatrixCacheMode cache_mode() const {
    return cache_mode_;
  }

 private:
  void lock(int token_id) { spin_locks_[token_id]->lock(); }
  void unlock(int token_id) { spin_locks_[token_id]->unlock(); }

  std::string to_key(int i) const { return std::to_string(i) + model_name_; }

  ModelName model_name_;
  std::vector<std::string> topic_name_;
  TokenCollection token_collection_;
  std::vector<std::shared_ptr<SpinLock> > spin_locks_;
  PhiMatrixCacheMode cache_mode_;
  mutable ThreadSafeCollectionHolder<int, std::vector<float>> cache_;
};

class RedisPhiMatrixAdapter {
 public:
  RedisPhiMatrixAdapter(std::shared_ptr<RedisPhiMatrix> phi_matrix, std::shared_ptr<RedisClient> redis_client)
      : phi_matrix_(phi_matrix)
      , redis_client_(redis_client) { }

  RedisPhiMatrixAdapter(std::shared_ptr<RedisClient> redis_client,
                        const ModelName& model_name,
                        const std::vector<std::string>& topic_name,
                        PhiMatrixCacheMode cache_mode = PhiMatrixCacheMode::NONE)
      : phi_matrix_(std::shared_ptr<RedisPhiMatrix>(new RedisPhiMatrix(model_name, topic_name, cache_mode)))
      , redis_client_(redis_client) { }

  int token_size() const { return phi_matrix_->token_size(); }
  int topic_size() const { return phi_matrix_->topic_size(); }
  std::vector<std::string> topic_name() const { return phi_matrix_->topic_name(); }

  const std::vector<std::string>& topic_name_ref() const { return phi_matrix_->topic_name_ref(); }
  const std::string& topic_name(int topic_id) const { return phi_matrix_->topic_name(topic_id); }

  void set_topic_name(int topic_id, const std::string& topic_name) {
    phi_matrix_->set_topic_name(topic_id, topic_name);
  }

  ModelName model_name() const { return phi_matrix_->model_name(); }

  const Token token(int token_id) const { return phi_matrix_->token(token_id); }
  bool has_token(const Token& token) const { return phi_matrix_->has_token(token); }
  int token_index(const Token& token) const { return phi_matrix_->token_index(token); }

  void set(int token_id, const std::vector<float>& buffer) {
    phi_matrix_->set(redis_client_, token_id, buffer);
  }

  float get(int token_id, int topic_id) const {
    return phi_matrix_->get(redis_client_, token_id, topic_id);
  }

  void get(int token_id, std::vector<float>* buffer) const {
    phi_matrix_->get(redis_client_, token_id, buffer);
  }

  void get_set(int token_id, std::vector<float>* buffer, const std::vector<float>& values) {
    phi_matrix_->get_set(redis_client_, token_id, buffer, values);
  }

  void increase(int token_id, const std::vector<float>& increment) {
    phi_matrix_->increase(redis_client_, token_id, increment);
  }

  int add_token(const Token& token, bool flag) {
    return phi_matrix_->add_token(redis_client_, token, flag);
  }
  
  int add_token(const Token& token, bool flag, const std::vector<float>& values) {
    return phi_matrix_->add_token(redis_client_, token, flag, values);
  }

  void clear_read_cache() { phi_matrix_->clear_read_cache(redis_client_); }
  void dump_write_cache(int token_begin_index, int token_end_index) {
    phi_matrix_->dump_write_cache(redis_client_, token_begin_index, token_end_index);
  }

  PhiMatrixCacheMode cache_mode() const { return phi_matrix_->cache_mode(); }

 private:
  std::shared_ptr<RedisPhiMatrix> phi_matrix_;
  std::shared_ptr<RedisClient> redis_client_;
};
