#include <algorithm>
#include <memory>
#include <numeric>

#include "glog/logging.h"

#include "redis_phi_matrix.h"

void SpinLock::lock() {
  while (state_.exchange(kLocked, std::memory_order_acquire) == kLocked) {
    /* busy-wait */
  }
}

void SpinLock::unlock() {
  state_.store(kUnlocked, std::memory_order_release);
}

int RedisPhiMatrix::token_size() const {
  return token_collection_.token_size();
}

const Token RedisPhiMatrix::token(int token_id) const {
  return token_collection_.token(token_id);
}

bool RedisPhiMatrix::has_token(const Token& token) const {
  return token_collection_.has_token(token);
}

int RedisPhiMatrix::token_index(const Token& token) const {
  return token_collection_.token_id(token);
}

// ATTN: this method should be used only for debugging, it's too slow for learning process!
float RedisPhiMatrix::get(std::shared_ptr<RedisClient> redis_client, int token_id, int topic_id) const {
  std::vector<float> buffer = redis_client->get_values(to_key(token_id), topic_size());
  return buffer[topic_id];
}

void RedisPhiMatrix::get(std::shared_ptr<RedisClient> redis_client, int token_id, std::vector<float>* buffer) const {
  if (cache_mode_ == PhiMatrixCacheMode::READ && cache_.has_key(token_id)) {
    auto values_ptr = cache_.get(token_id);
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      (*buffer)[topic_id] = (*values_ptr)[topic_id];
    }
  } else {
    std::vector<float> values = redis_client->get_values(to_key(token_id), topic_size());
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      (*buffer)[topic_id] = values[topic_id];
    }

    if (cache_mode_ == PhiMatrixCacheMode::READ) {
      cache_.set(token_id, std::make_shared<std::vector<float>>(values));
    }
  }
}

void RedisPhiMatrix::get_set(std::shared_ptr<RedisClient> redis_client, int token_id,
                             std::vector<float>* buffer, const std::vector<float>& values)
{
  lock(token_id);
  std::vector<float> temp = redis_client->get_set_values(to_key(token_id), values);
  for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
    (*buffer)[topic_id] = temp[topic_id];
  }
  unlock(token_id);
}

void RedisPhiMatrix::set(std::shared_ptr<RedisClient> redis_client, int token_id, const std::vector<float>& buffer) {
  lock(token_id);
  redis_client->set_values(to_key(token_id), buffer);
  unlock(token_id);
}

void RedisPhiMatrix::increase(std::shared_ptr<RedisClient> redis_client,
                              int token_id, const std::vector<float>& increment)
{
  lock(token_id);

  auto key = to_key(token_id);
  if (cache_mode_ == PhiMatrixCacheMode::WRITE) {
    if (cache_.has_key(token_id)) {
      auto values_ptr = cache_.get(token_id);

      for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
        (*values_ptr)[topic_id] += increment[topic_id];
      }
    } else {
      cache_.set(token_id, std::make_shared<std::vector<float>>(increment));
    }
  } else {
    if (!redis_client->increase_values(key, increment)) {
      LOG(WARNING) << "Update of token data " << key << " has failed" << std::endl;
    }
  }

  unlock(token_id);
}

int RedisPhiMatrix::add_token(std::shared_ptr<RedisClient> redis_client, const Token& token, bool flag) {
  auto values = std::vector<float>(topic_size(), 0.0f);
  return add_token(redis_client, token, flag, values);
}

int RedisPhiMatrix::add_token(std::shared_ptr<RedisClient> redis_client, const Token& token,
                              bool flag, const std::vector<float>& values)
{
  int token_id = token_collection_.token_id(token);
  if (token_id != -1) {
    return token_id;
  }
  
  spin_locks_.push_back(std::make_shared<SpinLock>());
  int index = token_collection_.add_token(token);
  if (flag) {
    redis_client->set_values(to_key(index), values);
  }
  return index;
}

void RedisPhiMatrix::dump_write_cache(std::shared_ptr<RedisClient> redis_client,
                                      int token_begin_index,
                                      int token_end_index)
{
  if (cache_mode_ != PhiMatrixCacheMode::WRITE) {
    return;
  }

  // randomize loop to minimize risk of collision between executors
  auto indices = std::make_shared<std::vector<int>>(std::vector<int>(token_end_index - token_begin_index));
  std::iota(indices->begin(), indices->end(), token_begin_index);
  std::random_shuffle(indices->begin(), indices->end());

  for (int token_id : *indices) {
    // No need in lock on token as each thread deal only with own set of tokens
    auto key = to_key(token_id);
    if (!cache_.has_key(token_id)) {
      continue;
    }

    auto values_ptr = cache_.get(token_id);

    if (!redis_client->increase_values(key, *values_ptr)) {
      LOG(ERROR) << "Update of token data from cache " << key << " has failed" << std::endl;
    }

    cache_.erase(token_id);
  }
}
