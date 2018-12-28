#include "redis_phi_matrix.h"

int RedisPhiMatrix::token_size() const {
  return token_collection_.token_size();
}

const Token RedisPhiMatrix::token(int index) const {
  return token_collection_.token(index);
}

bool RedisPhiMatrix::has_token(const Token& token) const {
  return token_collection_.has_token(token);
}

int RedisPhiMatrix::token_index(const Token& token) const {
  return token_collection_.token_id(token);
}

// ATTN: this method should be used only for debugging, it's too slow for learning process!
float RedisPhiMatrix::get(int token_id, int topic_id) const {
  std::vector<float> buffer = redis_client_.get_values(to_key(token_id), topic_size());
  return buffer[topic_id];
}

void RedisPhiMatrix::get(int token_id, std::vector<float>* buffer) const {
  auto iter = cache_.find(token_id);
  if (iter != cache_.end()) {
    const auto& temp = iter->second;
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      (*buffer)[topic_id] = temp[topic_id];
    }
  } else {
    std::vector<float> temp = redis_client_.get_values(to_key(token_id), topic_size());
    for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
      (*buffer)[topic_id] = temp[topic_id];
    }

    if (use_cache_) {
      cache_.emplace(token_id, temp);
    }
  }
}

void RedisPhiMatrix::get_set(int token_id, std::vector<float>* buffer, const std::vector<float>& values) {
  std::vector<float> temp = redis_client_.get_set_values(to_key(token_id), values);
  for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
    (*buffer)[topic_id] = temp[topic_id];
  }
}

void RedisPhiMatrix::set(int token_id, int topic_id, float value) {
  throw std::runtime_error("Redis matrix does not support single set");
}

void RedisPhiMatrix::set(int token_id, const std::vector<float>& buffer) {
  redis_client_.set_values(to_key(token_id), buffer);
}

void RedisPhiMatrix::increase(int token_id, int topic_id, float increment) {
  throw std::runtime_error("Redis matrix does not support single increase");
}

void RedisPhiMatrix::increase(int token_id, const std::vector<float>& increment) {
  auto key = to_key(token_id);
  if (!redis_client_.increase_values(key, increment)) {
    std::cout << "WARN: Update of token data " << key << " has failed" << std::endl;
  }
}

int RedisPhiMatrix::AddToken(const Token& token, bool flag) {
  auto values = std::vector<float>(topic_size(), 0.0f);
  return AddToken(token, flag, values);
}

int RedisPhiMatrix::AddToken(const Token& token, bool flag, const std::vector<float>& values) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1) {
    return token_id;
  }
  
  int index = token_collection_.AddToken(token);
  if (flag) {
    redis_client_.set_values(to_key(index), values);
  }
  return index;
}
