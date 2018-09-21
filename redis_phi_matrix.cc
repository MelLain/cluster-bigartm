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

float RedisPhiMatrix::get(int token_id, int topic_id) const {
  //throw std::runtime_error("Single get is forbidden");
  std::vector<float> buffer = redis_client_->redis_get(to_key(token_id), topic_size());
  return buffer[topic_id];
}

void RedisPhiMatrix::get(int token_id, std::vector<float>* buffer) const {
  std::vector<float> temp = redis_client_->redis_get(to_key(token_id), topic_size());
  for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
    (*buffer)[topic_id] = temp[topic_id];
  }
}

void RedisPhiMatrix::set(int token_id, int topic_id, float value) {
  throw std::runtime_error("Single set is forbidden");
  //std::vector<float> buffer = redis_get(to_key(token_id));
  //buffer[topic_id] = value;
  //redis_set(to_key(token_id), buffer);
}

void RedisPhiMatrix::set(int token_id, const std::vector<float>& buffer) {
  redis_client_->redis_set(to_key(token_id), buffer);
}

void RedisPhiMatrix::increase(int token_id, int topic_id, float increment) {
  throw std::runtime_error("Single increase is forbidden");
  //std::vector<float> buffer = redis_get(to_key(token_id));
  //buffer[topic_id] += increment;
  //redis_set(to_key(token_id), buffer);
}

void RedisPhiMatrix::increase(int token_id, const std::vector<float>& increment) {
  auto key = to_key(token_id);
  if (!redis_client_->redis_increase(key, increment)) {
    std::cout << "WARN: Update of token data " << key << " has failed" << std::endl;
  }
}

int RedisPhiMatrix::AddToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1) {
    return token_id;
  }
  
  int index = token_collection_.AddToken(token);
  auto temp = std::vector<float>(topic_size(), 0.0f);
  redis_client_->redis_set(to_key(index), temp);
  return index;
}
