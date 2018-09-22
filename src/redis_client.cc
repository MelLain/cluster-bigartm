#include "redis_client.h"

void RedisClient::redis_set(const std::string& key, const std::vector<float>& values) const {
  auto val_ptr = reinterpret_cast<const char*>(&(values[0]));
  auto val_size = (size_t) (values.size() * sizeof(float));

  reply_ = (redisReply*) redisCommand(context_, "SET %s %b", key.c_str(), val_ptr, val_size);
  clean_reply();
}

std::vector<float> RedisClient::redis_get(const std::string& key, int values_size) const {
  reply_ = (redisReply*) redisCommand(context_, "GET %s", key.c_str());

  auto values = reinterpret_cast<const float*>(reply_->str);
  auto retval = std::vector<float>(values, values + values_size);

  clean_reply();
  return retval;
}

void RedisClient::redis_set_value(const std::string& key, const std::string& value) const {
  reply_ = (redisReply*) redisCommand(context_, "SET %s %s", key.c_str(), value.c_str(), value.size());
  clean_reply();
}

std::string RedisClient::redis_get_value(const std::string& key, int value_size) const {
  reply_ = (redisReply*) redisCommand(context_, "GET %s", key.c_str());

  std::string retval = std::string(reply_->str, value_size);
  clean_reply();

  return retval;
}

bool RedisClient::redis_increase(const std::string& key, const std::vector<float>& increments) const {
  reply_ = (redisReply*) redisCommand(context_, "WATCH %s", key.c_str());
  clean_reply();

  // optimistic locking
  for (int i = 0; i < max_retries_; ++i) {
    reply_ = (redisReply*) redisCommand(context_, "GET %s", key.c_str());
    if (reply_->type == REDIS_REPLY_NIL) {
      clean_reply();
      continue;
    }

    auto values = reinterpret_cast<const float*>(reply_->str);
    auto buffer = std::vector<float>(values, values + increments.size());
    clean_reply();

    for (int j = 0; j < increments.size(); ++j) {
      buffer[j] += increments[j];
    }

    auto val_ptr = reinterpret_cast<const char*>(&(buffer[0]));
    auto val_size = (size_t) (buffer.size() * sizeof(float));

    reply_ = (redisReply*) redisCommand(context_, "SET %s %b", key.c_str(), val_ptr, val_size);

    if (reply_->type == REDIS_REPLY_NIL) {
      clean_reply();
      continue;
    }

    clean_reply();
    return true;
  }

  return false;
}
