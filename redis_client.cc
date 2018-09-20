#include "redis_client.h"

void RedisClient::redis_set(const std::string& key, const std::vector<float>& values) const {
  auto val_ptr = reinterpret_cast<const char*>(&(values[0]));
  auto val_size = (size_t) (values.size() * sizeof(float));
  reply_ = (redisReply*) redisCommand(context_, "SET %s %b", key.c_str(), val_ptr, val_size);

  freeReplyObject(reply_);
  reply_ = nullptr;
}

std::vector<float> RedisClient::redis_get(const std::string& key, int values_size) const {
  reply_ = (redisReply*) redisCommand(context_, "GET %s", key.c_str());

  auto dst = reinterpret_cast<const float*>(reply_->str);
  auto retval = std::vector<float>(dst, dst + values_size);

  freeReplyObject(reply_);
  reply_ = nullptr;

  return retval;
}
