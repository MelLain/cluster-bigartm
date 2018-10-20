#include "redis_client.h"

void RedisClient::set_values(const std::string& key, const std::vector<float>& values) const {
  auto val_ptr = reinterpret_cast<const char*>(&(values[0]));
  auto val_size = (size_t) (values.size() * sizeof(float));

  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "SET %s %b", key.c_str(), val_ptr, val_size);
  clean_reply();
}

std::vector<float> RedisClient::get_values(const std::string& key, int values_size) const {
  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "GET %s", key.c_str());

  auto values = reinterpret_cast<const float*>(reply_->str);
  auto retval = std::vector<float>(values, values + values_size);

  clean_reply();
  return retval;
}

void RedisClient::set_value(const std::string& key, const std::string& value) const {
  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(),
    "SET %s %b", key.c_str(), value.c_str(), value.size());

  clean_reply();
}

std::string RedisClient::get_value(const std::string& key) const {
  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "GET %s", key.c_str());

  std::string retval = std::string(reply_->str);
  clean_reply();

  return retval;
}

void RedisClient::set_hashmap(const std::string& key, const Normalizers& hashmap) const {
  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "DEL %s", key.c_str());
  clean_reply();

  for (const auto& kv : hashmap) {
    auto val_ptr = reinterpret_cast<const char*>(&(kv.second[0]));
    auto val_size = (size_t) (kv.second.size() * sizeof(double));

    reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(),
      "HSET %s %s %b", key.c_str(), kv.first.c_str(), val_ptr, val_size);

    clean_reply();
  }
}

Normalizers RedisClient::get_hashmap(const std::string& key, int values_size) const {
  reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "HKEYS %s", key.c_str());
  std::vector<std::string> hkeys;
  for (int i = 0; i < reply_->elements; ++ i) {
    hkeys.push_back(reply_->element[i]->str);
  }
  clean_reply();

  Normalizers retval;
  for (const auto& hkey : hkeys) {
    reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "HGET %s %s", key.c_str(), hkey.c_str());

    auto values = reinterpret_cast<const double*>(reply_->str);
    retval.emplace(std::make_pair(hkey, std::vector<double>(values, values + values_size)));

    clean_reply();    
  }

  return retval;
}

bool RedisClient::increase_values(const std::string& key, const std::vector<float>& increments) const {
  // optimistic locking
  for (int i = 0; i < max_retries_; ++i) {
    reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "WATCH %s", key.c_str());
    clean_reply();

    reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "GET %s", key.c_str());
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

    reply_ = (redisReply*) HiredisCommand<>::Command(context_, "MULTI", key.c_str());
    clean_reply();

    reply_ = (redisReply*) HiredisCommand<>::Command(context_, key.c_str(), "SET %s %b", key.c_str(), val_ptr, val_size);
    clean_reply();

    reply_ = (redisReply*) HiredisCommand<>::Command(context_, "EXEC", key.c_str());
    if (reply_->type == REDIS_REPLY_NIL) {
      clean_reply();
      continue;
    }

    clean_reply();
    return true;
  }

  std::cout << "Increments for key " << key << " were not applied" << std::endl;
  return false;
}
