#pragma once

#include <vector>
#include <string>
#include <sstream>

#include "hiredis/hiredis.h"

#include "common.h"

class RedisClient {
 public:
  RedisClient(const std::string& ip, int port, int max_retries, int num_connections, int timeout)
      : max_retries_(max_retries)
      , num_connections_(num_connections)
      , timeout_(timeout)
      , reply_(nullptr)
  {
    context_ = redisConnect(ip.c_str(), port);
    if (context_->err == true) {
        std::stringstream ss;
        ss << "Error while creating context: " << context_->errstr;
        throw std::runtime_error(ss.str());
    }

    if (context_ == nullptr) {
        throw std::runtime_error("Cannot allocate Redis context");
    }
  }

  ~RedisClient() {
    redisFree(context_);
    clean_reply();
  }

  // both set and get operations are atomic by default
  void redis_set(const std::string& key, const std::vector<float>& values) const;

  // compiler should return rvalue without coping, see
  // https://stackoverflow.com/questions/44065808/returning-stdvector-with-stdmove
  std::vector<float> redis_get(const std::string& key, int values_size) const;

  // this operation is atomic, see https://redis.io/topics/transactions
  bool redis_increase(const std::string& key, const std::vector<float>& increments) const;

 private:
  void clean_reply() const {
    if (reply_ != nullptr) {
      freeReplyObject(reply_);
      reply_ = nullptr;
    }
  }

  int max_retries_;
  int num_connections_;
  int timeout_;

  mutable redisReply* reply_;
  redisContext* context_;
};
