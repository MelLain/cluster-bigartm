#pragma once

#include <vector>
#include <string>
#include <sstream>

#include "hiredis/hiredis.h"

#include "common.h"

class RedisClient {
 public:
  RedisClient(const std::string& ip, int port, int maxRetries, int numConnections, int timeout)
      : maxRetries_(maxRetries)
      , numConnections_(numConnections)
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
    if (reply_ != nullptr) {
        freeReplyObject(reply_);
    }
  }

  void redis_set(const std::string& key, const std::vector<float>& values) const;

  // compiler should return rvalue without coping
  // https://stackoverflow.com/questions/44065808/returning-stdvector-with-stdmove
  std::vector<float> redis_get(const std::string& key, int values_size) const;

 private:
  int maxRetries_;
  int numConnections_;
  int timeout_;

  mutable redisReply* reply_;
  redisContext* context_;
};
