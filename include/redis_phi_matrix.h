#pragma once

#include <iterator>
#include <vector>

#include "boost/lexical_cast.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "boost/uuid/uuid_generators.hpp"

#include "common.h"
#include "phi_matrix.h"
#include "token.h"
#include "redis_client.h"

class RedisPhiMatrix : public PhiMatrix{
 public:
  RedisPhiMatrix(const ModelName& model_name,
  	             const std::vector<std::string>& topic_name,
  	             RedisClient& redis_client,
                 bool use_cache = false)
      : model_name_(model_name)
      , topic_name_(topic_name)
      , token_collection_()
      , redis_client_(redis_client)
      , use_cache_() { }

  virtual int token_size() const;

  virtual int topic_size() const { return topic_name_.size(); }
  virtual std::vector<std::string> topic_name() const { return topic_name_; }

  const std::vector<std::string>& topic_name_ref() const { return topic_name_; }

  virtual const std::string& topic_name(int topic_id) const { return topic_name_[topic_id]; }

  virtual void set_topic_name(int topic_id, const std::string& topic_name) {
  	topic_name_[topic_id] = topic_name;
  }

  virtual ModelName model_name() const { return model_name_; }

  virtual const Token token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;

  virtual void set(int token_id, int topic_id, float value);
  virtual void set(int token_id, const std::vector<float>& buffer);

  virtual float get(int token_id, int topic_id) const;
  virtual void get(int token_id, std::vector<float>* buffer) const;
  virtual void increase(int token_id, int topic_id, float increment);
  virtual void increase(int token_id, const std::vector<float>& increment);

  virtual void Clear() { };
  virtual int AddToken(const Token& token, bool flag);
  int AddToken(const Token& token, bool flag, const std::vector<float>& values);

  void ClearCache() {
    cache_.clear();
  }

  virtual ~RedisPhiMatrix() {
    ClearCache();
  }

  virtual bool use_cache() const {
    return use_cache_;
  }

  virtual std::shared_ptr<PhiMatrix> Duplicate() const {
    throw std::runtime_error("RedisPhiMatrix doesn't support duplication");
  };

 private:
  std::string to_key(int i) const { return std::to_string(i) + model_name_; }

  ModelName model_name_;
  std::vector<std::string> topic_name_;
  TokenCollection token_collection_;
  RedisClient& redis_client_;
  bool use_cache_;
  mutable std::unordered_map<int, std::vector<float>> cache_;
};
