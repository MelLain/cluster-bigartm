// Copyright 2017, Additive Regularization of Topic Models.

#include "token.h"

int TokenCollection::add_token(const Token& token) {
  int token_id = this->token_id(token);
  if (token_id != -1) {
    return token_id;
  }

  token_id = token_size();
  token_to_token_id_.insert(std::make_pair(token, token_id));
  token_id_to_token_.push_back(token);
  return token_id;
}

void TokenCollection::swap(TokenCollection* rhs) {
  token_to_token_id_.swap(rhs->token_to_token_id_);
  token_id_to_token_.swap(rhs->token_id_to_token_);
}

bool TokenCollection::has_token(const Token& token) const {
  return token_to_token_id_.count(token) > 0;
}

int TokenCollection::token_id(const Token& token) const {
  auto iter = token_to_token_id_.find(token);
  return (iter != token_to_token_id_.end()) ? iter->second : -1;
}

const Token& TokenCollection::token(int index) const {
  return token_id_to_token_[index];
}

void TokenCollection::clear() {
  token_to_token_id_.clear();
  token_id_to_token_.clear();
}

int TokenCollection::token_size() const {
  return static_cast<int>(token_to_token_id_.size());
}
