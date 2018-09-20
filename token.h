// Copyright 2017, Additive Regularization of Topic Models.

#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/functional/hash.hpp"

#include "common.h"

typedef std::string ClassId;
const std::string DefaultClass = "@default_class";

struct Token {
 public:
  Token(const ClassId& _class_id, const std::string& _keyword)
      : keyword(_keyword)
      , class_id(_class_id)
      , hash_(calcHash(_class_id, _keyword)) { }

  Token& operator=(const Token &rhs) {
    if (this != &rhs) {
      const_cast<std::string&>(keyword) = rhs.keyword;
      const_cast<ClassId&>(class_id) = rhs.class_id;
      const_cast<size_t&>(hash_) = rhs.hash_;
    }

    return *this;
  }

  bool operator<(const Token& token) const {
    if (keyword != token.keyword) {
      return keyword < token.keyword;
    }

    return class_id < token.class_id;
  }

  bool operator==(const Token& token) const {
    return (keyword == token.keyword && class_id == token.class_id);
  }

  bool operator!=(const Token& token) const {
    return !(*this == token);
  }

  size_t hash() const { return hash_; }

  const std::string keyword;
  const ClassId class_id;

 private:
  const size_t hash_;

  static size_t calcHash(const ClassId& class_id, const std::string& keyword) {
    size_t hash = 0;
    boost::hash_combine<std::string>(hash, keyword);
    boost::hash_combine<std::string>(hash, class_id);
    return hash;
  }
};

struct TokenHasher {
  size_t operator()(const Token& token) const {
    return token.hash();
  }
};

class TokenCollection {
 public:
  void Clear();
  int  AddToken(const Token& token);
  void Swap(TokenCollection* rhs);

  int token_size() const;
  bool has_token(const Token& token) const;
  int token_id(const Token& token) const;
  const Token& token(int index) const;

 private:
  std::unordered_map<Token, int, TokenHasher> token_to_token_id_;
  std::vector<Token> token_id_to_token_;
};
