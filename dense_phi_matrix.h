#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "boost/utility.hpp"

#include "common.h"
#include "phi_matrix.h"
#include "token.h"

class PhiMatrixFrame : public PhiMatrix {
 public:
  explicit PhiMatrixFrame(const ModelName& model_name,
                          const std::vector<std::string>& topic_name);

  virtual ~PhiMatrixFrame() { }

  virtual int topic_size() const { return static_cast<int>(topic_name_.size()); }
  virtual int token_size() const { return static_cast<int>(token_collection_.token_size()); }
  virtual const Token token(int index) const;
  virtual bool has_token(const Token& token) const;
  virtual int token_index(const Token& token) const;
  virtual std::vector<std::string> topic_name() const;
  virtual const std::string& topic_name(int topic_id) const;
  virtual void set_topic_name(int topic_id, const std::string& topic_name);
  virtual ModelName model_name() const;

  void Clear();
  virtual int AddToken(const Token& token);

  void Swap(PhiMatrixFrame* rhs);

  PhiMatrixFrame(const PhiMatrixFrame& rhs);
  PhiMatrixFrame& operator=(const PhiMatrixFrame&);

 private:
  ModelName model_name_;
  std::vector<std::string> topic_name_;

  TokenCollection token_collection_;
};

class DensePhiMatrix;

class PackedValues {
 public:
  PackedValues();
  explicit PackedValues(int size);
  explicit PackedValues(const PackedValues& rhs);
  PackedValues(const float* values, int size);

  bool is_packed() const;
  float get(int index) const;
  void get(std::vector<float>* buffer) const;
  float* unpack();
  void pack();
  void reset(int size);

 private:
  std::vector<float> values_;
  std::vector<bool> bitmask_;
  std::vector<int> ptr_;
};


class DensePhiMatrix : public PhiMatrixFrame {
 public:
  explicit DensePhiMatrix(const ModelName& model_name,
                          const std::vector<std::string>& topic_name);

  virtual ~DensePhiMatrix() { Clear(); }

  virtual std::shared_ptr<PhiMatrix> Duplicate() const;

  virtual float get(int token_id, int topic_id) const;
  virtual void get(int token_id, std::vector<float>* buffer) const;
  virtual void set(int token_id, int topic_id, float value);
  virtual void set(int token_id, const std::vector<float>& buffer);
  virtual void increase(int token_id, int topic_id, float increment);
  virtual void increase(int token_id, const std::vector<float>& increment);  // must be thread-safe

  virtual void Clear();
  virtual int AddToken(const Token& token);

  void Reset();
  void Reshape(const PhiMatrix& phi_matrix);

  void Print(std::ostream& stream);

 private:
  DensePhiMatrix(const DensePhiMatrix& rhs);
  DensePhiMatrix& operator=(const PhiMatrixFrame&);

  std::vector<PackedValues> values_;
};
