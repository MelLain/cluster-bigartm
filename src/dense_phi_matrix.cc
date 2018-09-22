#include "dense_phi_matrix.h"

#include <algorithm>

PhiMatrixFrame::PhiMatrixFrame(const ModelName& model_name,
                               const std::vector<std::string>& topic_name)
    : model_name_(model_name)
    , topic_name_()
    , token_collection_() {
  if (topic_name.size() == 0) {
    BOOST_THROW_EXCEPTION(std::runtime_error("Can not create model " + model_name + " with 0 topics"));
  }

  for (auto iter = topic_name.begin(); iter != topic_name.end(); ++iter) {
    topic_name_.push_back(*iter);
  }
}

PhiMatrixFrame::PhiMatrixFrame(const PhiMatrixFrame& rhs)
    : model_name_(rhs.model_name_),
      topic_name_(rhs.topic_name_),
      token_collection_(rhs.token_collection_) { }

const Token PhiMatrixFrame::token(int index) const {
  return token_collection_.token(index);
}

std::vector<std::string> PhiMatrixFrame::topic_name() const {
  std::vector<std::string> topic_name;
  for (const auto& elem : topic_name_) {
    topic_name.push_back(elem);
  }
  return topic_name;
}

const std::string& PhiMatrixFrame::topic_name(int topic_id) const {
  return topic_name_[topic_id];
}

void PhiMatrixFrame::set_topic_name(int topic_id, const std::string& topic_name) {
  topic_name_[topic_id] = topic_name;
}

std::string PhiMatrixFrame::model_name() const {
  return model_name_;
}

bool PhiMatrixFrame::has_token(const Token& token) const {
  return token_collection_.has_token(token);
}

int PhiMatrixFrame::token_index(const Token& token) const {
  return token_collection_.token_id(token);
}

void PhiMatrixFrame::Clear() {
  token_collection_.Clear();
}

int PhiMatrixFrame::AddToken(const Token& token) {
  int token_id = token_collection_.token_id(token);
  if (token_id != -1) {
    return token_id;
  }

  return token_collection_.AddToken(token);
}

void PhiMatrixFrame::Swap(PhiMatrixFrame* rhs) {
  model_name_.swap(rhs->model_name_);
  topic_name_.swap(rhs->topic_name_);
  token_collection_.Swap(&rhs->token_collection_);
}

// =======================================================
// PackedValues methods
// =======================================================

PackedValues::PackedValues()
    : values_()
    , bitmask_()
    , ptr_() { }

PackedValues::PackedValues(int size)
    : values_()
    , bitmask_()
    , ptr_() {
  bitmask_.resize(size, false);
}

PackedValues::PackedValues(const PackedValues& rhs)
    : values_(rhs.values_)
    , bitmask_(rhs.bitmask_)
    , ptr_(rhs.ptr_) { }

PackedValues::PackedValues(const float* values, int size)
    : values_()
    , bitmask_()
    , ptr_() {
  values_.resize(size); memcpy(&values_[0], values, sizeof(float) * size);
  pack();
}

bool PackedValues::is_packed() const {
  return !bitmask_.empty();
}

float PackedValues::get(int index) const {
  if (is_packed()) {
    if (!bitmask_[index]) {
      return 0.0f;
    }
    const auto sparse_ptr = std::lower_bound(ptr_.begin(), ptr_.end(), index);
    const int sparse_index = sparse_ptr - ptr_.begin();
    return values_[sparse_index];
  } else {
    return values_[index];
  }
}

void PackedValues::get(std::vector<float>* buffer) const {
  if (is_packed()) {
    buffer->assign(buffer->size(), 0.0f);
    for (int i = 0; i < (int64_t) ptr_.size(); i++) {
      buffer->at(ptr_[i]) = values_[i];
    }
  } else {
    buffer->assign(values_.begin(), values_.end());
  }
}

float* PackedValues::unpack() {
  if (is_packed()) {
    const int full_size = bitmask_.size();
    const int sparse_size = values_.size();
    if (values_.size() != ptr_.size()) {
      assert(values_.size() == ptr_.size());
    }

    std::vector<float> values(full_size, 0.0f);
    for (int i = 0; i < sparse_size; ++i) {
      values[ptr_[i]] = values_[i];
    }

    values_.swap(values);

    bitmask_.clear();
    ptr_.clear();
  }

  return &values_[0];
}

void PackedValues::pack() {
  if (is_packed()) {
    return;
  }

  int num_zeros = 0;
  for (auto value : values_) {
    if (value == 0) {
      num_zeros++;
    }
  }

  // pack iff at 60% of elements (or more) are zeros
  if (num_zeros < (int64_t) (3 * values_.size() / 5)) {
    return;
  }

  bitmask_.resize(values_.size(), false);
  ptr_.resize(values_.size() - num_zeros);
  std::vector<float> values(values_.size() - num_zeros, 0.0f);

  int sparse_index = 0;
  for (int i = 0; i < (int64_t) values_.size(); ++i) {
    if (values_[i] == 0) {
      continue;
    }

    ptr_[sparse_index] = i;
    values[sparse_index] = values_[i];
    bitmask_[i] = true;
    sparse_index++;
  }

  values_.swap(values);
}

void PackedValues::reset(int size) {
  bitmask_.resize(size, false);
  values_.clear();
  ptr_.clear();
}

// =======================================================
// DensePhiMatrix methods
// =======================================================

DensePhiMatrix::DensePhiMatrix(const ModelName& model_name,
                               const std::vector<std::string>& topic_name)
    : PhiMatrixFrame(model_name, topic_name), values_() { }

DensePhiMatrix::DensePhiMatrix(const DensePhiMatrix& rhs) : PhiMatrixFrame(rhs), values_() {
  for (int token_index = 0; token_index < rhs.token_size(); ++token_index) {
    values_.push_back(PackedValues(rhs.values_[token_index]));
  }
}

std::shared_ptr<PhiMatrix> DensePhiMatrix::Duplicate() const {
  return std::shared_ptr<PhiMatrix>(new DensePhiMatrix(*this));
}

float DensePhiMatrix::get(int token_id, int topic_id) const {
  return values_[token_id].get(topic_id);
}

void DensePhiMatrix::get(int token_id, std::vector<float>* buffer) const {
  assert(topic_size() > 0 && buffer->size() == topic_size());
  values_[token_id].get(buffer);
}

void DensePhiMatrix::set(int token_id, int topic_id, float value) {
  values_[token_id].unpack()[topic_id] = value;
  if ((topic_id + 1) == topic_size()) {
    values_[token_id].pack();
  }
}

void DensePhiMatrix::set(int token_id, const std::vector<float>& buffer) {
  for (int topic_id = 0; topic_id < topic_size(); ++topic_id) {
    values_[token_id].unpack()[topic_id] = buffer[topic_id];
    if ((topic_id + 1) == topic_size()) {
      values_[token_id].pack();
    }
  }
}

void DensePhiMatrix::increase(int token_id, int topic_id, float increment) {
  values_[token_id].unpack()[topic_id] += increment;
  if ((topic_id + 1) == topic_size()) {
    values_[token_id].pack();
  }
}

void DensePhiMatrix::increase(int token_id, const std::vector<float>& increment) {
  const int topic_size = this->topic_size();
  if (increment.size() != topic_size) {
    throw std::runtime_error("Inc size: " + std::to_string(increment.size()) + ", phi size: " + std::to_string(this->topic_size()));
  }

  float* values = values_[token_id].unpack();
  for (int topic_index = 0; topic_index < topic_size; ++topic_index) {
    values[topic_index] += increment[topic_index];
  }
  values_[token_id].pack();
}

void DensePhiMatrix::Clear() {
  values_.clear();
  PhiMatrixFrame::Clear();
}

int DensePhiMatrix::AddToken(const Token& token, bool flag) {
  int token_id = token_index(token);
  if (token_id != -1) {
    return token_id;
  }

  values_.push_back(PackedValues(topic_size()));
  int retval = PhiMatrixFrame::AddToken(token);
  assert(retval == (values_.size() - 1));
  return retval;
}

void DensePhiMatrix::Reset() {
  for (PackedValues& value : values_) {
    value.reset(topic_size());
  }
}

void DensePhiMatrix::Reshape(const PhiMatrix& phi_matrix) {
  Clear();
  for (int token_id = 0; token_id < phi_matrix.token_size(); ++token_id) {
    this->AddToken(phi_matrix.token(token_id), true);
  }
}

void DensePhiMatrix::Print(std::ostream& stream) {
  for (const auto& name : topic_name()) {
    stream << name << " ";
  }

  stream << "\n";
  for (int t = 0; t < token_size(); ++t) {
    for (int i = 0; i < topic_size(); ++i) {
      stream << values_[t].unpack()[i] << " "; 
    }
    stream << "\n";
  }
}
