#pragma once

#include <unordered_map>
#include <memory>
#include <vector>
#include <string>

#include "helpers.h"
#include "phi_matrix.h"
#include "protobuf_helpers.h"
#include "blas.h"

const float kProcessorEps = 1e-16f;

class NwtWriteAdapter {
 public:
  explicit NwtWriteAdapter(PhiMatrix* n_wt) : n_wt_(n_wt) { }

  void store(int nwt_token_id, const std::vector<float>& nwt_vector) {
    assert(nwt_vector.size() == n_wt_->topic_size());
    assert((nwt_token_id >= 0) && (nwt_token_id < n_wt_->token_size()));
    n_wt_->increase(nwt_token_id, nwt_vector);
  }

  PhiMatrix* n_wt() {
    return n_wt_;
  }

 private:
  PhiMatrix* n_wt_;
};

class ProcessorHelpers {
 public:
  static std::shared_ptr<LocalThetaMatrix<float>> initialize_theta(int topic_size, const artm::Batch& batch);

  static std::shared_ptr<CsrMatrix<float>> initialize_sparse_ndw(const artm::Batch& batch);

  static void find_batch_token_ids(const artm::Batch& batch, const PhiMatrix& phi_matrix, std::vector<int>* token_id);

  static void infer_theta_and_update_nwt_sparse(const artm::Batch& batch,
                                                const CsrMatrix<float>& sparse_ndw,
                                                const PhiMatrix& p_wt,
                                                LocalThetaMatrix<float>* theta_matrix,
                                                NwtWriteAdapter* nwt_writer,
                                                Blas* blas,
                                                int num_inner_iters,
                                                double* perplexity_value);

  ProcessorHelpers() = delete;
};
