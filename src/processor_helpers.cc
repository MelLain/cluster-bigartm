#include <algorithm>

#include "processor_helpers.h"

namespace {
  void NormalizeTheta(int item_index, int inner_iter, int topics_size, const float* n_td) {
    float sum = 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = n_td[topic_index];
      if (val > 0.0f) {
        sum += val;
      }
    }

    float sum_inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (int topic_index = 0; topic_index < topics_size; ++topic_index) {
      float val = sum_inv * (n_td[topic_index]);
      if (val < 1e-16f) {
        val = 0.0f;
      }

      const_cast<float*>(n_td)[topic_index] = val;
    }
  }
}

std::shared_ptr<LocalThetaMatrix<float>> ProcessorHelpers::InitializeTheta(int topic_size, const artm::Batch& batch) {
  auto Theta = std::make_shared<LocalThetaMatrix<float>>(topic_size, batch.item_size());

  Theta->InitializeZeros();
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    const float default_theta = 1.0f / topic_size;
    for (int iTopic = 0; iTopic < topic_size; ++iTopic) {
      (*Theta)(iTopic, item_index) = default_theta;
    }
  }
  return Theta;
}

std::shared_ptr<CsrMatrix<float>> ProcessorHelpers::InitializeSparseNdw(const artm::Batch& batch) {
  std::vector<float> n_dw_val;
  std::vector<int> n_dw_row_ptr;
  std::vector<int> n_dw_col_ind;

  // For sparse case
  for (int item_index = 0; item_index < batch.item_size(); ++item_index) {
    n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
    const auto& item = batch.item(item_index);

    for (int token_index = 0; token_index < item.token_id_size(); ++token_index) {
      const int token_id = item.token_id(token_index);
      const float token_weight = item.token_weight(token_index);

      n_dw_val.push_back(token_weight);
      n_dw_col_ind.push_back(token_id);
    }
  }
  n_dw_row_ptr.push_back(static_cast<int>(n_dw_val.size()));
  return std::make_shared<CsrMatrix<float>>(batch.token_size(), &n_dw_val, &n_dw_row_ptr, &n_dw_col_ind);
}

void ProcessorHelpers::FindBatchTokenIds(const artm::Batch& batch, const PhiMatrix& phi_matrix, std::vector<int>* token_id) {
  token_id->resize(batch.token_size(), -1);
  for (int token_index = 0; token_index < batch.token_size(); ++token_index) {
    token_id->at(token_index) = phi_matrix.token_index(Token(batch.class_id(token_index), batch.token(token_index)));
  }
}

void ProcessorHelpers::InferThetaAndUpdateNwtSparse(const artm::Batch& batch,
                                                    const CsrMatrix<float>& sparse_ndw,
                                                    const PhiMatrix& p_wt,
                                                    LocalThetaMatrix<float>* theta_matrix,
                                                    NwtWriteAdapter* nwt_writer,
                                                    Blas* blas,
                                                    int num_inner_iters,
                                                    float* perplexity_value) {
  LocalThetaMatrix<float> n_td(theta_matrix->num_topics(), theta_matrix->num_items());
  const int num_topics = p_wt.topic_size();
  const int docs_count = theta_matrix->num_items();
  const int tokens_count = batch.token_size();

  std::vector<int> token_id;
  ProcessorHelpers::FindBatchTokenIds(batch, p_wt, &token_id);

  int max_local_token_size = 0;  // find the longest document from the batch
  for (int d = 0; d < docs_count; ++d) {
    const int begin_index = sparse_ndw.row_ptr()[d];
    const int end_index = sparse_ndw.row_ptr()[d + 1];
    const int local_token_size = end_index - begin_index;
    max_local_token_size = std::max(max_local_token_size, local_token_size);
  }

  LocalPhiMatrix<float> local_phi(max_local_token_size, num_topics);
  std::vector<float> helper_vector(num_topics, 0.0f);

  for (int d = 0; d < docs_count; ++d) {
    float* ntd_ptr = &n_td(0, d);
    float* theta_ptr = &(*theta_matrix)(0, d);  // NOLINT

    const int begin_index = sparse_ndw.row_ptr()[d];
    const int end_index = sparse_ndw.row_ptr()[d + 1];
    local_phi.InitializeZeros();
    bool item_has_tokens = false;
    for (int i = begin_index; i < end_index; ++i) {
      int w = sparse_ndw.col_ind()[i];
      if (token_id[w] == PhiMatrix::kUndefIndex) {
        continue;
      }
      item_has_tokens = true;
      float* local_phi_ptr = &local_phi(i - begin_index, 0);
      p_wt.get(token_id[w], &helper_vector);
      for (int k = 0; k < num_topics; ++k) {
        local_phi_ptr[k] = helper_vector[k];
      }
    }

    if (!item_has_tokens) {
      continue;  // continue to the next item
    }

    for (int inner_iter = 0; inner_iter < num_inner_iters; ++inner_iter) {
      for (int k = 0; k < num_topics; ++k) {
        ntd_ptr[k] = 0.0f;
      }

      for (int i = begin_index; i < end_index; ++i) {
        const float* phi_ptr = &local_phi(i - begin_index, 0);

        float p_dw_val = 0;
        for (int k = 0; k < num_topics; ++k) {
          p_dw_val += phi_ptr[k] * theta_ptr[k];
        }
        if (p_dw_val == 0) {
          continue;
        }

        const float alpha = sparse_ndw.val()[i] / p_dw_val;
        for (int k = 0; k < num_topics; ++k) {
          ntd_ptr[k] += alpha * phi_ptr[k];
        }
      }

      for (int k = 0; k < num_topics; ++k) {
        theta_ptr[k] *= ntd_ptr[k];
      }

      NormalizeTheta(d, inner_iter, num_topics, theta_ptr);
    }
  }

  if (nwt_writer == nullptr) {
    return;
  }

  std::vector<int> token_nwt_id;
  ProcessorHelpers::FindBatchTokenIds(batch, *nwt_writer->n_wt(), &token_nwt_id);

  CsrMatrix<float> sparse_nwd(sparse_ndw);
  sparse_nwd.Transpose(blas);

  std::vector<float> p_wt_local(num_topics, 0.0f);
  std::vector<float> n_wt_local(num_topics, 0.0f);
  for (int w = 0; w < tokens_count; ++w) {
    if (token_nwt_id[w] == -1) {
      continue;
    }

    if (token_id[w] != -1) {
      p_wt.get(token_id[w], &p_wt_local);
    } else {
      p_wt_local.assign(num_topics, 1.0f);
    }

    for (int i = sparse_nwd.row_ptr()[w]; i < sparse_nwd.row_ptr()[w + 1]; ++i) {
      int d = sparse_nwd.col_ind()[i];
      float p_wd_val = blas->sdot(num_topics, &p_wt_local[0], 1, &(*theta_matrix)(0, d), 1);  // NOLINT
      if (p_wd_val < 1e-16f) {
        continue;
      }
      blas->saxpy(num_topics, sparse_nwd.val()[i] / p_wd_val,
        &(*theta_matrix)(0, d), 1, &n_wt_local[0], 1);  // NOLINT

      // compute perplexity
      float perp_value = 0.0f;
      for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
        perp_value += p_wt_local[topic_index] * (*theta_matrix)(topic_index, d);
      }
      *perplexity_value += sparse_nwd.val()[i] * log(perp_value > 0.0f ? perp_value : 1.0f);
    }

    std::vector<float> values(num_topics, 0.0f);
    for (int topic_index = 0; topic_index < num_topics; ++topic_index) {
      values[topic_index] = p_wt_local[topic_index] * n_wt_local[topic_index];
      n_wt_local[topic_index] = 0.0f;
    }

    nwt_writer->Store(token_nwt_id[w], values);
  }
}
