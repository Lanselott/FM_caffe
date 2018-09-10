#ifndef CAFFE_FM_LAYER_HPP_
#define CAFFE_FM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe
{

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class FmLayer : public Layer<Dtype>
{
  public:
    explicit FmLayer(const LayerParameter &param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);
    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "Fm"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    /*
    inline void conv_im2col_cpu(const Dtype *data, Dtype *col_buff, int *all_zero_mask = NULL)
    {
        im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                      col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                      pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
    */
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);
    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

    int M_;
    int K_;
    int N_;
    int conv_out_spatial_dim_;
    int col_offset_;

    bool csr_init;
    bool bias_term_;
    vector<int> col_buffer_shape_;
    Blob<Dtype> col_buffer_;
    Blob<Dtype> bias_multiplier_;
    Blob<Dtype> v_vector_;
    bool transpose_; ///< if true, assume transposed weights

    int k_value;
    bool do_sparse;
    Blob<int> col_buf_mask_;
    Blob<int> row_buf_mask_;
    vector<int> left_columns_; //the number of left columns of weight matrix for each group
    vector<int> left_rows_;
    Blob<Dtype> squeezed_weight_buffer_;

    Blob<Dtype> nz_weight_values_;//nonzero elements
    Blob<int> nz_weight_indices_;//index of nonzero
    Blob<int> nz_weight_index_pointers_;//pointer(index) of indices
};

} // namespace caffe

#endif // CAFFE_INNER_PRODUCT_LAYER_HPP_
