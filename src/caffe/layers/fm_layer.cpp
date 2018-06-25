#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
void FmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top)
{
  const int num_output = this->layer_param_.fm_param().num_output();
  bias_term_ = this->layer_param_.fm_param().bias_term();
  transpose_ = this->layer_param_.fm_param().transpose();
  k_value = this->layer_param_.fm_param().k_value();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.fm_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  /*
  //Ran: check if we need to set up the v_vector_
  if (this->blobs_.size() > 0)
  {
    LOG(INFO) << "Skipping v_vector initialization";
  }
  else
  {
    //v_vector_ size should be [num_output,k_value,n,1]
    //we do not support transpose yet
    this->blobs_.resize(1); //NO BIAS

    vector<int> v_vector_shape(4);

    v_vector_shape[0] = num_output;
    v_vector_shape[1] = k_value;
    v_vector_shape[2] = bottom[0]->shape(2);
    v_vector_shape[3] = 1;

    v_vector_[0].reset(new Blob<Dtype>(v_vector_shape));
    //fill the v_vector_
    shared_ptr<Filler<Dtype> > vector_filler(GetFiller<Dtype>(
        this->layer_param_.fm_param().vector_filler()));
    vector_filler->Fill(v_vector_[0].get());



  }
  */

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0)
  {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else
  {
    if (bias_term_)
    {
      this->blobs_.resize(16);
    }
    else
    {
      this->blobs_.resize(15);
    }
    vector<int> v_vector_shape(4);
    vector<int> vx_shape(4);
    vector<int> vx_sum_shape(4);
    vector<int> vx_sum_square(4);
    vector<int> v2x2(4);
    vector<int> v2x2_sum(4);

    vector<int> temp_(4);
    vector<int> v_forward_output(2);
    vector<int> vxx_shape(4);
    vector<int> x_sum_vx(4);

    vector<int> xx(2);
    vector<int> v_diff(4);

    //[num_output,k_value,1,CHW(K_)]
    v_vector_shape[0] = num_output;
    v_vector_shape[1] = k_value;
    v_vector_shape[2] = 1; //
    v_vector_shape[3] = K_;

    vx_shape[0] = bottom[0]->shape(0);
    vx_shape[1] = num_output;
    vx_shape[2] = k_value;
    vx_shape[3] = K_;

    vx_sum_shape[0] = bottom[0]->shape(0);
    vx_sum_shape[1] = num_output;
    vx_sum_shape[2] = k_value;
    vx_sum_shape[3] = 1;

    vx_sum_square[0] = bottom[0]->shape(0);
    vx_sum_square[1] = num_output;
    vx_sum_square[2] = k_value;
    vx_sum_square[3] = 1;

    v2x2[0] = bottom[0]->shape(0);
    v2x2[1] = num_output;
    v2x2[2] = k_value;
    v2x2[3] = K_;

    v2x2_sum[0] = bottom[0]->shape(0);
    v2x2_sum[1] = num_output;
    v2x2_sum[2] = k_value;
    v2x2_sum[3] = 1;

    temp_[0] = bottom[0]->shape(0);
    temp_[1] = num_output;
    temp_[2] = k_value;
    temp_[3] = 1;

    v_forward_output[0] = bottom[0]->shape(0);
    v_forward_output[1] = num_output;

    vxx_shape[0] = bottom[0]->shape(0);
    vxx_shape[1] = num_output;
    vxx_shape[2] = k_value;
    vxx_shape[3] = K_;

    x_sum_vx[0] = bottom[0]->shape(0);
    x_sum_vx[1] = num_output;
    x_sum_vx[2] = k_value; //
    x_sum_vx[3] = K_;

    xx[0] = bottom[0]->shape(0);
    xx[1] = K_;

    v_diff[0] = bottom[0]->shape(0);
    v_diff[1] = num_output;
    v_diff[2] = k_value;
    v_diff[3] = K_;
    //Backward part for update bottom_value:
    vector<int> v_sum(3);
    vector<int> x_diff(3);

    v_sum[0] = num_output;
    v_sum[1] = K_;
    v_sum[2] = K_;

    x_diff[0] = bottom[0]->shape(0);
    x_diff[1] = num_output;
    x_diff[2] = K_;

    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_)
    {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    }
    else
    {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.fm_param().weight_filler()));

    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, intiialize and fill the bias term
    if (bias_term_)
    {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.fm_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());

      this->blobs_[2].reset(new Blob<Dtype>(v_vector_shape));
      //fill the v_vector_

      shared_ptr<Filler<Dtype> > vector_filler(GetFiller<Dtype>(
          this->layer_param_.fm_param().vector_filler()));
      vector_filler->Fill(this->blobs_[2].get());

      this->blobs_[3].reset(new Blob<Dtype>(vx_shape));
      LOG(INFO) << "ALOHA";
      this->blobs_[4].reset(new Blob<Dtype>(vx_sum_shape));
      LOG(INFO) << "www";
      this->blobs_[5].reset(new Blob<Dtype>(vx_sum_square));
      LOG(INFO) << "www";
      this->blobs_[6].reset(new Blob<Dtype>(v2x2));
      LOG(INFO) << "www";
      this->blobs_[7].reset(new Blob<Dtype>(v2x2_sum));
      LOG(INFO) << "www";
      this->blobs_[8].reset(new Blob<Dtype>(temp_));
      LOG(INFO) << "www";
      this->blobs_[9].reset(new Blob<Dtype>(v_forward_output));
      LOG(INFO) << "www";
      this->blobs_[10].reset(new Blob<Dtype>(vxx_shape));
      LOG(INFO) << "www";
      this->blobs_[11].reset(new Blob<Dtype>(x_sum_vx));
      LOG(INFO) << "www";
      this->blobs_[12].reset(new Blob<Dtype>(xx));
      LOG(INFO) << "www";
      this->blobs_[13].reset(new Blob<Dtype>(v_diff));
      LOG(INFO) << "www";
      this->blobs_[14].reset(new Blob<Dtype>(v_sum));
      LOG(INFO) << "www";
      this->blobs_[15].reset(new Blob<Dtype>(x_diff));
      LOG(INFO) << "www";
    }
    else
    {
      this->blobs_[1].reset(new Blob<Dtype>(v_vector_shape));
      //fill the v_vector_

      shared_ptr<Filler<Dtype> > vector_filler(GetFiller<Dtype>(
          this->layer_param_.fm_param().vector_filler()));
      vector_filler->Fill(this->blobs_[1].get());

      this->blobs_[2].reset(new Blob<Dtype>(vx_shape));
      this->blobs_[3].reset(new Blob<Dtype>(vx_sum_shape));
      this->blobs_[4].reset(new Blob<Dtype>(vx_sum_square));
      this->blobs_[5].reset(new Blob<Dtype>(v2x2));
      this->blobs_[6].reset(new Blob<Dtype>(v2x2_sum));
      this->blobs_[7].reset(new Blob<Dtype>(temp_));
      this->blobs_[8].reset(new Blob<Dtype>(v_forward_output));
      this->blobs_[9].reset(new Blob<Dtype>(vxx_shape));
      this->blobs_[10].reset(new Blob<Dtype>(x_sum_vx));
      this->blobs_[11].reset(new Blob<Dtype>(xx));
      this->blobs_[12].reset(new Blob<Dtype>(v_diff));
      this->blobs_[13].reset(new Blob<Dtype>(v_sum));
      this->blobs_[14].reset(new Blob<Dtype>(x_diff));
    }

  } // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void FmLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top)
{
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.fm_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_)
  {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void FmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top)
{
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                        M_, N_, K_, (Dtype)1.,
                        bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_)
  {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(),
                          this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void FmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom)
{
  if (this->param_propagate_down_[0])
  {
    const Dtype *top_diff = top[0]->cpu_diff();
    const Dtype *bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_)
    {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            K_, N_, M_,
                            (Dtype)1., bottom_data, top_diff,
                            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
    else
    {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            N_, K_, M_,
                            (Dtype)1., top_diff, bottom_data,
                            (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1])
  {
    const Dtype *top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0])
  {
    const Dtype *top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_)
    {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            M_, K_, N_,
                            (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                            (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
    else
    {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            M_, K_, N_,
                            (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                            (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FmLayer);
#endif

INSTANTIATE_CLASS(FmLayer);
REGISTER_LAYER_CLASS(Fm);

} // namespace caffe
