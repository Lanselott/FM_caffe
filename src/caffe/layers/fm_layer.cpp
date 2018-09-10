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
  csr_init = false;

  const int num_output = this->layer_param_.fm_param().num_output();
  bias_term_ = this->layer_param_.fm_param().bias_term();
  transpose_ = this->layer_param_.fm_param().transpose();
  k_value = this->layer_param_.fm_param().k_value();
  do_sparse = this->layer_param_.fm_param().do_sparse();
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
      this->blobs_[4].reset(new Blob<Dtype>(vx_sum_shape));
      this->blobs_[5].reset(new Blob<Dtype>(vx_sum_square));
      this->blobs_[6].reset(new Blob<Dtype>(v2x2));
      this->blobs_[7].reset(new Blob<Dtype>(v2x2_sum));
      this->blobs_[8].reset(new Blob<Dtype>(temp_));
      this->blobs_[9].reset(new Blob<Dtype>(v_forward_output));
      this->blobs_[10].reset(new Blob<Dtype>(vxx_shape));
      this->blobs_[11].reset(new Blob<Dtype>(x_sum_vx));
      this->blobs_[12].reset(new Blob<Dtype>(xx));
      this->blobs_[13].reset(new Blob<Dtype>(v_diff));
      this->blobs_[14].reset(new Blob<Dtype>(v_sum));
      this->blobs_[15].reset(new Blob<Dtype>(x_diff));
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
//sparse on weight[num_classes ,CHW]
  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0));//pointer(index) of indices


}
/*
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
*/

template <typename Dtype>
void FmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top)
{
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *mutable_bottom_data = bottom[0]->mutable_cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const int num_output = this->layer_param_.fm_param().num_output();
  const int batch = bottom[0]->shape(0);
  int bottom_spatial_shape = bottom[0]->count(2);

  k_value = this->layer_param_.fm_param().k_value();
  const int axis = bottom[0]->CanonicalAxisIndex(
    this->layer_param_.fm_param().axis());
  K_ = bottom[0]->count(axis);
  //v_vector data & vx_matrix

  const Dtype *vx_matrix;     //[num_output, k_value,1,K_]
  const Dtype *vx_sum_matrix; //[num_output,k_value,1,1]
  const Dtype *vx_sum_square;
  const Dtype *v2x2_matrix;
  const Dtype *v2x2_sum_matrix;
  const Dtype *temp_;
  const Dtype *v_forward_output;

  Dtype *mutable_vx_matrix;     //[num_output, k_value,1,K_]
  Dtype *mutable_vx_sum_matrix; //[num_output,k_value,1,1]
  Dtype *mutable_vx_sum_square;
  Dtype *mutable_v2x2_matrix;
  Dtype *mutable_v2x2_sum_matrix;
  Dtype *mutable_temp_;
  Dtype *mutable_v_forward_output;

  const Dtype *v_vector;
  int v_spatial_size;
  int vx_spatial_size;
  int vx_sum_spatial_size;
  int a_count;

  int v_tensor_size;
  int vx_tensor_size;
  int vx_sum_tensor_size;

  if (bias_term_)
  {
    v_vector = this->blobs_[2]->cpu_data();
    //this->blobs_[2]->Reshape(num_output,k_value,1,K_);

    vx_matrix = this->blobs_[3]->cpu_data();
    mutable_vx_matrix = this->blobs_[3]->mutable_cpu_data();    
    this->blobs_[3]->Reshape(bottom[0]->shape(0),num_output,k_value,K_);
    //LOG(INFO)<<"2:"<<bottom[0]->shape(0);
    vx_sum_matrix = this->blobs_[4]->mutable_cpu_data();
    mutable_vx_sum_matrix = this->blobs_[4]->mutable_cpu_data();
    this->blobs_[4]->Reshape(bottom[0]->shape(0),num_output,k_value,1);
    LOG(INFO)<<"1:"<<bottom[0]->shape(0);

    vx_sum_square = this->blobs_[5]->cpu_data();
    mutable_vx_sum_square = this->blobs_[5]->mutable_cpu_data();
    this->blobs_[5]->Reshape(bottom[0]->shape(0),num_output,k_value,1);

    v2x2_matrix = this->blobs_[6]->cpu_data();
    mutable_v2x2_matrix = this->blobs_[6]->mutable_cpu_data();
    this->blobs_[6]->Reshape(bottom[0]->shape(0),num_output,k_value,K_);
LOG(INFO)<<"1:"<<bottom[0]->shape(0);
    v2x2_sum_matrix = this->blobs_[7]->cpu_data();
    mutable_v2x2_sum_matrix = this->blobs_[7]->mutable_cpu_data();
    this->blobs_[7]->Reshape(bottom[0]->shape(0),num_output,k_value,1);
LOG(INFO)<<"1:"<<bottom[0]->shape(0);
    temp_ = this->blobs_[8]->cpu_data();
    mutable_temp_ = this->blobs_[8]->mutable_cpu_data();
    this->blobs_[8]->Reshape(bottom[0]->shape(0),num_output,k_value,1);
LOG(INFO)<<"1:"<<bottom[0]->shape(0);
    v_forward_output = this->blobs_[9]->cpu_data();
    mutable_v_forward_output = this->blobs_[9]->mutable_cpu_data();
    vector<int> v_forward_output_shape = bottom[0]->shape();
    v_forward_output_shape.resize(2);
    v_forward_output_shape[1] = num_output;
    this->blobs_[9]->Reshape(v_forward_output_shape);

LOG(INFO)<<"1:"<<bottom[0]->shape(0); 

    v_spatial_size = this->blobs_[2]->count(2);
    v_tensor_size = this->blobs_[2]->count(1);

    vx_spatial_size = this->blobs_[3]->count(2);
    vx_tensor_size = this->blobs_[3]->count(1);

    vx_sum_spatial_size = this->blobs_[4]->count(2);
    vx_sum_tensor_size = this->blobs_[4]->count(1);

    a_count = this->blobs_[5]->count();
  }
  else
  {
    v_vector = this->blobs_[1]->cpu_data();

    vx_matrix = this->blobs_[2]->cpu_data();
    mutable_vx_matrix = this->blobs_[2]->mutable_cpu_data();

    vx_sum_matrix = this->blobs_[3]->cpu_data();
    mutable_vx_sum_matrix = this->blobs_[3]->mutable_cpu_data();

    vx_sum_square = this->blobs_[4]->cpu_data();
    mutable_vx_sum_square = this->blobs_[4]->mutable_cpu_data();

    v2x2_matrix = this->blobs_[5]->cpu_data();
    mutable_v2x2_matrix = this->blobs_[5]->mutable_cpu_data();

    v2x2_sum_matrix = this->blobs_[6]->cpu_data();
    mutable_v2x2_sum_matrix = this->blobs_[6]->mutable_cpu_data();

    temp_ = this->blobs_[7]->cpu_data();
    mutable_temp_ = this->blobs_[7]->mutable_cpu_data();

    v_forward_output = this->blobs_[8]->cpu_data();
    mutable_v_forward_output = this->blobs_[8]->mutable_cpu_data();

    v_spatial_size = this->blobs_[1]->count(2);
    v_tensor_size = this->blobs_[1]->count(1);

    vx_spatial_size = this->blobs_[2]->count(2);
    vx_tensor_size = this->blobs_[2]->count(1);

    vx_sum_spatial_size = this->blobs_[3]->count(2);
    vx_sum_tensor_size = this->blobs_[3]->count(1);

    a_count = this->blobs_[4]->count(1);
  }
  do_sparse = false;
  if (do_sparse)
  {
    //LOG(INFO)<<"do sparse";
    int M = this->blobs_[0]->shape(0);//classes
    int N = 1;
    int K = bottom[0]->shape(1);// kernel_dim_ = C*H*W
    /*
    LOG(INFO)<<"this->blobs_[0]->shape(0):"<<this->blobs_[0]->shape(0);
    LOG(INFO)<<"this->blobs_[0]->count(1):"<<this->blobs_[0]->count(1);

    LOG(INFO)<<"M:"<<M;
    LOG(INFO)<<"N:"<<N;
    LOG(INFO)<<"K:"<<K;
    */
    if(csr_init == false)
    {
      caffe_cpu_sparse_dense2csr(this->blobs_[0]->shape(0), this->blobs_[0]->count(1),
                  this->blobs_[0]->mutable_cpu_data(),
                  nz_weight_values_.mutable_cpu_data(),
                  nz_weight_indices_.mutable_cpu_data(),
                  nz_weight_index_pointers_.mutable_cpu_data());
      csr_init = true;//initial done
    }
    //LOG(INFO)<<"after dense2csr";
    for(int i = 0; i < bottom[0]->shape(0); i++)
    {
      caffe_cpu_sparse_mmcsr(M,
            N,
            K,
            (Dtype)1.,
            nz_weight_values_.cpu_data(),
            nz_weight_indices_.cpu_data(),
            nz_weight_index_pointers_.cpu_data(),
            nz_weight_index_pointers_.cpu_data() + 1,
            bottom[0]->cpu_data() + i * K,
            (Dtype)0.,top[0]->mutable_cpu_data() + i * M);
    }

    //LOG(INFO)<<"after mmcsr";
  }

  col_offset_ = this->blobs_[0]->count(1) * conv_out_spatial_dim_;
/*
  for(int i = 0; i < bottom[0]->shape(0); i++) 
  {
    const Dtype* col_buff = bottom[0]->cpu_data() + i * bottom[0]->count(1);
    int offset = col_offset_ * i;
    Dtype *col_buff_mutable = col_buffer_.mutable_cpu_data() + offset;

    conv_im2col_cpu(input, col_buff_mutable, col_buf_mask_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data() + offset;

    int left_rows = left_rows_[0];
    int left_cols = left_columns_[0];

    caffe_cpu_cblas_gemm(left_rows, conv_out_spatial_dim_, left_cols,
				  //(Dtype)1., squeezed_weight_groups_[g]->cpu_data(),
				  (Dtype)1., squeezed_weight_buffer_.cpu_data(),
				  left_cols , col_buff,
				conv_out_spatial_dim_, (Dtype)0., top[0]->mutable_cpu_data() + i * top[0]->count(1), conv_out_spatial_dim_);
  
  
  }
  */
  //NOTE:vx_matrix:[batch,num_output,k_value,K_] = batch * num_output * k_value * (V[K_ , 1] * X[1,K_])
  for (int i = 0; i < bottom[0]->shape(0); i++)
  {
    for (int k = 0; k < num_output; k++)
    {
      for (int j = 0; j < k_value; j++)
      {
        caffe_mul<Dtype>(K_, v_vector + j * K_ + k * K_ * k_value, bottom_data + i * K_,
                         mutable_vx_matrix + j * K_ + k * K_ * k_value + i * num_output * k_value * K_);
      }
    }
  }

  //NOTE:batch * ([num_output,k_value,1,CHW] * [CHW * 1]) => vx_sum shape:[batch,num_output,k_value,1]
  for (int i = 0; i < bottom[0]->shape(0); i++)
  {
    for (int k = 0; k < num_output; k++)
    {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k_value, 1, K_, (Dtype)1.,
                            v_vector + k * k_value * K_, bottom_data + i * K_,
                            (Dtype)0., mutable_vx_sum_matrix + k * k_value + i * num_output * k_value);
    }
  }

  //NOTE: square the vx_sum_matrix for each value
  //vx_sum_square shape [batch,num_output,k_value,1]
  caffe_mul<Dtype>(bottom[0]->shape(0) * num_output * k_value, vx_sum_matrix, vx_sum_matrix, mutable_vx_sum_square);
  //NOTE: square vx to v^2x^2, [batch,num_output,k_value,K_], and sum up to [batch,num_output,k_value,1]
  caffe_mul<Dtype>(bottom[0]->shape(0) * num_output * k_value * K_, vx_matrix, vx_matrix, mutable_v2x2_matrix);

  //initial the mutable_v2x2_sum_matrix to zero
  caffe_set<Dtype>(bottom[0]->shape(0) * num_output * k_value,0,mutable_v2x2_sum_matrix);

  for (int i = 0; i < k_value * num_output * bottom[0]->shape(0) * K_; i++)
  {
    //LOG(INFO)<<"kkkk"<<K_;
    //caffe_cpu_asum<Dtype>(K_,v2x2_matrix + i * K_,v2x2_sum_matrix + i);
    //v2x2_sum_matrix[i] = caffe_cpu_asum<Dtype>(K_,v2x2_matrix + i * K_);
    caffe_axpy<Dtype>(1, 1, v2x2_matrix + i, mutable_v2x2_sum_matrix + i / K_);
  }
  //LOG(INFO)<<"v2x2_sum done";
  //NOTE: vx_sum_matrix - v2x2_sum_matrix => temp_ [batch,num_output,k_value,1]
  caffe_sub<Dtype>(bottom[0]->shape(0) * num_output * k_value, vx_sum_square, v2x2_sum_matrix, mutable_temp_);
  //NOTE: final output,sum up on k_value dimension output shape [batch,num_output,k_value,1]->[batch,num_output]

  //initial the v_forward_output to zero
  caffe_set<Dtype>(bottom[0]->shape(0) * num_output,0,mutable_v_forward_output);

  for (int i = 0; i < bottom[0]->shape(0) * num_output * k_value; i++)
  {
    caffe_axpy<Dtype>(1, (Dtype)1., temp_ + i, mutable_v_forward_output + i / k_value);
  }

  if (M_ == 1)
  {
    caffe_cpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                          weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                        this->blobs_[1]->cpu_data(), top_data);
  }
  else
  { 
    if(!do_sparse)
      caffe_cpu_gemm<Dtype>(CblasNoTrans,
                            transpose_ ? CblasNoTrans : CblasTrans,
                            M_, N_, K_, (Dtype)1.,
                            bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.cpu_data(),
                            this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  //initial the v_forward_output
  //caffe_set<Dtype>();

  //NOTE:top + v_forward_output
  caffe_axpy<Dtype>(bottom[0]->shape(0) * num_output, (Dtype)1., v_forward_output, top_data);
}

/*
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
*/

template <typename Dtype>
void FmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom)
{
  //calculation on fm part
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *mutable_bottom_data = bottom[0]->mutable_cpu_data();
  Dtype *vx_matrix; //[num_output, k_value,1,K_]
  const int num_output = this->layer_param_.fm_param().num_output();
  const int batch = bottom[0]->shape(0);

  Dtype *v_bp;
  //Dtype* v_forward_output;
  Dtype *vxx_data;
  Dtype *vx_sum_matrix;
  Dtype *x_sum_vx;
  Dtype *xx;
  Dtype *v_diff;
  Dtype *v_sum;
  Dtype *x_diff;
  Dtype *v_vector_diff;
  int v_spatial_size;
  int v_tensor_size;
  int vx_sum_spatial_size;
  int vx_sum_tensor_size;
  const Dtype *v_vector;

  if (bias_term_)
  {
    v_vector = this->blobs_[2]->cpu_data();
    v_vector_diff = this->blobs_[2]->mutable_cpu_diff();
    vx_matrix = this->blobs_[3]->mutable_cpu_data();
    vx_sum_matrix = this->blobs_[4]->mutable_cpu_data();

    vxx_data = this->blobs_[10]->mutable_cpu_data();
    x_sum_vx = this->blobs_[11]->mutable_cpu_data();
    xx = this->blobs_[12]->mutable_cpu_data();
    v_diff = this->blobs_[13]->mutable_cpu_data();
    //v_forward_output = this->blobs_[9]->mutable_cpu_data();
    v_sum = this->blobs_[14]->mutable_cpu_data();
    x_diff = this->blobs_[15]->mutable_cpu_data();

    v_spatial_size = this->blobs_[2]->count(2);
    v_tensor_size = this->blobs_[2]->count(1);
    vx_sum_spatial_size = this->blobs_[4]->count(2);
    vx_sum_tensor_size = this->blobs_[4]->count(1);
  }
  else
  {
    v_vector = this->blobs_[1]->cpu_data();
    v_vector_diff = this->blobs_[1]->mutable_cpu_diff();
    vx_matrix = this->blobs_[2]->mutable_cpu_data();
    vx_sum_matrix = this->blobs_[3]->mutable_cpu_data();

    //v_forward_output = this->blobs_[8]->mutable_cpu_data();
    vxx_data = this->blobs_[9]->mutable_cpu_data();
    x_sum_vx = this->blobs_[10]->mutable_cpu_data();
    xx = this->blobs_[11]->mutable_cpu_data();
    v_diff = this->blobs_[12]->mutable_cpu_data();
    v_sum = this->blobs_[13]->mutable_cpu_data();
    x_diff = this->blobs_[14]->mutable_cpu_data();

    v_spatial_size = this->blobs_[1]->count(2);
    v_tensor_size = this->blobs_[1]->count(1);
    vx_sum_spatial_size = this->blobs_[3]->count(2);
    vx_sum_tensor_size = this->blobs_[3]->count(1);
  }

  //caffe_cpu_mul<Dtype>(top[0]->shape()*K_,bottom_data,bottom_data,v_forward_output);

  //LOG(INFO)<<"this->param_propagate_down_[0]:"<<this->param_propagate_down_[0];
  //LOG(INFO)<<"v_vector:"<<*v_vector;
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

      //fm part
      //NOTE:bottom_data * vx_sum_matrix
      //[batch,num_output,k_value,1] * [batch * CHW] => [batch,num_output,k_value,CHW]
      for (int i = 0; i < bottom[0]->shape(0); i++)
      {
        for (int j = 0; j < num_output; j++)
        {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                k_value, K_, 1,
                                (Dtype)1., vx_sum_matrix + j * k_value + i * k_value * num_output, bottom_data + i * K_,
                                (Dtype)0., x_sum_vx + j * k_value * K_ + i * k_value * num_output * K_);
        }
      }

      //NOTE: v_vector * x^2
      //x^2
      caffe_mul<Dtype>(bottom[0]->shape(0) * K_, bottom_data, bottom_data, xx);
      //x^2 * v: [batch,K_] * [num_output,k_value,1,K_] => [batch,num_output,k_value,CHW]
      for (int i = 0; i < bottom[0]->shape(0); i++)
      {
        for (int j = 0; j < num_output; j++)
        {
          for (int k = 0; k < k_value; k++)
          {
            caffe_mul<Dtype>(K_, xx + i * K_, v_vector + k * K_ + j * K_ * k_value,
                             vxx_data + k * K_ + j * K_ * k_value + i * K_ * k_value * num_output);
            //caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,);
          }
        }
      }
      //NOTE: x_sum_vx - vxx_data
      caffe_sub<Dtype>(bottom[0]->shape(0) * num_output * k_value * K_, x_sum_vx, vxx_data, v_diff);
      //NOTE: top_diff * v_diff: [num_output , batch] * [batch , num_output, k_value, K_]

      for (int i = 0; i < num_output; i++)
      {
        for (int j = 0; j < bottom[0]->shape(0); j++)
        {
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                1, k_value * K_, bottom[0]->shape(0),
                                (Dtype)1., top_diff + i * bottom[0]->shape(0) + j, v_diff + i * k_value * K_ + j * num_output * k_value * K_,
                                (Dtype)1., v_vector_diff + i * k_value * K_);
        }
      }
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

      //NOTE: get v_sum [num_output,k_value,1,K_] * [num_output,k_value,1,K_] => [num_output,K_,K_]
      for (int i = 0; i < num_output; i++)
      {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                              K_, K_, k_value,
                              (Dtype)1., v_vector + i * k_value * K_, v_vector + i * k_value * K_,
                              (Dtype)0., v_sum + i * K_ * K_);
      }
      //NOTE: v_sum * bottom :[num_output, K_,K_] * [batch,K_] => [batch,num_output,K_] (x_diff)
      for (int i = 0; i < bottom[0]->shape(0); i++)
      {
        for (int j = 0; j < num_output; j++)
        {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                K_, 1, K_,
                                (Dtype)1., v_sum + j * K_ * K_, bottom_data + i * K_,
                                (Dtype)0., x_diff + j * K_ + i * num_output * K_);
        }
      }

      //NOTE: update the bottom: [batch,num_output] * [batch,num_output,K_] => [batch,K_]
      for (int i = 0; i < bottom[0]->shape(0); i++)
      {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                              1., K_, num_output,
                              (Dtype)1., top_diff + i * num_output, x_diff + i * num_output * K_,
                              (Dtype)1., bottom[0]->mutable_cpu_diff() + i * K_);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FmLayer);
#endif

INSTANTIATE_CLASS(FmLayer);
REGISTER_LAYER_CLASS(Fm);

} // namespace caffe
