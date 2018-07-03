#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* mutable_bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int num_output = this->layer_param_.fm_param().num_output();
  int bottom_spatial_shape = bottom[0]->count(2);
  //v_vector data & vx_matrix

  Dtype* vx_matrix;//[num_output, k_value,1,K_]
  Dtype* vx_sum_matrix;//[num_output,k_value,1,1]
  Dtype* vx_sum_square;
  Dtype* v2x2_matrix;
  Dtype* v2x2_sum_matrix;
  Dtype* temp_;
  Dtype* v_forward_output;



  const Dtype* v_vector;
  int v_spatial_size;
  int vx_spatial_size;
  int vx_sum_spatial_size;
  int a_count;

  int v_tensor_size;
  int vx_tensor_size;
  int vx_sum_tensor_size;

  if(bias_term_)
  {
    v_vector = this->blobs_[2]->gpu_data();
    vx_matrix = this->blobs_[3]->mutable_gpu_data();
    vx_sum_matrix = this->blobs_[4]->mutable_gpu_data();
    vx_sum_square = this->blobs_[5]->mutable_gpu_data();
    v2x2_matrix = this->blobs_[6]->mutable_gpu_data();
    v2x2_sum_matrix = this->blobs_[7]->mutable_gpu_data();
    temp_ = this->blobs_[8]->mutable_gpu_data();
    v_forward_output = this->blobs_[9]->mutable_gpu_data();

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
    v_vector = this->blobs_[1]->gpu_data();
    vx_matrix = this->blobs_[2]->mutable_gpu_data();
    vx_sum_matrix = this->blobs_[3]->mutable_gpu_data();
    vx_sum_square = this->blobs_[4]->mutable_gpu_data();
    v2x2_matrix = this->blobs_[5]->mutable_gpu_data();
    v2x2_sum_matrix = this->blobs_[6]->mutable_gpu_data();
    temp_ = this->blobs_[7]->mutable_gpu_data();
    v_forward_output = this->blobs_[8]->mutable_gpu_data();


    v_spatial_size = this->blobs_[1]->count(2);
    v_tensor_size = this->blobs_[1]->count(1);

    vx_spatial_size = this->blobs_[2]->count(2);
    vx_tensor_size = this->blobs_[2]->count(1);

    vx_sum_spatial_size = this->blobs_[3]->count(2);
    vx_sum_tensor_size = this->blobs_[3]->count(1);

    a_count = this->blobs_[4]->count(1);
  }


//NOTE:vx_matrix:[batch,num_output,k_value,K_] = batch * num_output * k_value * (V[K_ , 1] * X[1,K_])
for(int i = 0; i < bottom[0]->shape(0); i++)
{
  for(int k = 0; k < num_output; k++)
  {
    for(int j = 0; j < k_value; j++)
    {
      caffe_gpu_mul<Dtype>(K_,v_vector + j * K_ + k * K_ * k_value,bottom_data + i * K_,
            vx_matrix + j * K_ + k * K_ * k_value + i * num_output * k_value * K_);
    }
  }
}



//NOTE:batch * ([num_output,k_value,1,CHW] * [CHW * 1]) => vx_sum shape:[batch,num_output,k_value,1]
for(int i = 0; i < bottom[0]->shape(0); i++)
{
  for(int k = 0; k < num_output; k++)
  {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasTrans,1,1,K_,(Dtype)1.,
                            v_vector + k * k_value * K_, bottom_data + i * K_,
                           (Dtype)0.,vx_sum_matrix + k * k_value + i * num_output * k_value);
    
  }
}

//NOTE: square the vx_sum_matrix for each value
//vx_sum_square shape [batch,num_output,k_value,1]
  caffe_gpu_mul<Dtype>(bottom[0]->shape(0)*num_output*k_value,vx_sum_matrix,vx_sum_matrix,vx_sum_square);
//NOTE: square vx to v^2x^2, [batch,num_output,k_value,K_], and sum up to [batch,num_output,k_value,1]
  caffe_gpu_mul<Dtype>(bottom[0]->shape(0)*num_output*k_value*K_,vx_matrix,vx_matrix,v2x2_matrix);
  for(int i = 0; i < k_value * num_output * bottom[0]->shape(0) * K_; i++)
  {
    //LOG(INFO)<<"kkkk"<<K_;
    //caffe_gpu_asum<Dtype>(K_,v2x2_matrix + i * K_,v2x2_sum_matrix + i);
    //v2x2_sum_matrix[i] = caffe_cpu_asum<Dtype>(K_,v2x2_matrix + i * K_);
    caffe_gpu_axpy<Dtype>(1,1,v2x2_matrix + i,v2x2_sum_matrix + i / K_);
  }
//LOG(INFO)<<"v2x2_sum done";
  //NOTE: vx_sum_matrix - v2x2_sum_matrix => temp_ [batch,num_output,k_value,1]
  caffe_gpu_sub<Dtype>(bottom[0]->shape(0)*num_output*k_value,vx_sum_matrix,v2x2_sum_matrix,temp_);
  //NOTE: final output,sum up on k_value dimension output shape [batch,num_output,k_value,1]->[batch,num_output]

  for(int i = 0; i < bottom[0]->shape(0) * num_output * k_value;i++)
  {
    caffe_gpu_axpy<Dtype>(1,(Dtype)1.,temp_ + i,v_forward_output + i / k_value );
  }

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
//NOTE:top + v_forward_output
caffe_gpu_axpy<Dtype>(bottom[0]->shape(0)*num_output,(Dtype)1.,v_forward_output,top_data);

}

template <typename Dtype>
void FmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
//calculation on fm part
const Dtype* bottom_data = bottom[0]->gpu_data();
Dtype* mutable_bottom_data = bottom[0]->mutable_cpu_data();
Dtype* vx_matrix;//[num_output, k_value,1,K_]
const int num_output = this->layer_param_.fm_param().num_output();
const int batch = bottom[0]->shape(0);

Dtype* v_bp;
//Dtype* v_forward_output;
Dtype* vxx_data;
Dtype* vx_sum_matrix;
Dtype* x_sum_vx;
Dtype* xx;
Dtype* v_diff;
Dtype* v_sum;
Dtype* x_diff;
Dtype* v_vector_diff;
int v_spatial_size;
int v_tensor_size;
int vx_sum_spatial_size;
int vx_sum_tensor_size;
const Dtype* v_vector;

if(bias_term_)
{
  v_vector = this->blobs_[2]->gpu_data();
  v_vector_diff = this->blobs_[2]->mutable_gpu_diff();
  vx_matrix = this->blobs_[3]->mutable_gpu_data();
  vx_sum_matrix = this->blobs_[4]->mutable_gpu_data();


  vxx_data = this->blobs_[10]->mutable_gpu_data();
  x_sum_vx = this->blobs_[11]->mutable_gpu_data();
  xx = this->blobs_[12]->mutable_gpu_data();
  v_diff = this->blobs_[13]->mutable_gpu_data();
  //v_forward_output = this->blobs_[9]->mutable_gpu_data();
  v_sum = this->blobs_[14]->mutable_gpu_data();
  x_diff = this->blobs_[15]->mutable_gpu_data();


  v_spatial_size = this->blobs_[2]->count(2);
  v_tensor_size = this->blobs_[2]->count(1);
  vx_sum_spatial_size = this->blobs_[4]->count(2);
  vx_sum_tensor_size = this->blobs_[4]->count(1);
}
else
{
  v_vector = this->blobs_[1]->gpu_data();
  v_vector_diff = this->blobs_[1]->mutable_gpu_diff();
  vx_matrix = this->blobs_[2]->mutable_gpu_data();
  vx_sum_matrix = this->blobs_[3]->mutable_gpu_data();

  //v_forward_output = this->blobs_[8]->mutable_gpu_data();
  vxx_data = this->blobs_[9]->mutable_gpu_data();
  x_sum_vx = this->blobs_[10]->mutable_gpu_data();
  xx = this->blobs_[11]->mutable_gpu_data();
  v_diff = this->blobs_[12]->mutable_gpu_data();
  v_sum = this->blobs_[13]->mutable_gpu_data();
  x_diff = this->blobs_[14]->mutable_gpu_data();


  v_spatial_size = this->blobs_[1]->count(2);
  v_tensor_size = this->blobs_[1]->count(1);
  vx_sum_spatial_size = this->blobs_[3]->count(2);
  vx_sum_tensor_size = this->blobs_[3]->count(1);
}

//caffe_gpu_mul<Dtype>(top[0]->shape()*K_,bottom_data,bottom_data,v_forward_output);


  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

          //fm part
          //NOTE:bottom_data * vx_sum_matrix
          //[batch,num_output,k_value,1] * [batch * CHW] => [batch,num_output,k_value,CHW]
          for(int i = 0; i < bottom[0]->shape(0); i++)
          {
            for(int j = 0; j < num_output; j++)
            {
              caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
                                  k_value,K_,1,
                                  (Dtype)1.,vx_sum_matrix + j * k_value + i * k_value * num_output,bottom_data + i * K_,
                                  (Dtype)0.,x_sum_vx + j * k_value * K_ + i * k_value * num_output * K_);

            }
          }

          //NOTE: v_vector * x^2
          //x^2
          caffe_gpu_mul<Dtype>(bottom[0]->shape(0)*K_,bottom_data,bottom_data,xx);
          //x^2 * v: [batch,K_] * [num_output,k_value,1,K_] => [batch,num_output,k_value,CHW]
          for(int i = 0; i < bottom[0]->shape(0); i++)
          {
              for(int j = 0; j < num_output; j++)
              {
                for(int k = 0; k < k_value; k++)
                {
                  caffe_gpu_mul<Dtype>(K_,xx + i * K_ ,v_vector + k * K_ + j * K_ * k_value,
                                      vxx_data + k * K_ +  j * K_ * k_value + i * K_ * k_value * num_output);
                  //caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,);
                }

              }
          }
          //NOTE: x_sum_vx - vxx_data
          caffe_gpu_sub<Dtype>(bottom[0]->shape(0) * num_output * k_value * K_,x_sum_vx,vxx_data,v_diff);
          //NOTE: top_diff * v_diff: [num_output , batch] * [batch , num_output, k_value, K_]

          for(int i = 0; i < num_output; i++)
          {
            for(int j = 0; j < bottom[0]->shape(0); j++)
            {
              caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
                  1,k_value * K_,bottom[0]->shape(0),
                  (Dtype)1.,top_diff + i * bottom[0]->shape(0) + j,v_diff + i * k_value * K_ + j * num_output * k_value * K_,
                  (Dtype)1.,v_vector_diff + i * k_value * K_);
            }
          }
    }
  }


  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());

      //NOTE: get v_sum [num_output,k_value,1,K_] * [num_output,k_value,1,K_] => [num_output,K_,K_]
      for(int i = 0; i < num_output; i++)
      {
        caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,
          K_,K_,k_value,
          (Dtype)1., v_vector + i * k_value * K_ ,v_vector + i * k_value * K_ ,
          (Dtype)0., v_sum + i * K_ * K_);

      }
      //NOTE: v_sum * bottom :[num_output, K_,K_] * [batch,K_] => [batch,num_output,K_] (x_diff)
      for(int i = 0; i < bottom[0]->shape(0); i++)
      {
        for(int j  = 0;  j < num_output; j++)
        {
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
              K_,1,K_,
              (Dtype)1.,v_sum + j * K_ * K_,bottom_data + i * K_,
              (Dtype)0.,x_diff + j * K_ + i * num_output * K_);
        }
      }

      //NOTE: update the bottom: [batch,num_output] * [batch,num_output,K_] => [batch,K_]
      for(int i = 0; i < bottom[0]->shape(0);i++)
      {
        caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,
              1.,K_,num_output,
              (Dtype)1.,top_diff + i * num_output,x_diff + i * num_output * K_,
              (Dtype)1.,bottom[0]->mutable_gpu_diff() + i * K_);
      }

    }

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FmLayer);

}  // namespace caffe
