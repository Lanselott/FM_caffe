#include <vector>
#include <omp.h>
#include "caffe/layers/sparse_layer.hpp"
//#include "caffe/util/winograd.hpp"




namespace caffe {

extern double padding_time;

template <typename Dtype>
void SparseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const LayerParameter& layerparam = this->layer_param();
  step = this->layer_param_.sparse_param().step();
 // int mkl_max_threads_saved = mkl_get_max_threads();
  //LOG(INFO)<<"MAX Threads:"<<mkl_max_threads_saved;
//  mkl_set_num_threads(8);
  //LOG(INFO)<<"STEP CHECKING: "<<step;
/*
  int height = this->conv_input_shape_.gpu_data()[1];
  int width = this->conv_input_shape_.gpu_data()[2];
  int kernel_h = this->kernel_shape_.gpu_data()[0];
  int kernel_w = this->kernel_shape_.gpu_data()[1];
  int pad_h = this->pad_.gpu_data()[0];
  int pad_w = this->pad_.gpu_data()[1];
  int stride_h = this->stride_.gpu_data()[0];
  int stride_w = this->stride_.gpu_data()[1];
  int dilation_h = this->dilation_.gpu_data()[0];
  int dilation_w = this->dilation_.gpu_data()[1];
*/
  const Dtype* weight = this->blobs_[0]->gpu_data();

  //LOG(INFO)<<"BIAS shape in sparse"<<counter;

  // JSP: by some reason, if nested omp parallelism is used for MKL, I get a wrong results.
  // Disable nested omp parallelization for now. We don't need nested parallelism as long as
  // batch size is big enough. Still, need more investigation.


if(step == 3)
{
//  LOG(INFO)<<"start sparse_merge cal";
  //bottom[1] => u; bottom[2] => v weight(blob_[0]) => sparse
  const Dtype* v = bottom[2]->gpu_data();
  const Dtype* u = bottom[1]->gpu_data();
  const Dtype* xuv = this->xuv_buffer_.gpu_data();
  const Dtype* xs = this->xs_buffer_.gpu_data();

  int xu_dim_ = this->xu_buffer_.count(1);
  int xuv_dim_ = this->xuv_buffer_.count(1);
  int xs_dim_ = this->xs_buffer_.count(1);
  //int top_dim_ = top[0]->count(1);
 // for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* xu_data = this->xu_buffer_.mutable_gpu_data();

    Dtype* xuv_data = this->xuv_buffer_.mutable_gpu_data();
    Dtype* xs_data = this->xs_buffer_.mutable_gpu_data();

    const Dtype alpha = 1.0;
    const int top_count = top[0]->count();
//initial for xs
if(xuv_dim_ != xs_dim_) LOG(FATAL)<<"xuv and xs dimension are not equal";
//LOG(INFO)<<this->blobs_[1]->shape(0);
/*
LOG(INFO)<<"GPU MODE";
LOG(INFO)<<"CHECK TOP SIZE";
LOG(INFO)<<"TOP[0] SIZE: "<<top[0]->shape(0)<<","<<top[0]->shape(1)<<","<<top[0]->shape(2)<<","<<top[0]->shape(3);
LOG(INFO)<<"XU SIZE: "<<this->xu_buffer_.shape(0)<<","<<this->xu_buffer_.shape(1)<<","<<this->xu_buffer_.shape(2)<<","<<this->xu_buffer_.shape(3);
LOG(INFO)<<"XS SIZE: "<<this->xs_buffer_.shape(0)<<","<<this->xs_buffer_.shape(1)<<","<<this->xs_buffer_.shape(2)<<","<<this->xs_buffer_.shape(3);
LOG(INFO)<<"XUV SIZE: "<<this->xuv_buffer_.shape(0)<<","<<this->xuv_buffer_.shape(1)<<","<<this->xuv_buffer_.shape(2)<<","<<this->xuv_buffer_.shape(3);
LOG(INFO)<<"check point 0";
*/
bool check_zero = this->forward_gpu_gemm_ccnmm_merge(bottom_data, weight,
    xs_data , 0, bottom);
  //  LOG(INFO)<<layerparam.name()<<"check_zero value:"<<check_zero;
//LOG(INFO)<<"xu top: "<<
//#pragma omp parallel for
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      //calculate xu,xuv together
//  #pragma omp parallel sections num_threads(1)
//  {
    //#pragma omp section

      this->forward_gpu_gemm_xu_xuv(bottom_data + n * this->bottom_dim_,xu_data + n * xu_dim_,
      u,v,xu_data + n * xu_dim_, xuv_data + n * xuv_dim_,n,bottom);

//#pragma omp section
  /*###If sparsity==1, jump the sparse kernel###*/
    if(n!=0 && check_zero == true) {
      this->forward_gpu_gemm_ccnmm_merge(bottom_data + n * this->bottom_dim_, weight,
                  xs_data + n * xs_dim_, n, bottom);
    //  LOG(INFO)<<layerparam.name();
    }
//#pragma omp section
      // (xuv)+(xs)
      if (this->bias_term_) {
        // bias term is fused with direct convolution for second layer

        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);

      }

  //    }
    }

/*
#pragma omp parallel for
    for (int n = 0; n < this->num_; ++n) {
    //  this->forward_cpu_gemm_xuv(xu_data + n * xu_dim_, v, xuv_data + n * xuv_dim_, n,bottom);
      if (this->bias_term_) {
        // bias term is fused with direct convolution for second layer
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    */
    if(check_zero == true){
    //  LOG(INFO)<<layerparam.name()<<"sparsity is not 1";
     caffe_gpu_axpy(top_count,alpha, xuv, xs_data );
     caffe_copy(top_count,xs,top_data);
    }
    else{
      caffe_copy(top_count,xuv,top_data);
    }
}

}

template <typename Dtype>
void SparseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseLayer);

}  // namespace caffe
