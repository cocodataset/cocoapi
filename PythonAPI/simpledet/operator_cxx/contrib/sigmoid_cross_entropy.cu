/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file sigmoid_cross_entropy.cu
 * \brief
 * \author Yuntao Chen
*/

#include "./sigmoid_cross_entropy-inl.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                               \
for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
     i += blockDim.x * gridDim.x)

constexpr int CAFFE_CUDA_NUM_THREADS = 512;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min((N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                  CAFFE_MAXIMUM_NUM_BLOCKS);
}

namespace mshadow {
namespace cuda {

template<typename T>
__global__ void SigmoidCrossEntropyLossKernel(
    const int n,
    const T* logits,
    const T* targets,
    T* losses,
    T* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      losses[index] = 0.;
      counts[index] = 0.;
    } else {
      losses[index] =
          -1. * logits[index] * (targets[index] - (logits[index] >= 0)) +
          logf(
              1 +
              expf(logits[index] - 2 * logits[index] * (logits[index] >= 0)));
      counts[index] = 1.;
    }
  }
}

template<typename T>
__global__ void SigmoidCrossEntropyLossGradientKernel(
    const int n,
    const T* logits,
    const T* targets,
    T* d_logits,
    T* counts) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    if (targets[index] == -1) {
      d_logits[index] = 0.;
      counts[index] = 0.;
    } else {
      d_logits[index] = 1. / (1. + expf(-logits[index])) - targets[index];
      counts[index] = 1.;
    }
  }
}

template<typename T>
inline void SigmoidCrossEntropyForward(const Tensor<gpu, 2, T> &data,
                                       const Tensor<gpu, 2, T> &label,
                                       Tensor<gpu, 2, T> &loss,
                                       Tensor<gpu, 1, T> &loss_sum,
                                       Tensor<gpu, 2, T> &count,
                                       Tensor<gpu, 1, T> &count_sum,
                                       Tensor<gpu, 1, T> &out,
                                       T scale) {
  using namespace mshadow::expr;
  SigmoidCrossEntropyLossKernel<<<CAFFE_GET_BLOCKS(data.shape_.Size()), CAFFE_CUDA_NUM_THREADS, 0>>>(
    data.shape_.Size(), data.dptr_, label.dptr_, loss.dptr_, count.dptr_);
  loss_sum = sumall_except_dim<0>(loss);
  count_sum = sumall_except_dim<0>(count);
  count_sum += static_cast<T>(1e-5);
  out = loss_sum / count_sum;
  int count_num = (count.size(0) * count.size(1));
  //out /= static_cast<T>(count_num);
  // mx.metric.Loss will take care of this
  // out *= scale; 
}

template<typename T>
inline void SigmoidCrossEntropyBackward(const Tensor<gpu, 2, T> &data,
                                        const Tensor<gpu, 2, T> &label,
                                        Tensor<gpu, 2, T> &d_data,
                                        Tensor<gpu, 2, T> &count,
                                        Tensor<gpu, 1, T> &count_sum,
                                        T scale) {
  using namespace mshadow::expr;
  SigmoidCrossEntropyLossGradientKernel<<<CAFFE_GET_BLOCKS(data.shape_.Size()), CAFFE_CUDA_NUM_THREADS, 0>>>(
    data.shape_.Size(), data.dptr_, label.dptr_, d_data.dptr_, count.dptr_);
  count_sum = sumall_except_dim<0>(count);
  count_sum += static_cast<T>(1e-5);
  d_data /= broadcast<0>(count_sum, d_data.shape_);
  int count_num = (count.size(0) * count.size(1));
  //d_data /= static_cast<T>(count_num);
  d_data *= scale;
}

} // namespace cuda

template<typename T>
inline void SigmoidCrossEntropyForward(const Tensor<gpu, 2, T> &data,
                                       const Tensor<gpu, 2, T> &label,
                                       Tensor<gpu, 2, T> &loss,
                                       Tensor<gpu, 1, T> &loss_sum,
                                       Tensor<gpu, 2, T> &count,
                                       Tensor<gpu, 1, T> &count_sum,
                                       Tensor<gpu, 1, T> &out,
                                       T scale) {
  cuda::SigmoidCrossEntropyForward(data, label, loss, loss_sum, count, count_sum, out, scale);
}

template<typename T>
inline void SigmoidCrossEntropyBackward(const Tensor<gpu, 2, T> &data,
                                        const Tensor<gpu, 2, T> &label,
                                        Tensor<gpu, 2, T> &d_data,
                                        Tensor<gpu, 2, T> &count,
                                        Tensor<gpu, 1, T> &count_sum,
                                        T scale) {
  cuda::SigmoidCrossEntropyBackward(data, label, d_data, count, count_sum, scale);
}

} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(SigmoidCrossEntropyParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SigmoidCrossEntropyOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

