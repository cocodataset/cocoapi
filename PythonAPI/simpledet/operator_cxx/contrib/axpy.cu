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
 * \file axpy.cu
 * \brief port from https://github.com/hujie-frank/SENet
 * \author Yuntao Chen
*/
#include "./axpy-inl.h"
#include "../../common/cuda_utils.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
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

template <typename Dtype>
__global__ void AxpyForward(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* x_data, const Dtype* y_data,
    Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    out_data[index] = scale_data[index / spatial_dim] * x_data[index]
        + y_data[index];
  }
}

template <typename Dtype>
__global__ void AxpyBackwardScale(const int outer_num, const int spatial_dim, 
    const Dtype* x_data, const Dtype* top_diff, Dtype* scale_diff) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  unsigned int tid = threadIdx.x;
  buffer[tid] = 0;
  __syncthreads();

  for (int j = tid; j < spatial_dim; j += blockDim.x) {
    int offset = blockIdx.x * spatial_dim + j;
    buffer[tid] += top_diff[offset] * x_data[offset];
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_diff[blockIdx.x] = buffer[0];
  }
}

template <typename Dtype>
__global__ void AxpyBackwardX(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* top_diff, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = scale_data[index / spatial_dim] * top_diff[index];
  }
}

template <typename Dtype>
inline void AxpyForwardLauncher(const Tensor<gpu, 2, Dtype> &scale_data,
                                const Tensor<gpu, 4, Dtype> &x_data,
                                const Tensor<gpu, 4, Dtype> &y_data,
                                const Tensor<gpu, 1, Dtype> &out_data) {
  const Dtype* scale_data_ptr = scale_data.dptr_;
  const Dtype* x_data_ptr = x_data.dptr_;
  const Dtype* y_data_ptr = y_data.dptr_;
  Dtype* out_ptr = out_data.dptr_;
  const int count = x_data.shape_.Size();
  const int channels = x_data.shape_.Size();
  AxpyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, scale_data_ptr, x_data_ptr, y_data_ptr, out_ptr);  
}

template <typename Dtype>
inline void AxpyBackwardLauncher(const Tensor<gpu, 2, Dtype> &scale_data,
                                 const Tensor<gpu, 4, Dtype> &x_data,
                                 const Tensor<gpu, 4, Dtype> &y_data,
                                 const Tensor<gpu, 2, Dtype> &scale_grad,
                                 const Tensor<gpu, 4, Dtype> &x_grad,
                                 const Tensor<gpu, 4, Dtype> &y_grad,
                                 const Tensor<gpu, 4, Dtype> &out_grad,
                                 Stream<gpu> *s) {
    const int count = out_grad.shape_.Size();

    // backward to scale
    const int outer_num = x_data.shape_.ProdShape(0, 2);
    const int spatial_num = x_data.shape_.ProdShape(2, 4);
    AxpyBackwardScale<Dtype><<<outer_num, CAFFE_CUDA_NUM_THREADS>>>(
        outer_num, 
        spatial_num,
        x_data.dptr_, 
        out_grad.dptr_,
        scale_grad.dptr_); 

    // backward to x
    AxpyBackwardX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, 
        spatial_num, 
        scale_data.dptr_, 
        out_grad.dptr_, 
        x_grad.dptr_);

    // backward to y
    Copy(y_grad, out_grad, s);
}

}  // namespace cuda


template <typename Dtype>
inline void AxpyForwardLauncher(const Tensor<gpu, 2, Dtype> &scale_data,
                                const Tensor<gpu, 4, Dtype> &x_data,
                                const Tensor<gpu, 4, Dtype> &y_data,
                                const Tensor<gpu, 1, Dtype> &out_data) {
    cuda::AxpyForwardLauncher(scale_data, x_data, y_data, out_data);  
}

template <typename Dtype>
inline void AxpyBackwardLauncher(const Tensor<gpu, 2, Dtype> &scale_data,
                                 const Tensor<gpu, 4, Dtype> &x_data,
                                 const Tensor<gpu, 4, Dtype> &y_data,
                                 const Tensor<gpu, 2, Dtype> &scale_grad,
                                 const Tensor<gpu, 4, Dtype> &x_grad,
                                 const Tensor<gpu, 4, Dtype> &y_grad,
                                 const Tensor<gpu, 4, Dtype> &out_grad,
                                 Stream<gpu> *s) {
    cuda::AxpyBackwardLauncher(scale_data, x_data, y_data, scale_grad, x_grad, y_grad, out_grad, s);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(AxpyParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AxpyOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
