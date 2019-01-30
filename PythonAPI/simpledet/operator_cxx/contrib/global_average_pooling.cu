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
 * \file global_average_pooling.cu
 * \brief port from https://github.com/hujie-frank/SENet
 * \author Chenxia Han
*/
#include <vector>
#include <algorithm>
#include "../mxnet_op.h"
#include "../../common/cuda_utils.h"
#include "./global_average_pooling-inl.h"

#define GAP_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
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
__global__ void GlobalAvePoolForwardKernel(const int spatial_dim, 
    const Dtype* bottom_data, Dtype* top_data) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  unsigned int tid = threadIdx.x;
  buffer[tid] = 0;
  __syncthreads();

  for (int j = tid; j < spatial_dim; j += blockDim.x) {
    buffer[tid] += bottom_data[blockIdx.x * spatial_dim + j];
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    top_data[blockIdx.x] = buffer[0] / spatial_dim;
  }
}

template<typename DType>
inline void GAPForward(const Tensor<gpu, 4, DType> &out,
					   const Tensor<gpu, 4, DType> &data) {
  const DType *bottom_data = data.dptr_;
  DType *top_data = out.dptr_;
  const int nblocks = data.shape_.ProdShape(0, 2);
  const int spatial_dim = data.shape_.ProdShape(2, 4);
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  GlobalAvePoolForwardKernel<DType> << <nblocks, CAFFE_CUDA_NUM_THREADS,
	0, stream >> >(spatial_dim, bottom_data, top_data);
  GAP_CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Dtype> 
__global__ void GlobalAvePoolBackwardKernel(const int nthreads, const int spatial_dim, 
    const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    bottom_diff[index] = top_diff[n] / spatial_dim;
  }
}

template<typename DType>
inline void GAPBackward(const Tensor<gpu, 4, DType> &in_grad,
					    const Tensor<gpu, 4, DType> &out_grad) {
  const DType *top_diff = out_grad.dptr_;
  DType *bottom_diff = in_grad.dptr_;
  const int count = in_grad.shape_.Size();
  const int spatial_dim = in_grad.shape_.ProdShape(2, 4);
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  GlobalAvePoolBackwardKernel<DType> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,
    0, stream >> >(count, spatial_dim, top_diff, bottom_diff);
  GAP_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace cuda

template<typename DType>
inline void GAPForward(const Tensor<gpu, 4, DType> &out,
                       const Tensor<gpu, 4, DType> &data) {
  cuda::GAPForward(out, data);
}

template<typename DType>
inline void GAPBackward(const Tensor<gpu, 4, DType> &in_grad,
                        const Tensor<gpu, 4, DType> &out_grad) {
  cuda::GAPBackward(in_grad, out_grad);
}

}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GAPParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GAPOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

