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
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
*/
#include "./roi_pooling_v1-inl.h"
#include "../common/cuda_utils.h"
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

template <typename T>
__global__ void ROIPoolForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    T* top_data,
    T* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    if (argmax_data) {
      argmax_data[index] = maxidx;
    }
  }
}

template <typename T>
__global__ void ROIPoolBackward(
    const int nthreads,
    const T* top_diff,
    const T* argmax_data,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const T* offset_argmax_data = argmax_data + top_offset;

    int argmax = static_cast<int>(offset_argmax_data[ph * pooled_width + pw]);
    if (argmax != -1) {
      // gpu_atomic_add(
      //     static_cast<T>(offset_top_diff[ph * pooled_width + pw]),
      //     offset_bottom_diff + argmax);
      atomicAdd(offset_bottom_diff + argmax, static_cast<T>(offset_top_diff[ph * pooled_width + pw]));
    }
  }
}

template<typename Dtype>
inline void ROIPoolForward_v1(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);

  ROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
}

template<typename Dtype>
inline void ROIPoolBackwardAcc_v1(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);

  ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(out_grad.shape_.Size()), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
    out_grad.shape_.Size(), top_diff, argmax_data, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois);
}

}  // namespace cuda

template<typename Dtype>
inline void ROIPoolForward_v1(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 4, Dtype> &max_idx,
                           const float spatial_scale) {
  cuda::ROIPoolForward_v1(out, data, bbox, max_idx, spatial_scale);
}

template<typename Dtype>
inline void ROIPoolBackwardAcc_v1(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 4, Dtype> &max_idx,
                               const float spatial_scale) {
  cuda::ROIPoolBackwardAcc_v1(in_grad, out_grad, bbox, max_idx, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ROIPoolingParam_v1 param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ROIPoolingOp_v1<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
