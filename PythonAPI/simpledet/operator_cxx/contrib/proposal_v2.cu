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
 * \file proposal_v2.cu
 * \brief Proposal Operator for SNIP
 * \author Shaoqing Ren, Jian Guo, Pengfei Chen, Yuntao Chen, Yanghao Li
*/
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "../tensor/sort_op.h"

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iterator>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./proposal_v2-inl.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
} while (0)

namespace mshadow {
namespace cuda {
namespace {
// scores are (b, anchor, h, w)
// proposals are (h * w * anchor, 5)
// w defines "x" and h defines "y"
// count should be total anchors numbers, h * w * anchors
template<typename Dtype>
__global__ void ProposalGridKernel(const int count,
                                   const int num_anchors,
                                   const int height,
                                   const int width,
                                   const int feature_stride,
                                   const Dtype* scores,
                                   Dtype* proposals) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % width;
    int h = index / num_anchors / width;

    proposals[index * 5 + 0] = proposals[a * 5 + 0] + w * feature_stride;
    proposals[index * 5 + 1] = proposals[a * 5 + 1] + h * feature_stride;
    proposals[index * 5 + 2] = proposals[a * 5 + 2] + w * feature_stride;
    proposals[index * 5 + 3] = proposals[a * 5 + 3] + h * feature_stride;
    proposals[index * 5 + 4] = scores[(a * height + h) * width + w];
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void BBoxPredKernel(const int count,
                               const int num_anchors,
                               const int feat_height,
                               const int feat_width,
                               const int real_height,
                               const int real_width,
                               const float im_height,
                               const float im_width,
                               const Dtype* boxes,
                               const Dtype* deltas,
                               Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float width = boxes[index * 5 + 2] - boxes[index * 5 + 0] + 1.0f;
    float height = boxes[index * 5 + 3] - boxes[index * 5 + 1] + 1.0f;
    float ctr_x = boxes[index * 5 + 0] + 0.5f * (width - 1.0f);
    float ctr_y = boxes[index * 5 + 1] + 0.5f * (height - 1.0f);

    float dx = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dw = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dh = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
    float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
    float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
    float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);

    pred_x1 = max(min(pred_x1, im_width - 1.0f), 0.0f);
    pred_y1 = max(min(pred_y1, im_height - 1.0f), 0.0f);
    pred_x2 = max(min(pred_x2, im_width - 1.0f), 0.0f);
    pred_y2 = max(min(pred_y2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// boxes are (h * w * anchor, 5)
// deltas are (b, 4 * anchor, h, w)
// out_pred_boxes are (h * w * anchor, 5)
// count should be total anchors numbers, h * w * anchors
// in-place write: boxes and out_pred_boxes are the same location
template<typename Dtype>
__global__ void IoUPredKernel(const int count,
                              const int num_anchors,
                              const int feat_height,
                              const int feat_width,
                              const int real_height,
                              const int real_width,
                              const float im_height,
                              const float im_width,
                              const Dtype* boxes,
                              const Dtype* deltas,
                              Dtype* out_pred_boxes) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    int a = index % num_anchors;
    int w = (index / num_anchors) % feat_width;
    int h = index / num_anchors / feat_width;

    float x1 = boxes[index * 5 + 0];
    float y1 = boxes[index * 5 + 1];
    float x2 = boxes[index * 5 + 2];
    float y2 = boxes[index * 5 + 3];

    float dx1 = deltas[((a * 4) * feat_height + h) * feat_width + w];
    float dy1 = deltas[((a * 4 + 1) * feat_height + h) * feat_width + w];
    float dx2 = deltas[((a * 4 + 2) * feat_height + h) * feat_width + w];
    float dy2 = deltas[((a * 4 + 3) * feat_height + h) * feat_width + w];

    float pred_x1 = max(min(x1 + dx1, im_width - 1.0f), 0.0f);
    float pred_y1 = max(min(y1 + dy1, im_height - 1.0f), 0.0f);
    float pred_x2 = max(min(x2 + dx2, im_width - 1.0f), 0.0f);
    float pred_y2 = max(min(y2 + dy2, im_height - 1.0f), 0.0f);

    out_pred_boxes[index * 5 + 0] = pred_x1;
    out_pred_boxes[index * 5 + 1] = pred_y1;
    out_pred_boxes[index * 5 + 2] = pred_x2;
    out_pred_boxes[index * 5 + 3] = pred_y2;

    if (h >= real_height || w >= real_width) {
      out_pred_boxes[index * 5 + 4] = -1.0f;
    }
  }
}

// filter box with stride less than rpn_min_size
// filter: set score to zero
// dets (n, 5)
template<typename Dtype>
__global__ void FilterBoxKernel(const int count,
                                const float min_size,
                                Dtype* dets,
                                const bool filter_scale,
                                const float valid_min,
                                const float valid_max) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    float iw = dets[index * 5 + 2] - dets[index * 5 + 0] + 1.0f;
    float ih = dets[index * 5 + 3] - dets[index * 5 + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[index * 5 + 0] -= min_size / 2;
      dets[index * 5 + 1] -= min_size / 2;
      dets[index * 5 + 2] += min_size / 2;
      dets[index * 5 + 3] += min_size / 2;
      dets[index * 5 + 4] = -1.0f;
    }else if (filter_scale && (iw * ih < valid_min || iw * ih > valid_max)) {
      dets[index * 5 + 4] = -1.0f;
    }
  }
}

// copy score and init order
// dets (n, 5); score (n, ); order (n, )
// count should be n (total anchors or proposals)
template<typename Dtype>
__global__ void CopyScoreKernel(const int count,
                                const Dtype* dets,
                                Dtype* score,
                                int* order) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    score[index] = dets[index * 5 + 4];
    order[index] = index;
  }
}

// reorder proposals according to order and keep the top_n proposals
// prev_dets (n, 5); order (n, ); dets (n, 5)
// count should be output anchor numbers (top_n)
template<typename Dtype>
__global__ void ReorderProposalsKernel(const int count,
                                       const Dtype* prev_dets,
                                       const int* order,
                                       Dtype* dets) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    const int order_i = order[index];
    for (int j = 0; j < 5; j ++) {
      dets[index * 5 + j] = prev_dets[order_i * 5 + j];
    }
  }
}

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, uint64_t *dev_mask) {
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _nms(const mshadow::Tensor<gpu, 2>& boxes,
          const float nms_overlap_thresh,
          int *keep,
          int *num_out,
          uint64_t *mask_dev,
          uint64_t *mask_host) {
  /*
  @input  boxes: (pre_nms_top_n, 5)
  @return keep
  @return num_out
  @tmp    mask_dev
  @tmp    mask_host
  */
  const int threadsPerBlock = sizeof(uint64_t) * 8;
  const int boxes_num = boxes.size(0);
  const int boxes_dim = boxes.size(1);

  float* boxes_dev = boxes.dptr_;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
  FRCNN_CUDA_CHECK(cudaPeekAtLastError());

  // TODO: need to be rewritten
  FRCNN_CUDA_CHECK(cudaMemcpy(mask_host,
                              mask_dev,
                              sizeof(uint64_t) * boxes_num * col_blocks,
                              cudaMemcpyDeviceToHost));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep[num_to_keep++] = i;
      uint64_t *p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;
}

// copy proposals to output
// dets (top_n, 5); keep (top_n, ); out (top_n, )
// count should be top_n (total anchors or proposals)
template<typename Dtype>
__global__ void PrepareOutput(const int count,
                              const Dtype* dets,
                              const int* keep,
                              const int out_size,
                              Dtype* out,
                              Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    if (index < out_size) {
      int keep_i = keep[index];
      for (int j = 0; j < 4; ++j) {
        out[index * 4 + j] = dets[keep_i * 5 + j];
      }
      score[index] = dets[keep_i * 5 + 4];
    } else {
      //int keep_i = keep[index % out_size];
      for (int j = 0; j < 4; ++j) {
        out[index * 4 + j] = 0.0f;
      }
      score[index] = 0;
    }
  }
}

}  // namespace
}  // namespace cuda
}  // namespace mshadow

namespace mxnet {
namespace op {

template<typename xpu>
class ProposalGPUOp_v2 : public Operator{
 public:
  explicit ProposalGPUOp_v2(ProposalParam_v2 param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow::cuda;

    CHECK_EQ(in_data.size(), 4);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    // CHECK_EQ(req[proposal_v2::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4> scores = in_data[proposal_v2::kClsProb].get<xpu, 4, float>(s); // batch_idx, anchor_idx, height_idx, width_idx
    Tensor<xpu, 4> bbox_deltas = in_data[proposal_v2::kBBoxPred].get<xpu, 4, float>(s); // batch_idx, height_idx, width_idx, anchor_idx
    Tensor<xpu, 2> im_info = in_data[proposal_v2::kImInfo].get<xpu, 2, float>(s); // batch_idx, 3(height, width, scale)
    Tensor<xpu, 2> valid_ranges = in_data[proposal_v2::kValidRanges].get<xpu, 2, float>(s); //batch_idx, 2(min_scale, max_scale)

    Tensor<xpu, 3> out = out_data[proposal_v2::kOut].get<xpu, 3, float>(s); // batch_idx, rois_idx, 4(x1, y1, x2, y2), batch_idx is needed after flatten
    Tensor<xpu, 3> out_score = out_data[proposal_v2::kScore].get<xpu, 3, float>(s); // batch_idx, rois_idx, 1(score)

    uint64_t WORKSPACE_LIMIT = 1024 * 1024 * param_.workspace; // 256 MB should be sufficient
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[proposal_v2::kTempSpace].get_space_typed<xpu, 1, uint8_t>(Shape1(WORKSPACE_LIMIT), s);
    uint64_t allocated_bytes = 0ULL;
    uint64_t allocated_bytes_outside_loop = 0ULL;

    int nbatch = scores.size(0);
    int num_anchors = scores.size(1) / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;  // count of total anchors
    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    // Generate first anchors based on base anchor
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.info.size() * param_.scales.info.size());
    std::vector<float> anchors;
    proposal_v2_utils::GenerateAnchors(base_anchor,
                           param_.ratios.info,
                           param_.scales.info,
                           &anchors);

    // Copy generated anchors to GPU
    Tensor<xpu, 3> proposals(reinterpret_cast<float *>(workspace.dptr_ + allocated_bytes), Shape3(nbatch, count, 5));
    allocated_bytes += nbatch * count * 5 * sizeof(float);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    // im_info is small, we want to copy them to cpu
    std::vector<float> cpu_im_info(nbatch * 3);
    FRCNN_CUDA_CHECK(cudaMemcpy(cpu_im_info.data(), 
                                im_info.dptr_,
                                sizeof(float) * cpu_im_info.size(),
                                cudaMemcpyDeviceToHost)); // less than 64K

    // copy valid_ranges to cpu
    std::vector<float> cpu_valid_ranges(nbatch * 2);
    FRCNN_CUDA_CHECK(cudaMemcpy(cpu_valid_ranges.data(),
                                valid_ranges.dptr_,
                                sizeof(float) * cpu_valid_ranges.size(),
                                cudaMemcpyDeviceToHost)); // less than 64K

    Shape<3> fg_scores_shape = Shape3(in_data[proposal_v2::kClsProb].shape_[1] / 2,
                                      in_data[proposal_v2::kClsProb].shape_[2],
                                      in_data[proposal_v2::kClsProb].shape_[3]);

    allocated_bytes_outside_loop = allocated_bytes;
    /* copy anchors for all images in batch */
    for (int i = 0; i < nbatch; i++) {
      // prevent padded predictions
      int real_height = static_cast<int>(cpu_im_info[i*3 + 0] / param_.feature_stride);
      int real_width = static_cast<int>(cpu_im_info[i*3 + 1] / param_.feature_stride);
      CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
      CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

      float* batch_proposals = proposals.dptr_ + i * 5 * count;
      FRCNN_CUDA_CHECK(cudaMemcpy(batch_proposals,
                                  &anchors[0], 
                                  sizeof(float) * anchors.size(),
                                  cudaMemcpyHostToDevice)); // less than 64K

      /* get current batch foreground score */
      float* foreground_score_ptr = reinterpret_cast<float *>(scores.dptr_) + i * 2 * count + fg_scores_shape.Size();
      Tensor<xpu, 3> fg_scores = Tensor<xpu, 3>(foreground_score_ptr, fg_scores_shape);

      /* copy proposals to a mesh grid */
      dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
      dim3 dimBlock(kMaxThreadsPerBlock);
      CheckLaunchParam(dimGrid, dimBlock, "ProposalGrid");
      ProposalGridKernel<<<dimGrid, dimBlock>>>(
        count, num_anchors, height, width, param_.feature_stride,
        fg_scores.dptr_, batch_proposals);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* transform anchors and bbox_deltas into bboxes */
      CheckLaunchParam(dimGrid, dimBlock, "BBoxPred");
      if (param_.iou_loss) {
        IoUPredKernel<<<dimGrid, dimBlock>>>(
          count, num_anchors, height, width, real_height, real_width,
          cpu_im_info[i * 3 + 0], cpu_im_info[i * 3 + 1],
          batch_proposals, bbox_deltas.dptr_ + i * 4 * count, batch_proposals);
      } else {
        BBoxPredKernel<<<dimGrid, dimBlock>>>(
          count, num_anchors, height, width, real_height, real_width,
          cpu_im_info[i * 3 + 0], cpu_im_info[i * 3 + 1],
          batch_proposals, bbox_deltas.dptr_ + i * 4 * count, batch_proposals);
      }
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* filter boxes with less than rpn_min_size */
      CheckLaunchParam(dimGrid, dimBlock, "FilterBox");
      FilterBoxKernel<<<dimGrid, dimBlock>>>(
        count, param_.rpn_min_size * cpu_im_info[i * 3 + 2], batch_proposals,
        param_.filter_scales, cpu_valid_ranges[i * 2 + 0] * cpu_valid_ranges[i * 2 + 0],
        cpu_valid_ranges[i * 2 + 1] * cpu_valid_ranges[i * 2 + 1]);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* copy score to a continuous memory */
      Tensor<xpu, 1> score(reinterpret_cast<float *>(workspace.dptr_ + allocated_bytes), Shape1(count));
      allocated_bytes += count * sizeof(float);
      CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";      

      Tensor<xpu, 1, int> order(reinterpret_cast<int *>(workspace.dptr_ + allocated_bytes), Shape1(count));
      allocated_bytes += count * sizeof(int);
      CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";      

      CheckLaunchParam(dimGrid, dimBlock, "CopyScore");
      CopyScoreKernel<<<dimGrid, dimBlock>>>(
        count, batch_proposals, score.dptr_, order.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* argsort score, save order */
      thrust::stable_sort_by_key(thrust::device,
                                 score.dptr_,
                                 score.dptr_ + score.size(0),
                                 order.dptr_,
                                 thrust::greater<float>());
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* Reorder proposals according to order */
      Tensor<xpu, 2> ordered_proposals(reinterpret_cast<float *>(workspace.dptr_ + allocated_bytes), Shape2(rpn_pre_nms_top_n, 5));
      allocated_bytes += rpn_pre_nms_top_n * 5 * sizeof(float);
      CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";      

      dimGrid.x = (rpn_pre_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "ReorderProposals");
      ReorderProposalsKernel<<<dimGrid, dimBlock>>>(
        rpn_pre_nms_top_n, batch_proposals, order.dptr_, ordered_proposals.dptr_);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());

      /* perform nms */
      std::vector<int> _keep(rpn_pre_nms_top_n);
      int out_size = 0;
      const int boxes_num = rpn_pre_nms_top_n;
      const int col_blocks = DIVUP(boxes_num, sizeof(uint64_t) * 8);
      // take special care when allocate memory of 8-byte alignment.
      allocated_bytes += allocated_bytes % sizeof(uint64_t);
      Tensor<xpu, 1, uint64_t> mask_tensor(reinterpret_cast<uint64_t *>(workspace.dptr_ + allocated_bytes), Shape1(boxes_num * col_blocks));
      allocated_bytes += boxes_num * col_blocks * sizeof(uint64_t); 
      CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";   
      // the following line does not need change since it the only place where requires host workspace
      Tensor<cpu, 1, uint64_t> mask_host_tensor = ctx.requested[proposal_v2::kTempSpace].get_host_space_typed<1, uint64_t>(Shape1(boxes_num * col_blocks));
      uint64_t *mask_dev = mask_tensor.dptr_;
      uint64_t *mask_host = mask_host_tensor.dptr_;
      _nms(ordered_proposals,
           param_.threshold,
           &_keep[0],
           &out_size,
           mask_dev,
           mask_host);

      /* copy nms result to gpu */
      Tensor<xpu, 1, int> keep(reinterpret_cast<int *>(workspace.dptr_ + allocated_bytes), Shape1(_keep.size()));
      allocated_bytes += _keep.size() * sizeof(int);
      CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";
      
      FRCNN_CUDA_CHECK(cudaMemcpy(keep.dptr_, 
                                  &_keep[0], 
                                  sizeof(int) * _keep.size(),
                                  cudaMemcpyHostToDevice)); // less than 64K

      /* copy results after nms */
      dimGrid.x = (rpn_post_nms_top_n + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
      CheckLaunchParam(dimGrid, dimBlock, "PrepareOutput");
      PrepareOutput<<<dimGrid, dimBlock>>>(
        rpn_post_nms_top_n, ordered_proposals.dptr_, keep.dptr_, out_size,
        out.dptr_ + i * 4 * rpn_post_nms_top_n,
        out_score.dptr_ + i * rpn_post_nms_top_n);
      FRCNN_CUDA_CHECK(cudaPeekAtLastError());
      
      // recycle all bytes allocated within loop
      allocated_bytes = allocated_bytes_outside_loop;
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 4);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal_v2::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal_v2::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal_v2::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> granges = in_grad[proposal_v2::kValidRanges].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal_v2::kClsProb], 0);
    Assign(gbbox, req[proposal_v2::kBBoxPred], 0);
    Assign(ginfo, req[proposal_v2::kImInfo], 0);
    Assign(granges, req[proposal_v2::kValidRanges], 0);
  }

 private:
  ProposalParam_v2 param_;
};  // class ProposalGPUOp_v2

template<>
Operator* CreateOp<gpu>(ProposalParam_v2 param) {
  return new ProposalGPUOp_v2<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
