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
 * \file nms.cu
 * \brief NMS Operator
 * \author Yanghao Li
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
#include <iterator>

#include "../operator_common.h"
#include "../mshadow_op.h"
#include "./nms-inl.h"

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
                              const int batchIdx,
                              Dtype* out,
                              Dtype* score) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
   // out[index * 5] = batchIdx;
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
class NMSGPUOp : public Operator{
 public:
  explicit NMSGPUOp(NMSParam param) {
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

    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    // CHECK_EQ(req[proposal::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> proposals = in_data[nms::kBBox].get<xpu, 3, float>(s); // batch_idx, rois_idx, 5(x1, y1, x2, y2, score)

    Tensor<xpu, 3> out = out_data[nms::kOut].get<xpu, 3, float>(s); // batch_idx, rois_idx, 4(x1, y1, x2, y2)
    Tensor<xpu, 3> out_score = out_data[nms::kScore].get<xpu, 3, float>(s); // batch_idx, rois_idx, 1(score)

    uint64_t WORKSPACE_LIMIT = 1024 * 1024 * param_.workspace; // 256 MB should be sufficient
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[nms::kTempSpace].get_space_typed<xpu, 1, uint8_t>(Shape1(WORKSPACE_LIMIT), s);
    uint64_t allocated_bytes = 0ULL;
    uint64_t allocated_bytes_outside_loop = 0ULL;

    int nbatch = proposals.size(0);
    int count = proposals.size(1);
    // set to -1 for max
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    /* copy anchors for all images in batch */
    for (int i = 0; i < nbatch; i++) {
      float* batch_proposals = proposals.dptr_ + i * 5 * count;

      /* copy score to a continuous memory */
      dim3 dimGrid((count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock);
      dim3 dimBlock(kMaxThreadsPerBlock);
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

      if (!param_.already_sorted) {
        /* argsort score, save order */
        thrust::stable_sort_by_key(thrust::device,
                                   score.dptr_,
                                   score.dptr_ + score.size(0),
                                   order.dptr_,
                                   thrust::greater<float>());
        FRCNN_CUDA_CHECK(cudaPeekAtLastError());
      }

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
      Tensor<cpu, 1, uint64_t> mask_host_tensor = ctx.requested[nms::kTempSpace].get_host_space_typed<1, uint64_t>(Shape1(boxes_num * col_blocks));
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
        rpn_post_nms_top_n, ordered_proposals.dptr_, keep.dptr_, out_size, i,
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
    CHECK_EQ(in_grad.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> gbbox = in_grad[nms::kBBox].get<xpu, 3, real_t>(s);

    Assign(gbbox, req[nms::kBBox], 0);
  }

 private:
  NMSParam param_;
};  // class NMSGPUOp

template<>
Operator* CreateOp<gpu>(NMSParam param) {
  return new NMSGPUOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
