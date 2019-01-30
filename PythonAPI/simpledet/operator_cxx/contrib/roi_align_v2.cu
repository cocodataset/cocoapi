/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align_v2.cu
 * \brief roi align operator
 * \author Yuchen Guo, Zehao Shi, Chenxia Han
*/
#include "./roi_align_v2-inl.h"
#include "../../common/cuda_utils.h"


namespace mxnet {
namespace op {

/*!
    * \brief Kernel for backward pass of ROIAlign.
*/
struct ROIAlignBackwardKernelGPU_v2 {
/*!
* \param index              loop index
* \param top_diff           gradient of output data
* \param argmax_x           index of value in pooled feature map on x axis
* \param argmax_y           index of value in pooled feature map on y axis
* \param num_rois_per_batch number of rois per batch
* \param spatial_scale      ratio of input feature map height (or width)
                                to raw image height (or width)
* \param channels           channels of input data
* \param height             height of input data
* \param width              width of input data
* \param pooled_height      height of fix pooled size
* \param pooled_width       width of fix pooled size
* \param bottom_diff        gradient of input 4D feature map
* \param bottom_rois        gradient of input rois
*/
template<typename T>
MSHADOW_XINLINE static void Map(
const int index,
const T* top_diff,
const T* argmax_data_x,
const T* argmax_data_y,
const int num_rois_per_batch,
const float spatial_scale,
const int channels,
const int height,
const int width,
const int pooled_height,
const int pooled_width,
T* bottom_diff,
const T* bottom_rois) {
    using namespace mxnet::op::mshadow_op;
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int roi_batch_ind = n / num_rois_per_batch;
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const T* offset_argmax_data_x = argmax_data_x + top_offset;
    const T* offset_argmax_data_y = argmax_data_y + top_offset;

    int pool_index = ph * pooled_width + pw;
    T a_x = offset_argmax_data_x[pool_index];
    T a_y = offset_argmax_data_y[pool_index];
    if (a_x != static_cast<T>(-1) && a_y != static_cast<T>(-1)) {
        int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_y)), 0), height-1);
        int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_y)), 0), height-1);
        int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_x)), 0), width-1);
        int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_x)), 0), width-1);
        int topleft = hlow * width + wleft;
        int topright = hlow * width + wright;
        int bottomleft = hhigh * width + wleft;
        int bottomright = hhigh * width + wright;

        T alpha = (hlow == hhigh) ? static_cast<T>(0.5) : (a_y - hlow) / (hhigh - hlow);
        T beta = (wleft == wright) ? static_cast<T>(0.5) : (a_x - wleft) / (wright - wleft);
        atomicAdd(offset_bottom_diff + topleft, offset_top_diff[pool_index] * (1 - alpha) * (1 - beta));
        atomicAdd(offset_bottom_diff + topright, offset_top_diff[pool_index] * (1 - alpha) * beta);
        atomicAdd(offset_bottom_diff + bottomleft, offset_top_diff[pool_index] * alpha * (1 - beta));
        atomicAdd(offset_bottom_diff + bottomright, offset_top_diff[pool_index] * alpha * beta);
    }
} // Map
};

template<>
void ROIAlignBackward_v2<gpu>(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 4);
  CHECK_EQ(outputs.size(), 2);
  // the order here relates to the order in ROIAlignGrad_v2
  std::vector<TBlob> out_grad(1, inputs[0]);
  std::vector<TBlob> in_data(1, inputs[1]);
  std::vector<TBlob> out_data(inputs.begin() + 2, inputs.begin() + 4);

  CHECK_EQ(out_grad[0].shape_[0], in_data[0].shape_[0]);
  CHECK_EQ(out_data[0].shape_[0], in_data[0].shape_[0]);
  CHECK_EQ(out_data[1].shape_[0], in_data[0].shape_[0]);
  CHECK_NE(req[0], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";
  CHECK_NE(req[1], kWriteInplace) <<
    "ROIAlign: Backward doesn't support kWriteInplace.";

  const ROIAlignParam_v2 param = nnvm::get<ROIAlignParam_v2>(attrs.parsed);

  const int count = out_grad[0].Size();
  const int num_rois_per_batch = in_data[0].size(1);
  const int channels = outputs[0].size(1);
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(3);
  const int pooled_width = out_grad[0].size(4);

  Stream<gpu> *s = ctx.get_stream<gpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(out_grad[0].type_flag_, DType, {
    const DType *top_diff = out_grad[0].dptr<DType>();
    const DType *bottom_rois = in_data[0].dptr<DType>();
    DType *argmax_x = out_data[0].dptr<DType>();
    DType *argmax_y = out_data[1].dptr<DType>();
    DType *grad_in = outputs[0].dptr<DType>();
    DType *grad_roi = outputs[1].dptr<DType>();

    if (kAddTo == req[roialign_v2::kData] || kWriteTo == req[roialign_v2::kData]) {
      if (kWriteTo == req[roialign_v2::kData]) {
        Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
      }
      mxnet_op::Kernel<ROIAlignBackwardKernelGPU_v2, gpu>::Launch(s,
        count, top_diff, argmax_x, argmax_y, num_rois_per_batch,
        param.spatial_scale, channels, height, width,
        pooled_height, pooled_width, grad_in, bottom_rois);
    }
    if (kWriteTo == req[roialign_v2::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}

NNVM_REGISTER_OP(_contrib_ROIAlign_v2)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignForward_v2<gpu>);

NNVM_REGISTER_OP(_backward_ROIAlign_v2)
.set_attr<FCompute>("FCompute<gpu>", ROIAlignBackward_v2<gpu>);

}  // namespace op
}  // namespace mxnet
