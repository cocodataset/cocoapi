/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align_v2.cc
 * \brief roi align operator
 * \author Yuchen Guo, Zehao Shi, Chenxia Han
*/
#include "./roi_align_v2-inl.h"


namespace mxnet {
namespace op {

/*!
 * \brief Kernel for backward pass of ROIAlign.
 */
struct ROIAlignBackwardKernelCPU {
  /*!
   * \param index              loop index
   * \param top_diff           gradient of output data
   * \param argmax_x           index of value in pooled feature map on x axis
   * \param argmax_y           index of value in pooled feature map on y axis
   * \param num_rois_per_batch number of rois per batch
   * \param num_rois           number of rois
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
  template<typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType* top_diff,
                                  const DType* argmax_x, const DType* argmax_y,
                                  const int num_rois_per_batch,
                                  const int num_rois, const float spatial_scale,
                                  const int channels, const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  DType* bottom_diff, const DType* bottom_rois) {
    using namespace mxnet::op::mshadow_op;
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    DType gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const DType* offset_bottom_rois = bottom_rois + roi_n * 4;
      int roi_batch_ind = roi_n / num_rois_per_batch;
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      DType roi_start_w = (offset_bottom_rois[0]) * spatial_scale;
      DType roi_start_h = (offset_bottom_rois[1]) * spatial_scale;
      DType roi_end_w = (offset_bottom_rois[2]) * spatial_scale;
      DType roi_end_h = (offset_bottom_rois[3]) * spatial_scale;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w > roi_start_w - 1.0 && w < roi_end_w + 1.0 &&
                           h > roi_start_h - 1.0 && h < roi_end_h + 1.0);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const DType* offset_top_diff = top_diff + offset;
      const DType* offset_argmax_x = argmax_x + offset;
      const DType* offset_argmax_y = argmax_y + offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          const int pool_index = ph * pooled_width + pw;
          DType a_x = offset_argmax_x[pool_index];
          DType a_y = offset_argmax_y[pool_index];
          int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_y)), 0), height-1);
          int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_y)), 0), height-1);
          int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(a_x)), 0), width-1);
          int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(a_x)), 0), width-1);
          // (w, h) is not around (a_x, a_y)
          if (h != hlow && h != hhigh && w != wleft && w != wright)
            continue;

          DType alpha = (hlow == hhigh) ? static_cast<DType>(0.5)
                                        : (a_y - hlow) / (hhigh - hlow);
          DType beta = (wleft == wright) ? static_cast<DType>(0.5)
                                         : (a_x - wleft) / (wright - wleft);
          if (h == hlow && w == wleft) {
            gradient += offset_top_diff[pool_index] * (1 - alpha) * (1 - beta);
          } else if (h == hlow && w == wright) {
            gradient += offset_top_diff[pool_index] * (1 - alpha) * beta;
          } else if (h == hhigh && w == wleft) {
            gradient += offset_top_diff[pool_index] * alpha * (1 - beta);
          } else if (h == hhigh && w == wright) {
            gradient += offset_top_diff[pool_index] * alpha * beta;
          }
        }
      }
    }
    bottom_diff[index] += gradient;
  }
};

template<>
void ROIAlignBackward_v2<cpu>(const nnvm::NodeAttrs& attrs,
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

  const int count = outputs[0].Size();
  const int num_rois_per_batch = in_data[0].size(1);
  const int num_rois = in_data[0].shape_.ProdShape(0, 2);
  const int channels = outputs[0].size(1);
  const int height = outputs[0].size(2);
  const int width = outputs[0].size(3);
  const int pooled_height = out_grad[0].size(3);
  const int pooled_width = out_grad[0].size(4);

  Stream<cpu> *s = ctx.get_stream<cpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(out_grad[0].type_flag_, DType, {
    const DType *top_diff = out_grad[0].dptr<DType>();
    const DType *bottom_rois = in_data[0].dptr<DType>();
    DType *argmax_x = out_data[0].dptr<DType>();
    DType *argmax_y = out_data[1].dptr<DType>();
    DType *grad_in = outputs[0].dptr<DType>();

    if (kAddTo == req[roialign_v2::kData] || kWriteTo == req[roialign_v2::kData]) {
      if (kWriteTo == req[roialign_v2::kData]) {
        Fill<false>(s, outputs[0], kWriteTo, static_cast<DType>(0));
      }
      mxnet_op::Kernel<ROIAlignBackwardKernelCPU, cpu>::Launch(s,
        count, top_diff, argmax_x, argmax_y, num_rois_per_batch, num_rois,
        param.spatial_scale, channels, height, width,
        pooled_height, pooled_width, grad_in, bottom_rois);
    }
    if (kWriteTo == req[roialign_v2::kBox]) {
      Fill<false>(s, outputs[1], kWriteTo, static_cast<DType>(0));
    }
  })
}

DMLC_REGISTER_PARAMETER(ROIAlignParam_v2);


NNVM_REGISTER_OP(_contrib_ROIAlign_v2)
.describe("ROIAlign foward.")
.set_num_inputs(2)
.set_num_outputs(3)
.set_attr_parser(ParamParser<ROIAlignParam_v2>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "rois"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "maxidx_x", "maxidx_y"};
})
.set_attr<nnvm::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      std::vector<TShape> *in_shape, std::vector<TShape> *out_shape){
  using namespace mshadow;
  const ROIAlignParam_v2 param = nnvm::get<ROIAlignParam_v2>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2) << "Input:[data, rois]";
  // data: [batch_size, c, h, w]
  TShape dshape = in_shape->at(roialign_v2::kData);
  CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
  // bbox: [batch_size, num_rois_per_batch, 4]
  TShape bshape = in_shape->at(roialign_v2::kBox);
  CHECK_EQ(bshape.ndim(), 3) << "bbox should be a 3D tensor of shape [batch, rois, 4]";
  CHECK_EQ(bshape[2], 4) << "bbox should be a 3D tensor of shape [batch, rois, 4]";
  // out: [batch_size, num_rois_per_batch, c, pooled_h, pooled_w]
  // max_idx_x: [batch_size, num_rois_per_batch, c, pooled_h, pooled_w]
  // max_idx_y: [batch_size, num_rois_per_batch, c, pooled_h, pooled_w]
  out_shape->clear();
  out_shape->push_back(
       Shape5(bshape[0], bshape[1], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  out_shape->push_back(
       Shape5(bshape[0], bshape[1], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  out_shape->push_back(
       Shape5(bshape[0], bshape[1], dshape[1], param.pooled_size[0], param.pooled_size[1]));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2);
  int dtype = (*in_type)[0];
  CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  out_type->push_back(dtype);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", ROIAlignForward_v2<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ROIAlignGrad_v2{"_backward_ROIAlign_v2"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 3D array")
.add_arguments(ROIAlignParam_v2::__FIELDS__());


NNVM_REGISTER_OP(_backward_ROIAlign_v2)
.describe("ROIAlign backward.")
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<ROIAlignParam_v2>)
.set_attr<FCompute>("FCompute<cpu>", ROIAlignBackward_v2<cpu>);

}  // namespace op
}  // namespace mxnet
