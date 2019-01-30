/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align_v2-inl.h
 * \brief roi align operator and symbol
 * \author Yuchen Guo, Zehao Shi, Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_ROI_ALIGN_V2_INL_H_
#define MXNET_OPERATOR_CONTRIB_ROI_ALIGN_V2_INL_H_

#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {


// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roialign_v2 {
enum ROIAlignOpInputs {kData, kBox};
enum ROIAlignOpOutputs {kOut, kMaxIdx_x, kMaxIdx_y};
}  // roialign


struct ROIAlignParam_v2 : public dmlc::Parameter<ROIAlignParam_v2> {
  TShape pooled_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(ROIAlignParam_v2) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("fix pooled size: (h, w)");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};


/*!
 * \brief Kernel for forward pass of ROIAlign.
 */
struct ROIAlignForwardKernel_v2 {
  /*!
   * \param index              loop index
   * \param bottom_data        input data which is a 4D feature map
   * \param num_rois_per_batch number of rois per batch
   * \param spatial_scale      ratio of input feature map height (or width)
                                   to raw image height (or width)
   * \param channels           channels of input data
   * \param height             height of input data
   * \param width              width of input data
   * \param pooled_height      height of fix pooled size
   * \param pooled_width       width of fix pooled size
   * \param bottom_rois        input rois of shape (batch, num_rois_per_batch, 4)
   * \param top_data           output data of shape (batch, num_rois_per_batch, channels, height, width)
   * \param argmax_x           index of value in pooled feature map on x axis, -1 if nothing is pooled
   * \param argmax_y           index of value in pooled feature map on y axis, -1 if nothing is pooled
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int index, const DType* bottom_data,
                                  const int num_rois_per_batch,
                                  const float spatial_scale, const int channels,
                                  const int height, const int width,
                                  const int pooled_height, const int pooled_width,
                                  const DType* bottom_rois, DType* top_data,
                                  DType* argmax_x, DType* argmax_y) {
    using namespace mxnet::op::mshadow_op;
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 4;
    int roi_batch_ind = n / num_rois_per_batch;

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_x[index] = 0;
      argmax_y[index] = 0;
      return;
    }

    DType roi_start_w = (bottom_rois[0]) * spatial_scale;
    DType roi_start_h = (bottom_rois[1]) * spatial_scale;
    DType roi_end_w = (bottom_rois[2]) * spatial_scale;
    DType roi_end_h = (bottom_rois[3]) * spatial_scale;
    // Force malformed ROIs to be 1x1
    DType roi_width = roi_end_w - roi_start_w;
    DType roi_height = roi_end_h - roi_start_h;
    DType bin_size_h = static_cast<DType>(roi_height)
                       / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width)
                       / static_cast<DType>(pooled_width);

    DType hstart = static_cast<DType>((ph) * bin_size_h);
    DType wstart = static_cast<DType>((pw) * bin_size_w);
    DType hend = static_cast<DType>((ph + 1) * bin_size_h);
    DType wend = static_cast<DType>((pw + 1) * bin_size_w);
    // Add roi offsets and clip to input boundaries
    hstart = minimum::Map(maximum::Map(hstart + roi_start_h, static_cast<DType>(0)),
                 static_cast<DType>(height - 1));
    hend = minimum::Map(maximum::Map(hend + roi_start_h, static_cast<DType>(0)),
               static_cast<DType>(height - 1));
    wstart = minimum::Map(maximum::Map(wstart + roi_start_w, static_cast<DType>(0)),
                 static_cast<DType>(width - 1));
    wend = minimum::Map(maximum::Map(wend + roi_start_w, static_cast<DType>(0)),
               static_cast<DType>(width - 1));
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    DType maxidx_x = -1;
    DType maxidx_y = -1;
    DType maxval = static_cast<DType>(0);
    if (!is_empty) {
      maxval = mshadow::red::limits::MinValue<DType>();
      bottom_data += (roi_batch_ind * channels + c) * height * width;
      DType h_stride = (hend - hstart)/3.0;
      DType w_stride = (wend - wstart)/3.0;
      for (DType h = hstart+h_stride; h <= hend-h_stride+0.01;
            h += maximum::Map(h_stride, static_cast<DType>(0.01))) {
        for (DType w = wstart+w_stride; w <= wend-w_stride+0.01;
              w += maximum::Map(w_stride, static_cast<DType>(0.01))) {
          int hlow = minimum::Map(maximum::Map(static_cast<int>(floor::Map(h)), 0), height-1);
          int hhigh = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(h)), 0), height-1);
          int wleft = minimum::Map(maximum::Map(static_cast<int>(floor::Map(w)), 0), width-1);
          int wright = minimum::Map(maximum::Map(static_cast<int>(ceil::Map(w)), 0), width-1);
          int topleft = hlow * width + wleft;
          int topright = hlow * width + wright;
          int bottomleft = hhigh * width + wleft;
          int bottomright = hhigh * width + wright;

          DType alpha = (hlow == hhigh) ? static_cast<DType>(0.5) : (h - hlow) / (hhigh - hlow);
          DType beta = (wleft == wright) ? static_cast<DType>(0.5) : (w - wleft) / (wright - wleft);
          DType value = (1 - alpha) * (1 - beta) * bottom_data[topleft]
                          + alpha * (1 - beta) * bottom_data[bottomleft]
                          + (1 - alpha) * beta * bottom_data[topright]
                          + alpha * beta * bottom_data[bottomright];

          if (value > maxval) {
            maxval = value;
            maxidx_x = w;
            maxidx_y = h;
          }
        }
      }
    }
    top_data[index] = maxval;
    argmax_x[index] = (DType)maxidx_x;
    argmax_y[index] = (DType)maxidx_y;
  }
};


template<typename xpu>
void ROIAlignForward_v2(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& in_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  size_t expected_in = 2;
  size_t expected_out = 3;
  CHECK_EQ(in_data.size(), expected_in);
  CHECK_EQ(out_data.size(), expected_out);
  CHECK_EQ(out_data[roialign_v2::kOut].shape_[0], in_data[roialign_v2::kBox].shape_[0]);
  CHECK_EQ(out_data[roialign_v2::kMaxIdx_x].shape_[0], in_data[roialign_v2::kBox].shape_[0]);
  CHECK_EQ(out_data[roialign_v2::kMaxIdx_y].shape_[0], in_data[roialign_v2::kBox].shape_[0]);

  const ROIAlignParam_v2 param = nnvm::get<ROIAlignParam_v2>(attrs.parsed);

  const int count = out_data[roialign_v2::kOut].Size();
  const int num_rois_per_batch = in_data[roialign_v2::kBox].size(1);
  const int channels = in_data[roialign_v2::kData].size(1);
  const int height = in_data[roialign_v2::kData].size(2);
  const int width = in_data[roialign_v2::kData].size(3);
  const int pooled_height = out_data[roialign_v2::kOut].size(3);
  const int pooled_width = out_data[roialign_v2::kOut].size(4);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // assume all the data and gradient have the same type
  MSHADOW_REAL_TYPE_SWITCH(in_data[0].type_flag_, DType, {
    const DType *bottom_data = in_data[roialign_v2::kData].dptr<DType>();
    const DType *bottom_rois = in_data[roialign_v2::kBox].dptr<DType>();
    DType *top_data = out_data[roialign_v2::kOut].dptr<DType>();
    DType *argmax_x = out_data[roialign_v2::kMaxIdx_x].dptr<DType>();
    DType *argmax_y = out_data[roialign_v2::kMaxIdx_y].dptr<DType>();

    mxnet_op::Kernel<ROIAlignForwardKernel_v2, xpu>::Launch(s,
      count, bottom_data, num_rois_per_batch, param.spatial_scale, channels, height,
      width, pooled_height, pooled_width, bottom_rois, top_data, argmax_x, argmax_y);
  })
}


template<typename xpu>
void ROIAlignBackward_v2(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs);


struct ROIAlignGrad_v2 {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    heads.push_back(ograds[roialign_v2::kOut]);
    heads.push_back(n->inputs[roialign_v2::kBox]);
    heads.emplace_back(nnvm::NodeEntry{n, roialign_v2::kMaxIdx_x, 0});
    heads.emplace_back(nnvm::NodeEntry{n, roialign_v2::kMaxIdx_y, 0});

    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_ROI_ALIGN_V2_INL_H_
