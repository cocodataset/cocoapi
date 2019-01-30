/*!
 * Copyright (c) 2017 by TuSimple
 * \file proposal_mask_target-inl.h
 * \brief C++ version proposal target
 * \author Yuntao Chen, Ziyang Zhou
 */
#ifndef MXNET_OPERATOR_PROPOSAL_MASK_TARGET_INL_H_
#define MXNET_OPERATOR_PROPOSAL_MASK_TARGET_INL_H_


#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "./operator_common.h"

#include <iostream>

namespace mshadow {
namespace proposal_mask_target {


template <typename DType>
inline void SampleROIMask(const Tensor<cpu, 2, DType> &all_rois,
                      const Tensor<cpu, 2, DType> &gt_boxes,
                      const Tensor<cpu, 2, DType> &gt_polys,
                      const Tensor<cpu, 1, DType> &bbox_mean,
                      const Tensor<cpu, 1, DType> &bbox_std,
                      const Tensor<cpu, 1, DType> &bbox_weight,
                      const index_t fg_rois_per_image,
                      const index_t rois_per_image,
                      const index_t mask_size,
                      const index_t num_classes,
                      const float fg_thresh,
                      const float bg_thresh_hi,
                      const float bg_thresh_lo,
                      const int image_rois,
                      const bool class_agnostic,
                      Tensor<cpu, 2, DType> &&rois,
                      Tensor<cpu, 1, DType> &&labels,
                      Tensor<cpu, 2, DType> &&bbox_targets,
                      Tensor<cpu, 2, DType> &&bbox_weights,
                      Tensor<cpu, 1, DType> &&match_gt_ious,
                      Tensor<cpu, 4, DType> &&mask_targets);

template <typename DType>
void BBoxOverlap(
  const Tensor<cpu, 2, DType> &boxes,
  const Tensor<cpu, 2, DType> &query_boxes,
  Tensor<cpu, 2, DType> &overlaps
);

template <typename DType>
void ExpandBboxRegressionTargets(
  const Tensor<cpu, 2, DType> &bbox_target_data,
  Tensor<cpu, 2, DType> &bbox_targets,
  Tensor<cpu, 2, DType> &bbox_weights,
  const Tensor<cpu, 1, DType> &bbox_weight
);

template <typename DType>
void NonLinearTransformAndNormalization(
  const Tensor<cpu, 2, DType> &ex_rois,
  const Tensor<cpu, 2, DType> &gt_rois,
  Tensor<cpu, 2, DType> &targets,
  const Tensor<cpu, 1, DType> &bbox_mean,
  const Tensor<cpu, 1, DType> &bbox_std
);

} // namespace proposal_mask_target
} // namespace mshadow

namespace mxnet {
namespace op {

namespace proposal_mask_target_enum {
enum ProposalMaskTargetInputs {kRois, kGtBboxes, kGtPolys};
enum ProposalMaskTargetOutputs {kRoiOutput, kLabel, kBboxTarget, kBboxWeight, kMatch_gt_iou, kMaskTarget};
}

struct ProposalMaskTargetParam : public dmlc::Parameter<ProposalMaskTargetParam> {
  index_t num_classes;
  index_t batch_images;
  int image_rois;
  index_t mask_size;
  float fg_fraction;
  float fg_thresh;
  float bg_thresh_hi;
  float bg_thresh_lo;
  bool proposal_without_gt;
  bool class_agnostic;
  bool ohem;
  bool output_iou;
  nnvm::Tuple<float> bbox_mean;
  nnvm::Tuple<float> bbox_std;
  nnvm::Tuple<float> bbox_weight;

  DMLC_DECLARE_PARAMETER(ProposalMaskTargetParam) {
    DMLC_DECLARE_FIELD(num_classes).describe("Number of classes for detection");
    DMLC_DECLARE_FIELD(batch_images).describe("Number of samples in a batch");
    DMLC_DECLARE_FIELD(image_rois).describe("Number of ROIs for one image");
    DMLC_DECLARE_FIELD(mask_size).describe("Size of mask target");
    DMLC_DECLARE_FIELD(fg_thresh).describe("Foreground IOU threshold");
    DMLC_DECLARE_FIELD(bg_thresh_hi).describe("Background IOU upper bound");
    DMLC_DECLARE_FIELD(bg_thresh_lo).describe("Background IOU lower bound");
    DMLC_DECLARE_FIELD(fg_fraction).set_default(0.25f).describe("Fraction of foreground proposals");
    DMLC_DECLARE_FIELD(proposal_without_gt).describe("Do not append ground-truth bounding boxes to output");
    DMLC_DECLARE_FIELD(class_agnostic).set_default(false).describe("class agnostic bbox_target");
    DMLC_DECLARE_FIELD(ohem).set_default(false).describe("Do online hard sample mining");
    DMLC_DECLARE_FIELD(output_iou).set_default(false).describe("output match_gt_iou");
    float tmp[] = {0.f, 0.f, 0.f, 0.f};
    DMLC_DECLARE_FIELD(bbox_mean).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box mean");
    tmp[0] = 0.1f; tmp[1] = 0.1f; tmp[2] = 0.2f; tmp[3] = 0.2f;
    DMLC_DECLARE_FIELD(bbox_std).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box std");
    tmp[0] = 1.f; tmp[1] = 1.f; tmp[2] = 1.f; tmp[3] = 1.f;
    DMLC_DECLARE_FIELD(bbox_weight).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Foreground bounding box weight");
  }
};

template<typename xpu, typename DType>
class ProposalMaskTargetOp : public Operator {
 public:
  explicit ProposalMaskTargetOp(ProposalMaskTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 6);
    CHECK_GT(req.size(), 5);
    CHECK_EQ(req[proposal_mask_target_enum::kRoiOutput], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kLabel], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kBboxTarget], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kBboxWeight], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kMaskTarget], kWriteTo);
 //   printf("PROPOSAL_TARGET START\n");
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const index_t num_image         = param_.batch_images;
    const index_t num_roi           = in_data[proposal_mask_target_enum::kRois].Size() / (num_image * 4);
    const index_t num_gtbbox        = in_data[proposal_mask_target_enum::kGtBboxes].Size() / (num_image * 5);
    // when image_rois = -1 , keep out all rois without subsampling.
    const int image_rois            = param_.image_rois;
    const index_t num_roi_per_image = image_rois == -1 ? num_roi : image_rois;
    Tensor<xpu, 3, DType> xpu_rois      = in_data[proposal_mask_target_enum::kRois].
                                          get_with_shape<xpu, 3, DType>(Shape3(num_image, num_roi, 4), s);
    Tensor<xpu, 3, DType> xpu_gt_bboxes = in_data[proposal_mask_target_enum::kGtBboxes].
                                          get_with_shape<xpu, 3, DType>(Shape3(num_image, num_gtbbox, 5), s);

    // Tensor<xpu, 3, DType> xpu_gt_polys = in_data[proposal_mask_target_enum::kGtPolys].
    //                                       get_with_shape<xpu, 3, DType>(Shape3(num_image, num_gtbbox, poly_len), s);

    Tensor<xpu, 3, DType> xpu_gt_polys = in_data[proposal_mask_target_enum::kGtPolys].get<xpu, 3, DType>(s);

    TensorContainer<cpu, 3, DType> rois     (xpu_rois.shape_);
    TensorContainer<cpu, 3, DType> gt_bboxes(xpu_gt_bboxes.shape_);
    TensorContainer<cpu, 3, DType> gt_polys(xpu_gt_polys.shape_);

    Copy(rois, xpu_rois, s);
    Copy(gt_bboxes, xpu_gt_bboxes, s);
    Copy(gt_polys, xpu_gt_polys, s);


    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_rois;
    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_gtbboxes;
    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_gtpolys;

    // clean up bboxes
    for (index_t i = 0; i < num_image; ++i) {
        kept_gtbboxes.push_back(std::vector<Tensor<cpu, 1, DType>>());
        kept_gtpolys.push_back(std::vector<Tensor<cpu, 1, DType>>());
        for (index_t j = 0; j < gt_bboxes.size(1); ++j) {
            if (gt_bboxes[i][j][4] != -1) {
                kept_gtbboxes[i].push_back(gt_bboxes[i][j]);
                kept_gtpolys[i].push_back(gt_polys[i][j]);
            }
        }
    }

    //for append gt
    std::vector<TensorContainer<cpu, 1, DType>> gt_bboxes_tmp(num_image * num_gtbbox);
    for (index_t i = 0; i < gt_bboxes_tmp.size(); ++i) {
      gt_bboxes_tmp[i].Resize(Shape1(4));
    }
    index_t start = 0;
    for (index_t i = 0; i < num_image; ++i) {
      kept_rois.push_back(std::vector<Tensor<cpu, 1, DType>>());
      for (index_t j = 0; j < rois.size(1); ++j) {
        // y2 == 0 indicates padding
        if (rois[i][j][3] > 0)
          kept_rois[i].push_back(rois[i][j]);
      }
      if (!param_.proposal_without_gt) {
        // all gt bboxes are appended
        for (index_t j = 0; j < kept_gtbboxes[i].size(); ++j) {
          Tensor<cpu, 1, DType> gt_tmp = gt_bboxes_tmp[start++];
        //  gt_tmp[0] = i;
          Copy(gt_tmp, kept_gtbboxes[i][j].Slice(0, 4));
          kept_rois[i].push_back(gt_tmp);
        }
      }
    }
  //  printf("PROPOSAL_TARGET MID\n");
    TensorContainer<cpu, 3, DType> cpu_output_rois(Shape3(num_image, image_rois, 4), 0.f);
    TensorContainer<cpu, 2, DType> cpu_labels(Shape2(num_image, image_rois), 0.f);
    TensorContainer<cpu, 3, DType> cpu_bbox_targets(Shape3(num_image, image_rois, param_.num_classes * 4), 0.f);
    TensorContainer<cpu, 3, DType> cpu_bbox_weights(Shape3(num_image, image_rois, param_.num_classes * 4), 0.f);
    TensorContainer<cpu, 2, DType> cpu_match_gt_ious(Shape2(num_image, image_rois), 0.f);
    TensorContainer<cpu, 5, DType> cpu_mask_targets(Shape5(num_image,  (index_t)(image_rois * param_.fg_fraction), 
                                                           param_.num_classes, param_.mask_size, param_.mask_size), -1.f);

    if (param_.ohem) {
        LOG(FATAL) << "OHEM not Implemented.";
    } else {
        index_t fg_rois_per_image = static_cast<index_t>(num_roi_per_image * param_.fg_fraction);
        TensorContainer<cpu, 1, DType> bbox_mean(Shape1(4));
        TensorContainer<cpu, 1, DType> bbox_std(Shape1(4));
        TensorContainer<cpu, 1, DType> bbox_weight(Shape1(4));
        bbox_mean[0] = param_.bbox_mean[0];
        bbox_mean[1] = param_.bbox_mean[1];
        bbox_mean[2] = param_.bbox_mean[2];
        bbox_mean[3] = param_.bbox_mean[3];
        bbox_std[0] = param_.bbox_std[0];
        bbox_std[1] = param_.bbox_std[1];
        bbox_std[2] = param_.bbox_std[2];
        bbox_std[3] = param_.bbox_std[3];
        bbox_weight[0] = param_.bbox_weight[0];
        bbox_weight[1] = param_.bbox_weight[1];
        bbox_weight[2] = param_.bbox_weight[2];
        bbox_weight[3] = param_.bbox_weight[3];
        for (index_t i = 0; i < num_image; ++i) {
          TensorContainer<cpu, 2, DType> kept_rois_i    (Shape2(kept_rois[i].size(),     rois.size(2)));
          TensorContainer<cpu, 2, DType> kept_gtbboxes_i(Shape2(kept_gtbboxes[i].size(), gt_bboxes.size(2)));
          for (index_t j = 0; j < kept_rois_i.size(0); ++j) {
              Copy(kept_rois_i[j], kept_rois[i][j]);
          }
          for (index_t j = 0; j < kept_gtbboxes_i.size(0); ++j) {
              Copy(kept_gtbboxes_i[j], kept_gtbboxes[i][j]);
          }
          index_t kept_gtbboxes_size_i = kept_gtbboxes[i].size();
          TensorContainer<cpu, 2, DType> kept_gtpolys_i(Shape2(kept_gtbboxes_size_i, gt_polys.size(2)), 0.f);
          for (index_t j = 0; j < kept_gtpolys[i].size(); ++j) {
            Copy(kept_gtpolys_i[j], kept_gtpolys[i][j]);
          }
          //printf("BEFORE SAMPLEROI");
          proposal_mask_target::SampleROIMask(kept_rois_i,
                        kept_gtbboxes_i,
                        kept_gtpolys_i,
                        bbox_mean,
                        bbox_std,
                        bbox_weight,
                        fg_rois_per_image,
                        num_roi_per_image,
                        param_.mask_size,
                        param_.num_classes,
                        param_.fg_thresh,
                        param_.bg_thresh_hi,
                        param_.bg_thresh_lo,
                        param_.image_rois,
                        param_.class_agnostic,
                        cpu_output_rois[i],
                        cpu_labels[i],
                        cpu_bbox_targets[i],
                        cpu_bbox_weights[i],
                        cpu_match_gt_ious[i],
                        cpu_mask_targets[i]);
        }
    }

    Tensor<xpu, 3, DType> xpu_output_rois  = out_data[proposal_mask_target_enum::kRoiOutput].
                                             get_with_shape<xpu, 3, DType>(Shape3(num_image, image_rois, 4), s);
    Tensor<xpu, 2, DType> xpu_labels       = out_data[proposal_mask_target_enum::kLabel].
                                             get_with_shape<xpu, 2, DType>(Shape2(num_image, image_rois), s);
    Tensor<xpu, 3, DType> xpu_bbox_targets = out_data[proposal_mask_target_enum::kBboxTarget].
                                             get_with_shape<xpu, 3, DType>(Shape3(num_image, image_rois, param_.num_classes * 4), s);
    Tensor<xpu, 3, DType> xpu_bbox_weights = out_data[proposal_mask_target_enum::kBboxWeight].
                                             get_with_shape<xpu, 3, DType>(Shape3(num_image, image_rois, param_.num_classes * 4), s);
    Tensor<xpu, 2, DType> xpu_match_gt_ious = out_data[proposal_mask_target_enum::kMatch_gt_iou].
                                             get_with_shape<xpu, 2, DType>(Shape2(num_image, image_rois), s);
    Tensor<xpu, 5, DType> xpu_mask_targets = out_data[proposal_mask_target_enum::kMaskTarget].get<xpu, 5, DType>(s);
                                            //  get_with_shape<xpu, 5, DType>(Shape5((index_t)(num_image,  (index_t)(image_rois * param_.fg_fraction), 
                                            //                param_.num_classes, param_.mask_size, param_.mask_size), s);

    Copy(xpu_output_rois, cpu_output_rois, s);
    Copy(xpu_labels, cpu_labels, s);
    Copy(xpu_bbox_targets, cpu_bbox_targets, s);
    Copy(xpu_bbox_weights, cpu_bbox_weights, s);
    Copy(xpu_match_gt_ious, cpu_match_gt_ious, s);
    Copy(xpu_mask_targets, cpu_mask_targets, s);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 2);
    const index_t num_image         = param_.batch_images;
    const index_t num_gtbbox        = in_data[proposal_mask_target_enum::kGtBboxes].Size() / (num_image * 5);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> rois      = in_grad[proposal_mask_target_enum::kRois].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> gt_bboxes = in_grad[proposal_mask_target_enum::kGtBboxes].get_with_shape<xpu, 3, DType>(Shape3(num_image, num_gtbbox, 5), s);

    rois = 0.f;
    gt_bboxes = 0.f;
  }

 private:
  ProposalMaskTargetParam param_;
};  // class ProposalTargetOp

template<typename xpu>
Operator *CreateOp(ProposalMaskTargetParam param, int dtype);

#if DMLC_USE_CXX11
class ProposalMaskTargetProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  int NumVisibleOutputs() const override {
    if (param_.output_iou) {
      return 6;
    } else {
      return 5;
    }
  }

  int NumOutputs() const override {
    return 6;
  }

  std::vector<std::string> ListArguments() const override {
    return {"rois", "gt_boxes", "gt_polys"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"roi_output", "label", "bbox_target", "bbox_weight", "match_gt_iou", "mask_target"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[rois, gt_boxes, gt_polys]";
    const TShape &dshape = in_shape->at(proposal_mask_target_enum::kRois);
    const int image_rois  = param_.image_rois == -1 ? dshape[1]: param_.image_rois;

    auto output_rois_shape = Shape3(dshape[0], image_rois, 4);
    auto label_shape = Shape2(dshape[0], image_rois);
    auto bbox_target_shape = Shape3(dshape[0], image_rois, param_.num_classes * 4);
    auto bbox_weight_shape = Shape3(dshape[0], image_rois, param_.num_classes * 4);
    auto match_gt_iou_shape = Shape2(dshape[0], image_rois);
    auto mask_target_shape = Shape5(dshape[0], (index_t)(image_rois * param_.fg_fraction), param_.num_classes, param_.mask_size, param_.mask_size);

    out_shape->clear();
    out_shape->push_back(output_rois_shape);
    out_shape->push_back(label_shape);
    out_shape->push_back(bbox_target_shape);
    out_shape->push_back(bbox_weight_shape);
    out_shape->push_back(match_gt_iou_shape);
    out_shape->push_back(mask_target_shape);
    aux_shape->clear();

    return true;
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {};
  }

  std::string TypeString() const override {
    return "ProposalMaskTarget";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new ProposalMaskTargetProp();
    ptr->param_ = param_;
    return ptr;
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ProposalMaskTargetParam param_;
};  // class ProposalTargetProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_PROPOSAL_TARGET_INL_H_
