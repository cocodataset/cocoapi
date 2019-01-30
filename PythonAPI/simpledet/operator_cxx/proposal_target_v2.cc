/*!
 * Copyright (c) 2017 by TuSimple
 * \file proposal_target.cc
 * \brief C++ version proposal target for SNIP
 * \author Yuntao Chen, Zehao Huang, Ziyang Zhou, Yanghao Li
 */
#include "./proposal_target_v2-inl.h"
#include <algorithm>
#include <cstdio>
using std::min;
using std::max;
using std::vector;
using std::begin;
using std::end;
using std::random_shuffle;
using std::log;


namespace mshadow {
namespace proposal_target_v2 {
template <typename DType>
inline void SampleROI(const Tensor<cpu, 2, DType> &all_rois,
                      const Tensor<cpu, 2, DType> &gt_boxes,
                      const Tensor<cpu, 1, DType> &bbox_mean,
                      const Tensor<cpu, 1, DType> &bbox_std,
                      const Tensor<cpu, 1, DType> &bbox_weight,
                      const index_t fg_rois_per_image,
                      const index_t rois_per_image,
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
                      Tensor<cpu, 1, DType> &&match_gt_ious) {
  /*
  overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
  gt_assignment = overlaps.argmax(axis=1)
  overlaps = overlaps.max(axis=1)
  labels = gt_boxes[gt_assignment, 4]
  */
  TensorContainer<cpu, 2, DType> IOUs(Shape2(all_rois.size(0), gt_boxes.size(0)), 0.f);
  BBoxOverlap(all_rois, gt_boxes, IOUs);

  vector<DType> max_overlaps(all_rois.size(0), 0.f);
  vector<DType> all_labels(all_rois.size(0), 0.f);
  vector<DType> gt_assignment(all_rois.size(0), 0.f);
  for (index_t i = 0; i < IOUs.size(0); ++i) {
      DType max_value = IOUs[i][0];
      index_t max_index = 0;
      for (index_t j = 1; j < IOUs.size(1); ++j) {
        if (max_value < IOUs[i][j]) {
            max_value = IOUs[i][j];
            max_index = j;
        }
      }
      gt_assignment[i] = max_index;
      max_overlaps[i] = max_value;
      all_labels[i] = gt_boxes[max_index][4];
  }
  /*
  fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
  fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
  if len(fg_indexes) > fg_rois_per_this_image:
    fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)
  */
  vector<index_t> fg_indexes;
  vector<index_t> neg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= fg_thresh) {
      fg_indexes.push_back(i);
    } else {
      neg_indexes.push_back(i);
    }
  }
  // when image_rois != -1, subsampling rois
  index_t fg_rois_this_image;
  if (image_rois != -1)
    fg_rois_this_image = min<index_t>(fg_rois_per_image, fg_indexes.size());
  else
    fg_rois_this_image = fg_indexes.size();
  if (fg_indexes.size() > fg_rois_this_image) {
    random_shuffle(begin(fg_indexes), end(fg_indexes));
    fg_indexes.resize(fg_rois_this_image);
  }

  /*
  bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0] 
  bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
  bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
  if len(bg_indexes) > bg_rois_per_this_image:
    bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)
  */
  vector<index_t> bg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= bg_thresh_lo && max_overlaps[i] < bg_thresh_hi) {
        bg_indexes.push_back(i);
    }
  }
  index_t bg_rois_this_image = min<index_t>(rois_per_image - fg_rois_this_image, bg_indexes.size());
  if (bg_indexes.size() > bg_rois_this_image) {
      random_shuffle(begin(bg_indexes), end(bg_indexes));
      bg_indexes.resize(bg_rois_this_image);
  }
  //printf("fg %d bg %d\n", fg_rois_this_image,bg_rois_this_image);
  // keep_indexes = np.append(fg_indexes, bg_indexes)
  vector<index_t> kept_indexes;
  for (index_t i = 0; i < fg_rois_this_image; ++i) {
      kept_indexes.push_back(fg_indexes[i]);
  }
  for (index_t i = 0; i < bg_rois_this_image; ++i) {
      kept_indexes.push_back(bg_indexes[i]);
  }

  // pad with negative rois, original code is GARBAGE and omitted
  while (kept_indexes.size() < rois_per_image && neg_indexes.size() > 0) {
      index_t gap = rois_per_image - kept_indexes.size();
      random_shuffle(begin(neg_indexes), end(neg_indexes));
      for (index_t idx = 0;idx < gap && idx < neg_indexes.size();++idx) {
          kept_indexes.push_back(neg_indexes[idx]);
      }
  }
  // printf("fg %d bg %d %d\n", fg_rois_this_image,bg_rois_this_image, kept_indexes.size());
  // pad with positive rois if no negative rois
//  while (kept_indexes.size() < rois_per_image && fg_indexes.size() > 0) {
//      index_t gap = rois_per_image - kept_indexes.size();
//      random_shuffle(begin(fg_indexes), end(fg_indexes));
//      for (auto idx = 0;idx < gap && idx < fg_indexes.size();++idx) {
//          kept_indexes.push_back(fg_indexes[idx]);
//      }
//  }
  // printf("fg %d bg %d %d\n", fg_rois_this_image,bg_rois_this_image, kept_indexes.size());
  /*
  labels = labels[keep_indexes]
  labels[fg_rois_per_this_image:] = 0
  rois = rois[keep_indexes]
  */
  for (index_t i = 0; i < kept_indexes.size(); ++i) {
    if (i < fg_rois_this_image){
      labels[i] = all_labels[kept_indexes[i]];
    }
    Copy(rois[i], all_rois[kept_indexes[i]]);
    match_gt_ious[i] = max_overlaps[kept_indexes[i]];
  }


  TensorContainer<cpu, 2, DType> rois_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i) {
      Copy(rois_tmp[i], rois[i]);
  }

  TensorContainer<cpu, 2, DType> gt_bboxes_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i) {
      Copy(gt_bboxes_tmp[i], gt_boxes[gt_assignment[kept_indexes[i]]].Slice(0, 4));
  }
  TensorContainer<cpu, 2, DType> targets(Shape2(rois.size(0), 4));
  NonLinearTransformAndNormalization(rois_tmp, gt_bboxes_tmp, targets, bbox_mean, bbox_std);

  TensorContainer<cpu, 2, DType> bbox_target_data(Shape2(targets.size(0), 5));
  for (index_t i = 0; i < bbox_target_data.size(0); ++i) {
      if (class_agnostic){
        //class-agnostic regression class index = {0, 1}
//        bbox_target_data[i][0] = min(1.f, labels[i]);
        bbox_target_data[i][0] = labels[i]< static_cast<DType>(1) ? labels[i]:static_cast<DType>(1);
      }else{
        bbox_target_data[i][0] = labels[i];
      }
      Copy(bbox_target_data[i].Slice(1, 5), targets[i]);
  }

  ExpandBboxRegressionTargets(bbox_target_data, bbox_targets, bbox_weights, bbox_weight);

}

template <typename DType>
void BBoxOverlap(const Tensor<cpu, 2, DType> &boxes,
                 const Tensor<cpu, 2, DType> &query_boxes,
                 Tensor<cpu, 2, DType> &overlaps) {
    const index_t n = boxes.size(0);
    const index_t k = query_boxes.size(0);
    for (index_t j = 0; j < k; ++j) {
        DType query_box_area = (query_boxes[j][2] - query_boxes[j][0] + 1.f) * (query_boxes[j][3] - query_boxes[j][1] + 1.f);
        for (index_t i = 0; i < n; ++i) {
            DType iw = min(boxes[i][2], query_boxes[j][2]) - max(boxes[i][0], query_boxes[j][0]) + 1.f;
            if (iw > 0) {
                DType ih = min(boxes[i][3], query_boxes[j][3]) - max(boxes[i][1], query_boxes[j][1]) + 1.f;
                if (ih > 0) {
                    DType box_area = (boxes[i][2] - boxes[i][0] + 1.f) * (boxes[i][3] - boxes[i][1] + 1.f);
                    DType union_area = box_area + query_box_area - iw * ih;
                    overlaps[i][j] = iw * ih / union_area;
                }
            }
        }
    }
}

template <typename DType>
void ExpandBboxRegressionTargets(const Tensor<cpu, 2, DType> &bbox_target_data,
                                 Tensor<cpu, 2, DType> &bbox_targets,
                                 Tensor<cpu, 2, DType> &bbox_weights,
                                 const Tensor<cpu, 1, DType> &bbox_weight) {
  index_t num_bbox = bbox_target_data.size(0);
  for (index_t i = 0; i < num_bbox; ++i) {
    if (bbox_target_data[i][0] > 0) {
      index_t cls = bbox_target_data[i][0];
      index_t start = 4 * cls;
      index_t end = start + 4;
      Copy(bbox_targets[i].Slice(start, end), bbox_target_data[i].Slice(1, 5));
      Copy(bbox_weights[i].Slice(start, end), bbox_weight);
    }
  }
}

template <typename DType>
void NonLinearTransformAndNormalization(const Tensor<cpu, 2, DType> &ex_rois,
                                        const Tensor<cpu, 2, DType> &gt_rois,
                                        Tensor<cpu, 2, DType> &targets,
                                        const Tensor<cpu, 1, DType> &bbox_mean,
                                        const Tensor<cpu, 1, DType> &bbox_std) {
  index_t num_roi = ex_rois.size(0);
  for (index_t i = 0; i < num_roi; ++i) {
      DType ex_width  = ex_rois[i][2] - ex_rois[i][0] + 1.f;
      DType ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.f;
      DType ex_ctr_x  = ex_rois[i][0] + 0.5 * (ex_width - 1.f);
      DType ex_ctr_y  = ex_rois[i][1] + 0.5 * (ex_height - 1.f);
      DType gt_width  = gt_rois[i][2] - gt_rois[i][0] + 1.f;
      DType gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.f;
      DType gt_ctr_x  = gt_rois[i][0] + 0.5 * (gt_width - 1.f);
      DType gt_ctr_y  = gt_rois[i][1] + 0.5 * (gt_height - 1.f);
      targets[i][0]   = (gt_ctr_x - ex_ctr_x) / (ex_width + 1e-14f);
      targets[i][1]   = (gt_ctr_y - ex_ctr_y) / (ex_height + 1e-14f);
      targets[i][2]   = log(gt_width / ex_width);
      targets[i][3]   = log(gt_height / ex_height);
      targets[i] -= bbox_mean;
      targets[i] /= bbox_std;
  }
}

}  // namespace proposal_target_v2
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ProposalTargetParam_v2 param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalTargetOp_v2<cpu, DType>(param);
  })
  return op;
}

template<>
Operator *CreateOp<gpu>(ProposalTargetParam_v2 param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalTargetOp_v2<gpu, DType>(param);
  })
  return op;
}

Operator *ProposalTargetProp_v2::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ProposalTargetParam_v2);

MXNET_REGISTER_OP_PROPERTY(ProposalTarget_v2, ProposalTargetProp_v2)
.describe("C++ version proposal target v2")
.add_argument("rois", "Symbol", "rois")
.add_argument("gt_boxes", "Symbol", "gtboxes")
.add_argument("valid_ranges", "Symbol", "valid ranges for this scale")
.add_arguments(ProposalTargetParam_v2::__FIELDS__());

}  // namespace op
}  // namespace mxnet
