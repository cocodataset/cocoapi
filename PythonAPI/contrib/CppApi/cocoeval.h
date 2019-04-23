// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_

#include <map>
#include <set>
#include "absl/container/flat_hash_map.h"
#include "third_party/py/pycocotools/coco.h"

namespace pycocotools {

typedef std::vector<std::vector<float>> IoUMatrix;

//  Params for coco evaluation api
struct Params {
 public:
  Params(std::string input_iou_type = "bbox");
  void SetDetParams();
  std::string iou_type;
  std::vector<int64_t> img_ids;
  std::vector<int64_t> cat_ids;
  std::vector<double> iou_thrs;
  std::vector<double> rec_thrs;
  std::vector<int64_t> max_dets;
  std::vector<std::vector<float>> area_rng;
  std::vector<std::string> area_rng_lbl;
  bool use_cats;
};

// Eval results for each image.
struct EvalImgs {
  int64_t image_id;
  int64_t category_id;
  int64_t max_det;
  // dt_matches - [TxD] matching gt id at each IoU or 0
  std::vector<int64_t> dt_matches;
  // dt_scores   - [1xD] confidence of each dt
  std::vector<double> dt_scores;
  // gt_ignore   - [1xG] ignore flag for each gt
  std::vector<bool> gt_ignore;
  // dt_ignore   - [TxD] ignore flag for each dt at each IoU
  std::vector<bool> dt_ignore;
};

struct Eval {
  Params params;
  std::vector<int64_t> counts;
  std::vector<float> precision;
  std::vector<float> recall;
};

// Interface for evaluating detection on the Microsoft COCO dataset.
//
// The usage for CocoEval is as follows:
//  COCO coco_gt=..., coco_dt =...       // load dataset and results
//  COCOeval E = Cocoeval(coco_gt,coco_dt); # initialize CocoEval object
//  E.Evaluate();                // run per image evaluation
//  E.Accumulate();              // accumulate per image results
//  E.Summarize();               // display summary metrics of results
//
// The evaluation parameters are as follows (defaults in brackets):
//  img_ids     - [all] N img ids to use for evaluation
//  cat_ids     - [all] K cat ids to use for evaluation
//  iou_thrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
//  rec_thrs    - [0:.01:1] R=101 recall thresholds for evaluation
//  area_rng    - [...] A=4 object area ranges for evaluation
//  max_dets    - [1 10 100] M=3 thresholds on max detections per image
//  iou_type    - ['bbox'] set iouType to 'segm', 'bbox' or 'keypoints'
//  use_cats    - [1] if true use category labels for evaluation
//  Note: if use_cats == 0 category labels are ignored as in proposal scoring.
//  Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
//
//  Evaluate(): evaluates detections on every image and every category and
//  concats the results into the "evalImgs" with fields:
//   dt_ids      - [1xD] id for each of the D detections (dt)
//   gt_ids      - [1xG] id for each of the G ground truths (gt)
//   dt_matches  - [TxD] matching gt id at each IoU or 0
//   gt_matches  - [TxG] matching dt id at each IoU or 0
//   dt_scores   - [1xD] confidence of each dt
//   gt_ignore  - [1xG] ignore flag for each gt
//   dt_ignore  - [TxD] ignore flag for each dt at each IoU
//
//  Accumulate(): accumulates the per-image, per-category evaluation
//  results in "evalImgs" into the dictionary "eval" with fields:
//   params     - parameters used for evaluation
//   counts     - [T,R,K,A,M] parameter dimensions (see above)
//   precision  - [TxRxKxAxM] precision for every evaluation setting
//   recall     - [TxKxAxM] max recall for every evaluation setting
//   Note: precision and recall==-1 for settings with no gt objects.
class COCOeval {
 public:
  // For test only.
  COCOeval() {}
  COCOeval(COCO &coco_gt, COCO &coco_dt, std::string iou_type)
      : coco_gt_(coco_gt), coco_dt_(coco_dt), iou_type_(iou_type) {
    params_ = Params(iou_type);
    std::vector<int64_t> img_ids, cat_ids;
    std::vector<std::string> cat_names, sup_names;
    params_.img_ids = coco_gt_.GetImgIds(img_ids, cat_ids);
    params_.cat_ids = coco_gt_.GetCatIds(cat_names, sup_names, cat_ids);
    cat_id_size_ = params_.cat_ids.back();
  }

  void CocoEvalClean(){
    coco_gt_.CocoFreeMem();
    coco_dt_.CocoFreeMem();
  }
  // Computes IoU of each predicted detections dt and ground truth
  // detections dt correspond to a img_id and cat_id,
  // complexity: O(dt.size() * gt.size()).
  std::vector<std::vector<float>> ComputeIoU(int64_t img_id, int64_t cat_id);
  // evaluates detections on img_id and every cat_id on each threshold level and
  // concats the results into the evalImgs.
  // Complexity: O(T * D * G)
  // T, D, G see above.
  void EvaluateImg(int64_t img_id, int64_t cat_id,
                   std::vector<std::vector<float>> &area_rng, int64_t max_det);
  // Run per image evaluation on given images and store results in eval_imgs_
  // return: None
  // Complexity: O (#images * #categories * T * D * G)
  void Evaluate();

  // accumulates the per-image, per-category evaluation
  // results in "evalImgs" into the dictionary "eval" with fields:
  // params     - parameters used for evaluation
  // date       - date evaluation was performed
  // counts     - [T,R,K,A,M] parameter dimensions (see above)
  // precision  - [TxRxKxAxM] precision for every evaluation setting
  // recall     - [TxKxAxM] max recall for every evaluation setting
  // Complexity: O(T * R * K * A * M)
  void Accumulate();

  void Summarize();

  std::vector<float> GetStats() { return stats_; }

 private:
  // Prepare gts_ and dts_ for evaluation based on params
  // :return: None
  // Only consider "bbox" type.
  void Prepare();
  std::pair<float, std::vector<float>> SummarizeInternal(
      bool ap = true, double iou_thr = -1.0, std::string area_rng = "all",
      int64_t max_dets = 100);
  void SummarizeDets();
  size_t Key(int64_t image_id, int64_t cat_id) {
    return image_id * cat_id_size_ + cat_id;
  }

  // ground truth COCO API.
  COCO coco_gt_;
  // detections COCO API
  COCO coco_dt_;
  // iou_type.
  std::string iou_type_;
  // evaluation parameters.
  Params params_;
  // per-image per-category evaluation results [KxAxI] elements.
  std::vector<EvalImgs> eval_imgs_;
  // accumulated evaluation results.
  Eval eval_;
  // gt for evaluation.
  absl::flat_hash_map<int64_t, std::vector<Annotation>> gts_;
  // dt for evaluation.
  absl::flat_hash_map<int64_t, std::vector<Annotation>> dts_;
  // parameters for evaluation.
  Params params_eval_;
  // result summarization.
  std::vector<float> stats_;
  // category result
  std::vector<std::vector<float>> category_stats_;
  // ious between all gts and dts.
  absl::flat_hash_map<int64_t, IoUMatrix> ious_;
  int64_t cat_id_size_;
};
}  // namespace pycocotools

#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_COCOEVAL_H_
