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
#ifndef THIRD_PARTY_PY_PYCOCOTOOLS_COCO_H_
#define THIRD_PARTY_PY_PYCOCOTOOLS_COCO_H_

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include "include/json/json.h"
#include "numpy/core/include/numpy/arrayobject.h"

extern "C" {
  #include "maskApi.h"
}

namespace pycocotools {

struct Annotation {
  float area = 0.0;
  bool iscrowd = false;
  int64_t image_id;
  int64_t category_id;
  int64_t id;
  std::vector<float> bbox;
  // Compressed RLE format
  RLE* R = nullptr;
  bool ignore = false;
  bool uignore = false;
  float score = 0.0;
};

class COCO {
 public:
  COCO(std::string annotation_file = "", bool use_mask = false);

  void CocoFreeMem(){
    for (auto ann : anns_vec_) {
      if (ann.R != nullptr) {
        rleFree(ann.R);
        delete ann.R;
      }
    }
  }
  void CreateIndex();
  void CreateIndexWithAnns();
  // Get ann ids that satisfy given filter conditions. default skips that filter
  // param: img_ids  (int array)     : get anns for given imgs
  // param: cat_ids  (int array)     : get anns for given cats
  // param: area_area (float array)   : get anns for given area range (e.g. [0
  // inf]) param: is_crowd (boolean)       : get anns for given crowd label
  // (False or True) return: ids (int array)       : integer array of ann ids
  std::vector<int64_t> GetAnnIds(std::vector<int64_t>& img_ids,
                                 std::vector<int64_t>& cat_ids,
                                 std::vector<float>& area_rng, bool is_crowd,
                                 bool respect_is_crowd = false);

  // filtering parameters. default skips that filter.
  // param: cat_names (str array)  : get cats for given cat names
  // param: sup_names (str array)  : get cats for given supercategory names
  // param: cat_ids (int array)  : get cats for given cat ids
  // return: ids (int array)   : integer array of cat ids
  std::vector<int64_t> GetCatIds(std::vector<std::string>& cat_names,
                                 std::vector<std::string>& sup_names,
                                 std::vector<int64_t>& cat_ids);

  // Get img ids that satisfy given filter conditions.
  // param: img_ids (int array) : get imgs for given ids
  // param: cat_ids (int array) : get imgs with all given cats
  // return: ids (int array)  : integer array of img ids
  std::vector<int64_t> GetImgIds(std::vector<int64_t>& img_ids,
                                 std::vector<int64_t>& cat_ids);

  // Load anns with the specified ids.
  // param: ids (int array)       : integer ids specifying anns
  // return: anns (object array) : loaded ann objects
  std::vector<Annotation> LoadAnns(std::vector<int64_t>& ids);

  // Load cats with the specified ids.
  // param: ids (int array)       : integer ids specifying cats
  // return: cats (object array) : loaded cat objects
  std::vector<Json::Value> LoadCats(std::vector<int64_t>& ids);

  // Load imgs with the specified ids.
  // param: ids (int array)       : integer ids specifying img
  // return: imgs (object array) : loaded img objects
  std::vector<Json::Value> LoadImgs(std::vector<int64_t>& ids);

  // Load result numpy array and return a result api object.
  // param:   py_array    :  numpy array of the result
  // return: res (obj)         : result api object
  // Only supports 'bbox' mode.
  COCO LoadRes(PyObject* py_object);
  COCO LoadResJson(std::string annotation_file);
  COCO LoadResMask(PyObject* py_object, PyObject* py_mask_object);

  // Convert result data from a numpy array [Nx7] where each row contains
  // {imageID,x1,y1,w,h,score,class}
  // param:  py_array (numpy.ndarray)
  // return: annotations (Json Value array)
  std::vector<Annotation> LoadNumpyAnnotations(
      PyArrayObject* py_array, PyObject* py_mask_array = nullptr);

  // Set value in dataset_;
  void SetMember(std::string name, Json::Value& value);

  // Set Ann vec_;
  void SetAnn(std::vector<Annotation>& anns) { anns_vec_ = anns; }

 private:
  Annotation Convert(const Json::Value& json_item);
  Json::Value dataset_;
  std::map<int64_t, int64_t> anns_;
  std::vector<Annotation> anns_vec_;
  std::map<int64_t, Json::Value> cats_;
  std::map<int64_t, Json::Value> imgs_;
  std::map<int64_t, std::vector<int64_t>> img_to_anns_;
  std::map<int64_t, std::vector<int64_t>> cat_to_imgs_;
  bool use_mask_;
};
}  // namespace pycocotools

#endif  // THIRD_PARTY_PY_PYCOCOTOOLS_COCO_H_
