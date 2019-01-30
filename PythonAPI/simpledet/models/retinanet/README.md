## RetinaNet

This repository implements [**RetinaNet**](https://arxiv.org/abs/1708.02002) in the **SimpleDet** framework. RetinaNet is state-of-the-art single stage detector, preventing the vast number of easy negatives from overwhelming the detector with focal loss.

### How we build RetinaNet

#### Input

The pyramid label parts of **RetinaNet** is similar with **Feature Pyramid Network**, you can refer to [FPN README](../FPN/README.md) . In addition, the label assignment method is different compared with **Faster R-CNN**, thus we overrides ```_assign_label_to_anchor``` and ```apply``` of ```AnchorTarget2D```, named ```PyramidAnchorTarget2DBase```, to obtain class-aware labels and avoid sampling RoIs.

#### Operators

- **bbox_norm**, passes data in forward, and normalizes gradient by number of positive samples in backward
- **focal_loss**, acts same as Sigmoid in forward, and return focal loss gradient in backward
- **decode_retina**, reuses the code from [**Detectron**](https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test_retinanet.py) to decode boxes and scores. Note that ```min_det_score``` is moved to ```RpnParam.proposal``` as it requires different threshold for results from **P7** level.

#### Symbol

- ``` RetinaNet```, detector only with RPN
- ```RetinaNetHead```, classification and regression header with sharing weights
- ```RetinaNetNeck```, top-down pathway for **FPN** in **RetinaNet**

#### Config

- ```min_det_score``` in ```TestParam``` is set to 0 to remove those appended boxes with zero scores
- To avoid sharing parameter of the same field from config in pyramid levels, i.e. ```stride```, we move ```stride```, ```long``` and ```short``` to ```__init__```, and pass ```AnchorTarget2DParam``` instance rather than class for anchor generation.
