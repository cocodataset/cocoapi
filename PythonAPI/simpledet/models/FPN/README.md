## Feature Pyramid Networks for Object Detection

Here we introduce how is [**Feature Pyramid Network**](https://arxiv.org/abs/1612.03144) built in **simpledet** framework. The following sections explain detail implementation.

#### AnchorTarget

Since **FPN** uses **Feature Pyramid** as backbone,  we cannot use ```AnchorTarget2D``` directly, which only generates anchor target for single stride declared in ```RpnParam```. Instead, we implement ```PyramidAnchorTarget2D``` to create a list of ```AnchorTarget2D```, each generating anchor target for single pyramid stride, then collect them together. More specifically, we create instances for each pyramid stride to generate anchor. To collect anchors from different pyramid levels, we overrides ```v_all_anchor``` and ```h_all_anchor``` property, which returns the concatenation of anchors from different levels, then assign to primary instances. Also, we override ```apply``` function to obtain label, sample anchor, target and weight from primary instances, then split and concat them in a certain axis.

#### Operators

- **get_topk_proposal**, since **FPN** has mutli-scale proposals, we should concat the multi-scale proposals together and get the topK proposals for roi-pooling or roi-align
- **assign_layer_fpn**, **FPN** assign the proposals to target levels(P2, P3, P4, P5) according to the areas, so we use this Operator to assign feature levels for proposals


#### Symbol

- ``` Detector```, detector is the same as FasterRcnn
- ```FPNConvTopDown```, top-down pathway for **Feature Pyramid Network**
- ```FPNRpnHead```, classification and regression header with sharing weights for FPN-RPN
- ```FPNRoiAlign```, we use this module to get the proposal feature for the proposals of different levels respectively, then add the feature from different level proposals together for next rcnn head

#### Config

- ```TestParam``` is the same as the setting in [**Detectron**](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)
- To avoid sharing parameter of the same field from config in pyramid levels, i.e. ```stride```, we move ```stride```, ```long``` and ```short``` to ```__init__```, and pass ```AnchorTarget2DParam``` instance rather than class for anchor generation.
