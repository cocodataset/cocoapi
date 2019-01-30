## Mask-RCNN

This repository implements [**Mask-RCNN**](https://arxiv.org/abs/1703.06870) in the SimpleDet framework.
Mask-RCNN is a simple and effective approach for object instance segmentation. By simply extending Faster-RCNN with a mask branch, Mask-RCNN can generate a high-quality segmentation mask for each instance. In the following, we will introduce how we build Mask-RCNN in the SimpleDet framework. Currently, we only provide FPN based Mask-RCNN.

### Qucik Start
```bash
# train
python3 detection_train.py --config config/mask_r50v1_fpn_1x.py

# test
python3 mask_test.py --config config/mask_r50v1_fpn_1x.py
```
### How we build Mask-RCNN
#### Input
First, we need mask label.

Instead of providing binary masks to the network, we adopt poly format in the current implementation. Since each instance may contain several parts, we adopt a list of lists ([[ax1, ay1, ax2, ay2,...], [bx1, by1, bx2, by2,...], ...) to represent each instance following COCO. For simplicity, we note [ax1, ay1, ax2, ay2, ...] as a segm.

We implement these transforms for poly format mask label:
- **PreprocessGtPoly**: convert each segm in a instance into ndarray.
- **EncodeGtPoly**: encode each instance into a fixed length format ([class_id, num_segms, len_segm1, len_segm2, segm1, segm2]).

For data augmentation, we extend several transfroms from Faster-RCNN:
- **Resize2DImageBboxMask**: based on **Resize2DImageBbox**
- **Flip2DImageBboxMask**: based on **Flip2DImageBbox**
- **Pad2DImageBboxMask**: based on **Pad2DImageBbox**

#### Operators
Then, we extend proposal_target to get sampled mask target for mask branch training:
- **proposal_mask_target**, decodes encoded gt poly into binary mask and samples a fixed amount of masks as mask target. For acceleration, we only provide mask target for fg roi. So the number of mask target is ```int(image_roi * fg_fraction)```. Currently we only support class specific mask target. So the shape of mask target is ```(batch_size, int(image_roi * fg_fraction), num_class (81 in COCO), mask_size, mask_size)```.

In order to test mask in an end-to-end manner, we reuses the code from detection_test.py and implement a bbox post processing operator:
- **bbox_post_processing**, adopts NMS for multi-class bbox and get final bbox results.

For loss function, we implement sigmoid cross entropy:
- **sigmoid_cross_entropy**, a general sigmoid cross entropy loss function.

#### Symbol
- **MaskFasterRcnn**, detector for MaskRCNN
- **MaskFPNRpnHead**, a new RpnHead inherited from FPNRpnHead, note that we slice the proposal sampled from proposal_mask_target since the mask target provided by this operator is only for fg roi.
- **MaskFasterRcnnHead**, mask head for MaskRCNN
- **MaskFasterRcnn4ConvHead**, a specific mask head with 4 convolutions.
- **BboxPostProcessor**, a bbox post processor for end-to-end test.

### How to build Mask-RCNN without FPN
- Implement **MaskRpnHead** following **MaskFPNRpnHead**.
- Implement your own MaskHead following **MaskFasterRcnn4ConvHead**
- Write your own config following **mask_r50v1_fpn_1x.py** and **faster_r50v1c4_c5_512roi_1x.py**

