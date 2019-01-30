## Cascade R-CNN

This repository implements [**Cascade R-CNN**](https://arxiv.org/abs/1712.00726) in the **SimpleDet** framework. Cascade R-CNN is a multi-stage object detector, aiming to reduce the overfitting problem by resampling of progressively improved hypotheses. Currently, we only provide Cascade R-CNN based on Faster R-CNN without FPN.

### How we build Cascade R-CNN

#### Input

Cascade R-CNN can share the origin Faster R-CNN input, so there is no need to implement an extra one.

#### Symbol

- ```CascadeRcnn```: detector with three ```R-CNN``` stages

- ```CascadeNeck```: neck to reduce feature channels for speedup, default ```conv_channel``` is set to 1024
- ```CascadeRoiAlign```: ```RoiExtractor``` for ```R-CNN``` stages
- ```CascadeBbox2fcHead```: header for ```R-CNN``` stages. Note that it is also required to generate proposal for next ```R-CNN``` stages, thus we add ```get_all_proposal``` to decode boxes predicted in this stage and ```get_sampled_proposal``` to generate ```bbox_target```.

#### Config

- ```BboxParam```, ```BboxParam2nd```, ```BboxParam3rd```: config for ```R-CNN``` stages, ```mean``` and ```std``` in ```regress_target``` aim to decode boxes predicted in this stage, and those in ```bbox_target``` is prepared to generate ```bbox_target``` for next ```R-CNN``` stage. Note that we add ```stage``` field to specify the weight used by ```R-CNN```, as in **test phase** ```bbox_head_1st``` and ```bbox_head_2nd``` forward twice with different input feature.

