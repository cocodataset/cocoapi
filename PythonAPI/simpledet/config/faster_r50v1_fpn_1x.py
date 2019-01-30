from symbol.builder import FasterRcnn as Detector
from models.FPN.builder import MSRAResNet50V1FPN as FPNBackbone
from models.FPN.builder import FPNConvTopDown
from models.FPN.builder import FPNRpnHead
from models.FPN.builder import FPNRoiAlign as FPNRoiExtractor
from models.FPN.builder import FPNBbox2fcHead as Bbox2fcHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = False


    class KvstoreParam:
        kvstore     = "nccl"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scale = (8,)
            ratio = (0.5, 1.0, 2.0)
            stride = [64, 32, 16, 8, 4]
            image_anchor = 256

        class head:
            conv_channel = 256
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 2000 if is_train else 1000
            post_nms_top_n = 2000 if is_train else 1000
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = False
            image_roi = 512
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 81
            class_agnostic = False
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class BboxParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 512
        batch_image = General.batch_image

        class regress_target:
            class_agnostic = False
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 7
        stride = [32, 16, 8, 4]


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014")
        else:
            image_set = ("coco_minival2014", )

    backbone = FPNBackbone(BackboneParam)
    neck = FPNConvTopDown(NeckParam)
    rpn_head = FPNRpnHead(RpnParam)
    roi_extractor = FPNRoiExtractor(RoiParam)
    bbox_head = Bbox2fcHead(BboxParam)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(backbone, neck, rpn_head, roi_extractor, bbox_head)


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-v1-50"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = None

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image / 3.0
            iter = 500


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = 6

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "data/coco/annotations/instances_minival2014.json"


    # data processing
    class NormParam:
        mean = (122.7717, 115.9465, 102.9801) # RGB order
        std = (1.0, 1.0, 1.0)

    # data processing
    class ResizeParam:
        short = 800
        long = 1333


    class PadParam:
        short = 800
        long = 1333
        max_num_gt = 100


    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate:
            def __init__(self):
                self.stride = (4, 8, 16, 32, 64)
                self.short = (200, 100, 50, 25, 13)
                self.long = (334, 167, 84, 42, 21)
            scales = (8)
            aspects = (0.5, 1.0, 2.0)

        class assign:
            allowed_border = 0
            pos_thr = 0.7
            neg_thr = 0.3
            min_pos_thr = 0.0

        class sample:
            image_anchor = 256
            pos_fraction = 0.5


    class RenameParam:
        mapping = dict(image="data")


    from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
        ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
        RenameRecord, Norm2DImage
    
    from models.FPN.input import PyramidAnchorTarget2D

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            PyramidAnchorTarget2D(AnchorTarget2DParam()),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
    else:
        transform = [
            ReadRoiRecord(None),
            Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            ConvertImageFromHwcToChw(),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "im_id", "rec_id"]
        label_name = []

    import core.detection_metric as metric

    rpn_acc_metric = metric.AccWithIgnore(
        "RpnAcc",
        ["rpn_cls_loss_output"],
        ["rpn_cls_label"]
    )
    rpn_l1_metric = metric.L1(
        "RpnL1",
        ["rpn_reg_loss_output"],
        ["rpn_cls_label"]
    )
    # for bbox, the label is generated in network so it is an output
    box_acc_metric = metric.AccWithIgnore(
        "RcnnAcc",
        ["bbox_cls_loss_output", "bbox_label_blockgrad_output"],
        []
    )
    box_l1_metric = metric.L1(
        "RcnnL1",
        ["bbox_reg_loss_output", "bbox_label_blockgrad_output"],
        []
    )

    metric_list = [rpn_acc_metric, rpn_l1_metric, box_acc_metric, box_l1_metric]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
