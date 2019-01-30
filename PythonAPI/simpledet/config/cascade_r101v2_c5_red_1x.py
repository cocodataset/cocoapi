from models.cascade_rcnn.builder import CascadeRcnn as Detector
from symbol.builder import MXNetResNet101V2C4C5 as Backbone
from models.cascade_rcnn.builder import CascadeNeck as Neck
from symbol.builder import RpnHead
from models.cascade_rcnn.builder import CascadeRoiAlign as RoiExtractor
from models.cascade_rcnn.builder import CascadeBbox2fcHead as BboxHead
from mxnext.complicate import normalizer_factory


def get_config(is_train):
    class General:
        log_frequency = 10
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_image = 2 if is_train else 1
        fp16 = False


    class KvstoreParam:
        kvstore     = "local"
        batch_image = General.batch_image
        gpus        = [0, 1, 2, 3, 4, 5, 6, 7]
        fp16        = General.fp16


    class NormalizeParam:
        # normalizer = normalizer_factory(type="syncbn", ndev=len(KvstoreParam.gpus))
        normalizer = normalizer_factory(type="fixbn")


    class BackboneParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer


    class NeckParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        conv_channel = 1024


    class RpnParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scale = (2, 4, 8, 16, 32)
            ratio = (0.5, 1.0, 2.0)
            stride = 16
            image_anchor = 256

        class head:
            conv_channel = 512
            mean = (0, 0, 0, 0)
            std = (1, 1, 1, 1)

        class proposal:
            pre_nms_top_n = 12000 if is_train else 6000
            post_nms_top_n = 2000 if is_train else 1000
            nms_thr = 0.7
            min_bbox_side = 0

        class subsample_proposal:
            proposal_wo_gt = True
            image_roi = 256
            fg_fraction = 0.25
            fg_thr = 0.5
            bg_thr_hi = 0.5
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 2
            class_agnostic = True
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)


    class BboxParam:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 256
        batch_image = General.batch_image
        stage       = "1st"

        class regress_target:
            class_agnostic = True
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.1, 0.1, 0.2, 0.2)
            loss_weight = 1.0

        class subsample_proposal:
            proposal_wo_gt = True
            image_roi = 256
            fg_fraction = 0.25
            fg_thr = 0.6
            bg_thr_hi = 0.6
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 2
            class_agnostic = True
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.05, 0.05, 0.1, 0.1)


    class BboxParam2nd:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 256
        batch_image = General.batch_image
        stage       = "2nd"

        class regress_target:
            class_agnostic = True
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.05, 0.05, 0.1, 0.1)
            loss_weight = 0.5

        class subsample_proposal:
            proposal_wo_gt = True
            image_roi = 256
            fg_fraction = 0.25
            fg_thr = 0.7
            bg_thr_hi = 0.7
            bg_thr_lo = 0.0

        class bbox_target:
            num_reg_class = 2
            class_agnostic = True
            weight = (1.0, 1.0, 1.0, 1.0)
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.033, 0.033, 0.067, 0.067)

    class BboxParam3rd:
        fp16        = General.fp16
        normalizer  = NormalizeParam.normalizer
        num_class   = 1 + 80
        image_roi   = 256
        batch_image = General.batch_image
        stage       = "3rd"

        class regress_target:
            class_agnostic = True
            mean = (0.0, 0.0, 0.0, 0.0)
            std = (0.033, 0.033, 0.067, 0.067)
            loss_weight = 0.25

        class subsample_proposal:
            proposal_wo_gt = None
            image_roi = None
            fg_fraction = None
            fg_thr = None
            bg_thr_hi = None
            bg_thr_lo = None

        class bbox_target:
            num_reg_class = None
            class_agnostic = None
            weight = None
            mean = None
            std = None


    class RoiParam:
        fp16 = General.fp16
        normalizer = NormalizeParam.normalizer
        out_size = 7
        stride = 16


    class DatasetParam:
        if is_train:
            image_set = ("coco_train2014", "coco_valminusminival2014")
        else:
            image_set = ("coco_minival2014", )

    backbone = Backbone(BackboneParam)
    neck = Neck(NeckParam)
    rpn_head = RpnHead(RpnParam)
    roi_extractor = RoiExtractor(RoiParam)
    bbox_head = BboxHead(BboxParam)
    bbox_head_2nd = BboxHead(BboxParam2nd)
    bbox_head_3rd = BboxHead(BboxParam3rd)
    detector = Detector()
    if is_train:
        train_sym = detector.get_train_symbol(
            backbone,
            neck,
            rpn_head,
            roi_extractor,
            bbox_head,
            bbox_head_2nd,
            bbox_head_3rd
        )
        test_sym = None
    else:
        train_sym = None
        test_sym = detector.get_test_symbol(
            backbone,
            neck,
            rpn_head,
            roi_extractor,
            bbox_head,
            bbox_head_2nd,
            bbox_head_3rd
        )


    class ModelParam:
        train_symbol = train_sym
        test_symbol = test_sym

        from_scratch = False
        random = True
        memonger = False
        memonger_until = "stage3_unit21_plus"

        class pretrain:
            prefix = "pretrain_model/resnet-101"
            epoch = 0
            fixed_param = ["conv0", "stage1", "gamma", "beta"]


    class OptimizeParam:
        class optimizer:
            type = "sgd"
            lr = 0.01 / 8 * len(KvstoreParam.gpus) * KvstoreParam.batch_image
            momentum = 0.9
            wd = 0.0001
            clip_gradient = 35

        class schedule:
            begin_epoch = 0
            end_epoch = 6
            lr_iter = [60000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image),
                       80000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)]

        class warmup:
            type = "gradual"
            lr = 0.0
            iter = 3000 * 16 // (len(KvstoreParam.gpus) * KvstoreParam.batch_image)


    class TestParam:
        min_det_score = 0.05
        max_det_per_image = 100

        process_roidb = lambda x: x
        process_output = lambda x, y: x

        class model:
            prefix = "experiments/{}/checkpoint".format(General.name)
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = "nms"
            thr = 0.5

        class coco:
            annotation = "data/coco/annotations/instances_minival2014.json"

    # data processing
    class ResizeParam:
        short = 800
        long = 1200 if is_train else 2000


    class PadParam:
        short = 800
        long = 1200
        max_num_gt = 100


    class AnchorTarget2DParam:
        class generate:
            short = 800 // 16
            long = 1200 // 16
            stride = 16
            scales = (2, 4, 8, 16, 32)
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
        RenameRecord, AnchorTarget2D

    if is_train:
        transform = [
            ReadRoiRecord(None),
            Resize2DImageBbox(ResizeParam),
            Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            AnchorTarget2D(AnchorTarget2DParam),
            RenameRecord(RenameParam.mapping)
        ]
        data_name = ["data", "im_info", "gt_bbox"]
        label_name = ["rpn_cls_label", "rpn_reg_target", "rpn_reg_weight"]
    else:
        transform = [
            ReadRoiRecord(None),
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
    # stage1 metric
    box_acc_metric_1st = metric.AccWithIgnore(
        "RcnnAcc_1st",
        ["bbox_cls_loss_1st_output", "bbox_label_blockgrad_1st_output"],
        []
    )
    box_l1_metric_1st = metric.L1(
        "RcnnL1_1st",
        ["bbox_reg_loss_1st_output", "bbox_label_blockgrad_1st_output"],
        []
    )
    # stage2 metric
    box_acc_metric_2nd = metric.AccWithIgnore(
        "RcnnAcc_2nd",
        ["bbox_cls_loss_2nd_output", "bbox_label_blockgrad_2nd_output"],
        []
    )
    box_l1_metric_2nd = metric.L1(
        "RcnnL1_2nd",
        ["bbox_reg_loss_2nd_output", "bbox_label_blockgrad_2nd_output"],
        []
    )
    # stage3 metric
    box_acc_metric_3rd = metric.AccWithIgnore(
        "RcnnAcc_3rd",
        ["bbox_cls_loss_3rd_output", "bbox_label_blockgrad_3rd_output"],
        []
    )
    box_l1_metric_3rd = metric.L1(
        "RcnnL1_3rd",
        ["bbox_reg_loss_3rd_output", "bbox_label_blockgrad_3rd_output"],
        []
    )

    metric_list = [
        rpn_acc_metric,
        rpn_l1_metric,
        box_acc_metric_1st,
        box_l1_metric_1st,
        box_acc_metric_2nd,
        box_l1_metric_2nd,
        box_acc_metric_3rd,
        box_l1_metric_3rd
    ]

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, \
           ModelParam, OptimizeParam, TestParam, \
           transform, data_name, label_name, metric_list
