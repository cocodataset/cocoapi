import os

from core.detection_module import DetModule
from core.detection_input import Loader
from utils.load_model import load_checkpoint

from six.moves import reduce
from six.moves.queue import Queue
from threading import Thread
import argparse
import importlib
import mxnet as mx
import numpy as np
import six.moves.cPickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    # general
    parser.add_argument('--config', help='config file path', type=str)
    args = parser.parse_args()

    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    return config


if __name__ == "__main__":
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    config = parse_args()

    pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
    transform, data_name, label_name, metric_list = config.get_config(is_train=False)

    sym = pModel.test_symbol

    image_sets = pDataset.image_set
    roidbs = [pkl.load(open("data/cache/{}.roidb".format(i), "rb"), encoding="latin1") for i in image_sets]
    roidb = reduce(lambda x, y: x + y, roidbs)
    roidb = pTest.process_roidb(roidb)
    for i, x in enumerate(roidb):
        x["rec_id"] = i

    loader = Loader(roidb=roidb,
                    transform=transform,
                    data_name=data_name,
                    label_name=label_name,
                    batch_size=1,
                    shuffle=False,
                    num_worker=4,
                    num_collector=2,
                    worker_queue_depth=2,
                    collector_queue_depth=2,
                    kv=None)

    data_names = [k[0] for k in loader.provide_data]

    execs = []
    for i in pKv.gpus:
        ctx = mx.gpu(i)
        arg_params, aux_params = load_checkpoint(pTest.model.prefix, pTest.model.epoch)
        mod = DetModule(sym, data_names=data_names, context=ctx)
        mod.bind(data_shapes=loader.provide_data, for_training=False)
        mod.set_params(arg_params, aux_params, allow_extra=False)
        execs.append(mod)

    all_outputs = []

    data_queue = Queue()
    result_queue = Queue()

    def eval_worker(exe, data_queue, result_queue):
        while True:
            batch = data_queue.get()
            exe.forward(batch, is_train=False)
            out = [x.asnumpy() for x in exe.get_outputs()]
            result_queue.put(out)

    workers = [Thread(target=eval_worker, args=(exe, data_queue, result_queue)) for exe in execs]
    for w in workers:
        w.daemon = True
        w.start()

    import time
    t1_s = time.time()

    for batch in loader:
        data_queue.put(batch)

    for _ in range(loader.total_record):
        r = result_queue.get()

        rid, id, info, post_cls_score, post_box, post_cls, mask = r
        rid, id, info, post_cls_score, post_box, post_cls, mask = rid.squeeze(), id.squeeze(), info.squeeze(), \
                                                                  post_cls_score.squeeze(), post_box.squeeze(), \
                                                                  post_cls.squeeze(), mask.squeeze()
        # TODO: POTENTIAL BUG, id or rid overflows float32(int23, 16.7M)
        id = np.asscalar(id)
        rid = np.asscalar(rid)

        scale = info[2]  # h_raw, w_raw, scale
        mask = mask[:, 1:, :, :] # remove bg
        post_box = post_box / scale  # scale to original image scale
        post_cls = post_cls.astype(np.int32)

        # remove pad bbox and mask
        valid_inds = np.where(post_cls > -1)[0]
        bbox_xyxy = post_box[valid_inds]
        cls_score = post_cls_score[valid_inds]
        cls = post_cls[valid_inds]
        mask = mask[valid_inds]

        output_record = dict(
            rec_id=rid,
            im_id=id,
            im_info=info,
            bbox_xyxy=bbox_xyxy,
            cls_score=cls_score,
            cls=cls,
            mask=mask
        )

        all_outputs.append(output_record)

    t2_s = time.time()
    print("network uses: %.1f" % (t2_s - t1_s))

    # let user process all_outputs
    all_outputs = pTest.process_output(all_outputs, roidb)

    t3_s = time.time()
    print("output processing uses: %.1f" % (t3_s - t2_s))

    output_dict = {}
    for rec in all_outputs:
        im_id = rec["im_id"]
        if im_id not in output_dict:
            output_dict[im_id] = dict(
                bbox_xyxy=[rec["bbox_xyxy"]],
                cls_score=[rec["cls_score"]],
                cls=[rec["cls"]],
                segm=[rec["segm"]]
            )
        else:
            output_dict[im_id]["bbox_xyxy"].append(rec["bbox_xyxy"])
            output_dict[im_id]["cls_score"].append(rec["cls_score"])
            output_dict[im_id]["cls"].append(rec["cls"])
            output_dict[im_id]["segm"].append(rec["segm"])
 
        output_dict[im_id]["bbox_xyxy"] = output_dict[im_id]["bbox_xyxy"][0]
        output_dict[im_id]["cls_score"] = output_dict[im_id]["cls_score"][0]
        output_dict[im_id]["cls"] = output_dict[im_id]["cls"][0]
        output_dict[im_id]["segm"] = output_dict[im_id]["segm"][0]


    t4_s = time.time()
    print("aggregate uses: %.1f" % (t4_s - t3_s))

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(pTest.coco.annotation)

    for k in output_dict:
        bbox_xyxy = output_dict[k]["bbox_xyxy"]
        cls_score = output_dict[k]["cls_score"]
        cls = output_dict[k]["cls"]
        segm = output_dict[k]["segm"]
        final_dets = {}
        final_segms = {}

        for cid in np.unique(cls):
            class_inds = np.where(cls == cid)[0]
            box = bbox_xyxy[class_inds]
            score = cls_score[class_inds]
            det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
            dataset_cid = coco.getCatIds()[cid]
            final_dets[dataset_cid] = det
            final_segms[dataset_cid] = segm[class_inds]
 
        del output_dict[k]["bbox_xyxy"]
        del output_dict[k]["cls_score"]
        del output_dict[k]["cls"]
        del output_dict[k]["segm"]
        output_dict[k]["det_xyxys"] = final_dets
        output_dict[k]["segmentations"] = final_segms

    t5_s = time.time()
    print("post process uses: %.1f" % (t5_s - t4_s))
 
    coco_result = []
    for iid in output_dict:
        result = []
        for cid in output_dict[iid]["det_xyxys"]:
            det = output_dict[iid]["det_xyxys"][cid]
            seg = output_dict[iid]["segmentations"][cid]
            if det.shape[0] == 0:
                continue
            scores = det[:, -1]
            xs = det[:, 0]
            ys = det[:, 1]
            ws = det[:, 2] - xs + 1
            hs = det[:, 3] - ys + 1
            result += [
                {'image_id': int(iid),
                 'category_id': int(cid),
                 'bbox': [float(xs[k]), float(ys[k]), float(ws[k]), float(hs[k])],
                 'score': float(scores[k]),
                 'segmentation': {"size": seg[k]["size"],
                                  "counts": seg[k]["counts"].decode("utf8")}}
                for k in range(det.shape[0])
            ]
        result = sorted(result, key=lambda x: x['score'])[-pTest.max_det_per_image:]
        coco_result += result

    t6_s = time.time()
    print("convert to coco format uses: %.1f" % (t6_s - t5_s))

    import json
    json.dump(coco_result,
              open("experiments/{}/{}_result.json".format(pGen.name, pDataset.image_set[0]), "w"),
              sort_keys=True, indent=2)

    ann_type = 'bbox'
    coco_dt = coco.loadRes(coco_result)
    coco_eval = COCOeval(coco, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ann_type = 'segm'
    coco_eval = COCOeval(coco, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    t7_s = time.time()
    print("coco eval uses: %.1f" % (t7_s - t6_s))
