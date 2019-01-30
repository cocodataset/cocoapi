import unittest
import mxnet as mx

from six.moves import cPickle as pkl
from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
    ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, \
    RenameRecord, AnchorTarget2D, AnchorLoader
from config import detection_config


class TestLoader(unittest.TestCase):

    def test_empty_v_loader(self):
        pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = detection_config.get_config(is_train=True)
        roidbs = pkl.load(open("unittest/data/coco_micro_test.roidb", "rb"), encoding="latin1")
        all_v_roidbs = [roidb for roidb in roidbs if roidb['h'] >= roidb['w']]

        loader = AnchorLoader(
            roidb=all_v_roidbs,
            transform=transform,
            data_name=data_name,
            label_name=label_name,
            batch_size=1,
            shuffle=True,
            num_thread=1,
            kv=mx.kvstore.create(pKv.kvstore)
        )
        with self.assertRaises(StopIteration):
            while True:
                data_batch = loader.next()

    def test_empty_h_loader(self):
        pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = detection_config.get_config(is_train=True)
        roidbs = pkl.load(open("unittest/data/coco_micro_test.roidb", "rb"), encoding="latin1")
        all_h_roidbs = [roidb for roidb in roidbs if roidb['h'] < roidb['w']]

        loader = AnchorLoader(
            roidb=all_h_roidbs,
            transform=transform,
            data_name=data_name,
            label_name=label_name,
            batch_size=1,
            shuffle=True,
            num_thread=1,
            kv=mx.kvstore.create(pKv.kvstore)
        )
        with self.assertRaises(StopIteration):
            while True:
                data_batch = loader.next()

    def test_record_num(self):
        pGen, pKv, pRpn, pRoi, pBbox, pDataset, pModel, pOpt, pTest, \
        transform, data_name, label_name, metric_list = detection_config.get_config(is_train=True)
        roidbs = pkl.load(open("unittest/data/coco_micro_test.roidb", "rb"), encoding="latin1")
        batch_size = 4

        loader = AnchorLoader(
            roidb=roidbs,
            transform=transform,
            data_name=data_name,
            label_name=label_name,
            batch_size=batch_size,
            shuffle=True,
            num_thread=1,
            kv=mx.kvstore.create(pKv.kvstore)
        )

        num_batch = 0
        while True:
            try:
                data_batch = loader.next()
                num_batch += 1
            except StopIteration:
                break
        self.assertEqual(batch_size * num_batch, loader.total_record)


