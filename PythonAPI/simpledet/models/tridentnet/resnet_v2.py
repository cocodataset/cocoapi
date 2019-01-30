from __future__ import print_function

import mxnet as mx
import mxnext as X
from mxnext.backbone.resnet_v2 import Builder


bn_count = [10000]

class TridentResNetV2Builder(Builder):
    def __init__(self):
        super(TridentResNetV2Builder, self).__init__()

    @staticmethod
    def bn_shared(data, name, normalizer, branch_ids=None, share_weight=True):
        if branch_ids is None:
            branch_ids = range(len(data))

        gamma = X.var(name + "_gamma")
        beta = X.var(name + "_beta")
        moving_mean = X.var(name + "_moving_mean")
        moving_var = X.var(name + "_moving_var")

        bn_layers = []
        for i, data_i in zip(branch_ids, data):
            if share_weight:
                bn_i = normalizer(data=data_i, name=name + "_shared%d" % i,
                                  gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var)
            else:
                bn_i = normalizer(data=data_i, name=name + "_branch%d" % i)
            bn_layers.append(bn_i)

        return bn_layers

    @staticmethod
    def conv_shared(data, name, kernel, num_filter, branch_ids=None, no_bias=True, share_weight=True,
                    pad=(0, 0), stride=(1, 1), dilate=(1, 1)):
        if branch_ids is None:
            branch_ids = range(len(data))

        weight = X.var(name + '_weight')
        if no_bias:
            bias = None
        else:
            bias = X.var(name + '_bias')

        conv_layers = []
        for i in range(len(data)):
            data_i = data[i]
            stride_i = stride[i] if type(stride) is list else stride
            dilate_i = dilate[i] if type(dilate) is list else dilate
            pad_i = pad[i] if type(pad) is list else pad
            branch_i = branch_ids[i]
            if share_weight:
                conv_i = X.conv(data=data_i, kernel=kernel, filter=num_filter, stride=stride_i, dilate=dilate_i, pad=pad_i,
                                name=name + '_shared%d' % branch_i, no_bias=no_bias, weight=weight, bias=bias)
            else:
                conv_i = X.conv(data=data_i, kernel=kernel, filter=num_filter, stride=stride_i, dilate=dilate_i, pad=pad_i,
                                name=name + '_branch%d' % branch_i, no_bias=no_bias)
            conv_layers.append(conv_i)

        return conv_layers

    @staticmethod
    def deform_conv_shared(data, name, conv_offset, kernel, num_filter, branch_ids=None, no_bias=True, share_weight=True,
                           num_deformable_group=4, pad=(0, 0), stride=(1, 1), dilate=(1, 1)):
        if branch_ids is None:
            branch_ids = range(len(data))

        weight = X.var(name + '_weight')
        if no_bias:
            bias = None
        else:
            bias = X.var(name + '_bias')

        conv_layers = []
        for i in range(len(data)):
            data_i = data[i]
            stride_i = stride[i] if type(stride) is list else stride
            dilate_i = dilate[i] if type(dilate) is list else dilate
            pad_i = pad[i] if type(pad) is list else pad
            conv_offset_i = conv_offset[i] if type(conv_offset) is list else conv_offset
            branch_i = branch_ids[i]
            if share_weight:
                conv_i = mx.contrib.symbol.DeformableConvolution(
                    data=data_i, offset=conv_offset_i, kernel=kernel, num_filter=num_filter, stride=stride_i, num_deformable_group=4,
                    dilate=dilate_i, pad=pad_i, no_bias=no_bias, weight=weight, bias=bias, name=name + '_shared%d' % branch_i)
            else:
                conv_i = mx.contrib.symbol.DeformableConvolution(
                    data=data_i, offset=conv_offset_i, kernel=kernel, num_filter=num_filter, stride=stride_i, num_deformable_group=4,
                    dilate=dilate_i, pad=pad_i, no_bias=no_bias, name=name + '_branch%d' % branch_i)
            conv_layers.append(conv_i)

        return conv_layers

    @staticmethod
    def stack_branch_symbols(data_list):
        data = mx.symbol.stack(*data_list, axis=1)
        data = mx.symbol.Reshape(data, (-3, -2))

        return data

    @classmethod
    def resnet_trident_unit(cls, data, name, filter, stride, dilate, proj, norm_type, norm_mom, ndev,
                            branch_ids, branch_bn_shared, branch_conv_shared, branch_deform=False):
        """
        One resnet unit is comprised of 2 or 3 convolutions and a shortcut.
        :param data:
        :param name:
        :param filter:
        :param stride:
        :param dilate:
        :param proj:
        :param norm_type:
        :param norm_mom:
        :param ndev:
        :param branch_ids:
        :param branch_bn_shared:
        :param branch_conv_shared:
        :param branch_deform:
        :return:
        """
        if branch_ids is None:
            branch_ids = range(len(data))

        norm = X.normalizer_factory(type=norm_type, ndev=ndev, mom=norm_mom)

        bn1 = cls.bn_shared(
            data, name=name + "_bn1", normalizer=norm, branch_ids=branch_ids, share_weight=branch_bn_shared)
        relu1 = [X.relu(bn) for bn in bn1]
        conv1 = cls.conv_shared(
            relu1, name=name + "_conv1", num_filter=filter // 4, kernel=(1, 1),
            branch_ids=branch_ids, share_weight=branch_conv_shared)

        bn2 = cls.bn_shared(
            conv1, name=name + "_bn2", normalizer=norm, branch_ids=branch_ids, share_weight=branch_bn_shared)
        relu2 = [X.relu(bn) for bn in bn2]
        if not branch_deform:
            conv2 = cls.conv_shared(
                relu2, name=name + "_conv2", num_filter=filter // 4, kernel=(3, 3),
                pad=dilate, stride=stride, dilate=dilate,
                branch_ids=branch_ids, share_weight=branch_conv_shared)
        else:
            conv2_offset = cls.conv_shared(
                relu2, name=name + "_conv2_offset", num_filter=72, kernel=(3, 3),
                pad=(1, 1), stride=(1, 1), dilate=(1, 1), no_bias=False,
                branch_ids=branch_ids, share_weight=branch_conv_shared)
            conv2 = cls.deform_conv_shared(
                relu2, name=name + "_conv2", conv_offset=conv2_offset,  num_filter=filter // 4, kernel=(3, 3),
                pad=dilate, stride=stride, dilate=dilate, num_deformable_group=4,
                branch_ids=branch_ids, share_weight=branch_conv_shared)

        bn3 = cls.bn_shared(
            conv2, name=name + "_bn3", normalizer=norm, branch_ids=branch_ids, share_weight=branch_bn_shared)
        relu3 = [X.relu(bn) for bn in bn3]
        conv3 = cls.conv_shared(
            relu3, name=name + "_conv3", num_filter=filter, kernel=(1, 1),
            branch_ids=branch_ids, share_weight=branch_conv_shared)

        if proj:
            shortcut = cls.conv_shared(
                relu1, name=name + "_sc", num_filter=filter, kernel=(1, 1),
                branch_ids=branch_ids, share_weight=branch_conv_shared)
        else:
            shortcut = data

        return [X.add(conv3_i, shortcut_i, name=name + "_plus_branch{}".format(i)) \
                for i, conv3_i, shortcut_i in zip(branch_ids, conv3, shortcut)]

    @classmethod
    def resnet_trident_stage(cls, data, name, num_block, filter, stride, dilate, norm_type, norm_mom, ndev,
                             num_branch, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform):
        """
        One resnet stage is comprised of multiple resnet units. Refer to depth config for more information.
        :param data:
        :param name:
        :param num_block:
        :param filter:
        :param stride:
        :param dilate:
        :param norm_type:
        :param norm_mom:
        :param ndev:
        :param num_branch:
        :param branch_ids:
        :param branch_bn_shared:
        :param branch_conv_shared:
        :return:
        """
        assert isinstance(dilate, list) and len(dilate) == num_branch, 'dilate should be a list with num_branch items.'

        d = [(d, d) for d in dilate]

        data = cls.resnet_unit(data, "{}_unit1".format(name), filter, stride, 1, True, norm_type, norm_mom, ndev)
        data = [data] * num_branch
        for i in range(2, num_block + 1):
            if branch_deform and i >= num_block - 2:
                unit_deform = True
            else:
                unit_deform = False
            data = cls.resnet_trident_unit(
                data, "{}_unit{}".format(name, i), filter, (1, 1), d, False, norm_type, norm_mom, ndev,
                branch_ids, branch_bn_shared, branch_conv_shared, branch_deform=unit_deform)

        return data

    @classmethod
    def resnet_trident_c4(cls, data, num_block, stride, dilate, norm_type, norm_mom, ndev,
                          num_branch, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform):
        return cls.resnet_trident_stage(
            data, "stage3", num_block, 1024, stride, dilate, norm_type, norm_mom, ndev,
            num_branch, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform)

    @classmethod
    def resnet_factory(cls, depth, use_3x3_conv0, use_bn_preprocess,
                       num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                       norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        num_c2_unit, num_c3_unit, num_c4_unit, num_c5_unit = TridentResNetV2Builder.depth_config[depth]

        data = X.var("data")
        if fp16:
            data = X.to_fp16(data, "data_fp16")
        c1 = cls.resnet_c1(data, use_3x3_conv0, use_bn_preprocess, norm_type, norm_mom, ndev)
        c2 = cls.resnet_c2(c1, num_c2_unit, 1, 1, norm_type, norm_mom, ndev)
        c3 = cls.resnet_c3(c2, num_c3_unit, 2, 1, norm_type, norm_mom, ndev)
        c4 = cls.resnet_trident_c4(c3, num_c4_unit, 2, branch_dilates, norm_type, norm_mom, ndev,
                                   num_branch, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform)
        # stack branch features and merge into batch dim
        c4 = cls.stack_branch_symbols(c4)
        c5 = cls.resnet_c5(c4, num_c5_unit, 1, 2, norm_type, norm_mom, ndev)

        return c1, c2, c3, c4, c5

    @classmethod
    def resnet_c4_factory(cls, depth, use_3x3_conv0, use_bn_preprocess,
                          num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                          norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess,
                                                num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                                                norm_type, norm_mom, ndev, fp16)

        return c4

    @classmethod
    def resnet_c4c5_factory(cls, depth, use_3x3_conv0, use_bn_preprocess,
                            num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                            norm_type="local", norm_mom=0.9, ndev=None, fp16=False):
        c1, c2, c3, c4, c5 = cls.resnet_factory(depth, use_3x3_conv0, use_bn_preprocess,
                                                num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                                                norm_type, norm_mom, ndev, fp16)
        c5 = X.fixbn(c5, "bn1")
        c5 = X.relu(c5)

        return c4, c5

    def get_backbone(self, variant, depth, endpoint, normalizer, fp16,
                     num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform):
        # parse variant
        if variant == "mxnet":
            use_bn_preprocess = True
            use_3x3_conv0 = False
        elif variant == "tusimple":
            use_bn_preprocess = False
            use_3x3_conv0 = True
        else:
            raise KeyError("Unknown backbone variant {}".format(variant))

        # parse endpoint
        if endpoint == "c4":
            factory = self.resnet_c4_factory
        elif endpoint == "c4c5":
            factory = self.resnet_c4c5_factory
        else:
            raise KeyError("Unknown backbone endpoint {}".format(endpoint))

        return factory(depth, use_3x3_conv0, use_bn_preprocess,
                       num_branch, branch_dilates, branch_ids, branch_bn_shared, branch_conv_shared, branch_deform,
                       norm_type=normalizer, fp16=fp16)
