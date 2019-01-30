import numpy as np
import mxnet as mx
import glob


def get_latest_ckpt_epoch(prefix):
    """
    Get latest checkpoint epoch by prefix
    """
    def get_checkpoint_epoch(prefix):
        return int(prefix[prefix.rfind('.params')-4:prefix.rfind('.params')])

    checkpoints = glob.glob(prefix + '*.params')
    assert len(checkpoints), 'can not find params startswith {}'.prefix
    return max([get_checkpoint_epoch(x) for x in checkpoints])


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    print('load %s-%04d.params' % (prefix, epoch))
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def convert_context(params, ctx):
    """
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params

