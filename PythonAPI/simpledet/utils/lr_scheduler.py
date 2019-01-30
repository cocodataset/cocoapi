import logging
from mxnet.lr_scheduler import LRScheduler

class WarmupMultiFactorScheduler(LRScheduler):
    def __init__(self, step, factor=1, warmup=False, warmup_type='constant', warmup_lr=0, warmup_step=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        if warmup:
            if warmup_step >= step[0]:
                raise ValueError("Warmup step must be smaller than schedule step")
            if warmup_type not in ['constant', 'gradual']:
                raise ValueError("Warmup scheduler only support constant or gradual")

        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_type = warmup_type
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

    def __call__(self, num_update):
        if self.warmup and num_update <= self.warmup_step:
            if self.warmup_type == 'constant':
                return self.warmup_lr
            elif self.warmup_type == 'gradual':
                return (self.base_lr - self.warmup_lr) / self.warmup_step * num_update + self.warmup_lr

        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr
