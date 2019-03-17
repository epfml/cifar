# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer


def define_scheduler(args, optimizer):
    # define the learning rate scheduler.
    scheduler = LRScheduler(args, optimizer)
    return scheduler


# define the learning rate scheduler.
class LRScheduler(object):
    def __init__(self, args, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        # init
        self.args = args
        self.optimizer = optimizer
        self.local_index = 0
        self.init_learning_rate()
        self.init_scheduler()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def init_learning_rate(self):
        # init the learning rates.
        self.init_warmup_lr = self.args.lr
        self.learning_rate_per_samples = self.args.lr / self.args.base_batch_size

        if self.args.lr_scaleup:
            if self.args.lr_scaleup_type == 'linear':
                _lr = self.learning_rate_per_samples * self.args.batch_size
                _scale = self.args.graph.n_nodes
            elif self.args.lr_scaleup_type == 'sqrt':
                _lr = self.args.lr
                _scale = (
                    1. * self.args.graph.n_nodes * self.args.batch_size /
                    self.args.base_batch_size) ** 0.5
            else:
                raise NotImplementedError
        else:
            _lr = self.learning_rate_per_samples * self.args.batch_size
            _scale = 1

        # get the eventual learning the backup.
        self.learning_rate = _lr * _scale
        self.old_learning_rate = self.learning_rate

    def init_scheduler(self):
        self.scheduler = Scheduler(self.args).get_lr_scheduler()

    def get_lr(self):
        return self.scheduler(self.epoch_)

    def step(self):
        self.update_training_progress()

        # get the new learning rate.
        lr = self.get_lr()
        if lr is None:
            lr = self.old_learning_rate

        # apply the new learning rate.
        if self.old_learning_rate != lr:
            self.old_learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def update_training_progress(self):
        self.local_index += 1
        self.epoch_ = self.local_index / self.args.num_batches_train_per_device_per_epoch
        self.epoch = int(self.epoch_)

    def is_stop(self):
        if self.args.stop_criteria == 'epoch':
            return self.epoch >= self.args.num_epochs
        elif self.args.stop_criteria == 'iteration':
            return self.local_index >= self.args.num_iterations_per_worker


class Scheduler(object):
    def __init__(self, args):
        self.args = args

    def get_lr_scheduler(self):
        epoch_fields, lr_fields, scale_indicators = self.get_scheduling_setup()
        lr_schedulers = self.build_lr_schedulers(epoch_fields, lr_fields, scale_indicators)
        return self._get_lr_scheduler(epoch_fields, lr_schedulers)

    def _get_lr_scheduler(self, epoch_fields, lr_schedulers):
        def f(epoch_index):
            def _is_fall_in(index, left_index, right_index):
                return left_index <= index < right_index

            for ind, (epoch_left, epoch_right) in enumerate(epoch_fields):
                if _is_fall_in(epoch_index, epoch_left, epoch_right):
                    return lr_schedulers[ind](epoch_index)
        return f

    def get_scheduling_setup(self):
        if self.args.lr_schedule_scheme == 'strict':
            return _get_scheduling_setup_for_strict(self.args)
        elif 'custom_one_cycle' == self.args.lr_schedule_scheme:
            # NOTE: The scheme yet does not support multi-GPU training.
            # No warmup and no linear scale are applied.
            return _get_scheduling_setup_for_onecycle(self.args)
        elif 'custom_multistep' == self.args.lr_schedule_scheme:
            return _get_scheduling_setup_for_multistep(self.args)
        elif 'custom_convex_decay' == self.args.lr_schedule_scheme:
            return _get_scheduling_setup_for_convex_decay(self.args)
        else:
            raise NotImplementedError

    def build_lr_schedulers(self, epoch_fields, lr_fields, scale_indicators):
        lr_schedulers = dict()

        for field_id, (epoch_field, lr_field, indicator) in \
                enumerate(zip(epoch_fields, lr_fields, scale_indicators)):
            lr_scheduler = self._build_lr_scheduler(epoch_field, lr_field, indicator)
            lr_schedulers[field_id] = lr_scheduler
        return lr_schedulers

    def _build_lr_scheduler(self, epoch_field, lr_field, scale_indicator):
        lr_left, lr_right = lr_field
        epoch_left, epoch_right = epoch_field
        n_steps = epoch_right - epoch_left

        if scale_indicator == 'linear':
            return _linear_scale(lr_left, lr_right, n_steps, epoch_left)
        elif scale_indicator == 'poly':
            return _poly_scale(lr_left, lr_right, n_steps, epoch_left)
        elif scale_indicator == 'convex':
            assert self.args.lr_gamma is not None
            assert self.args.lr_mu is not None
            assert self.args.lr_alpha is not None
            return _convex_scale(
                self.args.lr_gamma, self.args.lr_mu, self.args.lr_alpha)
        else:
            raise NotImplementedError


"""Define the scheduling step,
    e.g., logic of epoch_fields, lr_fields and scale_indicators.

    We should be able to determine if we only use the pure info from parser,
    or use a mixed version (the second one might be more common in practice)

    For `epoch_fields`, we define it by a string separated by ',',
    e.g., '10,20,30' to indicate different ranges.
    More precisely, previous `epoch_fields` example
    is equivalent to three different epoch ranges,
    i.e., [0, 10), [10, 20), [20, 30).

    For `lr_fields`, it is corresponding to the `epoch_fields`,
    indicating the left lr and right lr for each epoch range.

    For scale_indicators,
    it is used to define how to scale the left lr and right lr
    in the corresponding epoch range.
"""

# define the formal procedure of setting up the scheduling.


def _get_scheduling_setup(args):
    assert args.lr_change_epochs is not None
    assert args.lr_fields is not None
    assert args.lr_scale_indicators is not None

    # define lr_fields
    lr_fields = _get_lr_fields(args.lr_fields)

    # define scale_indicators
    scale_indicators = _get_lr_scale_indicators(args.lr_scale_indicators)

    # define epoch_fields
    epoch_fields = _get_lr_epoch_fields(args.lr_change_epochs)

    return epoch_fields, lr_fields, scale_indicators


def _get_lr_fields(lr_fields):
    return [map(float, l.split(',')) for l in lr_fields.split('/')]


def _get_lr_scale_indicators(lr_scale_indicators):
    def digital2name(x):
        return {
            '0': 'linear',
            '1': 'poly',
            '2': 'convex'  # lr = \gamma / (\mu (t + a))
        }[x]
    return [digital2name(l) for l in lr_scale_indicators.split(',')]


def _get_lr_epoch_fields(lr_change_epochs):
    """note that the change points exclude the head and tail of the epochs.
    """
    lr_change_epochs = [int(l) for l in lr_change_epochs.split(',')]
    from_s = lr_change_epochs[:-1]
    to_s = lr_change_epochs[1:]
    return list(zip(from_s, to_s))


# case: _get scheduling setup for "strict learnign rate" configuration from the parser.


def _get_scheduling_setup_for_strict(args):
    # define lr_fields
    args.lr_change_epochs = '0,{original},{full}'.format(
        original=args.lr_change_epochs, full=args.num_epochs
    )

    return _get_scheduling_setup(args)


# case: _get scheduling setup for "onecycle learning rate" scheme.


def _get_scheduling_setup_for_onecycle(args):
    args.lr_fields = '{low},{high}/{high},{low}/{low},{extra_low}'.format(
        low=args.lr_onecycle_low,
        high=args.lr_onecycle_high,
        extra_low=args.lr_onecycle_extra_low
    )
    args.lr_change_epochs = '0,{half_cycle},{cycle},{full}'.format(
        half_cycle=args.lr_onecycle_num_epoch // 2,
        cycle=args.lr_onecycle_num_epoch,
        full=args.num_epochs
    )
    args.lr_scale_indicators = '0,0,0'
    return _get_scheduling_setup(args)


# case: _get scheduling setup for "multiple-step constant learning rates" scheme.


def _get_scheduling_setup_for_multistep(args):
    # define lr_fields
    args.lr_fields = _build_multistep_lr_fields(
        args.lr_change_epochs,
        args.lr_warmup, args.learning_rate, args.init_warmup_lr, args.lr_decay)

    # define lr_change_epochs
    args.lr_change_epochs, num_intervals = _build_multistep_lr_change_epochs(
        args.lr_change_epochs, args.lr_warmup, args.lr_warmup_epochs,
        args.num_epochs)

    # define scale_indicators
    args.lr_scale_indicators = ','.join(['0'] * num_intervals)
    return _get_scheduling_setup(args)


def _build_multistep_lr_fields(
        lr_change_epochs, lr_warmup, learning_rate, init_warmup_lr, lr_decay):
    if lr_change_epochs is not None:
        _lr_fields = [
            learning_rate * ((1. / lr_decay) ** l)
            for l in range(len(lr_change_epochs.split(',')) + 1)
        ]
    else:
        _lr_fields = [learning_rate]

    lr_fields = '/'.join(['{lr},{lr}'.format(lr=lr) for lr in _lr_fields])

    if lr_warmup:
        return '{},{}/'.format(init_warmup_lr, learning_rate) + lr_fields
    else:
        return lr_fields


def _build_multistep_lr_change_epochs(
        lr_change_epochs, lr_warmup, lr_warmup_epochs, num_epochs):
    if lr_change_epochs is not None:
        lr_change_epochs = [0] + lr_change_epochs.split(',') + [num_epochs]
    else:
        lr_change_epochs = [0, num_epochs]

    if lr_warmup:
        lr_change_epochs = [0, lr_warmup_epochs] + lr_change_epochs[1:]
    return ','.join([str(x) for x in lr_change_epochs]), len(lr_change_epochs) - 1


# case: _get scheduling setup for "convex learning" scheme.


def _get_scheduling_setup_for_convex_decay(args):
    # define lr_fields
    args.lr_fields = '{},{}'.format(args.learning_rate, 0)

    # define lr_change_epochs
    args.lr_change_epochs = '0,{full}'.format(full=args.num_epochs)

    # define scale_indicators
    args.lr_scale_indicators = '2'
    return _get_scheduling_setup(args)


"""define choice of scaling learning rate within the range."""


def _linear_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        step = (lr_right - lr_left) / n_steps
        return (index - abs_index) * step + lr_left
    return f


def _poly_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        return lr_left * ((1 - (index - abs_index) / n_steps) ** 2)
    return f


def _convex_scale(gamma, mu, alpha):
    # it is expected in the form of lr = \gamma / (\mu (t + a))
    def f(index):
        return gamma / (mu * (alpha + index))
    return f
