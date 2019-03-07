# -*- coding: utf-8 -*-


def define_scheduler(args):
    # get the learning rate per sample.
    args.learning_rate_per_samples = args.lr / args.base_batch_size

    # get a valid learning rate.
    args.init_warmup_lr = args.lr

    if args.lr_scaleup:
        if args.lr_scaleup_type == 'linear':
            _lr = args.learning_rate_per_samples * args.batch_size
            _scale = args.graph.n_nodes
        elif args.lr_scaleup_type == 'sqrt':
            _lr = args.lr
            _scale = (
                1. * args.graph.n_nodes * args.batch_size /
                args.base_batch_size) ** 0.5
        else:
            raise NotImplementedError
        args.learning_rate = _lr * _scale
    else:
        _lr = args.learning_rate_per_samples * args.batch_size
        _scale = 1
    args.learning_rate = _lr * _scale

    # just backup the current learning rate.
    args.old_learning_rate = args.learning_rate

    # define the learning rate scheduler.
    lr_scheduler = get_lr_scheduler(args)
    return lr_scheduler


def adjust_learning_rate(args, optimizer, lr_scheduler):
    """Sets the learning rate to the initial LR decayed by # of accessed sample
        We should decay the learning rate based on the number of samples that
        we have accessed.
    """
    # adjust and assign learning rate.
    lr = lr_scheduler(args.epoch_)

    if lr is None:
        lr = args.old_learning_rate

    if args.old_learning_rate != lr:
        args.old_learning_rate = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr



"""the stuff below related to get_lr_scheduler."""


def get_lr_scheduler(args):
    epoch_fields, lr_fields, scale_indicators = get_scheduling_setup(args)
    lr_schedulers = build_lr_schedulers(args, epoch_fields, lr_fields, scale_indicators)
    return _get_lr_scheduler(epoch_fields, lr_schedulers)


# get scheduling setup.


def get_scheduling_setup(args):
    if args.lr_schedule_scheme == 'strict':
        return _get_scheduling_setup_for_strict(args)
    elif 'custom_one_cycle' == args.lr_schedule_scheme:
        # NOTE: The scheme yet does not support multi-GPU training.
        # No warmup and no linear scale are applied.
        return _get_scheduling_setup_for_onecycle(args)
    elif 'custom_multistep' == args.lr_schedule_scheme:
        return _get_scheduling_setup_for_multistep(args)
    elif 'custom_convex_decay' == args.lr_schedule_scheme:
        return _get_scheduling_setup_for_convex_decay(args)
    else:
        raise NotImplementedError


# build lr schedulers.


def build_lr_schedulers(args, epoch_fields, lr_fields, scale_indicators):
    lr_schedulers = dict()

    for field_id, (epoch_field, lr_field, indicator) in \
            enumerate(zip(epoch_fields, lr_fields, scale_indicators)):
        lr_scheduler = _build_lr_scheduler(args, epoch_field, lr_field, indicator)
        lr_schedulers[field_id] = lr_scheduler
    return lr_schedulers


def _build_lr_scheduler(args, epoch_field, lr_field, scale_indicator):
    lr_left, lr_right = lr_field
    epoch_left, epoch_right = epoch_field
    n_steps = epoch_right - epoch_left

    if scale_indicator == 'linear':
        return _linear_scale(lr_left, lr_right, n_steps, epoch_left)
    elif scale_indicator == 'poly':
        return _poly_scale(lr_left, lr_right, n_steps, epoch_left)
    elif scale_indicator == 'convex':
        assert args.lr_gamma is not None
        assert args.lr_mu is not None
        assert args.lr_alpha is not None
        return _convex_scale(args.lr_gamma, args.lr_mu, args.lr_alpha)
    else:
        raise NotImplementedError


# get lr scheduler.


def _get_lr_scheduler(epoch_fields, lr_schedulers):
    def f(epoch_index):
        def _is_fall_in(index, left_index, right_index):
            return left_index <= index < right_index

        for ind, (epoch_left, epoch_right) in enumerate(epoch_fields):
            if _is_fall_in(epoch_index, epoch_left, epoch_right):
                return lr_schedulers[ind](epoch_index)
    return f


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

