# -*- coding: utf-8 -*-
from pcode.components.optim.optimizers.sgd import SGD
from pcode.components.optim.optimizers.adam import Adam
from pcode.components.optim.optimizers.sparse_sgd import SparseSGD


def define_optimizer(conf, model):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': conf.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]

    # define the optimizer.
    if conf.optimizer == 'sgd':
        return SGD(
            params, lr=conf.learning_rate,
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov, conf=conf)
    elif conf.optimizer == 'adam':
        return Adam(
            params, lr=conf.learning_rate,
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps, conf=conf)
    elif conf.optimizer == 'sparse':
        return SparseSGD(
            params, lr=conf.learning_rate,
            momentum=conf.momentum_factor,
            conf=conf)
    else:
        raise NotImplementedError
