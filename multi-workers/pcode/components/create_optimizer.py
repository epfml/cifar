# -*- coding: utf-8 -*-
from pcode.components.optim.optimizers.sgd import SGD
from pcode.components.optim.optimizers.adam import Adam
from pcode.components.optim.optimizers.sparse_sgd import SparseSGD


def define_optimizer(args, model):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': args.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]

    # define the optimizer.
    if args.optimizer == 'sgd':
        return SGD(
            params, lr=args.learning_rate,
            momentum=args.momentum_factor,
            nesterov=args.use_nesterov, args=args)
    elif args.optimizer == 'adam':
        return Adam(
            params, lr=args.learning_rate,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=args.adam_eps, args=args)
    elif args.optimizer == 'sparse':
        return SparseSGD(
            params, lr=args.learning_rate,
            momentum=args.momentum_factor,
            args=args)
    else:
        raise NotImplementedError
