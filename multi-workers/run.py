# -*- coding: utf-8 -*-
import os
import itertools

import parameters as para


def run_cmd(cmd):
    # run the cmd.
    print('\nRun the following cmd:\n' + cmd)
    os.system(cmd)


def single_run(conf):
    cmd = build_cmd(conf)
    run_cmd(cmd)


def build_cmd(conf, replacement=None):
    # get the world size.
    if replacement is not None and 'blocks' in replacement[0]:
        blocks = replacement[1][replacement[0].index('blocks')]
    else:
        blocks = conf.blocks
    world_size = sum([int(l) for l in blocks.split(',')])

    # get prefix_cmd.
    if world_size > 1:
        prefix_cmd = 'mpirun -n {} --hostfile {} --mca btl_tcp_if_exclude docker0,lo --mca orte_base_help_aggregate 0 --prefix {} '

        # assign mpi environment.
        prefix_cmd = prefix_cmd . format(
            world_size, conf.hostfile, conf.mpi_path)
    else:
        prefix_cmd = ''

    # build complete script.
    cmd = '{} main.py '.format(conf.python_path)
    for k, v in conf.__dict__.items():
        if replacement is not None and k in replacement[0]:
            cmd += ' --{} {} '.format(k, replacement[1][replacement[0].index(k)])
        elif v is not None:
            cmd += ' --{} {} '.format(k, v)
    return prefix_cmd + cmd


def generate_combinations(params):
    names = sorted(params)
    combinations = itertools.product(*(params[name] for name in names))
    return names, list(combinations)


def multi_run(conf):
    # define parameters to vary.
    # please refer parameters.py to use the correct name and datatype!
    params = {
        'world': ['0,0,0,0'],
        'blocks': ['1', '2', '3', '4'],
    }
    names, combinations = generate_combinations(params)

    # generate the runnable cmds.
    for combination in combinations:
        cmd = build_cmd(conf, replacement=(names, combination))
        run_cmd(cmd)


if __name__ == '__main__':
    args = para.get_args()
    single_run(args)
    # multi_run(args)
