# -*- coding: utf-8 -*-
import os

import parameters as para


def main(conf):
    world_size = sum([int(l) for l in conf.blocks.split(',')])

    # build cmd.
    if conf.is_kube:
        prefix_cmd = 'mpirun -n {} --hostfile {} --prefix {} '
    else:
        prefix_cmd = 'mpirun -n {} --hostfile {} --mca btl_tcp_if_exclude docker0,lo --mca orte_base_help_aggregate 0 --prefix {} '

    # assign mpi environment.
    prefix_cmd = prefix_cmd . format(
        world_size, conf.hostfile, conf.mpi_path)

    # build complete script.
    cmd = '{} main.py '.format(conf.python_path)
    for k, v in conf.__dict__.items():
        if v is not None:
            cmd += ' --{} {} '.format(k, v)

    cmd = prefix_cmd + cmd

    # init the databse.
    run_cmd(args, cmd)


def run_cmd(args, cmd):
    # run the cmd.
    print('\nRun the following cmd:\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    args = para.get_args()
    main(args)
