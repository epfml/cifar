# -*- coding: utf-8 -*-
import os

import parameters as para
from pcode.tools.db import init_mongo, announce_job_termination_to_mongo


def main(args):
    world_size = sum([int(l) for l in args.blocks.split(',')])

    # build cmd.
    if args.is_kube:
        prefix_cmd = 'mpirun -n {} --hostfile {} --prefix {} '
    else:
        prefix_cmd = 'mpirun -n {} --hostfile {} --mca btl_tcp_if_exclude docker0,lo --mca orte_base_help_aggregate 0 --prefix {} '

    # assign mpi environment.
    prefix_cmd = prefix_cmd . format(
        world_size, args.hostfile, args.mpi_path)

    # build complete script.
    cmd = '{} main.py '.format(args.python_path)
    for k, v in args.__dict__.items():
        if v is not None:
            cmd += ' --{} {} '.format(k, v)

    cmd = prefix_cmd + cmd

    # init the databse.
    run_cmd(args, cmd)


def run_cmd(args, cmd):
    if args.use_db:
        # define the extra database to record the metadata.
        print('Init the mongo db.')
        init_mongo(args)

    # run the cmd.
    print('\nRun the following cmd:\n' + cmd)
    os.system(cmd)

    # announce the termination.
    if args.use_db:
        announce_job_termination_to_mongo(args)
        print('Announce the completition of the job to the mongodb.')


if __name__ == '__main__':
    args = para.get_args()
    main(args)
