# -*- coding: utf-8 -*-

import os
from os.path import join
import time
import logging

from pcode.utils.op_files import write_txt


logger = logging.getLogger('o')


def init_logging(args):
    """Setup logging modules.
    A stream handler and file handler are added to default logger `mlbench`.
    """
    logging_file = join(args.checkpoint_dir, 'record' + str(args.graph.rank))

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = args.graph.rank
            return True

    logger.setLevel(logging.DEBUG)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s',
        "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logging_file)
    # fh.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _warp(string, symbol='*', length=80):
    one_side_length = (length - len(string) - 2) // 2
    if one_side_length > 0:
        return symbol * one_side_length + ' ' + string + ' ' + symbol * one_side_length
    else:
        return string


def info(content, display=True):
    if display:
        logger.info(content)


def debug(content, display=True):
    if display:
        logger.debug(content)


def critical(content, display=True):
    if display:
        logger.critical('{}'.format(content))


def log_args(args):
    info('parameters: ')
    for arg in vars(args):
        info(str(arg) + '\t' + str(getattr(args, arg)))
    for name in ['n_nodes', 'world', 'rank',
                 'ranks_with_blocks', 'blocks_with_ranks',
                 'device', 'on_cuda', 'get_neighborhood']:
        info('{}: {}'.format(name, getattr(args.graph, name)))


def logging_display_training(args, tracker):
    log_info = 'Epoch: {epoch:.3f}. Local index: {local_index}. Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
        epoch=args.epoch_,
        local_index=args.local_index,
        loss=tracker['losses'].avg,
        top1=tracker['top1'].avg,
        top5=tracker['top5'].avg)
    debug(log_info)


def logging_display_val(args):
    info('best accuracy for rank {} at local index {} \
        (best epoch {:.3f}, current epoch {:.3f}): {}.'.format(
        args.graph.rank, args.local_index,
        args.best_epoch[-1] if len(args.best_epoch) != 0 else '',
        args.epoch_, args.best_prec1))


def update_performance_tracker(tracker, loss, performance, size):
    tracker['losses'].update(loss.item(), size)

    if len(performance) == 2:
        tracker['top5'].update(performance[1], size)
    tracker['top1'].update(performance[0], size)
    return tracker
