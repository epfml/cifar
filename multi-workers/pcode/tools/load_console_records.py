# -*- coding: utf-8 -*-
import re

import torch

from pcode.utils.auxiliary import str2time, is_float
from pcode.utils.op_files import read_txt


def _parse_record(lines, parse_fn, pattern, var_names):
    parsed_lines = []
    for line in lines:
        parsed_line = parse_fn(line, pattern, var_names)
        if parsed_line is not None:
            parsed_lines.append(parsed_line)

    return parsed_lines


def _parse_record_for_train_fn(line, pattern, var_names):
    try:
        # print(line)
        matched_line = re.findall(pattern, line, re.DOTALL)

        if len(matched_line) != 0:
            # get parsed line.
            matched_line = [x.strip() for x in matched_line[0]]
            # convert the string to time.
            matched_line[0] = str2time(matched_line[0], '%Y-%m-%d %H:%M:%S')
            # map str to float
            matched_line = [
                float(x) if isinstance(x, str) and is_float(x) else x
                for x in matched_line]
            # build dictionary
            zip_line = zip(var_names, matched_line)
            line = dict(zip_line)
            return line
        else:
            return None
    except Exception as e:
        print(' the error: {}'.format(e))
        return None


def _parse_record_for_train(lines):
    pattern = r'(.*?)o(.*?)\sDEBUG: Epoch: (.*?)\. Local index: (.*?)\. Load: (.*?)s \| Data: (.*?)s \| Computing: (.*?)s \| Sync: (.*?)s \| Global: (.*?)s \| Loss: (.*?) \| top1: (.*?) \| top5: (.*?)$'
    var_names = [
        'time', 'rank', 'epoch', 'local_index',
        'load_time', 'data_time', 'compute_time', 'sync_time',
        'global_time', 'loss', 'top1', 'top5']
    return _parse_record(lines, _parse_record_for_train_fn, pattern, var_names)


def _parse_record_for_test_fn(line, pattern, var_names):
    try:
        matched_line = re.findall(pattern, line, re.DOTALL)

        if len(matched_line) != 0:
            # get parsed line.
            matched_line = [x.strip() for x in matched_line[0]]
            # convert the string to time.
            matched_line[0] = str2time(matched_line[0], '%Y-%m-%d %H:%M:%S')
            # map str to float
            matched_line = [
                float(x) if isinstance(x, str) and is_float(x) else x
                for x in matched_line]
            # build dictionary
            zip_line = zip(var_names, matched_line)
            line = dict(zip_line)
            return line
        else:
            return None
    except Exception as e:
        print(' the error: {}'.format(e))
        return None


def _parse_record_for_test(lines):
    pattern = r'(.*?)o(.*?)INFO:(.*?)\(best epoch (.*?), current epoch (.*?)\):(.*?)\.$'
    var_names = ['time', 'rank', 'info', 'best_epoch', 'current_epoch', 'top1']
    lines = [line for line in lines if 'best epoch' in line]
    return _parse_record(lines, _parse_record_for_test_fn, pattern, var_names)


def _get_console_record(path):
    # load record file and parse args.
    lines = read_txt(path)

    # parse records.
    parsed_train_lines = _parse_record_for_train(lines)
    parsed_test_lines = _parse_record_for_test(lines)
    return parsed_train_lines, parsed_test_lines


def get_console_records(paths):
    return [_get_console_record(path) for path in paths]


def get_checkpoint_args(path):
    checkpoint = torch.load(path, map_location='cpu')
    arguments = vars(checkpoint['arguments'])

    arguments.update({
        'n_nodes': arguments['graph'].n_nodes,
        'world': arguments['graph'].world,
        'rank': arguments['graph'].rank,
        'ranks': arguments['graph'].ranks,
        'ranks_with_blocks': arguments['graph'].ranks_with_blocks,
        'blocks_with_ranks': arguments['graph'].blocks_with_ranks,
        'device': arguments['graph'].device,
        'on_cuda': arguments['graph'].on_cuda,
    })
    arguments['graph'] = None
    return arguments
