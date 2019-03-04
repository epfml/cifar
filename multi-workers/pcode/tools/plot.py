# -*- coding: utf-8 -*-
from operator import itemgetter

import pandas as pd

from pcode.tools.get_summary import reorder_records
from pcode.tools.plot_utils import \
    determine_color_and_lines, plot_one_case, \
    smoothing_func, configure_figure, build_legend, groupby_indices


"""plot the curve in terms of time."""


def plot_curve_wrt_time(
        ax, records,
        x_wrt_sth, y_wrt_sth, xlabel, ylabel, title=None, markevery_list=None,
        is_smooth=True, smooth_space=100, l_subset=0.0, r_subset=1.0,
        reorder_record_item=None, remove_duplicate=True, legend=None,
        legend_loc='lower right', legend_ncol=2, bbox_to_anchor=[0, 0],
        ylimit_bottom=None, ylimit_top=None, use_log=False):
    """Each info consists of
        ['tr_loss', 'tr_top1', 'tr_steps', 'tr_time', 'te_top1', 'te_steps', 'te_times'].
    """
    # parse a list of records.
    num_records = len(records)
    distinct_conf_set = set()

    # re-order the records.
    if reorder_record_item is not None:
        records = reorder_records(records, based_on=reorder_record_item)

    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.add(_legend)

        # determine the style of line, color and marker.
        line_style, color_style, mark_style = determine_color_and_lines(
            num_rows=num_records // 3, num_cols=3, ind=ind)

        if markevery_list is not None:
            mark_every = markevery_list[ind]
        else:
            mark_style = None
            mark_every = None

        # determine if we want to smooth the curve.
        if 'tr_epochs' == x_wrt_sth:
            x = info['tr_steps']
            x = [1.0 * _x / args['num_batches_train_per_device_per_epoch'] for _x in x]
        else:
            x = info[x_wrt_sth]
            if 'time' in x_wrt_sth:
                x = [(time - x[0]).seconds + 1 for time in x]            
        y = info[y_wrt_sth]

        if is_smooth:
            x, y = smoothing_func(x, y, smooth_space)

        # only plot subtset.
        _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
        _x = x[_l_subset: _r_subset]
        _y = y[_l_subset: _r_subset]

        # plot
        ax = plot_one_case(
            ax, x=_x, y=_y,
            label=_legend,
            line_style=line_style, color_style=color_style,
            mark_style=mark_style, mark_every=mark_every,
            remove_duplicate=remove_duplicate)

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax, xlabel=xlabel, ylabel=ylabel, title=title,
        has_legend=legend is not None,
        legend_loc=legend_loc, legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor
    )
    return ax


def plot_mean_curve_wrt_time(
        ax, records, conditions,
        x_wrt_sth, y_wrt_sth, xlabel, ylabel, title=None, markevery_list=None,
        is_smooth=True, smooth_space=100, l_subset=0.0, r_subset=1.0,
        remove_duplicate=True, legend=None,
        legend_loc='lower right', legend_ncol=2, bbox_to_anchor=[0, 0],
        ylimit_bottom=None, ylimit_top=None, use_log=False):
    """Each info consists of
        ['tr_loss', 'tr_top1', 'tr_steps', 'tr_time', 'te_top1', 'te_steps', 'te_times'].
    """
    # parse a list of records.
    list_of_df = []
    distinct_conf_set = set()
    condition_names = list(conditions.keys())

    # build data frame.
    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.update(_legend)

        # determine if we want to smooth the curve.
        if 'tr_epochs' == x_wrt_sth:
            x = [
                1.0 * _x / args['num_batches_train_per_device_per_epoch']
                for _x in info['tr_steps']
            ]
        else:
            x = info[x_wrt_sth]
        y = info[y_wrt_sth]
        if is_smooth:
            x, y = smoothing_func(x, y, smooth_space)

        # only plot subtset.
        _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
        _x = x[_l_subset: _r_subset]
        _y = y[_l_subset: _r_subset]

        # build dataframe
        _df = pd.DataFrame()
        _df['x'] = _x
        _df['y'] = _y
        for condition_name in condition_names:
            _df[condition_name] = args[condition_name]
        list_of_df.append(_df)

    # get grouped df.
    df = pd.concat(list_of_df)
    grouped_df = list(df.groupby(list(df.columns[2:])))
    num_records = len(grouped_df)

    # plot
    for ind in range(num_records):
        # determine the style of line, color and marker.
        line_style, color_style, mark_style = determine_color_and_lines(
            num_rows=num_records // 3, num_cols=3, ind=ind)

        if markevery_list is not None:
            mark_every = markevery_list[ind]
        else:
            mark_style = None
            mark_every = None

        _args, _df = grouped_df[ind]
        ax = plot_one_case(
            ax, sns_plot=_df,
            label=build_legend(_df, legend),
            line_style=line_style, color_style=color_style,
            mark_style=mark_style, mark_every=mark_every,
            remove_duplicate=remove_duplicate)

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax, xlabel=xlabel, ylabel=ylabel, title=title,
        has_legend=legend is not None,
        legend_loc=legend_loc, legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor
    )
    return ax, df


"""summary information."""


def _summary_info(record, arg_names):
    args, info = record
    train_top1_acc = sum(info['tr_top1'][-100:]) / 100
    test_top1_acc = max(info['te_top1'])
    return [
        args[arg_name] for arg_name in arg_names] + [
        train_top1_acc, test_top1_acc
    ]


def summary_info(records, arg_names):
    # define header.
    headers = arg_names + ['tr_top1_acc', 'te_top1_acc']
    # reorder records
    records = reorder_records(
        records,
        based_on='n_workers' if 'n_workers' in headers else 'n_nodes'
    )
    # extract test records
    test_records = [_summary_info(record, arg_names) for record in records]
    # aggregate test records
    aggregated_records = pd.DataFrame(test_records, columns=headers)
    # average test records
    averaged_records = aggregated_records.fillna(-1).groupby(
        headers[:-2], as_index=False).agg({
            'te_top1_acc': ['mean', 'std', 'max', 'min']}
            ).sort_values(('te_top1_acc', 'mean'), ascending=False)
    return aggregated_records, averaged_records
