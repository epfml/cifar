# -*- coding: utf-8 -*-
from copy import deepcopy
from datetime import datetime


def deepcopy_model(conf, model):
    # a dirty hack....
    tmp_model = deepcopy(model)
    if conf.track_model_aggregation:
        for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
            tmp_para.grad = para.grad.clone()
    return tmp_model


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def get_fullname(o):
    """get the full name of the class."""
    return '%s.%s' % (o.__module__, o.__class__.__name__)


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a,
                        [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)
