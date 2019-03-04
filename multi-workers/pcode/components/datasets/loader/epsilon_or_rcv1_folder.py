# -*- coding: utf-8 -*-

from pcode.tracking.logging import info
from pcode.components.datasets.loader.utils import IMDBPT


def define_epsilon_or_rcv1_folder(root):
    info('load epsilon_or_rcv1 from lmdb: {}.'.format(root))
    return IMDBPT(root, is_image=False)
