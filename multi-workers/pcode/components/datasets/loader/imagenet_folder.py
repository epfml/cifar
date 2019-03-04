# -*- coding: utf-8 -*-
import torchvision.datasets as datasets

from pcode.tracking.logging import info
from pcode.components.datasets.preprocess_toolkit import get_transform
from pcode.components.datasets.loader.utils import IMDBPT


def define_imagenet_folder(name, root, flag, cuda=True):
    is_train = 'train' in root
    transform = get_transform(name, augment=is_train, color_process=False)

    if flag:
        info('load imagenet from lmdb: {}'.format(root))
        return IMDBPT(root, transform=transform, is_image=True)
    else:
        info("load imagenet using pytorch's default dataloader.")
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=None)