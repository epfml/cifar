# -*- coding: utf-8 -*-
import torchvision.datasets as datasets

from pcode.components.datasets.loader.preprocess_toolkit import get_transform
from pcode.components.datasets.loader.utils import IMDBPT


def define_imagenet_folder(name, root, flag, cuda=True):
    is_train = 'train' in root
    transform = get_transform(name, augment=is_train, color_process=False)

    if flag:
        print('load imagenet from lmdb: {}'.format(root))
        return IMDBPT(root, transform=transform, is_image=True)
    else:
        print("load imagenet using pytorch's default dataloader.")
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=None)
