# -*- coding: utf-8 -*-
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pcode.components.datasets.loader.imagenet_folder import define_imagenet_folder
from pcode.components.datasets.loader.svhn_folder import define_svhn_folder
from pcode.components.datasets.loader.epsilon_or_rcv1_folder import define_epsilon_or_rcv1_folder


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = (split == 'train')

    # decide normalize parameter.
    if name == 'cifar10':
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif name == 'cifar100':
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # decide data type.
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    return dataset_loader(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_mnist(root, split, transform, target_transform, download):
    is_train = (split == 'train')

    if is_train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return datasets.MNIST(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(root=root,
                          split=split,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_svhn(root, split, transform, target_transform, download):
    is_train = (split == 'train')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return define_svhn_folder(root=root,
                              is_train=is_train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)


def _get_imagenet(args, name, datasets_path, split):
    is_train = (split == 'train')
    root = os.path.join(
        datasets_path,
        'lmdb' if 'downsampled' not in name else 'lmdb_32x32'
        ) if args.use_lmdb_data else datasets_path

    if is_train:
        root = os.path.join(root, 'train{}'.format(
            '' if not args.use_lmdb_data else '.lmdb')
        )
    else:
        root = os.path.join(root, 'val{}'.format(
            '' if not args.use_lmdb_data else '.lmdb')
        )
    return define_imagenet_folder(
        name=name, root=root, flag=args.use_lmdb_data,
        cuda=args.graph.on_cuda)


def _get_epsilon_or_rcv1(root, name, split):
    root = os.path.join(root, '{}_{}.lmdb'.format(name, split))
    return define_epsilon_or_rcv1_folder(root)


def get_dataset(
        args, name, datasets_path, split='train', transform=None,
        target_transform=None, download=True):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == 'cifar10' or name == 'cifar100':
        return _get_cifar(
            name, root, split, transform, target_transform, download)
    elif name == 'svhn':
        return _get_svhn(root, split, transform, target_transform, download)
    elif name == 'mnist':
        return _get_mnist(root, split, transform, target_transform, download)
    elif name == 'stl10':
        return _get_stl10(root, split, transform, target_transform, download)
    elif 'imagenet' in name:
        return _get_imagenet(args, name, datasets_path, split)
    elif name == 'epsilon':
        return _get_epsilon_or_rcv1(root, name, split)
    elif name == 'rcv1':
        return _get_epsilon_or_rcv1(root, name, split)
    else:
        raise NotImplementedError
