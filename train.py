#!/usr/bin/env python3

import os
import time

import numpy as np
import torch
import torchvision

import models
import cifar_utils.accumulators

def main(config, output_dir, gpu_id, pretrained_model=None, pretrained_dataset=None):
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    # Set the seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.cuda.set_device(gpu_id)
    # We will run on CUDA if there is a GPU available
    # device = torch.device('cuda:{}'.format(str(gpu_id)) if torch.cuda.is_available() else 'cpu')

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    if pretrained_dataset is not None:
        training_loader, test_loader = pretrained_dataset
    else:
        training_loader, test_loader = get_dataset(config)

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = get_model(config, gpu_id)

    optimizer, scheduler = get_optimizer(config, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = cifar_utils.accumulators.Max()

    print("number of epochs would be ", config['num_epochs'])
    for epoch in range(config['num_epochs']):
        print('Epoch {:03d}'.format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = cifar_utils.accumulators.Mean()
        mean_train_loss = cifar_utils.accumulators.Mean()

        # Update the optimizer's learning rate
        scheduler.step(epoch)

        for batch_x, batch_y in training_loader:
            batch_x, batch_y = batch_x.cuda(gpu_id), batch_y.cuda(gpu_id)

            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            loss.backward()

            # Do an optimizer steps
            optimizer.step()

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Log training stats
        log_metric(
            'accuracy',
            {'epoch': epoch, 'value': mean_train_accuracy.value()},
            {'split': 'train'}
        )
        log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': mean_train_loss.value()},
            {'split': 'train'}
        )

        # Evaluation
        model.eval()
        mean_test_accuracy = cifar_utils.accumulators.Mean()
        mean_test_loss = cifar_utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.cuda(gpu_id), batch_y.cuda(gpu_id)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            'accuracy',
            {'epoch': epoch, 'value': mean_test_accuracy.value()},
            {'split': 'test'}
        )
        log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': mean_test_loss.value()},
            {'split': 'test'}
        )

        # Store checkpoints for the best model so far
        is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
        if is_best_so_far:
            store_checkpoint(output_dir, "best.checkpoint", model, epoch, mean_test_accuracy.value())

    # Store a final checkpoint
    store_checkpoint(output_dir, "final.checkpoint", model, config['num_epochs'] - 1, mean_test_accuracy.value())

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    """
    Log timeseries data.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    print("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def get_dataset(config, test_batch_size=100, shuffle_train=True, num_workers=2, data_root='./data', unit_batch_train=False):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config['dataset'] == 'Cifar10':
        dataset = torchvision.datasets.CIFAR10
    elif config['dataset'] == 'Cifar100':
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError('Unexpected value for config[dataset] {}'.format(config['dataset']))

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(data_mean, data_stddev),
    ])

    training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
    test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

    if unit_batch_train:
        train_batch_size = 1
    else:
        train_batch_size = config['batch_size']

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return training_loader, test_loader


def get_optimizer(config, model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config['optimizer_learning_rate'],
            momentum=config['optimizer_momentum'],
            weight_decay=config['optimizer_weight_decay'],
        )
    else:
        raise ValueError('Unexpected value for optimizer')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['optimizer_decay_at_epochs'],
        gamma=1.0/config['optimizer_decay_with_factor'],
    )

    return optimizer, scheduler


def get_model(config, device=-1):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config['dataset'] == 'Cifar100' else 10

    model = {
        'vgg11_nobias': lambda: models.VGG('VGG11', num_classes, batch_norm=False, bias=False),
        'vgg11':     lambda: models.VGG('VGG11', num_classes, batch_norm=False),
        'vgg11_bn':  lambda: models.VGG('VGG11', num_classes, batch_norm=True),
        'vgg13':     lambda: models.VGG('VGG13', num_classes, batch_norm=False),
        'vgg13_bn':  lambda: models.VGG('VGG13', num_classes, batch_norm=True),
        'vgg16':     lambda: models.VGG('VGG16', num_classes, batch_norm=False),
        'vgg16_bn':  lambda: models.VGG('VGG16', num_classes, batch_norm=True),
        'vgg19':     lambda: models.VGG('VGG19', num_classes, batch_norm=False),
        'vgg19_bn':  lambda: models.VGG('VGG19', num_classes, batch_norm=True),
        'resnet18':  lambda: models.ResNet18(num_classes=num_classes),
        'resnet34':  lambda: models.ResNet34(num_classes=num_classes),
        'resnet50':  lambda: models.ResNet50(num_classes=num_classes),
        'resnet101': lambda: models.ResNet101(num_classes=num_classes),
        'resnet152': lambda: models.ResNet152(num_classes=num_classes),
    }[config['model']]()

    if device != -1:
        # model.to(device)
        model = model.cuda(device)
        print("model parameters are \n", list([param.shape for param in model.parameters()]))
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True

    return model


def get_pretrained_model(config, path, device_id=-1):

    model = get_model(config, device_id)

    if device_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(device_id))
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )

    print("Loading model at path {} which had accuracy {}".format(path, state['test_accuracy']))
    model.load_state_dict(state['model_state_dict'])

    return model, state['test_accuracy']

def get_retrained_model(args, train_loader, test_loader, old_network, config, output_dir):
    # update the parameters
    config['num_epochs'] = args.retrain
    if args.retrain_lr_decay > 0:

        config['optimizer_learning_rate'] = args.cifar_init_lr / args.retrain_lr_decay
        print('optimizer_learning_rate is ', config['optimizer_learning_rate'])
    # retrain
    best_acc = main(config, output_dir, args.gpu_id, pretrained_model=old_network, pretrained_dataset=(train_loader, test_loader))
    # currently I don' return the best model, as it checkpointed
    return None, best_acc
def store_checkpoint(output_dir, filename, model, epoch, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    time.sleep(1) # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save({
        'epoch': epoch,
        'test_accuracy': test_accuracy,
        'model_state_dict': model.state_dict(),
    }, path)


if __name__ == '__main__':
    config = dict(
        dataset='Cifar10',
        model='resnet18',
        optimizer='SGD',
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=42,
    )

    output_dir = './output.tmp'  # Can be overwritten by a script calling this
    gpu_id = 0

    main(config, output_dir, gpu_id)
