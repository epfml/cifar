#!/usr/bin/env python3

import os

import numpy as np
import torch
import torchvision

import utils.accumulators

config = dict(
    dataset="Cifar10",
    model="resnet18",
    optimizer="SGD",
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=256,
    num_epochs=300,
    seed=42,
)


output_dir = "./output.tmp"  # Can be overwritten by a script calling this


def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    # Set the seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # We will run on CUDA if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = get_dataset()
    model = get_model(device)
    optimizer, scheduler = get_optimizer(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    for epoch in range(config["num_epochs"]):
        print("Epoch {:03d}".format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        for batch_x, batch_y in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            loss.backward()

            # Do an optimizer step
            optimizer.step()

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Update the optimizer's learning rate
        scheduler.step()

        # Log training stats
        log_metric(
            "accuracy",
            {"epoch": epoch, "value": mean_train_accuracy.value()},
            {"split": "train"},
        )
        log_metric(
            "cross_entropy",
            {"epoch": epoch, "value": mean_train_loss.value()},
            {"split": "train"},
        )

        # Evaluation
        model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric(
            "accuracy",
            {"epoch": epoch, "value": mean_test_accuracy.value()},
            {"split": "test"},
        )
        log_metric(
            "cross_entropy",
            {"epoch": epoch, "value": mean_test_loss.value()},
            {"split": "test"},
        )

        best_accuracy_so_far.add(mean_test_accuracy.value())

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


def get_dataset(
    test_batch_size=1000,
    shuffle_train=True,
    num_workers=2,
    data_root=os.getenv("DATA_DIR", "./data"),
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config["dataset"] == "Cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif config["dataset"] == "Cifar100":
        dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(
            "Unexpected value for config[dataset] {}".format(config["dataset"])
        )

    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = dataset(
        root=data_root, train=False, download=True, transform=transform_test
    )

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config["batch_size"],
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return training_loader, test_loader


def get_optimizer(model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config["optimizer_learning_rate"],
            momentum=config["optimizer_momentum"],
            weight_decay=config["optimizer_weight_decay"],
        )
    else:
        raise ValueError("Unexpected value for optimizer")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["optimizer_decay_at_epochs"],
        gamma=1.0 / config["optimizer_decay_with_factor"],
    )

    return optimizer, scheduler


def get_model(device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    num_classes = 100 if config["dataset"] == "Cifar100" else 10

    model = {
        "vgg11": lambda: torchvision.models.vgg11(num_classes=num_classes),
        "vgg11_bn": lambda: torchvision.models.vgg11_bn(num_classes=num_classes),
        "resnet18": lambda: torchvision.models.resnet18(num_classes=num_classes),
        "resnet50": lambda: torchvision.models.resnet50(num_classes=num_classes),
        "resnet101": lambda: torchvision.models.resnet101(num_classes=num_classes),
    }[config["model"]]()

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model


if __name__ == "__main__":
    main()
