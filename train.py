#!/usr/bin/env python3

import json
import os

import numpy as np
import torch
import torchvision
import wandb

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

CHECKPOINT_FILE = "checkpoint.pt"

# Override config values from environment variables:
for key, default_value in config.items():
    if os.getenv(key) is not None:
        if not isinstance(default_value, str):
            try:
                config[key] = json.loads(os.getenv(key, ""))
            except json.decoder.JSONDecodeError:
                print(f"Failed to decode environment variable {key} with value {os.getenv(key)}}.")
                exit(1)
        else:
            config[key] = os.getenv(key, "")


def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    wandb.init(config=config, resume="allow")

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

    start_epoch = 0

    # If the run was resumed, load the latest checkpoint.
    if wandb.run is not None and wandb.run.resumed:
        wandb.restore(CHECKPOINT_FILE)
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_accuracy_so_far.add(checkpoint["best_accuracy_so_far"])

    for epoch in range(start_epoch, config["num_epochs"]):
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
        wandb.log({
            "train/accuracy": mean_train_accuracy.value(),
            "train/loss": mean_train_loss.value(),
        }, step=epoch + 1)

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
        wandb.log({
            "test/accuracy": mean_test_accuracy.value(),
            "test/loss": mean_test_loss.value(),
        }, step=epoch + 1)

        best_accuracy_so_far.add(mean_test_accuracy.value())
        wandb.summary["best_accuracy"] = best_accuracy_so_far.value()

        # Save a checkpoint. If we get preempted, we can resume from there.
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy_so_far": best_accuracy_so_far.value(),
        }, CHECKPOINT_FILE)
        wandb.save(CHECKPOINT_FILE)

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


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
