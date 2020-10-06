import train

def main():
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """
    config = train.config
    # Set the seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # We will run on CUDA if there is a GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    training_loader, test_loader = train.get_dataset()
    model = train.get_model(device)
    optimizer, scheduler = train.get_optimizer(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # We keep track of the best accuracy so far to store checkpoints
    best_accuracy_so_far = utils.accumulators.Max()

    for epoch in range(config['num_epochs']):
        print('Epoch {:03d}'.format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = utils.accumulators.Mean()
        mean_train_loss = utils.accumulators.Mean()

        # Update the optimizer's learning rate
        scheduler.step(epoch)

        for batch_x, batch_y in training_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = train.accuracy(prediction, batch_y)
            loss.backward()

            # Do an optimizer step
            optimizer.step()

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))
            batch_grad_norm = torch.zeros(1).to(device)
            for param in model.parameters():
                x = param.grad.view(-1)
                batch_grad_norm += torch.dot(x, x)
            train.log_metric(
                'batch_gradient_norm',
                {'epoch': epoch, 'value': batch_grad_norm.item()},
                {'split': 'train'},
                False
            )

        # Log training stats
        train.log_metric(
            'accuracy',
            {'epoch': epoch, 'value': mean_train_accuracy.value()},
            {'split': 'train'}
        )
        train.log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': mean_train_loss.value()},
            {'split': 'train'}
        )

        

        # Evaluation
        model.eval()
        mean_test_accuracy = utils.accumulators.Mean()
        mean_test_loss = utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = train.accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        train.log_metric(
            'accuracy',
            {'epoch': epoch, 'value': mean_test_accuracy.value()},
            {'split': 'test'}
        )
        train.log_metric(
            'cross_entropy',
            {'epoch': epoch, 'value': mean_test_loss.value()},
            {'split': 'test'}
        )


        # Store checkpoints for the best model so far
        is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
        if is_best_so_far:
            train.store_checkpoint("best.checkpoint", model, epoch, mean_test_accuracy.value())

    # Store a final checkpoint
    train.store_checkpoint("final.checkpoint", model, config['num_epochs'] - 1, mean_test_accuracy.value())

    # Return the optimal accuracy, could be used for learning rate tuning
    return best_accuracy_so_far.value()

train.main = main


import json
import os
import sys

import utils.logging
import torch
import numpy as np

for batch_size in [32, 64, 128, 256]:
    # Define a fresh output directory
    train.output_dir = 'sebastian-delay-largebatch-cifar/bs{}'.format(batch_size)
    os.makedirs(train.output_dir, exist_ok=True)

    # Configure the experiment
    train.config = dict(
        dataset='Cifar10',
        model='resnet18',
        optimizer='SGD',
        optimizer_decay_at_epochs=[80, 120, 160],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=5e-4 ,
        batch_size=batch_size,
        num_epochs=200,
        seed=42,
    )

    # Save the config
    with open(os.path.join(train.output_dir, 'config.json'), 'w') as fp:
        json.dump(train.config, fp, indent=' ')

    # Configure the logging of scalar measurements
    logfile = utils.logging.JSONLogger(os.path.join(train.output_dir, 'metrics.json'))
    train.log_metric = logfile.log_metric

    # Train
    best_accuracy = train.main()

    # Keep track of the accuracies achieved
    print(batch_size, best_accuracy)
