#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class CNNModel(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            # Block 1
            torch.nn.LazyConv2d(64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2),

            # Block 2
            torch.nn.LazyConv2d(128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3),

            # Block 3
            torch.nn.LazyConv2d(256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.4),

            # Classifier
            torch.nn.Flatten(),
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.model(x)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    cifar = CIFAR10()

    # TODO: Create the model and train it.
    train_images = cifar.train.data["images"].to(torch.float32) / 255.0
    train_labels = cifar.train.data["labels"]
    
    dev_images = cifar.dev.data["images"].to(torch.float32) / 255.0
    dev_labels = cifar.dev.data["labels"]

    train_ds = torch.utils.data.TensorDataset(train_images, train_labels)
    dev_ds = torch.utils.data.TensorDataset(dev_images, dev_labels)

    train = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size)
    model = CNNModel(args)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), weight_decay=1e-4),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=10)
        },
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)


    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        test_images = cifar.test.data["images"].to(torch.float32) / 255.0
        

        test_loader = torch.utils.data.DataLoader(test_images, batch_size=args.batch_size)
        

        for prediction in model.predict(test_loader):
            label = prediction.argmax().item()
            print(label, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
