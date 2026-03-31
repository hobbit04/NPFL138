#!/usr/bin/env python3
import argparse
import os

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
import torchmetrics

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--hidden_layers", default=512, type=int, help="Number of hidden layers in final layer")

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"])
        return image, item["label"]


class Network(npfl138.TrainableModule):
    def __init__(self, backbone, args):
        super().__init__()
        self._args = args
        self.backbone = backbone

        self._classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, self._args.hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(self._args.hidden_layers, 34)
            # torch.nn.Linear(1280, 34)
        )

    def forward(self, x):
        features = self.backbone(x)

        return self._classifier(features)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    train_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    test_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    for param in efficientnetv2_b0.parameters(): # Freeze!
        param.requires_grad = False

    model = Network(efficientnetv2_b0, args)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=34)
        },
        logdir=logdir
    )

    # Prepare the dataset
    train_data = DatasetWrapper(cags.train, train_preprocessing)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    dev_data = DatasetWrapper(cags.dev, test_preprocessing)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size)

    model.fit(dataloader=train_loader, epochs=args.epochs, dev=dev_loader)

    # Fine tune the whole model
    for param in efficientnetv2_b0.parameters(): # Un-freeze!
        param.requires_grad = True
    
    print("Unfreezed!")

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=34)
        },
        logdir=logdir
    )
    model.fit(dataloader=train_loader, epochs=3, dev=dev_loader)


    test_data = DatasetWrapper(cags.test, test_preprocessing)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test_loader, data_with_labels=True):
            predicted_class = prediction.argmax(dim=-1)
            print(predicted_class.item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
