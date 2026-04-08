#!/usr/bin/env python3
import argparse
import os

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
import torchmetrics

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--hidden_layers", default=256, type=int, help="Number of hidden layers in final layer")

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
            # torch.nn.Linear(1280, self._args.hidden_layers),
            # torch.nn.BatchNorm1d(self._args.hidden_layers),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.3),

            # torch.nn.Linear(self._args.hidden_layers, 34)

            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.backbone.num_features, 34)
        )

    def forward(self, x):
        features = self.backbone(x)

        return self._classifier(features)

def main(args: argparse.Namespace) -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    cags = CAGS(decode_on_demand=False)

    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b3.in21k_ft_in1k", pretrained=True, num_classes=0)

    train_preprocessing = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
        v2.RandomErasing(p=0.25),
    ])
    test_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    for param in efficientnetv2_b0.parameters():  # Freeze!
        param.requires_grad = False


    train_data = DatasetWrapper(cags.train, train_preprocessing)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    dev_data = DatasetWrapper(cags.dev, test_preprocessing)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size)

    model = Network(efficientnetv2_b0, args)

    steps_per_epoch = len(train_loader)
    total_steps_1 = args.epochs * steps_per_epoch

    optimizer_1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    warmup_epochs = 1
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = total_steps_1 - warmup_steps

    warmup_scheduler = LinearLR(
      optimizer_1,
      start_factor=0.01, 
      end_factor=1.0,
      total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
      optimizer_1,
      T_max=cosine_steps,
      eta_min=1e-5
    )
    scheduler_1 = SequentialLR(
      optimizer_1,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[warmup_steps]
    )
    model.configure(
        optimizer=optimizer_1,
        scheduler=scheduler_1,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        metrics={
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=34)
        },
        logdir=logdir
    )
    model.to(device)


    model.fit(dataloader=train_loader, epochs=args.epochs, dev=dev_loader)

    for param in efficientnetv2_b0.parameters():  # Un-freeze!
        param.requires_grad = True
    
    print("Unfreezed!")
    steps_per_epoch = len(train_loader)
    epochs_2 = 15
    total_steps_2 = epochs_2 * steps_per_epoch

    optimizer_2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    warmup_epochs = 1
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = total_steps_2 - warmup_steps

    warmup_scheduler = LinearLR(
        optimizer_2,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer_2,
        T_max=cosine_steps,
        eta_min=1e-5
    )
    scheduler_2 = SequentialLR(
        optimizer_2,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]   # step index to switch schedulers
    )
    model.configure(
        optimizer=optimizer_2,
        scheduler=scheduler_2,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=34)},
        logdir=logdir
    )
    model.to(device)

    model.fit(dataloader=train_loader, epochs=epochs_2, dev=dev_loader)


    test_data = DatasetWrapper(cags.test, test_preprocessing)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test_loader, data_with_labels=True):
            predicted_class = prediction.argmax(dim=-1)
            print(predicted_class.item(), file=predictions_file)
    print("Prediction file saved")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
