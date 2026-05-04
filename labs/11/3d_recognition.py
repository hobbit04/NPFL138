#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.11")
from npfl138.datasets.modelnet import ModelNet

import timm
import torch.nn.functional as F

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=32, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: ModelNet.Dataset) -> None:
        super().__init__()
        self.backbone = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=0, in_chans=3)
        self.classifier = torch.nn.Linear(self.backbone.num_features, 10)

    def forward(self, x):
        proj_xy = x.max(dim=2).values  # [B, 1, D, D]
        proj_xz = x.max(dim=3).values
        proj_yz = x.max(dim=4).values
        img = torch.cat([proj_xy, proj_xz, proj_yz], dim=1).float()  # [B, 3, D, D]
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.classifier(self.backbone(img))

class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ModelNet.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        grid = example["grid"].float() 
        label = example["label"].long()
        if self._training:
            for dim in [1, 2, 3]:
                if torch.rand(1) > 0.5:
                    grid = grid.flip(dim)
            k = torch.randint(0, 4, (1,)).item()
            grid = torch.rot90(grid, k, [2, 3])
        return grid, label
 
    def collate(self, batch):
        grids, labels = zip(*batch)
        return torch.stack(grids), torch.stack(labels)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    modelnet = ModelNet(args.modelnet)

    # TODO: Create the model and train it
    train = TrainableDataset(modelnet.train, True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(modelnet.dev, False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(modelnet.test, False).dataloader(batch_size=args.batch_size)

    model = Model(args, train=train)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train), eta_min=1e-5
    )
    model.configure(
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy('multiclass', num_classes=10)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
        scheduler=scheduler,
    ).to(device=device)

    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
