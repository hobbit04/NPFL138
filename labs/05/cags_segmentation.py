#!/usr/bin/env python3
import argparse
import os

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"])
        return image, item["mask"]


class Network(npfl138.TrainableModule):
    def __init__(self, backbone, args):
        super().__init__()
        self._args = args
        self.backbone = backbone

        self.up1 = torch.nn.ConvTranspose2d(1280, 112, kernel_size=2, stride=2)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(112 + 112, 112, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(112))
        
        self.up2 = torch.nn.ConvTranspose2d(112, 48, kernel_size=2, stride=2)
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(48 + 48, 48, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(48))
        
        self.up3 = torch.nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(32 + 32, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32))
        
        self.up4 = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(16 + 16, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(16))
        
        self.up5 = torch.nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = torch.nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        output, features = self.backbone.forward_intermediates(x)

        x = self.up1(output)
        x = torch.cat([x, features[3]], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.conv4(x)
        
        x = self.up5(x)
        x = self.final_conv(x)
        
        return x


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

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    train_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1]., 
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    test_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    for param in efficientnetv2_b0.parameters(): # Freeze!
        param.requires_grad = False

    # TODO: Create the model and train it.
    model = Network(efficientnetv2_b0, args)
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={
            "iou": CAGS.MaskIoUMetric()
        },
        logdir=logdir
    )
    model.to(device)

    train_data = DatasetWrapper(cags.train, train_preprocessing)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    dev_data = DatasetWrapper(cags.dev, test_preprocessing)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size)

    model.fit(dataloader=train_loader, epochs=args.epochs, dev=dev_loader)

    # Fine tune the whole model
    for param in efficientnetv2_b0.parameters(): # Un-freeze!
        param.requires_grad = True
    
    print("Unfreezed!")

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01),
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={
            "iou": CAGS.MaskIoUMetric()
        },
        logdir=logdir
    )
    model.fit(dataloader=train_loader, epochs=5, dev=dev_loader)

    test_data = DatasetWrapper(cags.test, test_preprocessing)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)


    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test_loader, data_with_labels=True, as_numpy=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.0, [-1]):  # I'll use logits
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
