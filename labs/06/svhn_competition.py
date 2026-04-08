#!/usr/bin/env python3
import argparse
import os

import timm
import torch
import torchvision.ops as ops
import torchvision.transforms.v2 as v2

import bboxes_utils
import npfl138
npfl138.require_version("2526.6")
from npfl138.datasets.svhn import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--iou_threshold", default=0.5, type=float, help="IoU threshold for anchor assignment.")
parser.add_argument("--score_threshold", default=0.3, type=float, help="Score threshold for predictions.")
parser.add_argument("--nms_threshold", default=0.5, type=float, help="NMS IoU threshold.")
parser.add_argument("--image_size", default=224, type=int, help="Input image size.")


ANCHOR_SCALES = [2 ** (i / 3) for i in range(3)]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
ANCHORS_PER_CELL = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
FEATURE_STRIDES = [8, 16, 32]
FEATURE_CHANNELS = [48, 112, 192]
FEATURE_INDICES = [2, 3, 4]


def build_anchors(image_size):
    all_anchors = []
    for stride in FEATURE_STRIDES:
        feature_size = image_size // stride
        base = stride * 4
        for y in range(feature_size):
            for x in range(feature_size):
                cy = (y + 0.5) * stride
                cx = (x + 0.5) * stride
                for scale in ANCHOR_SCALES:
                    for ratio in ANCHOR_RATIOS:
                        h = base * scale * (ratio ** 0.5)
                        w = base * scale / (ratio ** 0.5)
                        all_anchors.append([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2])
    return torch.tensor(all_anchors, dtype=torch.float32)


class SvhnDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, image_size):
        self.dataset = dataset
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        classes = item["classes"]
        bboxes = item["bboxes"].float()

        orig_h, orig_w = image.shape[1], image.shape[2]
        image = self.transform(image)

        scale_y = self.image_size / orig_h
        scale_x = self.image_size / orig_w
        bboxes = bboxes * torch.tensor([scale_y, scale_x, scale_y, scale_x])

        orig_size = torch.tensor([orig_h, orig_w], dtype=torch.float32)
        return (image, orig_size), (classes, bboxes)


def collate_fn(batch):
    images = torch.stack([item[0][0] for item in batch])
    orig_sizes = torch.stack([item[0][1] for item in batch])
    targets = [item[1] for item in batch]
    return (images, orig_sizes), targets


class DetectionHead(torch.nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.shared = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.cls_head = torch.nn.Conv2d(256, num_anchors * (num_classes + 1), 1)
        self.box_head = torch.nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, x):
        x = self.shared(x)
        cls = self.cls_head(x)
        box = self.box_head(x)
        N = x.shape[0]
        cls = cls.permute(0, 2, 3, 1).reshape(N, -1, self.num_classes + 1)
        box = box.permute(0, 2, 3, 1).reshape(N, -1, 4)
        return cls, box


class SvhnModel(npfl138.TrainableModule):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        self.backbone = backbone

        self.lateral_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(ch, 256, 1) for ch in FEATURE_CHANNELS
        ])

        self.detection_heads = torch.nn.ModuleList([
            DetectionHead(256, ANCHORS_PER_CELL, SVHN.LABELS)
            for _ in FEATURE_STRIDES
        ])

        self.register_buffer("anchors", build_anchors(args.image_size))

    def forward(self, images):
        _, features = self.backbone.forward_intermediates(images)

        all_cls, all_box = [], []
        for i, (feat_idx, lateral) in enumerate(zip(FEATURE_INDICES, self.lateral_convs)):
            feat = lateral(features[feat_idx])
            cls, box = self.detection_heads[i](feat)
            all_cls.append(cls)
            all_box.append(box)

        cls = torch.cat(all_cls, dim=1)
        box = torch.cat(all_box, dim=1)
        return cls, box

    def train_step(self, xs, ys):
        images = xs[0]
        targets = ys

        self.optimizer.zero_grad()
        cls_preds, box_preds = self(images)
        anchors = self.anchors

        batch_cls_targets = []
        batch_box_targets = []
        batch_pos_mask = []

        anchors_cpu = anchors.cpu()
        for gold_classes, gold_bboxes in targets:
            gold_classes = gold_classes.cpu()
            gold_bboxes = gold_bboxes.cpu()
            if len(gold_classes) == 0:
                anchor_cls = torch.zeros(len(anchors), dtype=torch.long, device=self.device)
                anchor_box = torch.zeros(len(anchors), 4, device=self.device)
            else:
                anchor_cls, anchor_box = bboxes_utils.bboxes_training(
                    anchors_cpu, gold_classes, gold_bboxes, self.args.iou_threshold
                )
                anchor_cls = anchor_cls.to(self.device)
                anchor_box = anchor_box.to(self.device)
            batch_cls_targets.append(anchor_cls)
            batch_box_targets.append(anchor_box)
            batch_pos_mask.append(anchor_cls > 0)

        batch_cls_targets = torch.stack(batch_cls_targets)
        batch_box_targets = torch.stack(batch_box_targets)
        batch_pos_mask = torch.stack(batch_pos_mask)

        cls_loss = torch.nn.functional.cross_entropy(
            cls_preds.reshape(-1, SVHN.LABELS + 1),
            batch_cls_targets.reshape(-1),
        )

        if batch_pos_mask.any():
            box_loss = torch.nn.functional.smooth_l1_loss(
                box_preds[batch_pos_mask],
                batch_box_targets[batch_pos_mask],
            )
        else:
            box_loss = cls_preds.sum() * 0

        loss = cls_loss + box_loss
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.detach()}

    def predict_step(self, xs):
        images, orig_sizes = xs[0], xs[1]
        with torch.no_grad():
            cls_preds, box_preds = self(images)

        anchors = self.anchors
        results = []

        for i in range(len(images)):
            orig_h = orig_sizes[i][0].item()
            orig_w = orig_sizes[i][1].item()
            scale = torch.tensor(
                [orig_h / self.args.image_size, orig_w / self.args.image_size,
                 orig_h / self.args.image_size, orig_w / self.args.image_size],
                device=self.device,
            )

            cls_scores = torch.softmax(cls_preds[i], dim=-1)
            fg_scores = cls_scores[:, 1:]
            best_scores, best_classes = fg_scores.max(dim=-1)

            mask = best_scores > self.args.score_threshold
            if not mask.any():
                best_idx = best_scores.argmax()
                mask = torch.zeros(len(best_scores), dtype=torch.bool, device=self.device)
                mask[best_idx] = True

            scores = best_scores[mask]
            classes = best_classes[mask]
            decoded = bboxes_utils.bboxes_from_rcnn(anchors[mask], box_preds[i][mask])
            decoded = decoded.clamp(0, self.args.image_size)

            boxes_xyxy = decoded[:, [1, 0, 3, 2]]
            keep = ops.nms(boxes_xyxy, scores, self.args.nms_threshold)

            final_classes = classes[keep].cpu().numpy()
            final_boxes = (decoded[keep] * scale).cpu().numpy()

            results.append((final_classes, final_boxes))

        return results

    def unpack_batch(self, y, *xs):
        yield from y


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    train_preprocessing = v2.Compose([
        v2.Resize((args.image_size, args.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    test_preprocessing = v2.Compose([
        v2.Resize((args.image_size, args.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    for param in efficientnetv2_b0.parameters():
        param.requires_grad = False

    model = SvhnModel(efficientnetv2_b0, args)
    model.configure(
        optimizer=torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=1e-4,
        ),
        logdir=logdir,
    )

    train_data = SvhnDataset(svhn.train, train_preprocessing, args.image_size)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )

    model.fit(dataloader=train_loader, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        test_data = SvhnDataset(svhn.test, test_preprocessing, args.image_size)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, collate_fn=collate_fn,
        )
        for predicted_classes, predicted_bboxes in model.predict(test_loader, data_with_labels=True):
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
