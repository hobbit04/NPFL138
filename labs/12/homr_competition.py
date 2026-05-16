#!/usr/bin/env python3
import argparse
import os

import torch
import torchvision.transforms.functional as TF

import npfl138
npfl138.require_version("2526.12")
from npfl138.datasets.homr_dataset import HOMRDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--rnn_dim", default=256, type=int, help="BiGRU hidden dim per direction.")
parser.add_argument("--rnn_layers", default=2, type=int, help="Number of BiGRU layers.")

TARGET_HEIGHT = 128
BLANK = 0  # [PAD] is at index 0 in MARKS_VOCAB
WIDTH_REDUCTION = 4  # two MaxPool2d((2,2)) halvings in the width dimension


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # [1, H, W] uint8
        marks = example["marks"]  # [num_marks]

        _, H, W = image.shape
        new_W = round(W * TARGET_HEIGHT / H)
        image = TF.resize(image, [TARGET_HEIGHT, new_W], antialias=True)
        image = image.float() / 255.0

        return image, marks.long()

    def collate(self, batch):
        images, marks = zip(*batch)

        image_widths = torch.tensor([img.shape[-1] for img in images], dtype=torch.long)
        mark_lens = torch.tensor([len(m) for m in marks], dtype=torch.long)

        max_W = max(img.shape[-1] for img in images)
        # Pad with 1.0 (white background) so CNN sees neutral content past image boundary
        padded = torch.ones(len(images), 1, TARGET_HEIGHT, max_W)
        for i, img in enumerate(images):
            padded[i, :, :, :img.shape[-1]] = img

        padded_marks = torch.nn.utils.rnn.pad_sequence(marks, batch_first=True, padding_value=BLANK)

        return (padded, image_widths), (padded_marks, mark_lens)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # CNN reduces height 128→1 via 7 halvings; width reduced 4x by the first two MaxPool(2,2)
        self.cnn = torch.nn.Sequential(
            # Block 1: [B, 1, 128, W] → [B, 32, 64, W/2]
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            # Block 2: → [B, 64, 32, W/4]
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            # Block 3: → [B, 128, 16, W/4]
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
            # Block 4: → [B, 256, 8, W/4]
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
            # Block 5: → [B, 256, 4, W/4]
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
            # Block 6: → [B, 256, 2, W/4]
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
            # Block 7: → [B, 256, 1, W/4]
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 1)),
        )
        self.rnn = torch.nn.GRU(
            input_size=256,
            hidden_size=args.rnn_dim,
            num_layers=args.rnn_layers,
            dropout=0.2 if args.rnn_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(args.rnn_dim * 2, HOMRDataset.MARKS)
        self.ctc_loss = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)

    def forward(self, images, image_widths) -> torch.Tensor:
        features = self.cnn(images)          # [B, 256, 1, W/4]
        B, C, H, W = features.shape
        features = features.reshape(B, C * H, W).permute(0, 2, 1)  # [B, W/4, 256]
        output, _ = self.rnn(features)       # [B, W/4, rnn_dim*2]
        logits = self.fc(output)             # [B, W/4, MARKS]
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def _output_lengths(self, image_widths):
        return image_widths // WIDTH_REDUCTION

    def compute_loss(self, y_pred, y_true, images, image_widths) -> torch.Tensor:
        labels, label_lens = y_true
        output_lengths = self._output_lengths(image_widths)
        return self.ctc_loss(y_pred.permute(1, 0, 2), labels, output_lengths, label_lens)

    def ctc_decode(self, y_pred, image_widths) -> list[torch.Tensor]:
        arg_maxes = torch.argmax(y_pred, dim=-1)  # [B, T]
        output_lengths = self._output_lengths(image_widths)
        predictions = []
        for i in range(len(arg_maxes)):
            seq = arg_maxes[i, :output_lengths[i]]
            decoded, last = [], None
            for token in seq:
                t = token.item()
                if t != last and t != BLANK:
                    decoded.append(t)
                last = t
            predictions.append(torch.tensor(decoded, dtype=torch.long))
        return predictions

    def compute_metrics(self, y_pred, y_true, images, image_widths) -> dict:
        if not self.training:
            labels, label_lens = y_true
            preds = self.ctc_decode(y_pred, image_widths)
            targets = [labels[i, :label_lens[i]] for i in range(len(labels))]
            self.metrics["edit_distance"].update(preds, targets)
        return self.metrics

    def predict_step(self, xs):
        with torch.no_grad():
            images, image_widths = xs
            yield from self.ctc_decode(self.forward(images, image_widths), image_widths)


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    homr = HOMRDataset(decode_on_demand=True)

    train = TrainableDataset(homr.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(homr.dev).dataloader(args.batch_size)
    test = TrainableDataset(homr.test).dataloader(args.batch_size)

    model = Model(args)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        metrics={"edit_distance": HOMRDataset.EditDistanceMetric(ignore_index=BLANK)},
        logdir=logdir,
    )

    model.fit(train, dev=dev, epochs=args.epochs)

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)
        for sequence in predictions:
            print(" ".join(HOMRDataset.MARKS_VOCAB.strings(sequence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
