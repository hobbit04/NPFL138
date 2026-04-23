#!/usr/bin/env python3
import argparse
import os

import torch
import torchaudio.models.decoder

import npfl138
npfl138.require_version("2526.8.1")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--rnn_dim", default=256, type=int)

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        self.rnn = torch.nn.GRU(
            input_size=CommonVoiceCs.MFCC_DIM,
            hidden_size=args.rnn_dim,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = torch.nn.Linear(args.rnn_dim * 2, CommonVoiceCs.LETTERS)

    def forward(self, features, feature_lens) -> torch.Tensor:
        # TODO: Compute the output of the model.
        output, _ = self.rnn(features)
        logits = self.fc(output)

        return torch.nn.functional.log_softmax(logits, dim=-1)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, features, feature_lens) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        labels, label_lens = y_true
        y_pred = y_pred.transpose(0, 1)
        criterion = torch.nn.CTCLoss(
            blank=CommonVoiceCs.LETTERS_VOCAB.index('[PAD]'), 
            zero_infinity=True
        )
        
        loss = criterion(y_pred, labels, feature_lens, label_lens)
        return loss
    
    def ctc_decoding(self, y_pred: torch.Tensor, feature_lens) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        arg_maxes = torch.argmax(y_pred, dim=-1) # [Batch, Time]
        
        predictions = []
        blank_index = CommonVoiceCs.LETTERS_VOCAB.index('[PAD]')
        
        for i in range(len(arg_maxes)):
            seq = arg_maxes[i, :feature_lens[i]]
            decoded = []
            last_token = None
            for token in seq:
                if token != last_token and token != blank_index: 
                    decoded.append(token)
                last_token = token
            predictions.append(torch.tensor(decoded))
            
        return predictions

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, features, feature_lens
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        if not self.training:
            labels, label_lens = y_true
            predictions = self.ctc_decoding(y_pred, feature_lens)
            targets = [labels[i, :label_lens[i]] for i in range(len(labels))]
            
            self.metrics["edit_distance"].update(predictions, targets)
            
        return self.metrics

    def predict_step(self, xs):
        with torch.no_grad():
            # Perform constrained decoding.
            features, feature_lens = xs
            yield from self.ctc_decoding(self.forward(features, feature_lens), feature_lens)

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # You can use `CommonVoiceCs.LETTER_NAMES : list[str]` or `CommonVoiceCs.LETTERS_VOCAB : npfl138.Vocabulary`
        # to convert between letters and their indices. While the letters do not explicitly contain
        # a blank token, the [PAD] token can be employed as one.
        mfccs = torch.tensor(example["mfccs"], dtype=torch.float32)
        sentence = example["sentence"]
        label_indices = torch.tensor(
            CommonVoiceCs.LETTERS_VOCAB.indices(sentence),
            dtype=torch.long
        )

        return mfccs, label_indices

    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        feature_lens = torch.tensor([len(f) for f in features], dtype=torch.long)
        label_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True,
            padding_value=CommonVoiceCs.LETTERS_VOCAB.index('[PAD]'),
        )

        return (features, feature_lens), (labels, label_lens)
    
def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test).dataloader(args.batch_size)

    # TODO: Create the model and train it. The `Model.compute_metrics` method assumes you
    # passed the following metric to the `configure` method under the name "edit_distance":
    #   CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)
    model = Model(args, common_voice.train)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={"edit_distance": CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)


    # Generate test set annotations, but in `model.logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
