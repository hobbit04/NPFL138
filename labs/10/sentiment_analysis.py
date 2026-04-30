#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import torchmetrics
import transformers
from transformers import get_linear_schedule_with_warmup


import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.3, type=float, help="Probability of dropout layer")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate")
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self._eleczech = eleczech
        hidden_dim = eleczech.config.hidden_size

        self._dropout = torch.nn.Dropout(args.dropout)
        self._classifier = torch.nn.Linear(hidden_dim, len(dataset.label_vocab))

    # TODO: Implement the model computation.
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self._eleczech(input_ids, attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self._dropout(cls)
        return self._classifier(cls)  


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer, max_length: int = 512) -> None:
        super().__init__(dataset)
        self._tokenizer = tokenizer
        self._max_length = max_length
    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        enc = self._tokenizer(
            example["document"],
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        label_str = example["label"]
        if not label_str:
            label = torch.tensor(-1, dtype=torch.long)
        else:
            label = torch.tensor(
                self.dataset.label_vocab.index(label_str),
                dtype=torch.long,
            )
        return (input_ids, attention_mask), label

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        inputs, labels = zip(*batch)
        input_ids, attention_masks = zip(*inputs)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self._tokenizer.pad_token_id,
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0,
        )
        labels = torch.stack(labels)
        return (input_ids, attention_masks), labels


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train = TrainableDataset(facebook.train, tokenizer).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(facebook.dev, tokenizer).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(facebook.test, tokenizer).dataloader(batch_size=args.batch_size) 

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    steps_per_epoch = len(train)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model.configure(
        optimizer=optimizer,  # maybe also a scheduler, but not required
        scheduler=scheduler,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(
        "multiclass", num_classes=len(facebook.train.label_vocab))},
        logdir=logdir,
    )
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test, data_with_labels=True)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(document_logits.argmax().item()), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
