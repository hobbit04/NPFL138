#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import transformers



import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Model(npfl138.TrainableModule):
    def __init__(self, robeczech: transformers.PreTrainedModel) -> None:
        super().__init__()
        self._robeczech = robeczech

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self._robeczech(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.start_logits, outputs.end_logits

    def compute_loss(self, y_pred, y, *xs):
        start_logits, end_logits = y_pred
        start_positions, end_positions = y
        loss_fct = torch.nn.CrossEntropyLoss()
        return (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, examples: list, tokenizer, max_length: int = 512, training: bool = True) -> None:
        super().__init__(examples)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._training = training

    def transform(self, example):
        encoding = self._tokenizer(
            example["question"],
            example["context"],
            max_length=self._max_length,
            truncation="only_second",  # never cut the question
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=False,
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        sequence_ids   = encoding.sequence_ids(0)  # None=special, 0=question, 1=context

        if self._training:
            ans_start_char = example["answer"]["start"]
            ans_end_char   = ans_start_char + len(example["answer"]["text"])

            start_position = end_position = 0

            for i, (seq_id, (char_s, char_e)) in enumerate(zip(sequence_ids, offset_mapping)):
                if seq_id != 1:
                    continue
                if char_s <= ans_start_char < char_e:
                    start_position = i
                if char_s < ans_end_char <= char_e:
                    end_position = i

            return (
                (input_ids, attention_mask),
                (torch.tensor(start_position), torch.tensor(end_position)),
            )
        else:
            context_start = next(i for i, s in enumerate(sequence_ids) if s == 1)
            return (
                (input_ids, attention_mask),
                (offset_mapping, example["context"], context_start),
            )
        
    def collate(self, batch):
        inputs, targets = zip(*batch)
        input_ids, attention_masks = zip(*inputs)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self._tokenizer.pad_token_id,
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0,
        )

        if self._training:
            start_positions, end_positions = zip(*targets)
            return (
                (input_ids, attention_masks),
                (torch.stack(list(start_positions)), torch.stack(list(end_positions))),
            )
        else:
            offset_mappings, contexts, context_starts = zip(*targets)
            return (
                (input_ids, attention_masks),
                (list(offset_mappings), list(contexts), list(context_starts)),
            )


def flatten(split) -> list:
    examples = []
    for p in split.paragraphs:
        for qa in p["qas"]:
            answer = qa["answers"][0] if qa["answers"] else {"text": "", "start": 0}
            examples.append({
                "context":  p["context"],
                "question": qa["question"],
                "answer":   answer,
            })
    return examples

def predict_answers(model, dataloader):
    predictions = []
    model.eval()
    with torch.no_grad():
        for (input_ids, attention_masks), (offset_mappings, contexts, context_starts) in dataloader:
            input_ids       = input_ids.to(model.device)
            attention_masks = attention_masks.to(model.device)
            start_logits, end_logits = model(input_ids, attention_masks)

            for b in range(len(contexts)):
                s_log        = start_logits[b].cpu()
                e_log        = end_logits[b].cpu()
                offset_map   = offset_mappings[b]
                context      = contexts[b]
                ctx_start    = context_starts[b]

                ctx_end = len(offset_map) - 1
                while ctx_end > ctx_start and offset_map[ctx_end] == (0, 0):
                    ctx_end -= 1

                best_score = float("-inf")
                best_start = best_end = ctx_start
                for s in range(ctx_start, ctx_end + 1):
                    for e in range(s, ctx_end + 1):
                        score = s_log[s].item() + e_log[e].item()
                        if score > best_score:
                            best_score = score
                            best_start, best_end = s, e

                char_s = offset_map[best_start][0]
                char_e = offset_map[best_end][1]
                predictions.append(context[char_s:char_e])
    return predictions


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.AutoModelForQuestionAnswering.from_pretrained("ufal/robeczech-base")

    # Load the data
    dataset = ReadingComprehensionDataset()
    train     = TrainableDataset(flatten(dataset.train), tokenizer, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev_train = TrainableDataset(flatten(dataset.dev),   tokenizer, training=True).dataloader(batch_size=args.batch_size)
    dev       = TrainableDataset(flatten(dataset.dev),   tokenizer, training=False).dataloader(batch_size=args.batch_size)
    test      = TrainableDataset(flatten(dataset.test),  tokenizer, training=False).dataloader(batch_size=args.batch_size)

    # train_paragraphs = dataset.train.paragraphs
    # for i in range(2):
    #     p = train_paragraphs[i]
    #     print(f"\n[Paragraph {i+1}]")
    #     print(f"Context: {p['context'][:150]}...")

    #     for j, qa in enumerate(p["qas"]):
    #         print(f"  Q{j+1}: {qa['question']}")
            
    #         for k, answer in enumerate(qa["answers"]):
    #             ans_text = answer["text"]
    #             ans_start = answer["start"]
                
    #             print(f"    A{k+1}: {ans_text:20s} (start_offset={ans_start})")
                
    # TODO: Create the model and train it.
    model = Model(robeczech)
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01),
        logdir=logdir,
    )
    model.fit(train, dev=dev_train, epochs=args.epochs)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = predict_answers(model, test)

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)