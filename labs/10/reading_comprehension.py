#!/usr/bin/env python3
import argparse
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import torch
import transformers
from transformers import get_linear_schedule_with_warmup

import npfl138
npfl138.require_version("2526.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="Learning rate.")
parser.add_argument("--max_length", default=384, type=int, help="Max sequence length.")
parser.add_argument("--doc_stride", default=128, type=int, help="Sliding window stride.")
parser.add_argument("--n_best", default=20, type=int, help="Top-N start/end candidates.")
parser.add_argument("--max_answer_len", default=30, type=int, help="Max answer span in tokens.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def find_token_span(answer_start_char, answer_end_char, offsets, seq_ids):
    """Convert character-level [start, end) to token indices, or (0, 0) if not in window."""
    ctx_start = next((i for i, s in enumerate(seq_ids) if s == 1), None)
    ctx_end = next((i for i in range(len(seq_ids) - 1, -1, -1) if seq_ids[i] == 1), None)
    if ctx_start is None:
        return 0, 0

    ctx_char_start = offsets[ctx_start][0]
    ctx_char_end = offsets[ctx_end][1]
    if answer_start_char < ctx_char_start or answer_end_char > ctx_char_end:
        return 0, 0

    start_pos = end_pos = None
    for i in range(ctx_start, ctx_end + 1):
        tok_s, tok_e = offsets[i]
        if start_pos is None and tok_s <= answer_start_char < tok_e:
            start_pos = i
        if tok_s < answer_end_char <= tok_e:
            end_pos = i

    if start_pos is None or end_pos is None or start_pos > end_pos:
        return 0, 0
    return start_pos, end_pos


class QADataset(torch.utils.data.Dataset):
    """Flat dataset of tokenized QA features with sliding window over long contexts."""

    def __init__(self, paragraphs, tokenizer, max_length, doc_stride):
        self.input_ids = []
        self.attention_masks = []
        self.start_positions = []
        self.end_positions = []

        # Inference metadata (one entry per feature)
        self.example_ids = []
        self.offset_mappings = []
        self.sequence_ids_list = []
        self.contexts = []
        self.n_examples = 0

        for para in paragraphs:
            context = para["context"]
            for qa in para["qas"]:
                example_id = self.n_examples
                self.n_examples += 1
                answers = qa["answers"]

                encoding = tokenizer(
                    qa["question"],
                    context,
                    max_length=max_length,
                    truncation="only_second",
                    stride=doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                for w in range(encoding["input_ids"].shape[0]):
                    offsets = encoding["offset_mapping"][w].tolist()
                    seq_ids = encoding.sequence_ids(w)

                    self.input_ids.append(encoding["input_ids"][w])
                    self.attention_masks.append(encoding["attention_mask"][w])
                    self.example_ids.append(example_id)
                    self.offset_mappings.append(offsets)
                    self.sequence_ids_list.append(seq_ids)
                    self.contexts.append(context)

                    if answers:
                        ans = answers[0]
                        start_pos, end_pos = find_token_span(
                            ans["start"], ans["start"] + len(ans["text"]), offsets, seq_ids
                        )
                    else:
                        start_pos, end_pos = 0, 0

                    self.start_positions.append(torch.tensor(start_pos, dtype=torch.long))
                    self.end_positions.append(torch.tensor(end_pos, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            (self.input_ids[idx], self.attention_masks[idx]),
            (self.start_positions[idx], self.end_positions[idx]),
        )


class Model(npfl138.TrainableModule):
    def __init__(self):
        super().__init__()
        self._qa = transformers.AutoModelForQuestionAnswering.from_pretrained("ufal/robeczech-base")

    def forward(self, input_ids, attention_mask):
        out = self._qa(input_ids=input_ids, attention_mask=attention_mask)
        return out.start_logits, out.end_logits

    def train_step(self, xs, y):
        input_ids, attention_mask = xs
        start_pos, end_pos = y

        out = self._qa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_pos,
            end_positions=end_pos,
        )
        loss = self.track_loss(out.loss)
        loss.backward()

        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler is not None and self.scheduler.step()

        return {**({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {}), **self.losses}

    def test_step(self, xs, y):
        with torch.no_grad():
            input_ids, attention_mask = xs
            start_pos, end_pos = y
            out = self._qa(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            self.track_loss(out.loss)
        return {**self.losses}


def predict_answers(model, qa_dataset, n_best, max_answer_len):
    """Run model on all windows, aggregate logits per question, extract best text span."""
    loader = torch.utils.data.DataLoader(qa_dataset, batch_size=64, shuffle=False)

    all_start_logits = []
    all_end_logits = []

    model.eval()
    with torch.no_grad():
        for (input_ids, attn_mask), _ in loader:
            input_ids = input_ids.to(model.device)
            attn_mask = attn_mask.to(model.device)
            out = model._qa(input_ids=input_ids, attention_mask=attn_mask)
            all_start_logits.append(out.start_logits.cpu().numpy())
            all_end_logits.append(out.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)

    example_to_features = {}
    for feat_idx, eid in enumerate(qa_dataset.example_ids):
        example_to_features.setdefault(eid, []).append(feat_idx)

    predictions = []
    for example_id in range(qa_dataset.n_examples):
        best_score = float("-inf")
        best_text = ""

        for feat_idx in example_to_features.get(example_id, []):
            sl = all_start_logits[feat_idx]
            el = all_end_logits[feat_idx]
            offsets = qa_dataset.offset_mappings[feat_idx]
            seq_ids = qa_dataset.sequence_ids_list[feat_idx]
            context = qa_dataset.contexts[feat_idx]

            start_candidates = np.argsort(sl)[-n_best:][::-1]
            end_candidates = np.argsort(el)[-n_best:][::-1]

            for s in start_candidates:
                if seq_ids[s] != 1:
                    continue
                for e in end_candidates:
                    if seq_ids[e] != 1 or e < s or e - s + 1 > max_answer_len:
                        continue
                    score = sl[s] + el[e]
                    if score > best_score:
                        best_text = context[offsets[s][0]:offsets[e][1]]
                        best_score = score

        predictions.append(best_text)

    return predictions


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    dataset = ReadingComprehensionDataset()

    print("Preprocessing datasets...")
    train_ds = QADataset(dataset.train.paragraphs, tokenizer, args.max_length, args.doc_stride)
    dev_ds = QADataset(dataset.dev.paragraphs, tokenizer, args.max_length, args.doc_stride)
    test_ds = QADataset(dataset.test.paragraphs, tokenizer, args.max_length, args.doc_stride)
    print(f"  train: {len(train_ds)} features / {train_ds.n_examples} questions")
    print(f"  dev:   {len(dev_ds)} features / {dev_ds.n_examples} questions")
    print(f"  test:  {len(test_ds)} features / {test_ds.n_examples} questions")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
    )

    model = Model()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.configure(optimizer=optimizer, scheduler=scheduler, logdir=logdir)

    model.fit(train_loader, epochs=args.epochs)
    dev_preds = predict_answers(model, dev_ds, args.n_best, args.max_answer_len)
    acc = ReadingComprehensionDataset.evaluate(dataset.dev, dev_preds)
    print(f"Epoch {model.epoch} dev accuracy: {100 * acc:.2f}%")

    os.makedirs(logdir, exist_ok=True)
    test_preds = predict_answers(model, test_ds, args.n_best, args.max_answer_len)
    with open(os.path.join(logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as f:
        for ans in test_preds:
            print(ans, file=f)

    print(f"Saved {len(test_preds)} test predictions to {logdir}/reading_comprehension.txt")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
