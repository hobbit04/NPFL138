import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.9")
from npfl138.datasets.morpho_dataset import MorphoDataset


npfl138.startup(41, 0, False)
npfl138.global_keras_initializers()

# Load the data.
morpho = MorphoDataset("czech_cac", max_sentences=None)
# Print dataset sizes
print(f"Train: {len(morpho.train)} sentences, Dev: {len(morpho.dev)} sentences, Test: {len(morpho.test)} sentences")
print(f"Vocab size — words: {len(morpho.train.words.string_vocab)}, tags: {len(morpho.train.tags.string_vocab)}")

# # Print a few example sentences
# print("\n--- Sample sentences from train ---")
# for i in range(3):
#     example = morpho.train[i]
#     print(f"\nSentence {i+1}:")
#     for word, lemma, tag in zip(example["words"], example["lemmas"], example["tags"]):
#         print(f"  {word:20s}  lemma={lemma:20s}  tag={tag}")
print(morpho.train[0])

batch = [
    (["HI", "my", "name", "is"], ["hi", "me", "name", "be"]),
    (["Nice", "to", "meet", "you!"], ["nice", "to", "meet", "you"])
]
words, lemmas = zip(*batch)
print("words:", words)
print("\nlemmas:", lemmas)
