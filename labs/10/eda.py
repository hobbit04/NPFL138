from npfl138.datasets.morpho_dataset import MorphoDataset
import torch
import npfl138

morpho = MorphoDataset("czech_cac", max_sentences=10)
 
print("\n--- Sample sentences from train ---")
for i in range(3):
    example = morpho.train[i]
    print(f"\nSentence {i+1}:")
    for word, lemma, tag in zip(example["words"], example["lemmas"], example["tags"]):
        print(f"  {word:20s}  lemma={lemma:20s}  tag={tag}")
