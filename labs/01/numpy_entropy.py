#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    count = dict()
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if line in count:
                count[line] += 1
            else:
                count[line] = 1

    # TODO: Load model distribution, each line `string \t probability`.
    probabilties = dict()
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            key = line.split('\t')[0]
            prob = float(line.split('\t')[1])
            probabilties[key] = prob

    data_keys = sorted(count.keys())

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_counts = np.array([count[key] for key in data_keys])
    data_dist = data_counts / np.sum(data_counts)

    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array([probabilties.get(key, 0) for key in data_keys])

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -1 * np.sum(data_dist * np.log(data_dist))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.
    if np.any((model_dist == 0) & (data_dist > 0)):
        crossentropy = np.inf
    else:
        crossentropy = -1 * np.sum(data_dist * np.log(model_dist))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    if crossentropy == np.inf:
        kl_divergence = np.inf
    else:
        kl_divergence = crossentropy - entropy
    
    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print(f"Entropy: {entropy:.2f} nats")
    print(f"Crossentropy: {crossentropy:.2f} nats")
    print(f"KL divergence: {kl_divergence:.2f} nats")
