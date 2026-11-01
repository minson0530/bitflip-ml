import csv
import os
import random

import torch

from bits import flip_bit, float_to_bits
from models import load_mnist
from flip_one import load_trained_model

SEED = 0
ATTEMPTS = 1000              # number of simulated attack attempts
BIT = 30                     # the exponent's top bit in FP32
CRITICAL_DROP = 0.10         # an attempt is "critical" if accuracy falls by more than 10 points
RESULTS_PATH = "results/sweep_mnist_fp32.csv"


def load_test_tensors():
    """Load all 10,000 test images into memory once, as one big tensor."""
    _, test_set = load_mnist()
    images = test_set.data.float().div(255).unsqueeze(1)   # same 0-1 scaling as ToTensor
    labels = test_set.targets
    return images, labels


def fast_accuracy(model, images, labels):
    """Accuracy on all test images in a single pass."""
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def all_numbers(model):
    """Every weight and bias tensor in the model, flattened into one long row each."""
    return [(name, tensor.view(-1)) for name, tensor in model.named_parameters()]


def pick_random_number(params, rng):
    """Pick one of the model's 101,770 numbers uniformly at random.

    Returns the layer name, its flattened tensor, and the position inside it.
    """
    total = sum(flat.numel() for _, flat in params)
    k = rng.randrange(total)
    for name, flat in params:
        if k < flat.numel():
            return name, flat, k
        k -= flat.numel()


def run_sweep(model, images, labels, baseline):
    """Simulate ATTEMPTS random bit-30 flips and record what each one did."""
    rng = random.Random(SEED)
    params = all_numbers(model)
    rows = []
    for attempt in range(1, ATTEMPTS + 1):
        name, flat, index = pick_random_number(params, rng)
        original = flat[index].item()
        bit_before = (float_to_bits(original) >> BIT) & 1     # was the bit a 0 or a 1?
        with torch.no_grad():
            flat[index] = flip_bit(original, BIT)              # simulated Rowhammer flip
            accuracy = fast_accuracy(model, images, labels)
            flipped = flat[index].item()
            flat[index] = original                             # restore before the next attempt
        rows.append({
            "attempt": attempt,
            "layer": name,
            "index": index,
            "value_before": original,
            "value_after": flipped,
            "bit_before": bit_before,
            "accuracy": accuracy,
            "critical": baseline - accuracy > CRITICAL_DROP,
        })
    return rows


def save_rows(rows):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, baseline):
    critical = [r for r in rows if r["critical"]]
    rate = len(critical) / len(rows)
    print(f"baseline accuracy:   {baseline:.2%}")
    print(f"attempts:            {len(rows)}")
    print(f"critical attempts:   {len(critical)}  ({rate:.1%})")
    if critical:
        print(f"expected attempts until first critical: about {1 / rate:.0f}")
        print(f"first critical attempt in this run:     #{critical[0]['attempt']}")
        worst = min(rows, key=lambda r: r["accuracy"])
        print(f"worst attempt: {worst['layer']}[{worst['index']}] "
              f"{worst['value_before']:.4g} -> {worst['value_after']:.4g}, "
              f"accuracy {worst['accuracy']:.2%}")

    print("\nby layer:")
    for layer in sorted({r["layer"] for r in rows}):
        hits = [r for r in rows if r["layer"] == layer]
        bad = [r for r in hits if r["critical"]]
        print(f"  {layer:<15} attempts {len(hits):>4}   critical {len(bad):>3}")


def main():
    images, labels = load_test_tensors()
    model = load_trained_model()
    baseline = fast_accuracy(model, images, labels)
    rows = run_sweep(model, images, labels, baseline)
    save_rows(rows)
    print_summary(rows, baseline)
    print(f"\nsaved every attempt to {RESULTS_PATH}")


if __name__ == "__main__":
    main()