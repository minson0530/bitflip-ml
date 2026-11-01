import csv
import random

import torch

from flip_one import load_trained_model
from sweep import load_test_tensors, all_numbers, pick_random_number, SEED, ATTEMPTS, CRITICAL_DROP

RESULTS_PATH = "results/sweep_formats_mnist.csv"

# name: (float type, same-size integer type, exponent's top bit, bits per number)
FORMATS = {
    "fp32": (torch.float32, torch.int32, 30, 32),
    "fp16": (torch.float16, torch.int16, 14, 16),
    "bf16": (torch.bfloat16, torch.int16, 14, 16),
}


def accuracy(model, images, labels):
    """Accuracy on all test images, computed in the model's own format."""
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def flip_in_memory(flat, index, int_type, bit):
    """Flip one bit of flat[index] directly in memory. Calling it twice undoes it."""
    as_int = flat.view(int_type)        # the same memory, read as integers
    as_int[index] ^= (1 << bit)


def sweep_one_format(name, images, labels):
    float_type, int_type, bit, _ = FORMATS[name]
    model = load_trained_model().to(float_type)        # store every weight in this format
    images = images.to(float_type)
    baseline = accuracy(model, images, labels)

    rng = random.Random(SEED)                          # same seed -> same 1,000 weights
    params = all_numbers(model)
    rows = []
    for attempt in range(1, ATTEMPTS + 1):
        layer, flat, index = pick_random_number(params, rng)
        before = flat[index].item()
        bit_before = (flat.view(int_type)[index].item() >> bit) & 1
        with torch.no_grad():
            flip_in_memory(flat, index, int_type, bit)     # simulated Rowhammer flip
            after = flat[index].item()
            acc = accuracy(model, images, labels)
            flip_in_memory(flat, index, int_type, bit)     # flip again = restore
        rows.append({"format": name, "attempt": attempt, "layer": layer, "index": index,
                     "value_before": before, "value_after": after, "bit_before": bit_before,
                     "accuracy": acc, "critical": baseline - acc > CRITICAL_DROP})
        if attempt % 200 == 0:
            print(f"  {name}: {attempt}/{ATTEMPTS} attempts done")
    return baseline, rows


def main():
    images, labels = load_test_tensors()
    all_rows, summary = [], []
    for name in FORMATS:
        baseline, rows = sweep_one_format(name, images, labels)
        all_rows += rows
        critical = sum(r["critical"] for r in rows) / len(rows)
        bits = FORMATS[name][3]
        summary.append((name, baseline, critical, bits))

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'format':<6} {'baseline':>9} {'critical':>9} {'attempts':>9}  {'breaks per 1,000 random memory bits':>36}")
    for name, baseline, critical, bits in summary:
        attempts = f"{1 / critical:.1f}" if critical else "never"
        per_bit = 1000 * critical / bits   # chance a random bit IS the dangerous one, times the chance that breaks the model
        print(f"{name:<6} {baseline:>9.2%} {critical:>9.1%} {attempts:>9}  {per_bit:>36.1f}")
    print(f"\nsaved every attempt to {RESULTS_PATH}")


if __name__ == "__main__":
    main()