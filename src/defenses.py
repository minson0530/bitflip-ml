import csv
import hashlib
import time

import torch

from bits import flip_bit
from flip_one import load_trained_model
from models import load_mnist
from sweep import load_test_tensors, fast_accuracy, RESULTS_PATH, CRITICAL_DROP, BIT

TIMING_RUNS = 50


class NoDefense:
    name = "none"

    def __init__(self, model, train_images):
        pass

    def before_inference(self, model):
        pass


class RangeClamp:
    """Force every weight back into its layer's normal (recorded) range; NaN becomes 0."""
    name = "range clamp"

    def __init__(self, model, train_images):
        self.ranges = {name: (p.min().item(), p.max().item())
                       for name, p in model.named_parameters()}

    def before_inference(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                low, high = self.ranges[name]
                p.nan_to_num_(nan=0.0)      # NaN fails every comparison, so handle it first
                p.clamp_(low, high)


class ActivationClip:
    """Cap every hidden neuron's output at the largest value seen on normal training images."""
    name = "activation clip"

    def __init__(self, model, train_images):
        with torch.no_grad():
            hidden = model.relu(model.hidden(model.flatten(train_images)))
        self.limit = hidden.max().item()
        model.relu.register_forward_hook(self.cap)   # runs right after every ReLU

    def cap(self, module, inputs, output):
        return output.nan_to_num(nan=0.0).clamp(max=self.limit)

    def before_inference(self, model):
        pass                                          # the work happens inside the hook


def fingerprint(tensor):
    """A SHA-256 hash of a tensor's raw bytes: any single flipped bit changes it."""
    return hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest()


class ChecksumRestore:
    """Detect any changed layer by its fingerprint, and copy the clean backup back in."""
    name = "checksum + restore"

    def __init__(self, model, train_images):
        self.backup = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.hashes = {name: fingerprint(p) for name, p in model.named_parameters()}

    def before_inference(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if fingerprint(p) != self.hashes[name]:
                    p.copy_(self.backup[name])


def load_attempts():
    """The 1,000 random flips from the Step 4 sweep, to replay against each defense."""
    with open(RESULTS_PATH) as f:
        return [(r["layer"], int(r["index"])) for r in csv.DictReader(f)]


def replay(defense, model, images, labels, attempts):
    """Re-run every recorded flip with the defense active; return each attempt's accuracy."""
    params = dict(model.named_parameters())
    accuracies = []
    for layer, index in attempts:
        flat = params[layer].view(-1)
        original = flat[index].item()
        with torch.no_grad():
            flat[index] = flip_bit(original, BIT)    # the attack
            defense.before_inference(model)          # the defense gets its chance
            accuracies.append(fast_accuracy(model, images, labels))
            flat[index] = original                   # restore before the next attempt
    return accuracies


def overhead_ms(defense, model):
    """Average time the defense adds before each use of the model, in milliseconds."""
    start = time.perf_counter()
    for _ in range(TIMING_RUNS):
        defense.before_inference(model)
    return (time.perf_counter() - start) / TIMING_RUNS * 1000


def main():
    images, labels = load_test_tensors()
    train_set, _ = load_mnist()
    train_images = train_set.data[:10000].float().div(255).unsqueeze(1)
    attempts = load_attempts()

    print(f"{'defense':<20} {'clean acc':>9} {'critical':>9} {'worst acc':>10} {'overhead':>10}")
    for Defense in (NoDefense, RangeClamp, ActivationClip, ChecksumRestore):
        model = load_trained_model()                  # a fresh, clean model for each defense
        defense = Defense(model, train_images)
        baseline = fast_accuracy(model, images, labels)
        accuracies = replay(defense, model, images, labels, attempts)
        critical = sum(baseline - a > CRITICAL_DROP for a in accuracies)
        print(f"{Defense.name:<20} {baseline:>9.2%} {critical:>5}/{len(attempts)} "
              f"{min(accuracies):>10.2%} {overhead_ms(defense, model):>8.2f}ms")


if __name__ == "__main__":
    main()