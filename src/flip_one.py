import torch

from bits import flip_bit, field_name
from models import DigitClassifier, load_mnist, evaluate, MODEL_PATH

BITS_TO_TEST = [31, 30, 29, 28, 25, 23, 22, 15, 0]


def load_trained_model():
    """Rebuild the network and fill it with the saved, trained weights."""
    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


def flip_and_measure(model, test_set, tensor, index):
    """Flip each bit of tensor[index] in turn, measure accuracy, then restore it."""
    original = tensor[index].item()
    results = []
    for position in BITS_TO_TEST:
        with torch.no_grad():
            tensor[index] = flip_bit(original, position)   # simulated Rowhammer flip
            accuracy = evaluate(model, test_set)
            new_value = tensor[index].item()
            tensor[index] = original                       # restore before the next test
        results.append((position, new_value, accuracy))
    return original, results


def print_results(name, original, results):
    print(f"\n=== {name} = {original:.4g}")
    print(f"{'bit':>3}  {'field':<8}  {'new value':>14}  accuracy")
    for position, new_value, accuracy in results:
        print(f"{position:>3}  {field_name(position):<8}  {new_value:>14.4g}  {accuracy:.2%}")


def main():
    _, test_set = load_mnist()
    model = load_trained_model()
    print(f"baseline accuracy: {evaluate(model, test_set):.2%}")

    weights = model.hidden.weight      # (128, 784): one weight per neuron per pixel
    biases = model.hidden.bias         # (128,): one constant per neuron

    best_pixel = weights[0].argmax().item()    # neuron 0's most positive weight
    best_neuron = biases.argmax().item()       # the neuron with the most positive bias

    targets = [
        ("hidden.weight[0, 300] (negative)", weights, (0, 300)),
        (f"hidden.weight[0, {best_pixel}] (positive)", weights, (0, best_pixel)),
        (f"hidden.bias[{best_neuron}] (positive)", biases, (best_neuron,)),
    ]
    for name, tensor, index in targets:
        original, results = flip_and_measure(model, test_set, tensor, index)
        print_results(name, original, results)


if __name__ == "__main__":
    main()