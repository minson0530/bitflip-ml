import csv

from sweep import load_test_tensors, RESULTS_PATH

PIXELS = 784


def load_rows():
    with open(RESULTS_PATH) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["index"] = int(r["index"])
        r["value_before"] = float(r["value_before"])
        r["critical"] = r["critical"] == "True"
    return rows


def critical_rate(rows):
    if not rows:
        return "n/a"
    bad = sum(r["critical"] for r in rows)
    return f"{bad:>3} / {len(rows):<4} = {bad / len(rows):.0%}"


def main():
    rows = [r for r in load_rows() if r["layer"] == "hidden.weight"]

    print("first-layer weights, by sign of the original value:")
    print(f"  positive: {critical_rate([r for r in rows if r['value_before'] > 0])}")
    print(f"  negative: {critical_rate([r for r in rows if r['value_before'] < 0])}")

    # How often each pixel has ink across the 10,000 test images (0.0 to 1.0).
    images, _ = load_test_tensors()
    inked = (images.view(-1, PIXELS) > 0).float().mean(dim=0)

    print("\npositive first-layer weights, by how often their pixel is inked:")
    positive = [r for r in rows if r["value_before"] > 0]
    buckets = [(0.00, 0.01, "almost never (<1%)"),
               (0.01, 0.10, "rarely (1-10%)"),
               (0.10, 0.30, "sometimes (10-30%)"),
               (0.30, 1.01, "often (>30%)")]
    for low, high, label in buckets:
        group = [r for r in positive if low <= inked[r["index"] % PIXELS] < high]
        print(f"  {label:<20} {critical_rate(group)}")


if __name__ == "__main__":
    main()