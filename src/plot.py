import matplotlib.pyplot as plt

from analyze import load_rows, PIXELS
from sweep import load_test_tensors, fast_accuracy
from flip_one import load_trained_model

FIGURE_PATH = "results/ink_vs_damage.png"
BLUE = "#2a78d6"     # positive weights (the ones that matter)
GREY = "#a3a29d"     # negative weights (background)
INK = "#0b0b0b"      # main text
MUTED = "#52514e"    # secondary text and guide lines


def ink_and_damage(rows, inked, baseline, positive):
    """For each flipped weight of one sign: (% of images inked at its pixel, accuracy points lost)."""
    chosen = [r for r in rows if (r["value_before"] > 0) == positive]
    xs = [inked[r["index"] % PIXELS].item() * 100 for r in chosen]
    ys = [(baseline - float(r["accuracy"])) * 100 for r in chosen]
    return xs, ys


def main():
    images, labels = load_test_tensors()
    baseline = fast_accuracy(load_trained_model(), images, labels)
    inked = (images.view(-1, PIXELS) > 0).float().mean(dim=0)
    rows = [r for r in load_rows() if r["layer"] == "hidden.weight"]

    neg_x, neg_y = ink_and_damage(rows, inked, baseline, positive=False)
    pos_x, pos_y = ink_and_damage(rows, inked, baseline, positive=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(neg_x, neg_y, s=14, color=GREY, alpha=0.6, label="negative weight")
    ax.scatter(pos_x, pos_y, s=14, color=BLUE, alpha=0.8, label="positive weight")

    top = max(pos_x + pos_y) + 5
    ax.plot([0, top], [0, top], linestyle="--", linewidth=1, color=MUTED)
    ax.text(top * 0.60, top * 0.72, "lost = inked", color=MUTED, fontsize=9)
    ax.axhline(10, linestyle=":", linewidth=1, color=MUTED)
    ax.text(top * 0.98, 11, "critical: lost > 10 points", color=MUTED,
            fontsize=9, ha="right")

    ax.set_xlim(0, top)
    ax.set_ylim(-2, top)
    ax.set_xlabel("Test images with ink on the weight's pixel (%)", color=INK)
    ax.set_ylabel("Accuracy lost after the flip (percentage points)", color=INK)
    ax.set_title("A flipped positive weight costs about as much accuracy\n"
                 "as its pixel is inked; negative weights cost nothing",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"saved {FIGURE_PATH}  ({len(pos_x)} positive, {len(neg_x)} negative weights)")


if __name__ == "__main__":
    main()