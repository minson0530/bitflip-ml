# bitflip-ml

**How much damage does one flipped bit do to a neural network, which bits cause it, and can cheap defenses stop it?**

[GPUHammer](https://arxiv.org/pdf/2507.08166) (USENIX Security 2025) showed that Rowhammer can flip bits in GPU memory, and that a single flip in a model weight can drop ImageNet accuracy from 80% to under 0.1%. This project simulates that attack in software to explain *which* flips break a model and *why*, compares weight formats (FP32, FP16, BF16), and tests four defenses.

All bit flips here are simulated by editing weights in memory. No hardware is attacked.

![A flipped positive weight costs about as much accuracy as its pixel is inked; negative weights cost nothing](results/ink_vs_damage.png)

## Key findings

1. **One bit is enough, but only the right one.** Flipping the exponent's most significant bit (0 → 1) turns a small weight like 0.05 into about 10^37. Flips in the 23 mantissa bits change accuracy by at most 0.05 points.
2. **Where the weight sits decides the damage.** On an MNIST classifier (97.23%), a flip in a hidden-layer bias dropped accuracy to 8.92% (the model answered "5" for every image), while flips in two first-layer weights barely mattered.
3. **Random flips are often critical, and far more so in convolutional networks.** On MNIST, 22.6% of 1,000 random flips cut accuracy by more than 10 points (about 4 attempts for a blind attacker). On ResNet50, 64.3% did (about 1.6 attempts), and the worst single flip took accuracy from 89.10% to 0.00%.
4. **Damage is predictable from two facts.** A first-layer flip is critical only if the weight is positive (ReLU silences negative ones: 0 of 471 critical) and its pixel is inked in more than about 10% of images. The accuracy lost is about 0.9 × the share of images inked at that pixel (figure above).
5. **Every positive convolution weight is a target.** In ResNet50, 100% (FP32) and 99% (FP16) of flipped positive conv weights broke the model. A conv weight is applied across the whole image, so unlike an MNIST pixel weight it can never sit unused.
6. **16-bit formats are 1.5–2× more exposed per memory bit.** FP16's worst flip is "only" 65,000× instead of 10^37×, so it breaks the model slightly less often per weight (MNIST 21.5% vs 22.6%; ResNet50 49.7% vs 64.3%). But the dangerous bit is 1 in 16 bits instead of 1 in 32, so a random flipped memory bit is still more likely to break an FP16 model (MNIST 13.4 vs 7.1, ResNet50 31.1 vs 20.1 per 1,000 bits). A 65,000× flip still took ResNet50 to 0.00%, matching GPUHammer's FP16 attack.
7. **Cheap defenses stop the obvious attack (MNIST).** A per-layer range clamp (with a NaN check) stopped all 226 critical flips for 0.45 ms per use. A checksum with a clean backup repaired every flip exactly, at 3× the cost and double the memory.

## Results

### Single flips (MNIST, FP32)

| Target | Before | After bit-30 flip | Accuracy |
| --- | --- | --- | --- |
| None (baseline) | | | 97.23% |
| `hidden.weight[0, 300]`, negative | −0.019 | −6.5 × 10^36 | 97.17% |
| `hidden.weight[0, 249]`, positive | 0.164 | 5.6 × 10^37 | 95.72% |
| `hidden.bias[120]`, positive | 0.215 | 7.3 × 10^37 | **8.92%** |

### 1,000 random flips (MNIST, FP32)

| First-layer weights hit | Critical (> 10 points lost) |
| --- | --- |
| Negative | 0 of 471 (0%) |
| Positive, pixel inked in under 10% of images | 0 of 295 (0%) |
| Positive, pixel inked in 10–30% of images | 65 of 69 (94%) |
| Positive, pixel inked in over 30% of images | 157 of 157 (100%) |

Overall: 226 of 1,000 critical (22.6%), including 4 of 7 output-layer weights. Worst single flip: 19.21%.

### Weight formats (MNIST, same 1,000 weights)

| Format | Baseline | Critical per random weight | Critical per 1,000 random memory bits |
| --- | --- | --- | --- |
| FP32 | 97.23% | 22.6% | 7.1 |
| FP16 | 97.23% | 21.5% | 13.4 |
| BF16 | 97.23% | 22.6% | 14.1 |

### Defenses (MNIST, same 1,000 flips replayed)

| Defense | Still critical | Worst accuracy | Overhead per use |
| --- | --- | --- | --- |
| None | 226 / 1000 | 19.21% | 0 ms |
| Range clamp + NaN check | 0 / 1000 | 97.13% | 0.45 ms |
| Activation clip (hidden layer) | 6 / 1000 | 19.21% | 0.01 ms |
| Checksum + restore | 0 / 1000 | 97.23% | 1.44 ms |

No defense changed clean accuracy. The activation clip cannot protect the output layer, which feeds the scores directly.

### ResNet50 (ImageNet model)

Pretrained ResNet50 (`IMAGENET1K_V2`, one of GPUHammer's five models) on a fixed sample of 1,000 [Imagenette](https://github.com/fastai/imagenette) validation photos, scored against all 1,000 ImageNet classes. 300 random exponent-MSB flips per format, the same 300 weights in both, on a Colab T4 GPU.

| | MNIST (FP32) | ResNet50 FP32 | ResNet50 FP16 |
| --- | --- | --- | --- |
| Baseline | 97.23% | 89.10% | 89.20% |
| Random flips critical | 22.6% | **64.3%** | **49.7%** |
| Expected attempts | about 4 | about 1.6 | about 2 |
| Worst single flip | 19.21% | **0.00%** | **0.00%** |
| Critical per 1,000 random memory bits | 7.1 | 20.1 | 31.1 |

Worst flips: FP32 `layer4.1.conv2.weight` 0.0186 → 6.3 × 10^36; FP16 `layer2.3.conv2.weight` 0.0223 → 1,463. For comparison, GPUHammer measured ResNet50 at 80.26% → 0.02–0.08% top-1 on full ImageNet with real FP16 flips.

| ResNet50 conv weights hit | FP32 critical | FP16 critical |
| --- | --- | --- |
| Positive | 143 of 143 (100%) | 141 of 143 (99%) |
| Negative | 41 of 133 (31%) | 0 of 133 (0%) |

The whole FP32-vs-FP16 gap comes from negative weights. In FP16 a flipped negative weight becomes a finite number around −1,000, which ReLU silences as on MNIST. In FP32 it becomes about −10^37, which can overflow the format: 3 of 10 critical negative flips tested produced NaN outputs, and 0 of 10 harmless ones did. The cause of the other 7 is an open question (see Limitations).

By layer type (FP32): conv 184 of 276 critical, final fully connected layer 9 of 23, batchnorm 0 of 1.

## Limitations

- **Simulation, not a hardware attack.** Flips are written directly into memory; which bits real hardware can flip, and in which direction, is not modelled. Real Rowhammer flips are usually one-directional, while this project flips regardless of direction (the original bit is recorded in the CSVs).
- **Sample sizes.** MNIST results use one seed and 1,000 flips. ResNet50 results use 300 flips per format and 1,000 Imagenette photos (10 easy ImageNet classes), not the full ImageNet validation set, so its rates carry more uncertainty (roughly ±5 points) and its baseline is higher than GPUHammer's.
- **Open question: negative ResNet50 weights in FP32.** 31% of flipped negative conv weights broke the FP32 model but none broke FP16. Overflow into NaN explains 3 of the 10 cases checked; the NaN check used only 250 of the 1,000 photos, so it may undercount. Candidates for the rest are overflow to infinity (without NaN) turned positive by a batchnorm layer with a negative scale, or the first conv layer, whose inputs can be negative.
- **Exposure per memory bit** assumes only the exponent's most significant bit is dangerous. Step 1 supports this for small weights, but every bit position was not swept in every format.
- **Defenses were tested on MNIST, against bit-30 flips only.** An attacker who keeps the damage inside the normal range (a sign flip, a lower exponent bit) would pass the range clamp and the activation clip. The checksum's backup lives in the same memory and could in principle be attacked too.

## Reproduce

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/bits.py            # Step 1: flip each bit of sample FP32 values
python src/models.py          # Step 2: train the MNIST model (about 1-2 min), saves results/mnist_fp32.pt
python src/flip_one.py        # Step 3: single flips on chosen weights and a bias
python src/sweep.py           # Step 4: 1,000 random flips, saves results/sweep_mnist_fp32.csv
python src/analyze.py         #         split results by weight sign and pixel ink
python src/plot.py            #         figure: results/ink_vs_damage.png
python src/sweep_formats.py   # Step 4c: FP32 vs FP16 vs BF16
python src/defenses.py        # Step 5: replay the 1,000 flips against four defenses
```

Every script uses a fixed seed, so results match the tables above. MNIST downloads automatically to `data/`.

The ResNet50 experiment is in `notebooks/resnet50_colab.ipynb`. Open it in [Google Colab](https://colab.research.google.com/), choose a GPU runtime, and run the cells in order (about 20 minutes on a T4). It downloads ResNet50 and Imagenette automatically and writes `resnet50_sweep.csv`.

## Repository layout

```
src/
  bits.py            flip a bit in an FP32 value; field names
  models.py          MNIST data, the network, training and evaluation
  flip_one.py        flip every tested bit of one chosen number
  sweep.py           random-flip sweep and fast evaluation
  analyze.py         critical rate by weight sign and pixel ink
  plot.py            ink-vs-damage figure
  sweep_formats.py   the sweep in FP32, FP16 and BF16
  defenses.py        four defenses, replayed against the same flips
notebooks/
  resnet50_colab.ipynb   ResNet50 sweep in FP32 and FP16, with outputs
results/             trained model, CSVs of every attempt, figures
```

## References

- S. Lin et al., [GPUHammer: Rowhammer Attacks on GPU Memories are Practical](https://arxiv.org/pdf/2507.08166), USENIX Security 2025.
- S. Hong et al., [Terminal Brain Damage: Exposing the Graceless Degradation in Deep Neural Networks Under Hardware Fault Attacks](https://www.usenix.org/conference/usenixsecurity19/presentation/hong), USENIX Security 2019. This project uses their simulation method.
- Y. Kim et al., [Flipping Bits in Memory Without Accessing Them](https://users.ece.cmu.edu/~yoonguk/papers/kim-isca14.pdf), ISCA 2014.