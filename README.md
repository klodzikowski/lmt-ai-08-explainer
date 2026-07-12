# lmt-ai-08-explainer

An interactive MNIST network explainer: a worked example for the Class 8 homework
("vibe-code a tool that helps you understand our neural network"). Draw a digit and watch
it travel through the course network (the 784-64-10 dense net) from raw pixels to
hidden-layer activations to the final 10 probabilities. Everything runs live in the
browser from one self-contained `index.html`: real trained weights, real forward passes,
no internet needed.

Live app: <https://klodzikowski.github.io/lmt-ai-08-explainer/>

## How to use it in class

- Open the [live app](https://klodzikowski.github.io/lmt-ai-08-explainer/), or open
  `index.html` locally (it works offline either way).
- Draw a digit with the mouse or a finger, or tap one of the ten real handwritten test
  digits, and walk the class through the four panels left to right: drawing → the 784
  numbers the network actually receives → which of the 64 hidden neurons fire → the
  10 output probabilities.
- Teaching moments that come up naturally: clear the canvas and note the network *still*
  outputs probabilities summing to 100% (softmax must place a bet); draw an ambiguous
  shape and watch the probabilities split; hover over the hidden-layer bars to see
  individual neurons fire or stay silent; hover over the row of numbers under the
  28 × 28 grid to see exactly which pixels they are.

## How it works, in plain language

**The network.** The same architecture as in our course: 784 inputs (one per pixel of a
28 × 28 image), a hidden layer of 64 neurons, and 10 outputs (one per digit). Each hidden
neuron multiplies all 784 pixel values by its own learnt weights, adds them up, and
passes the total through ReLU (negative totals become 0, so the neuron stays silent). The
10 output neurons do the same over the 64 hidden values, and softmax turns their scores
into percentages that always add up to 100%.

**The weights are real.** They were trained with `tools/train.js` (plain Node.js, no
libraries) on the 60,000 handwritten digits of the classic MNIST training set (the same
dataset from our course), using mini-batch gradient descent on cross-entropy loss, with
small random image shifts added for robustness. The trained network scores **97.7%** on
the 10,000 MNIST test digits it never saw during training. The weights are compressed
(8-bit quantisation) and baked into `index.html` as text, which is why one HTML file can
classify digits with zero downloads.

**What happens when you draw.** Your 280 × 280 sketch is shrunk to 28 × 28 by averaging
blocks of pixels, then cropped, scaled and centred the same way the original MNIST images
were prepared (that is why the "what the network sees" panel can look shifted compared
to your drawing). Those 784 numbers then flow through the network described above, on
every stroke, in a few microseconds.

**Checking the maths.** `tools/verify.js` extracts the weights *and* the forward-pass
code out of the shipped `index.html` and re-runs them in Node against the held-out MNIST
test set: it confirms 97.7% accuracy on 10,000 unseen digits, that all softmax outputs
sum to 1, and that the shipped code matches an independent reimplementation of the
network. Run it yourself:

```
node tools/verify.js index.html <folder with the gzipped MNIST files>
```

## Files

- `index.html`: the whole app (page, network, trained weights, sample digits).
- `tools/train.js`: trains the network on MNIST and writes `tools/weights.json`.
- `tools/verify.js`: proves the shipped file classifies held-out digits correctly.
