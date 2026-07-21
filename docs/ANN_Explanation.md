# Artificial Neural Network (ANN) — Complete Explanation

## Table of Contents
1. [The Big Picture — What Is An ANN?](#1-the-big-picture)
2. [The Biological Inspiration — The Human Neuron](#2-the-biological-inspiration)
3. [The Artificial Neuron (Perceptron)](#3-the-artificial-neuron)
4. [Weights and Biases — The Core Math](#4-weights-and-biases)
5. [Activation Functions — Why We Need Them](#5-activation-functions)
6. [ReLU — The Star of Modern Deep Learning](#6-relu)
7. [Putting It Together — A Full Neural Network](#7-putting-it-together)
8. [Forward Propagation — How the Network Makes a Prediction](#8-forward-propagation)
9. [Loss Function — How the Network Knows It's Wrong](#9-loss-function)
10. [Backpropagation — How the Network Learns](#10-backpropagation)
11. [Gradient Descent & Adam Optimizer](#11-gradient-descent--adam-optimizer)
12. [Dropout — Preventing Overfitting](#12-dropout)
13. [Softmax — Turning Numbers Into Probabilities](#13-softmax)
14. [Our Specific ANN Architecture (from ann.py)](#14-our-specific-ann-architecture)
15. [The Complete Training Loop (from ann.py)](#15-the-complete-training-loop)
16. [Quick Reference Cheat Sheet](#16-quick-reference-cheat-sheet)

---

## 1. The Big Picture

### What is an ANN?

An **Artificial Neural Network (ANN)** is a computer program that is loosely inspired by how the human brain works. It is a mathematical function that takes in some numbers, does a series of calculations, and spits out a decision.

In our project, the ANN is the **final decision-maker**. It takes in a supervector (a list of 2,496 numbers that mathematically describes an audio clip) and outputs a prediction like "This audio is Hindi" or "This audio is Punjabi."

### Why not just use IF-ELSE statements?

You might wonder: "Why can't we just write a bunch of IF-ELSE rules to identify languages?"

The problem is that language patterns are incredibly complex. There are thousands of subtle phonetic differences between Hindi and Urdu, or Bengali and Nepali. No human could ever write enough rules to capture all of them.

Instead, we let the ANN **learn** these rules automatically by showing it thousands of examples. After seeing enough training data, the ANN figures out the patterns on its own — patterns so subtle and complex that no human could have programmed them manually.

---

## 2. The Biological Inspiration

### How a Human Brain Neuron Works

Your brain contains approximately 86 billion neurons. Each neuron works like this:

```
              Dendrites (Inputs)
                 │  │  │
                 ▼  ▼  ▼
            ┌──────────────┐
            │   Cell Body  │  ← Processes the signals
            │  (Soma)      │
            └──────┬───────┘
                   │
                   ▼
               Axon (Output)
                   │
                   ▼
            Next Neuron...
```

1. **Dendrites** receive electrical signals from other neurons.
2. The **Cell Body (Soma)** adds up all the incoming signals.
3. If the total signal exceeds a certain **threshold**, the neuron "fires" and sends a signal down its **Axon** to the next neuron.
4. If the total signal is too weak, the neuron stays quiet (doesn't fire).

This "fire or don't fire" decision is the key concept that inspired artificial neural networks.

---

## 3. The Artificial Neuron

### The Perceptron (A Single Artificial Neuron)

An artificial neuron mimics the biological neuron using pure math:

```
   x₁ ──→ ×w₁ ──┐
                  │
   x₂ ──→ ×w₂ ──┼──→ Σ (sum) ──→ f(z) ──→ output
                  │
   x₃ ──→ ×w₃ ──┘
                  │
          bias b ─┘
```

Here's what each part does:

| Part | Biological Equivalent | What It Does |
|------|----------------------|--------------|
| `x₁, x₂, x₃` | Dendrites (inputs) | The input numbers (features) |
| `w₁, w₂, w₃` | Synapse strength | How important each input is |
| `Σ (sum)` | Cell Body (Soma) | Adds up all the weighted inputs |
| `f(z)` | Firing threshold | Decides whether to "fire" or not |
| `output` | Axon | The final signal sent to the next neuron |
| `b` (bias) | Resting potential | A baseline value added to the sum |

---

## 4. Weights and Biases — The Core Math

### The Formula for a Single Neuron

The math inside every single artificial neuron is actually very simple:

```
z = (w₁ × x₁) + (w₂ × x₂) + (w₃ × x₃) + b
```

Or more compactly:

```
z = Σ (wᵢ × xᵢ) + b
```

Where:
- `x` = the input values (the features)
- `w` = the weights (how important each input is)
- `b` = the bias (a constant that shifts the output)
- `z` = the raw output (before the activation function)

### What Are Weights?

**Weights** are the most important numbers in the entire neural network. They control how much attention the neuron pays to each input.

**Example:** Imagine a neuron is trying to decide if an audio clip is Hindi.
- If weight `w₁ = 0.9` (high), the neuron pays a LOT of attention to input `x₁`.
- If weight `w₂ = 0.01` (low), the neuron almost completely ignores input `x₂`.
- If weight `w₃ = -0.5` (negative), the neuron treats input `x₃` as **evidence against** Hindi.

When we say the neural network is "learning," what we really mean is: **the weights are being adjusted** over and over until the predictions become accurate.

### What Is the Bias?

The **bias** (`b`) is like a baseline. It allows the neuron to fire even if all the inputs are zero.

Think of it like this: If you're grading a test, the weights determine how much each question is worth, but the bias is like giving everyone 5 bonus points just for showing up. It shifts the entire curve up or down.

### Numerical Example

Let's say we have a neuron with 3 inputs:

```
Inputs:  x₁ = 2.0,  x₂ = 3.0,  x₃ = -1.0
Weights: w₁ = 0.5,  w₂ = -0.3, w₃ = 0.8
Bias:    b = 0.1

z = (0.5 × 2.0) + (-0.3 × 3.0) + (0.8 × -1.0) + 0.1
z = 1.0 + (-0.9) + (-0.8) + 0.1
z = -0.6
```

The raw output `z = -0.6`. But we don't use this number directly. We first pass it through an **activation function**.

---

## 5. Activation Functions — Why We Need Them

### The Problem Without Activation Functions

Without activation functions, a neural network is just a fancy way of doing **linear algebra** (multiplying and adding numbers). No matter how many layers you stack, the output would always be a straight line.

**Why is that a problem?** Because language classification is NOT a straight-line problem. The relationship between audio features and language is incredibly complex, curvy, and nonlinear.

### What Does an Activation Function Do?

An activation function takes the raw output `z` and **bends** it — it introduces curves, kinks, and nonlinear behavior into the math. This is what allows neural networks to learn complex, non-obvious patterns.

### Common Activation Functions

| Function | Formula | Range | Used For |
|----------|---------|-------|----------|
| **Sigmoid** | σ(z) = 1 / (1 + e⁻ᶻ) | (0, 1) | Old networks, binary classification |
| **Tanh** | tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ) | (-1, 1) | Old networks, centered data |
| **ReLU** | f(z) = max(0, z) | [0, ∞) | **Modern networks (OURS!)** |
| **Softmax** | See Section 13 | (0, 1) | Output layer for classification |

---

## 6. ReLU — The Star of Modern Deep Learning

### What Is ReLU?

**ReLU** stands for **Rectified Linear Unit**. Despite the scary name, it is the simplest activation function in existence:

```
ReLU(z) = max(0, z)
```

In plain English: **If the number is positive, keep it. If it's negative, make it zero.**

### Examples

```
ReLU(5.0)  = 5.0   (positive → keep it)
ReLU(2.3)  = 2.3   (positive → keep it)
ReLU(0.0)  = 0.0   (zero → stays zero)
ReLU(-3.7) = 0.0   (negative → becomes zero)
ReLU(-100) = 0.0   (negative → becomes zero)
```

### Visual Representation

```
output
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  │    ╱
  │   ╱
  │  ╱
  │ ╱
──┼──────────── input
  │
  │  (everything below zero
  │   is flattened to zero)
```

### Why Is ReLU So Popular?

Before ReLU was invented (around 2010), people used Sigmoid and Tanh. These older functions had a critical problem called the **Vanishing Gradient Problem**: when the network was very deep (many layers), the learning signals would become so tiny that the early layers would essentially stop learning.

ReLU solved this problem because:
1. **It's incredibly fast to compute.** It's just a comparison: is `z > 0`? Done.
2. **It doesn't squash large values.** If `z = 100`, ReLU keeps it as `100`. Sigmoid would squash it to `0.99999`, losing information.
3. **It creates sparsity.** Many neurons output `0` (when `z < 0`), which makes the network more efficient. Think of it as the neuron saying "I have nothing useful to say about this input, so I'll stay quiet."

### Continuing Our Numerical Example

Remember our neuron calculated `z = -0.6`. After applying ReLU:

```
ReLU(-0.6) = max(0, -0.6) = 0.0
```

The neuron "stays quiet" — it outputs zero, meaning it has nothing useful to contribute for this particular input. This is perfectly fine! Other neurons in the same layer will fire instead.

---

## 7. Putting It Together — A Full Neural Network

### Layers

A neural network is just many neurons organized into **layers**:

```
INPUT LAYER          HIDDEN LAYER 1       HIDDEN LAYER 2       OUTPUT LAYER
(2496 neurons)       (128 neurons)        (64 neurons)         (5 neurons)

  x₁  ──────────→  h₁  ──────────→  h₁  ──────────→  Bengali
  x₂  ──────────→  h₂  ──────────→  h₂  ──────────→  Hindi
  x₃  ──────────→  h₃  ──────────→  h₃  ──────────→  Nepali
  ...              ...               ...               Punjabi
  x₂₄₉₆ ────────→  h₁₂₈ ─────────→  h₆₄ ─────────→  Urdu
```

- **Input Layer:** Receives the raw data. In our project, this is the 2,496-number supervector.
- **Hidden Layers:** The "thinking" layers. Each neuron takes ALL the outputs from the previous layer, multiplies them by its own weights, adds them up, and applies ReLU.
- **Output Layer:** Produces the final answer. One neuron per language (5 total). The neuron with the highest value is the predicted language.

### Why "Hidden"?

They're called "hidden" because you never directly see or interact with them. The input layer is what you feed in, the output layer is what you read out, but the hidden layers are internal — they're the network's private "thoughts."

### How Many Weights Are There?

Each connection between two neurons has its own weight. The total number of weights in our network is:

```
Layer 1: 2496 inputs × 128 neurons = 319,488 weights + 128 biases = 319,616 parameters
Layer 2: 128 inputs  × 64 neurons  =   8,192 weights + 64 biases  =   8,256 parameters
Layer 3: 64 inputs   × 5 neurons   =     320 weights + 5 biases   =     325 parameters
                                                          TOTAL    = 328,197 parameters
```

When we say "training the neural network," we mean adjusting all **328,197** of these numbers simultaneously until the network makes accurate predictions!

---

## 8. Forward Propagation — How the Network Makes a Prediction

**Forward propagation** is the process of feeding an input through the network, layer by layer, until you get an output. It's called "forward" because data flows in one direction: input → hidden → output.

### Step-by-Step (Simplified to 3 inputs, 2 hidden neurons, 2 outputs)

**Step 1: Input Layer → Hidden Layer**

```
Given: x = [2.0, 3.0, -1.0]

Hidden Neuron 1:
z₁ = (w₁₁ × x₁) + (w₁₂ × x₂) + (w₁₃ × x₃) + b₁
z₁ = (0.5 × 2.0) + (-0.3 × 3.0) + (0.8 × -1.0) + 0.1
z₁ = 1.0 - 0.9 - 0.8 + 0.1 = -0.6
h₁ = ReLU(-0.6) = 0.0

Hidden Neuron 2:
z₂ = (w₂₁ × x₁) + (w₂₂ × x₂) + (w₂₃ × x₃) + b₂
z₂ = (0.2 × 2.0) + (0.7 × 3.0) + (-0.1 × -1.0) + 0.3
z₂ = 0.4 + 2.1 + 0.1 + 0.3 = 2.9
h₂ = ReLU(2.9) = 2.9
```

**Step 2: Hidden Layer → Output Layer**

```
Hidden layer output: h = [0.0, 2.9]

Output Neuron 1 (e.g., Hindi):
o₁ = (w₁ × h₁) + (w₂ × h₂) + b
o₁ = (0.4 × 0.0) + (0.6 × 2.9) + 0.2
o₁ = 0 + 1.74 + 0.2 = 1.94

Output Neuron 2 (e.g., Bengali):
o₂ = (w₁ × h₁) + (w₂ × h₂) + b
o₂ = (-0.3 × 0.0) + (0.5 × 2.9) + (-0.1)
o₂ = 0 + 1.45 - 0.1 = 1.35
```

**Step 3: Make the Decision**

```
Output: [1.94, 1.35]
Since 1.94 > 1.35, the network predicts: Hindi ✓
```

That's it! The entire forward propagation is just multiplication, addition, and ReLU — repeated layer by layer.

---

## 9. Loss Function — How the Network Knows It's Wrong

After the network makes a prediction, we need a way to measure **how wrong** it was. This measurement is called the **Loss** (or **Cost**).

### Cross-Entropy Loss

In our project, we use **Cross-Entropy Loss**, which is the standard loss function for classification problems.

**The Formula:**

```
Loss = -log(p_correct)
```

Where `p_correct` is the probability the network assigned to the **correct** answer (after softmax — see Section 13).

### Intuition

- If the network is **very confident and correct** (p_correct = 0.95):
  - Loss = -log(0.95) = **0.05** (very small loss — good!)
  
- If the network is **uncertain** (p_correct = 0.50):
  - Loss = -log(0.50) = **0.69** (moderate loss — needs improvement)
  
- If the network is **confident but WRONG** (p_correct = 0.01):
  - Loss = -log(0.01) = **4.60** (huge loss — very bad!)

The goal of training is to make the loss as close to zero as possible.

### In Our Code (ann.py, line 88):
```python
criterion = nn.CrossEntropyLoss()
```
This single line creates the loss function. PyTorch handles all the complex math internally.

---

## 10. Backpropagation — How the Network Learns

**Backpropagation** is the algorithm that actually teaches the network. It's called "back" propagation because it works **backwards** — from the output layer back to the input layer.

### The Core Idea

1. The network makes a prediction (forward propagation).
2. We calculate how wrong the prediction was (the loss).
3. We ask: **"For each weight in the network, if I increase or decrease it slightly, does the loss go up or down?"**
4. We adjust each weight in the direction that makes the loss go **down**.

### The Math: Gradients

The mathematical tool we use is called a **gradient** (also called a **derivative** or **partial derivative**).

The gradient of the loss with respect to a weight tells us:
- **Direction:** Should we increase or decrease this weight?
- **Magnitude:** By how much should we change it?

```
∂Loss
──────  = gradient of the loss with respect to weight w
∂w
```

- If the gradient is **positive** → increasing `w` increases the loss → we should **decrease** `w`
- If the gradient is **negative** → increasing `w` decreases the loss → we should **increase** `w`
- If the gradient is **zero** → we're at a perfect spot → don't change `w`

### The Chain Rule

Backpropagation uses a calculus trick called the **Chain Rule** to efficiently compute gradients for every single weight in the network, even in layers that are far from the output.

```
∂Loss     ∂Loss     ∂output    ∂hidden
────── = ────── × ──────── × ─────────
∂w₁      ∂output   ∂hidden     ∂w₁
```

This is like a chain of multiplication — the error signal flows backwards through the network, one layer at a time, getting multiplied at each step.

### In Our Code (ann.py, line 111):
```python
loss.backward()
```
This single line computes ALL the gradients for ALL 328,197 weights in the entire network using backpropagation. PyTorch does this automatically!

---

## 11. Gradient Descent & Adam Optimizer

Now that we have the gradients (from backpropagation), we need to actually **update** the weights. This is where the **optimizer** comes in.

### Basic Gradient Descent

The simplest approach is:

```
w_new = w_old - learning_rate × gradient
```

Where:
- `w_old` = the current weight value
- `gradient` = how much the loss changes when we change this weight
- `learning_rate` = a small number (like 0.001) that controls how big each step is
- `w_new` = the updated weight value

### What Is the Learning Rate?

The **learning rate** is one of the most important settings in the entire network.

- **Too large** (e.g., 1.0): The network takes huge steps and overshoots the optimal weights, bouncing around wildly and never converging.
- **Too small** (e.g., 0.0000001): The network takes tiny baby steps and would need millions of epochs to learn anything.
- **Just right** (e.g., 0.001): The network converges smoothly to a good solution.

### Adam Optimizer

In our project, we don't use basic Gradient Descent. We use a much smarter optimizer called **Adam** (Adaptive Moment Estimation).

Adam is like Gradient Descent with two superpowers:

1. **Momentum:** It remembers which direction it was moving and keeps going in that direction. Think of it like a ball rolling downhill — it builds up speed and can roll over small bumps instead of getting stuck.

2. **Adaptive Learning Rate:** It automatically adjusts the learning rate for each individual weight. Weights that are frequently updated get smaller learning rates (to fine-tune them), and weights that are rarely updated get larger learning rates (to help them catch up).

### In Our Code (ann.py, line 89):
```python
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
```

- `model.parameters()` — tells Adam about all 328,197 weights
- `lr=learning_rate` — the base learning rate (0.001 in our config)
- `weight_decay=0.01` — a regularization trick that slightly shrinks all weights toward zero to prevent overfitting (called **L2 regularization**)

---

## 12. Dropout — Preventing Overfitting

### What Is Overfitting?

**Overfitting** is when the neural network memorizes the training data instead of learning general patterns.

Imagine a student who memorizes every single practice exam answer word-for-word instead of understanding the concepts. They'll score 100% on the practice exams but fail the real exam because the questions are slightly different.

In our case:
- **Training accuracy: 99.49%** — the network almost perfectly memorizes the training data
- **Test accuracy: 90.13%** — on new, unseen audio, it's still very good but not perfect

The gap between these two numbers (about 9%) is caused by some overfitting. Dropout helps reduce this gap.

### How Dropout Works

During training, Dropout **randomly turns off** a percentage of neurons in each hidden layer for each training example.

```
Without Dropout:           With Dropout (20%):
h₁ = 2.3                  h₁ = 2.3
h₂ = 0.7                  h₂ = 0.0  ← randomly turned off!
h₃ = 1.5                  h₃ = 1.5
h₄ = 0.9                  h₄ = 0.9
h₅ = 3.1                  h₅ = 0.0  ← randomly turned off!
```

### Why Does This Help?

When neurons are randomly turned off, the network can't rely on any single neuron too much. It's forced to spread the knowledge across many neurons, making the network more robust.

Think of it like a sports team: if the coach randomly benches different players during practice, every player has to learn to play well. The team becomes stronger overall because they don't depend on one star player.

### In Our Code (ann.py, line 45):
```python
layers.append(nn.Dropout(dropout))  # dropout = 0.2 (20%)
```

This means that during every training step, each neuron in the hidden layers has a 20% chance of being temporarily silenced.

**IMPORTANT:** During inference (prediction), Dropout is **turned off**. All neurons are active. This is why we call `model.eval()` before making predictions.

---

## 13. Softmax — Turning Numbers Into Probabilities

### The Problem

The output layer of our network produces 5 raw numbers (called **logits**), one per language. These numbers can be anything — positive, negative, large, small:

```
Raw logits: [3.2, 1.1, 0.5, 8.7, 2.3]
             Bengali  Hindi  Nepali  Punjabi  Urdu
```

These raw numbers are not probabilities. They don't add up to 1, and some might be negative.

### The Softmax Formula

Softmax converts these raw numbers into proper probabilities that:
- Are all between 0 and 1
- Add up to exactly 1.0 (100%)

```
                    e^(zᵢ)
softmax(zᵢ) = ─────────────────
               e^(z₁) + e^(z₂) + ... + e^(z₅)
```

Where `e` is Euler's number (≈ 2.71828).

### Numerical Example

```
Raw logits: [3.2, 1.1, 0.5, 8.7, 2.3]

Step 1: Calculate e^z for each:
e^3.2 = 24.5
e^1.1 = 3.0
e^0.5 = 1.6
e^8.7 = 6,002.9
e^2.3 = 10.0

Step 2: Sum them all:
Total = 24.5 + 3.0 + 1.6 + 6,002.9 + 10.0 = 6,042.0

Step 3: Divide each by the total:
Bengali:  24.5 / 6,042.0 = 0.004  (0.4%)
Hindi:     3.0 / 6,042.0 = 0.001  (0.1%)
Nepali:    1.6 / 6,042.0 = 0.000  (0.0%)
Punjabi: 6002.9 / 6,042.0 = 0.993  (99.3%)  ← WINNER!
Urdu:     10.0 / 6,042.0 = 0.002  (0.2%)
                              ─────
                        Total: 1.000 (100%)
```

The network is 99.3% confident this audio is Punjabi!

### In Our Code (predict.py, line 110):
```python
probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
```

---

## 14. Our Specific ANN Architecture

Here is the exact neural network we use in our SLID project, as defined in `ann.py` and `config.yaml`:

```
┌─────────────────────────────────────────────────────┐
│                   INPUT LAYER                        │
│              2,496 neurons (supervector)              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               HIDDEN LAYER 1                         │
│    Linear: 2496 → 128  (319,616 parameters)          │
│    Activation: ReLU                                  │
│    Dropout: 20%                                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               HIDDEN LAYER 2                         │
│    Linear: 128 → 64   (8,256 parameters)             │
│    Activation: ReLU                                  │
│    Dropout: 20%                                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               OUTPUT LAYER                           │
│    Linear: 64 → 5     (325 parameters)               │
│    Activation: Softmax (applied by CrossEntropyLoss) │
│                                                      │
│    Output: [Bengali, Hindi, Nepali, Punjabi, Urdu]   │
└─────────────────────────────────────────────────────┘

Total trainable parameters: 328,197
```

### How to Read This Architecture

The data flows from top to bottom:

1. **Input (2,496 numbers):** The supervector from the GMM-UBM module enters the network.
2. **Hidden Layer 1 (128 neurons):** Each of the 128 neurons looks at all 2,496 input numbers, computes a weighted sum, and applies ReLU. Then 20% of these 128 neurons are randomly silenced (Dropout).
3. **Hidden Layer 2 (64 neurons):** Each of the 64 neurons looks at all 128 outputs from the previous layer, computes a weighted sum, and applies ReLU. Then 20% of these 64 neurons are randomly silenced.
4. **Output Layer (5 neurons):** Each of the 5 neurons looks at all 64 outputs from the previous layer and computes a final weighted sum. The neuron with the highest value corresponds to the predicted language.

---

## 15. The Complete Training Loop

Here is what happens during every single **epoch** (one complete pass through the training data):

### In Our Code (ann.py, lines 95–129):

```
For each epoch (we run 200 epochs total):
│
├── For each batch of 64 supervectors:
│   │
│   ├── Step 1: ZERO GRADIENTS
│   │   optimizer.zero_grad()
│   │   (Reset all gradient values to zero from the previous batch)
│   │
│   ├── Step 2: FORWARD PASS
│   │   outputs = model(inputs)
│   │   (Feed the supervectors through the network to get predictions)
│   │
│   ├── Step 3: CALCULATE LOSS
│   │   loss = criterion(outputs, labels)
│   │   (Compare predictions to the correct answers using Cross-Entropy)
│   │
│   ├── Step 4: BACKWARD PASS (Backpropagation)
│   │   loss.backward()
│   │   (Calculate gradients for all 328,197 weights)
│   │
│   └── Step 5: UPDATE WEIGHTS
│       optimizer.step()
│       (Adam optimizer adjusts all weights using the gradients)
│
└── Print epoch loss and accuracy every 10 epochs
```

### What Is a Batch?

Instead of showing the network one training example at a time, we show it **64 examples at once** (a "batch"). This is much faster because modern GPUs can process many examples in parallel.

Our config: `batch_size: 64`

### What Is an Epoch?

One **epoch** = one complete pass through the entire training dataset.

We have 5,687 training supervectors and a batch size of 64.
So each epoch consists of: 5,687 ÷ 64 ≈ **89 batches**.

We train for **200 epochs**, which means the network sees every training example **200 times** and adjusts its weights after each of the 89 × 200 = **17,800 batches**.

---

## 16. Quick Reference Cheat Sheet

| Term | What It Means |
|------|---------------|
| **ANN** | Artificial Neural Network — a math program that learns patterns from data |
| **Neuron** | A single unit that multiplies inputs by weights, adds them up, and applies an activation function |
| **Weight** | A number that controls how important each input is to a neuron |
| **Bias** | A constant added to shift the neuron's output |
| **Layer** | A group of neurons at the same depth in the network |
| **Hidden Layer** | An internal layer between input and output |
| **ReLU** | max(0, z) — if positive keep it, if negative make it zero |
| **Forward Propagation** | Feeding data through the network to get a prediction |
| **Loss / Cost** | A number measuring how wrong the prediction was |
| **Cross-Entropy** | The specific loss function for classification: -log(p_correct) |
| **Backpropagation** | Computing gradients by flowing error signals backwards through the network |
| **Gradient** | The direction and amount to change a weight to reduce the loss |
| **Gradient Descent** | Updating weights by subtracting (learning_rate × gradient) |
| **Adam** | A smart optimizer with momentum and adaptive learning rates |
| **Learning Rate** | Controls how big each weight update step is (ours: 0.001) |
| **Epoch** | One complete pass through all training data |
| **Batch** | A group of training examples processed together (ours: 64) |
| **Dropout** | Randomly silencing neurons during training to prevent memorization |
| **Softmax** | Converts raw output numbers into probabilities that sum to 1.0 |
| **Overfitting** | When the network memorizes training data instead of learning general patterns |
| **Inference** | Using the trained network to make predictions on new data |

---

## Summary: The Complete SLID Pipeline

```
Raw Audio (30 sec)
       │
       ▼
   Preprocessing (trim silence)
       │
       ▼
   MFCC Extraction (39 features per frame)
       │
       ▼
   GMM-UBM Supervector (2,496 numbers)
       │
       ▼
  ┌─────────────────────────────────────┐
  │         ANN (This Document!)         │
  │                                     │
  │  Input: 2,496 supervector numbers   │
  │     ↓                               │
  │  Hidden Layer 1: 128 neurons + ReLU │
  │     ↓                               │
  │  Hidden Layer 2: 64 neurons + ReLU  │
  │     ↓                               │
  │  Output: 5 probabilities            │
  │                                     │
  │  [0.4%, 0.1%, 0.0%, 99.3%, 0.2%]   │
  │   Ben   Hin   Nep   Pun    Urd     │
  │                                     │
  │  PREDICTION: Punjabi (99.3%)        │
  └─────────────────────────────────────┘
```

The GMM is the **ears** that process the sound into a mathematical description.
The ANN is the **brain** that reads that description and decides the language.

Together, they achieve **90.13% accuracy** on our 5-language classification task!
