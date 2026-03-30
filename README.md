# A Mathematical Proof of the Symmetry Problem in Neural Networks

This repository contains an independent research report and empirical validation studying why **zero-weight initialization** fails in neural networks, and how the failure mode depends on the **activation function**.

The work derives gradient expressions analytically (via the chain rule) to identify where learning becomes blocked or collapses into symmetry. It also includes a TensorFlow/Keras implementation to validate the theory experimentally.

---

## Research Overview

Although many references caution against zero initialization, the underlying mechanics are often presented informally. This project provides a step-by-step derivation of the gradients to pinpoint the exact divergence point:

- **Sigmoid** escapes the initial deadlock (because its output at zero is non-zero), but then updates symmetrically and collapses into a degenerate solution.
- **Tanh** and **ReLU** do not escape at all under zero initialization, because their output at zero eliminates the gradient signal at the source.

---

## Key Insights

### 1) The Sigmoid divergence
Because \( f(0) = 0.5 \), Sigmoid produces a non-zero output weight gradient at the first update step. This allows training to begin, unlike Tanh and ReLU.

### 2) Cascade to symmetry
Although Sigmoid updates, both hidden neurons receive identical gradients. This keeps hidden units perfectly synchronized and effectively reduces the network to a single-neuron linear model (a symmetry collapse).

### 3) Permanent deadlock for Tanh and ReLU
For Tanh and ReLU, \( f(0) = 0 \) makes the output weight gradient zero under zero initialization. As a result, all weights and biases remain unchanged and learning never starts.

---

## Project Structure

- `Mathematical_Proof_Symmetry_Problem.pdf`  
  Full research report (approximately 15 pages) with complete derivations and explanations.

- `symmetry_experiment.py`  
  TensorFlow/Keras experiment script used to empirically validate the theoretical results.

---

## Empirical Results (Summary)

Running the included experiment reproduces the theoretical predictions:

- **Sigmoid:** hidden weights become non-zero but remain exactly identical (e.g., \( w_{11} = w_{12} \)), demonstrating symmetry persistence.
- **Tanh/ReLU:** all weights and biases remain exactly `0.000` after training, indicating complete learning deadlock.

---
