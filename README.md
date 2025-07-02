# Look-Back: Implicit Visual Re-focusing in MLLM Reasoning

> **This repository is the official implementation of "Look-Back: Implicit Visual Re-focusing in MLLM Reasoning".**  
> Look-Back enables Multimodal Large Language Models (MLLMs) to autonomously determine **when**, **where**, and **how** to re-focus on visual inputs during reasoning—without explicit visual information injection.

---

## 🔍 Overview

Most MLLMs tend to ignore visual input during later reasoning stages, overly relying on text. Look-Back is a novel **implicit** method that introduces a `<back>` mechanism, prompting the model to **self-reflect** on the visual input when needed—without changing the model architecture or re-injecting images.

We achieve this via a **two-stage training framework**:
1. **Cold-start Supervised Fine-Tuning (SFT)**: Injects structured `<back>` tokens into the model’s reasoning traces.
2. **Reinforcement Learning (RL)**: Applies a reward based on the `<back>` token to encourage visual reflection.

<p align="center">
  <img src="assets/fig1.png" width="90%">
</p>

The model learns to re-focus attention to visual details in a human-like fashion, significantly improving performance across 8 multimodal reasoning benchmarks (e.g., MathVista, MME, HalluBench, etc.).

---

## 🚧 TODO

- [ ] 🧠 Release the trained **Look-Back models** (`Semantic-back` and `Solution-back`) based on Qwen-2.5-VL-7B
- [ ] 📊 Release **evaluation scripts and benchmark loaders**
- [ ] 🧹 The full codebase is being **cleaned and documented**

