#  Multimodal Emotion Recognition with EMG and LLMs

This project explores **multimodal emotion recognition** by bridging **Electromyography (EMG) biosignals** with **Large Language Models (LLMs)**.  
We propose a hybrid system that extracts robust features from EMG signals using **CNN/TCN encoders** and translates them into text-like embeddings for instruction-tuned LLMs.  

---

##  Project Overview

### Motivation
- EMG signals are rich in physiological information but often noisy and difficult to interpret.
- Classical models (e.g., SVM) achieve limited performance due to low signal-to-noise ratio (SNR).
- By combining **deep feature extraction** (CNN/TCN) with **LLM fine-tuning**, we enable interpretable, language-compatible emotion classification.

### Contributions
- **Hybrid architecture:** CNN/TCN for EMG embeddings + LoRA-tuned LLMs for decoding.
- **Numerical-to-textual bridging:** Mapping high-dimensional EMG features into discrete tokens.
- **Evaluation pipeline:** Classical baselines, encoder-based embeddings, and multimodal LLM fine-tuning.
- **SNR and sparsity analysis:** Quantitative insight into the dataset quality and embedding robustness.

---

## 📊 Dataset

- **Source:** Custom-collected EMG recordings (ALS patient cohort).  
- **Channels:** 8 EMG channels  
- **Sampling rate:** 15,000 Hz  
- **Recording length per sample:** 45,000 timesteps (~3s)  
- **Available emotions (12 classes):** anger, anxiety, contempt, delight, disgust, fear, happiness, neutral, perplexity, pride, sadness, surprise  
- **Selected emotions (used for LLM training):** anger, anxiety, fear, happiness  
- **Segments after preprocessing:** ~28,080 (with augmentation)  

---

## 🔬 Methodology

### 1. Feature Extraction
- **Time-domain features:** mean, variance, RMS  
- **Frequency features:** mean frequency (FFT)  
- **Concatenation:** raw EMG + extracted features → `(n_segments, segment_length, 12)`  

### 2. Classical Baseline
- **Model:** Support Vector Machine (SVM) with RBF kernel  
- **Performance:** 36.3% accuracy (vs. 16.6% random baseline)  
- **Conclusion:** Weak but detectable structure in EMG data.  

### 3. Deep Encoders
- **CNN Encoder:**  
  - 2D convolutions + residual blocks + attention layer  
  - Validation accuracy: **87%**  
- **TCN Encoder:**  
  - Dilated causal convolutions + residuals  
  - Validation accuracy: **89%**  
- CNN embeddings were primarily used in later experiments.

### 4. Multimodal LLM Fine-Tuning
- **LoRA (Low-Rank Adaptation):** Efficient adaptation of pretrained LLMs.  
- **Models tested:**  
  - TinyLlama-bnb-4bit → **26.91% accuracy**  
  - Phi-3.5-mini-instruct-bnb-4bit → **25.96% accuracy**  
  - LLaMA-3.2-3B-Instruct → **24.07% accuracy**  
- **Training format (ShareGPT-style):**
  ```python
  sharegpt_data.append({
      "conversations": [
          {"from": "user", "value": f"EMG features: {feature_str}"},
          {"from": "assistant", "value": f"Emotion: {label}"}
      ]
  })
