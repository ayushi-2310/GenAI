# Transformers in AI

## 1. Evolution of Sequence Models (Why Transformers?)
Before Transformers, NLP relied on processing text sequentially. This created significant bottlenecks in both speed and memory.

* **Early Approaches:** Rule-based systems and N-grams had "short-term memory." They could only see a few words back, failing to understand long sentences.
* **Recurrent Models (RNNs/LSTMs):** These processed words one by one. The "hidden state" acted as a memory, but it suffered from **Vanishing Gradients**—the model would "forget" the beginning of a sentence by the time it reached the end.
* **The Sequential Bottleneck:** Because RNNs are sequential, they cannot be easily parallelized on GPUs. You must compute word 1 to get word 2.
* **Motivation for Transformers:** They allow for **massive parallelization** (processing the whole sentence at once) and **Long-Range Dependencies** (any word can look at any other word regardless of distance).

---

## 2. Core Transformer Intuition
* **Context-Aware Representations:** In a Transformer, the meaning of a vector changes based on its neighbors (e.g., "Apple" in a tech context vs. fruit context).
* **Global vs. Local:** Unlike CNNs (local) or RNNs (sequential), Transformers have a "global" view of the entire input from layer one.
* **Learned Routing:** The model doesn't just process data; it learns *which* parts of the data are important to *other* parts.

---

## 3. Attention Mechanism (The Engine)


### 3.1 Purpose & Function
* **Dynamic Selection:** Attention allows a token to "attend" to relevant information. When processing the word "it," the model uses attention to find the noun "it" refers to.
* **Self-Attention vs. Cross-Attention:** Self-attention happens within a single sentence. Cross-attention happens between two different sequences (like English and French in translation).

### 3.2 Multi-Head Attention (MHA)
* **Intuition:** Instead of one "view" of the data, MHA uses multiple "heads." One head might focus on grammar, another on entity relationships, and another on punctuation.
* **Trade-off:** More heads allow for richer representations but increase computational cost and memory usage.

### 3.3 Practical Concerns
* **Quadratic Scaling:** The cost of attention grows by the square of the sequence length ($N^2$). This is why LLMs have "context window" limits.
* **Causal Masking:** In decoders, we hide future tokens so the model can't "cheat" during training.

---

## 4. Architecture Components
* **Embeddings & Positional Encoding:** Since Transformers process everything at once, they have no concept of "order." We add "Positional Encodings" to the word vectors to tell the model where each word sits in the sentence.
* **Feed-Forward Networks (FFN):** While Attention lets tokens "talk," the FFN is where the model does the "thinking." It processes each token individually to refine its meaning.
* **Residual Connections:** These act as "highways," allowing information to flow through deep layers without getting distorted, preventing training instability.
* **Layer Normalization:** This keeps the "numbers" in a healthy range, ensuring that no single feature dominates and the model trains faster.

---

## 5. Encoder vs. Decoder vs. Encoder-Decoder
[Image comparing Encoder-only, Decoder-only, and Encoder-Decoder architectures]

* **Encoder-Only (e.g., BERT):** Sees the whole sentence (Bidirectional). Best for **Understanding** (Sentiment analysis, classification).
* **Decoder-Only (e.g., GPT):** Sees only the past (Causal). Best for **Generation** (Chatbots, writing).
* **Encoder-Decoder (e.g., T5):** The encoder understands the input, and the decoder generates a new output. Best for **Transformation** (Translation, Summarization).

---

## 6. Data Flow & Inference
1.  **Input:** Text $\rightarrow$ Tokens $\rightarrow$ Embeddings + Positions.
2.  **Processing:** Multiple layers of Attention $\rightarrow$ FFN $\rightarrow$ LayerNorm.
3.  **Output Head:** Converts the final vectors into probabilities for words.
4.  **KV Caching:** In production, we store the "Key" and "Value" vectors of past tokens so we don't have to recompute them for every new word generated. This drastically reduces latency.

---

## 7. Scaling & Efficiency
* **Scaling Laws:** Bigger models (more layers/width) generally perform better, but they require exponentially more data and compute.
* **Quantization:** Reducing the precision of model weights (e.g., 16-bit to 4-bit) to make them fit on smaller GPUs without losing much intelligence.
* **Sparse Attention:** Techniques to avoid the $N^2$ cost by only looking at nearby tokens or a few "global" tokens.

---

## 8. LLMs and Prompting
* **Emergent Abilities:** Very large models suddenly "learn" how to do things like math or coding that weren't the primary focus of their training.
* **In-Context Learning:** Providing examples in a prompt ("Few-shot") helps the model understand the specific format you want.
* **Alignment (RLHF):** Using human feedback to ensure the model is helpful and safe, not just predicting the next most likely (but potentially toxic) word.

---

## 9. Failure Modes & Risks
* **Hallucination:** The model is a "stochastic parrot"—it prioritizes sounding confident over being factually correct.
* **Attention Misinterpretation:** Just because a model "attends" to a word doesn't mean it "understands" it the way a human does.
* **Data Bias:** If the training data contains biases, the Transformer will amplify them in its outputs.

---

## 💡 Interview Strategy
* **Why Transformers?** Focus on **Parallelization** and **Global Context**.
* **Encoder vs. Decoder?** Mention **Bidirectional** (Encoder) vs. **Autoregressive/Causal** (Decoder).
* **Production bottlenecks?** Mention **Memory bandwidth** and **Quadratic scaling of attention**.
