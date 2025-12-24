# Deep Learning Sequence Models 

## 1. Sequence Modeling – Motivation
Sequence models are used when **data order matters** and previous inputs affect future outputs.

**Examples:**
- Natural Language Processing (text, translation)
- Speech recognition
- Time-series forecasting
- Music generation

Traditional neural networks fail because they assume **independent inputs**.

---

## 2. RNN (Recurrent Neural Network)
**Core Idea:**  
An RNN introduces **memory** by feeding the previous hidden state back into the network.

**Working:**
- Processes one time step at a time
- Hidden state carries information forward

**Advantages:**
- Simple architecture
- Suitable for short sequences

**Limitations:**
- Cannot retain long-term dependencies
- Suffers from **vanishing and exploding gradients**
- Sequential processing → slow training

**Common Uses:** basic NLP tasks, simple time-series

---

## 3. LSTM (Long Short-Term Memory)
**Core Idea:**  
LSTM improves RNN by controlling information flow using **gates**.

### LSTM Gates
- **Forget Gate:** decides what past information to discard
- **Input Gate:** decides what new information to store
- **Output Gate:** decides what information to expose

**Why it works:**
- Maintains a separate **cell state**
- Prevents gradient vanishing

**Advantages:**
- Handles long-term dependencies
- Stable training

**Disadvantages:**
- Computationally expensive
- More parameters

**Common Uses:** machine translation, speech recognition

---

## 4. GRU (Gated Recurrent Unit)
**Core Idea:**  
GRU is a simplified LSTM.

**Key Differences from LSTM:**
- Combines forget and input gates into **update gate**
- No separate cell state

**Advantages:**
- Faster training
- Fewer parameters
- Comparable performance to LSTM

**Use Cases:** real-time systems, large datasets

---

## 5. Bidirectional RNN / LSTM
**Core Idea:**  
Processes the sequence in **both forward and backward directions**.

**Benefit:**
- Captures past and future context

**Example:**
- Understanding ambiguous words in a sentence

**Use Cases:** named entity recognition, translation, speech

---

## 6. Attention Mechanism
**Core Idea:**  
Instead of compressing the entire sequence into one vector, the model **selectively focuses on relevant parts**.

**Why Attention is Needed:**
- Long sequences degrade RNN/LSTM memory
- Important information may be far away

**Benefits:**
- Improves context understanding
- Eliminates information bottleneck
- Works well with long sequences

---

## 7. Attention Mechanism – High-Level Steps
1. Compare current state with all input states
2. Assign **importance weights**
3. Generate weighted sum of inputs
4. Use it to make prediction

---

## 8. Self-Attention
**Core Idea:**  
Each word attends to **every other word in the same sequence**.

**Key Benefit:**
- Captures global dependencies regardless of distance

**Example:**
> “The animal didn’t cross the street because it was tired.”  
Self-attention links **"it" → animal**

**Advantages:**
- Parallel computation
- No recurrence
- Better long-range understanding

---

## 9. Query, Key, Value (QKV)
Self-attention is computed using **Query, Key, and Value vectors**.

- **Query (Q):** what the word is looking for
- **Key (K):** what the word represents
- **Value (V):** information the word carries

### Attention Calculation (Conceptual)
1. Compare Q with all K
2. Convert similarity scores into weights
3. Use weights to combine V vectors

This produces a **context-aware representation**.

---

## 10. Soft Attention vs Hard Attention
**Soft Attention:**
- Uses all elements with different weights
- Fully differentiable
- Most commonly used

**Hard Attention:**
- Selects only a subset of inputs
- Non-differentiable
- Rare in practice

---

## 11. Multi-Head Attention
**Core Idea:**  
Run multiple self-attention mechanisms in parallel.

**Each head learns to focus on different aspects:**
- Syntax
- Semantics
- Word relationships
- Positional patterns

**Benefits:**
- Richer representations
- Better contextual learning

---

## 12. Attention with RNN/LSTM
**Idea:**  
Attention enhances RNN/LSTM by allowing them to revisit all hidden states.

**Used in:**
- Seq2Seq models
- Translation
- Summarization

---

## 13. Transformers
**Core Idea:**  
Transformers remove recurrence and rely **entirely on self-attention**.

### Main Components
- Self-attention layers
- Feed-forward networks
- Positional encoding

**Advantages:**
- Parallel processing
- Faster training
- Superior long-range dependency modeling

---

## 14. Transformers vs RNN/LSTM

| Feature | RNN/LSTM | Transformer |
|------|------|------|
| Sequential processing | Yes | No |
| Long-range dependency | Limited | Excellent |
| Parallelism | No | Yes |
| Training speed | Slow | Fast |

---

## 15. Popular Transformer Models
- **BERT:** Encoder-only, bidirectional, understanding tasks
- **GPT:** Decoder-only, autoregressive, generation tasks

---

## 16. One-Line Interview Summary
> Sequence models evolved from RNNs to LSTM/GRU to handle long dependencies, Attention enabled selective focus, and Transformers replaced recurrence with scalable self-attention for state-of-the-art performance.
