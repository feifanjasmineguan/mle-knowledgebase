# The Transformer Architecture

## Summary

The Transformer is a neural network architecture introduced in the paper "Attention is All You Need" (Vaswani et al., 2017) that replaces recurrence with self-attention mechanisms. It consists of stacked encoder and decoder layers connected by attention mechanisms, enabling efficient parallel processing and achieving state-of-the-art performance in machine translation and other sequence-to-sequence tasks. The architecture's key innovation is its ability to capture dependencies between words without sequential processing, making it highly parallelizable and suitable for TPU acceleration.

## Key Concepts

### Architecture Overview
- **Encoder Stack**: Multiple identical layers (typically 6) that process input sequences
- **Decoder Stack**: Multiple identical layers that generate output sequences autoregressively
- **Encoder-Decoder Attention**: Mechanism allowing decoders to focus on relevant input positions

### Self-Attention Mechanism
The core component enabling the Transformer to process sequences in parallel:

- **Query (Q), Key (K), Value (V) vectors**: Derived from embeddings via learned linear transformations (WQ, WK, WV matrices)
- **Attention score calculation**: Dot product of query with keys, scaled by √d_k (typically √64 ≈ 8) for gradient stability
- **Softmax normalization**: Converts scores into a probability distribution determining focus weights
- **Weighted sum**: Multiply value vectors by attention weights and aggregate

**Formula**: `Attention(Q,K,V) = softmax(QK^T / √d_k)V`

### Multi-Head Attention
- Uses **8 parallel attention heads** in the original architecture
- Each head has independent Q/K/V weight matrices, learning different representation subspaces
- Enables model to focus on different positions and semantic aspects simultaneously
- Outputs concatenated and projected through linear transformation (WO)
- Addresses two limitations of single-head attention:
  - Expands ability to attend to multiple positions
  - Provides multiple representation subspaces

### Positional Encoding
- Addresses the model's lack of inherent sequential understanding
- Sinusoidal positional encoding vectors added to embeddings
- Formula uses sine for even dimensions, cosine for odd dimensions
- Enables model to extrapolate to sequence lengths unseen during training
- Produces learnable relative position distances

### Sub-Layer Components

**Feed-Forward Network**: Applied independently to each position
- Fully connected layer with ReLU activation
- Two linear transformations with intermediate dimension expansion

**Residual Connections**: Each sub-layer has residual connection around it
- Input added to output: `LayerNorm(x + SubLayer(x))`
- Facilitates gradient flow during backpropagation

**Layer Normalization**: Applied after each sub-layer
- Stabilizes training and improves convergence

### Decoder-Specific Mechanisms

**Causal Self-Attention**: 
- Masked to prevent attending to future positions
- Future positions set to -∞ before softmax
- Ensures autoregressive generation

**Encoder-Decoder Attention**:
- Queries generated from decoder layer
- Keys and Values from encoder output
- Directs decoder to focus on relevant input regions

### Output Generation
1. **Linear Layer**: Projects decoder output to logits vector (vocabulary_size)
2. **Softmax Layer**: Converts logits to probability distribution
3. **Decoding Strategies**:
   - **Greedy decoding**: Select highest probability word at each step
   - **Beam search**: Maintain top-k hypotheses, exploring multiple paths

## Details

### Forward Pass Flow

**Encoding Phase**:
1. Input words embedded into 512-dimensional vectors
2. Positional encodings added to preserve word order
3. Sequence flows through each encoder layer:
   - Self-attention: allows each word to attend to all other words
   - Feed-forward: applied independently per position
4. Output of top encoder becomes K,V for decoder attention

**Decoding Phase**:
1. Target words embedded and positionally encoded
2. Each decoder layer processes:
   - Causal self-attention (masked to prevent future peeking)
   - Encoder-decoder attention (attends to encoder output)
   - Feed-forward network
3. Output projected to vocabulary probabilities
4. Process repeats autoregressively until end-of-sequence token

### Parallelization Advantages

Unlike RNNs requiring sequential processing:
- **Self-attention layer**: Dependencies exist but all computed in parallel via matrix operations
- **Feed-forward layer**: No inter-position dependencies; fully parallelizable
- **Encoder layers**: Can process multiple sequences in parallel (batching)
- **Google Cloud TPU recommendation**: Architecture designed for tensor acceleration

### Training Procedure

**Loss Calculation**:
- Compare model output probability distributions with target one-hot encoded words
- Use cross-entropy loss across all positions
- Backpropagation updates all model weights

**Training Considerations**:
- Teacher forcing: During training, provide ground truth previous tokens as input
- Cross-validation: Prevent data leakage; track separate validation metrics
- Gradient stability: Scaling by √d_k and layer normalization prevent vanishing/exploding gradients

### Mathematical Details

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention**:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Positional Encoding** (dimension d, position pos):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

## Interview Relevance

### Common Interview Questions

1. **Architecture & Design**
   - "Explain why the Transformer uses attention instead of RNNs"
     - Answer: Enables parallelization, captures long-range dependencies without vanishing gradients
   - "What is the purpose of Query, Key, and Value projections?"
     - Answer: Reduces dimensionality (512→64), learns different representations through weight matrices
   - "Why scale attention scores by √d_k?"
     - Answer: Prevents dot products from growing too large (exploding gradients), stabilizes training

2. **Self-Attention Mechanisms**
   - "Walk through the self-attention calculation step-by-step"
     - Expected: Discuss Q,K,V projections → dot products → scaling → softmax → weighted sum
   - "How does multi-head attention improve performance?"
     - Answer: Different heads learn to attend to different positions and representation subspaces
   - "Can you explain what 'self' means in self-attention?"
     - Answer: Attention over the same sequence (unlike cross-attention between different sequences)

3. **Positional Information**
   - "Why do Transformers need positional encoding?"
     - Answer: Attention is permutation-invariant; positional encoding signals word order
   - "Why use sinusoidal positional encoding instead of learned embeddings?"
     - Answer: Sinusoidal can extrapolate to unseen sequence lengths

4. **Decoder Design**
   - "Explain the difference between encoder self-attention and decoder self-attention"
     - Answer: Decoder uses causal masking (can't attend to future); encoder can attend to all positions
   - "What is encoder-decoder attention and why is it needed?"
     - Answer: Allows decoder to focus on relevant parts of input; different Q (from decoder) vs K,V (from encoder)
   - "Why use causal masking in the decoder?"
     - Answer: Prevents information leakage during training; ensures autoregressive generation during inference

5. **Training & Inference**
   - "How does teacher forcing work and why is it used?"
     - Answer: Feed ground-truth previous outputs during training for faster, more stable convergence
   - "Compare greedy decoding vs. beam search"
     - Answer: Greedy is fast but suboptimal; beam search explores multiple hypotheses (hyperparameter: beam_size)
   - "What is the role of the final softmax layer?"
     - Answer: Converts logits to probability distribution over vocabulary; enables sampling strategies

6. **Computational Efficiency**
   - "Why is the Transformer more parallelizable than RNNs?"
     - Answer: No sequential dependencies in attention or FFN layers; all positions computed simultaneously
   - "What is the time complexity of self-attention?"
     - Answer: O(n²d

## Backlinks
- [[_index]]