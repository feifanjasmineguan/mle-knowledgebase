# Attention Mechanisms in Machine Learning

## Summary

Attention mechanisms enable neural networks to dynamically focus on relevant parts of input data when making predictions. Inspired by human visual and linguistic attention, attention assigns importance weights to different input elements, allowing models to capture long-range dependencies without being constrained by fixed context vectors. The mechanism has evolved from addressing seq2seq limitations in machine translation to becoming a foundational component in transformers and modern deep learning architectures.

## Key Concepts

### Core Attention Definition
Attention computes a **context vector** as a weighted sum of encoder hidden states:
- **Context vector**: $\mathbf{c}_t = \sum_{i=1}^n \alpha_{t,i} \boldsymbol{h}_i$
- **Alignment weights**: $\alpha_{t,i} = \frac{\exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_i))}{\sum_{i'=1}^n \exp(\text{score}(\boldsymbol{s}_{t-1}, \boldsymbol{h}_{i'}))}$

The alignment score determines how strongly output position $t$ correlates with input position $i$.

### Why Attention Was Needed

**Seq2Seq Problem**: Traditional encoder-decoder models compress variable-length input into a fixed-size context vector, causing information loss on long sequences. The encoder's last hidden state becomes a bottleneck.

**Attention Solution**: Creates learnable shortcuts between decoder and entire encoder sequence, allowing the model to attend to relevant input positions at each decoding step.

### Attention Score Functions

| Type | Score Function | Key Property |
|------|---|---|
| **Content-Based** | $\text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$ | Memory address by similarity |
| **Additive (Bahdanau)** | $\mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_{t-1}; \boldsymbol{h}_i])$ | Learned alignment network |
| **Location-Based** | $\text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$ | Depends only on decoder position |
| **General (Luong)** | $\boldsymbol{s}_t^\top\mathbf{W}_a\boldsymbol{h}_i$ | Trainable weight matrix |
| **Dot-Product** | $\boldsymbol{s}_t^\top\boldsymbol{h}_i$ | Simple, efficient |
| **Scaled Dot-Product** | $\frac{\boldsymbol{s}_t^\top\boldsymbol{h}_i}{\sqrt{n}}$ | Prevents vanishing gradients |

### Soft vs Hard Attention

**Soft Attention**
- Attends to entire input space with learned weights across all positions
- Differentiable and smooth
- Computationally expensive for large inputs
- Used in NMT, image captioning

**Hard Attention**
- Selects single patch/position at each step
- Less computational cost
- Non-differentiable; requires reinforcement learning or variance reduction to train
- More interpretable but harder to optimize

### Global vs Local Attention

**Global (Soft)**
- Attends to all encoder positions
- Same as soft attention

**Local (Hybrid)**
- Predicts single aligned position for current step
- Computes context vector over window around predicted position
- Differentiable improvement over hard attention
- Balances efficiency and expressiveness

## Details

### Bidirectional Encoder Architecture

Encoders use bidirectional RNNs (Bi-LSTM/Bi-GRU) to incorporate both preceding and following context:
$$\boldsymbol{h}_i = [\overrightarrow{\boldsymbol{h}}_i^\top; \overleftarrow{\boldsymbol{h}}_i^\top]^\top$$

Concatenating forward and backward hidden states captures richer contextual information.

### Self-Attention (Intra-Attention)

Relates different positions **within the same sequence** to compute improved representation:
- Enables machine reading and abstractive summarization
- Allows each position to attend to all other positions in the sequence
- Particularly useful for capturing long-range dependencies within a single sequence

### Neural Turing Machines (NTM)

Couples neural networks with external memory using attention-based addressing:

**Components**:
- **Controller**: Neural network (feedforward or recurrent) executing operations
- **Memory**: $N \times M$ matrix storing information
- **Read/Write Heads**: Parallel heads with soft attention over memory locations

**Operations**:
- **Read**: $\mathbf{r}_t = \sum_{i=1}^N w_t(i)\mathbf{M}_t(i)$ (weighted sum of memory rows)
- **Write**: Erase old content then add new via gates: $\tilde{\mathbf{M}}_t(i) = \mathbf{M}_{t-1}(i)[1 - w_t(i)\mathbf{e}_t]$ then $\mathbf{M}_t(i) = \tilde{\mathbf{M}}_t(i) + w_t(i)\mathbf{a}_t$

**Addressing Mechanisms**:
1. **Content-Based**: $w_t^c(i) = \text{softmax}(\beta_t \cdot \text{cosine}[\mathbf{k}_t, \mathbf{M}_t(i)])$
2. **Interpolation**: Blend with previous attention via gate $g_t$
3. **Location-Based**: 1-D convolution with shift weights and sharpening $\gamma_t$

### Pointer Networks

Applies attention to **select input positions** rather than blend representations.

**Use Cases**: Sorting, traveling salesman, convex hull—problems where output must correspond to input positions.

**Key Difference**: Output is sequence of indices $\boldsymbol{c} = (c_1, \dots, c_m)$ where $1 \leq c_i \leq n$, not continuous values.

$$p(c_i \vert c_1, \dots, c_{i-1}, \boldsymbol{x}) = \text{softmax}(\text{score}(\boldsymbol{s}_t; \boldsymbol{h}_i))$$

### Transformer Architecture

**Revolutionary insight**: "Attention is All You Need"—build seq2seq models using only attention, eliminating recurrence.

#### Key, Query, Value Framework

- **Query** ($\mathbf{Q}$, dim $m$): Representation of what we're looking for (decoder state)
- **Key** ($\mathbf{K}$, dim $n$): Representation of what we have (encoder states)
- **Value** ($\mathbf{V}$, dim $n$): Information to retrieve (encoder states)

**Scaled Dot-Product Attention**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

The scaling factor $1/\sqrt{d_k}$ prevents large dot products from saturating softmax.

#### Multi-Head Self-Attention

Runs scaled dot-product attention multiple times in parallel with different learned projections:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1; \cdots; \text{head}_h]\mathbf{W}^O$$

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}^Q_i, \mathbf{K}\mathbf{W}^K_i, \mathbf{V}\mathbf{W}^V_i)$$

**Advantage**: Allows model to jointly attend to information from different representation subspaces at different positions; single head averaging would inhibit this capability.

#### Encoder Stack

- N=6 identical layers
- Each layer: multi-head self-attention + position-wise feedforward network
- **Residual connections** and **layer normalization** around each sub-layer
- Output dimension

## Backlinks
- [[_index]]