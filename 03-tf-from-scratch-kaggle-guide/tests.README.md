
### Testing as a Learning tool
https://pmc.ncbi.nlm.nih.gov/articles/PMC6920642/

> based on @03-tf-from-scratch-kaggle-guide help me learn by doing "testing as a learning tool", generate 10 questions that can test my foundational knowledge of this that and show if I have a knowledge tree of understanding. Be very thoughtful on what matters, be brief.

<br>

#### `1`
1. Walk through the attention formula Attention(Q,K,V) = softmax(QK^T/√d_k)V step by step. Explain what each matrix multiplication represents, why we divide by √d_k, and what the final output means in terms of information flow.

2. Trace a single token's journey through your transformer implementation from tokenization to final classification. At each step (embedding → positional encoding → attention → residual → FFN → classifier), explain what information is being added, transformed, or preserved, and why each step is necessary.

3. Your implementation uses multiple attention heads, each with embed_dim // num_heads dimensions. Explain why we split dimensions, how different heads can specialize during training, and what happens in the final output linear layer.

4. Your positional encoding uses sin/cos functions with different frequencies. Explain why transformers need positional information at all (vs RNNs), why this mathematical approach works, and what would happen if you removed positional encoding.

5. Your encoder uses x = x + self.attention(hidden_state). Explain the mathematical and practical reasons for residual connections, how they affect gradient flow during backpropagation, and why they're especially critical in deep transformer architectures.

6. Attention operations (Q@K^T, softmax, @V) are mostly linear, while FFN introduces non-linearity. Explain what types of patterns each can learn, why both are necessary, and give examples of what the model couldn't learn with only attention or only FFN.

7. Your implementation is encoder-only with a classifier head. Compare this to decoder-only (like GPT) and encoder-decoder (like T5) architectures. Explain when you'd use each, how attention masking differs, and why your choice fits the classification task.

8. Analyze the dimensionality flow in your implementation: token indices → embeddings → attention → FFN → classification. Explain how you'd decide on embedding_dimensions, num_heads, and num_encoder_layers for different tasks and computational budgets.

9. During training, a single classification loss must coordinate learning across embedding matrices, attention weights (W_Q, W_K, W_V), FFN parameters, and layer norms. Explain how gradients flow back through your architecture and what each component learns to optimize the final objective.

10. Your attention mechanism has O(n²) complexity in sequence length. Explain where this comes from mathematically, why it becomes a bottleneck for long sequences, and what trade-offs you'd consider when scaling to longer inputs or larger models.

`feedback`
- Overall Assessment: 5.4/10, I'd swear I did min 7.
- Strengths: You have good intuition about the big picture and understand key concepts like attention flow, residuals, and architectural differences.
- Weaknesses: Lack precision in mathematical details, confused about information flow specifics, and need deeper understanding of training dynamics.

`suggestions`
- Matrix projection mechanics
  - what it means geometrically, rotating/stretching vectors emphasizing features
  - action: implement the code for it in numpy
- Variance scaling in Attention (√d_k)
  - code random Q,K matrices, compute QK^T, with/out scaling, and see softmax changes
- Layer Normalization vs Batch Normalization
  - Understand math behind each
  - Implement in Transformer, and see how they affect training
- Positional encoding maths
  - check sin/cos patterns for each position, and it's distinguishable patterns
- Gradient flow through attention
  - Manually compute gradients for a 2-token attention example
- What Attention Heads Actually Learn
  - Use attention visualization tools on your trained model to see what patterns emerge.
- FFN's Role in Feature Learning
  - visualize ffn activations to see what patternsit learns
- Training without classification head
  - Implement a simple contrastive loss and train your transformer to create good sentence embeddings.


#### `2`
1. Token → Vector Journey
Given the sentence "The cat sat", trace how "cat" becomes a vector. What dimensions change at each step: tokenization → embedding → positional encoding?
2. Attention Computation
For tokens ["I", "love", "pizza"], manually compute the attention scores between "love" and all tokens. Show Q@K^T calculation and explain what high/low scores mean.
3. Why √d_k?
Without scaling by √d_k, what happens to softmax(QK^T) when d_k=512? Calculate actual values to demonstrate the problem.
4. Multi-Head Purpose
If you have 8 heads with d_k=64 each, how do they see different parts of the same input? Why not 1 head with d_k=512?
5. Residual Connection Math
Show mathematically why x + AttentionBlock(x) helps gradients flow better than just AttentionBlock(x) during backpropagation.
6. FFN's Non-linearity
Give a specific example of a pattern that attention (linear) cannot learn but FFN (non-linear) can. Use actual vector operations.
7. Positional Encoding Decode
Given two position encodings, show how the model can determine relative distance. Why sine/cosine specifically?
8. Loss → Weight Updates
For classification loss on "This movie is great" → positive, trace how the loss updates: classifier weights → FFN → attention weights → embeddings.
9. Encoder vs Decoder Masking
Show the attention mask matrix for "I love pizza" in: (a) encoder (your implementation), (b) decoder (GPT-style). Why the difference?
10. Scale Decision
You have 10K movie reviews. Justify choosing: embedding_dim=256 vs 512, num_heads=4 vs 8, num_layers=4 vs 6. What breaks first when scaling?


`feedback`

- Embedding mechanics: Study how nn.Embedding is just a learnable lookup table, implement them from scratch without pytorch.
- Positional encoding math: Work through the sine/cosine formula with actual positions, visualize them.
  - Answer why can the model decode relative positions?
  - Compute distance between position 5 and 15 and show how sin/cos patterns encode this distance.
- Concrete non-linearity examples: Implement XOR with linear vs non-linear layers. Build XOR classifier.
  - Prove why attention alone can't solve XOR
  - only linear layers (fail), add gelu (suceeds).
- Understand how gradients trace through mini-Transformer
  ```python
    # Build 2-token, 1-head, 4-dim transformer
    # Input: "I am" → classify positive/negative

    # Manually compute:
    # 1. Forward pass: embedding → attention → output
    # 2. Loss calculation
    # 3. Backward pass: ∂L/∂W for each weight matrix
    # 4. Verify with autograd
  ```
- Scaling laws: Learn the relationship between data size and model capacity. 
    ```python
    # Use IMDB movie reviews (or similar)
    # Train 4 models and measure:

    configs = [
        {"embed_dim": 128, "num_heads": 4, "layers": 2},  # Tiny
        {"embed_dim": 256, "num_heads": 4, "layers": 4},  # Small
        {"embed_dim": 512, "num_heads": 8, "layers": 4},  # Medium
        {"embed_dim": 512, "num_heads": 8, "layers": 8},  # Large
    ]

    # For each, track:
    # - Training loss curve
    # - Validation accuracy
    # - Memory usage
    # - Training time
    # - Where it breaks (overfitting? underfitting?)
    ```