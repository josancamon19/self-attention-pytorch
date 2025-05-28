"""
Positional Encoding Deep Dive Exercise
=====================================
Time estimate: 25-30 minutes

In this exercise, you'll explore:
1. Why we need positional encoding
2. How sin/cos waves encode position information
3. How transformers learn to use this information
4. Alternative approaches and their trade-offs
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from _0_tokenization import tokenize_input, tokenizer
from _1_config import config
from _2_embedding import embed
import seaborn as sns

# Exercise 1: Understanding Why Position Matters (5 minutes)
# ==========================================================
print("=" * 60)
print("Exercise 1: Why Position Matters in Transformers")
print("=" * 60)

# Let's see why transformers need positional information
sentences = [
    "The cat sat on the mat",
    "The mat sat on the cat",  # Same words, different meaning!
    "On the mat sat the cat",  # Yet another arrangement
]

print("\nThese sentences use the same words but mean different things:")
for i, sent in enumerate(sentences):
    tokens = tokenizer.encode(sent)
    print(f"{i+1}. '{sent}' -> tokens: {tokens}")

# TODO: Think about this question and write your answer:
# Question: If we just embed these tokens without position, what information is lost?
# Your answer here:
"""
YOUR ANSWER: 
we still have each token separate, why would we lose it's position?
If we have a list of embeddings, isn't that enough order?

kk, got it. (Check readme)

when we go through the attention layer, the only thing we have in parallel is the embedding's to do the operations, 
we need to put in any way the position, so the attention layer can understand.
"""

# Exercise 2: Visualizing the Positional Encoding Pattern (7 minutes)
# ==================================================================
print("\n" + "=" * 60)
print("Exercise 2: Visualizing Positional Encoding Patterns")
print("=" * 60)

def create_positional_encoding(max_len, d_model):
    """Create positional encoding from scratch to understand it better"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    # The magic formula: div_term creates different frequencies
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-np.log(10000.0) / d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cos to odd indices  
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Create and visualize a small positional encoding
pe_small = create_positional_encoding(50, 128)

# Plot heatmap
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
sns.heatmap(pe_small[:20, :64].numpy(), cmap='RdBu_r', center=0)
plt.title('Positional Encoding Heatmap (first 20 positions, 64 dimensions)')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')

# Plot specific dimensions to see the wave patterns
plt.subplot(2, 1, 2)
positions = list(range(50))
plt.plot(positions, pe_small[:, 0], label='dim 0 (sin)', alpha=0.8)
plt.plot(positions, pe_small[:, 1], label='dim 1 (cos)', alpha=0.8)
plt.plot(positions, pe_small[:, 10], label='dim 10 (sin)', alpha=0.8)
plt.plot(positions, pe_small[:, 11], label='dim 11 (cos)', alpha=0.8)
plt.plot(positions, pe_small[:, 20], label='dim 20 (sin)', alpha=0.8)
plt.title('Sinusoidal Patterns at Different Dimensions')
plt.xlabel('Position')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# TODO: Observe the patterns and answer:
# 1. What do you notice about the frequency of waves at different dimensions?
# 2. Why might using different frequencies be helpful?
"""
YOUR OBSERVATIONS:
1. 
2. 
"""

# Exercise 3: Understanding the Mathematical Intuition (8 minutes)
# ===============================================================
print("\n" + "=" * 60)
print("Exercise 3: Mathematical Intuition Behind Sin/Cos")
print("=" * 60)

# Let's understand why sin/cos are perfect for encoding positions
def analyze_positional_properties():
    """Explore key properties of positional encoding"""
    pe = create_positional_encoding(100, 128)
    
    # Property 1: Bounded values (always between -1 and 1)
    print(f"Min value: {pe.min():.3f}")
    print(f"Max value: {pe.max():.3f}")
    
    # Property 2: Unique encoding for each position
    # Check if first 10 positions have unique encodings
    print("\nChecking uniqueness of positional encodings:")
    for i in range(5):
        for j in range(i+1, 5):
            similarity = torch.cosine_similarity(pe[i], pe[j], dim=0)
            print(f"Similarity between position {i} and {j}: {similarity:.4f}")
    
    # Property 3: Relative positions have consistent relationships
    print("\nRelative position patterns:")
    # The encoding allows the model to learn "position 5 is 3 steps after position 2"
    
    # Let's check dot products between positions at fixed distances
    distances = [1, 2, 5, 10]
    for dist in distances:
        similarities = []
        for pos in range(0, 20):
            if pos + dist < 100:
                sim = torch.dot(pe[pos], pe[pos + dist])
                similarities.append(sim.item())
        avg_sim = np.mean(similarities)
        print(f"Average dot product for distance {dist}: {avg_sim:.4f}")
    
    return pe

pe_analysis = analyze_positional_properties()

# TODO: Think about why these properties matter:
"""
YOUR THOUGHTS:
- Why is it important that values are bounded?
- Why do we need unique encodings?
- What does the relative position pattern tell us?
"""

# Exercise 4: How Transformers Learn to Use Positional Information (10 minutes)
# ============================================================================
print("\n" + "=" * 60)
print("Exercise 4: How Transformers Use Positional Encoding")
print("=" * 60)

# Let's simulate how attention mechanisms can use positional information
class SimpleAttentionDemo(nn.Module):
    """Simplified attention to show how position info is used"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Simple linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: [seq_len, d_model]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.T) / np.sqrt(self.d_model)
        attention = torch.softmax(scores, dim=-1)
        
        return attention

# Create sample embeddings with and without positional encoding
sentence = "The cat sat on mat"
tokens = tokenize_input(sentence)
embeddings = embed(tokens).squeeze(0)  # Remove batch dimension

# Add positional encoding
pe = create_positional_encoding(embeddings.size(0), embeddings.size(1))
embeddings_with_pos = embeddings + pe

# Initialize attention module
attention_demo = SimpleAttentionDemo(embeddings.size(1))

# Compare attention patterns
with torch.no_grad():
    # Without positional encoding
    attn_without_pos = attention_demo(embeddings)
    
    # With positional encoding  
    attn_with_pos = attention_demo(embeddings_with_pos)

# Visualize the difference
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot attention without position
axes[0].imshow(attn_without_pos.numpy(), cmap='Blues')
axes[0].set_title('Attention WITHOUT Positional Encoding')
axes[0].set_xlabel('Key Position')
axes[0].set_ylabel('Query Position')

# Plot attention with position
axes[1].imshow(attn_with_pos.numpy(), cmap='Blues')
axes[1].set_title('Attention WITH Positional Encoding')
axes[1].set_xlabel('Key Position')
axes[1].set_ylabel('Query Position')

# Plot the difference
diff = attn_with_pos - attn_without_pos
axes[2].imshow(diff.numpy(), cmap='RdBu_r')
axes[2].set_title('Difference (Impact of Positional Encoding)')
axes[2].set_xlabel('Key Position')
axes[2].set_ylabel('Query Position')

plt.tight_layout()
plt.show()

# TODO: Analyze the attention patterns:
"""
YOUR ANALYSIS:
1. What differences do you see in the attention patterns?
2. How might positional encoding help the model understand word order?
"""

# Exercise 5: Alternative Approaches and Trade-offs (5 minutes)
# ============================================================
print("\n" + "=" * 60)
print("Exercise 5: Alternative Positional Encoding Methods")
print("=" * 60)

# 1. Learned Positional Embeddings (like BERT)
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(0), device=x.device)
        return x + self.pos_embedding(positions)

# 2. Relative Positional Encoding (used in some modern transformers)
def relative_position_demo():
    """Show how relative positions work"""
    seq_len = 6
    # Create relative position matrix
    rel_pos = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos[i, j] = j - i  # Relative distance
    
    print("Relative Position Matrix:")
    print("(shows distance from each position to every other position)")
    print(rel_pos.numpy().astype(int))
    
    # Visualize
    plt.figure(figsize=(6, 5))
    sns.heatmap(rel_pos.numpy(), annot=True, fmt='.0f', cmap='RdBu_r', center=0)
    plt.title('Relative Position Encoding')
    plt.xlabel('Position j')
    plt.ylabel('Position i')
    plt.show()

relative_position_demo()

print("\nComparison of Positional Encoding Methods:")
print("-" * 60)
print("1. Sinusoidal (Original Transformer):")
print("   ✓ Deterministic, no parameters to learn")
print("   ✓ Can extrapolate to longer sequences")
print("   ✓ Captures relative positions naturally")
print("   ✗ Fixed pattern, less flexible")

print("\n2. Learned Embeddings (BERT):")
print("   ✓ Can learn task-specific patterns")
print("   ✓ More flexible")
print("   ✗ Requires training")
print("   ✗ Cannot handle sequences longer than training")

print("\n3. Relative Positional Encoding:")
print("   ✓ Directly models relative distances")
print("   ✓ Better for tasks where relative position matters more")
print("   ✗ More complex to implement")

# Final Challenge: Implement a custom positional encoding
# ======================================================
print("\n" + "=" * 60)
print("FINAL CHALLENGE: Create Your Own Positional Encoding")
print("=" * 60)

def create_custom_positional_encoding(max_len, d_model):
    """
    TODO: Implement your own positional encoding scheme!
    
    Ideas to try:
    - Linear interpolation between positions
    - Gaussian curves centered at each position
    - Binary encoding of positions
    - Combination of multiple patterns
    
    Requirements:
    - Output shape should be [max_len, d_model]
    - Values should be bounded (preferably -1 to 1)
    - Each position should have a unique encoding
    """
    # YOUR CODE HERE
    pe = torch.zeros(max_len, d_model)
    
    # Example starter: Linear decay
    # for pos in range(max_len):
    #     pe[pos, :] = ...
    
    return pe

# Test your implementation
# custom_pe = create_custom_positional_encoding(50, 128)
# plt.figure(figsize=(10, 6))
# sns.heatmap(custom_pe[:20, :64].numpy(), cmap='RdBu_r')
# plt.title('Your Custom Positional Encoding')
# plt.show()

print("\n" + "=" * 60)
print("Congratulations! You've completed the Positional Encoding exercise!")
print("=" * 60)

# Summary of Key Insights:
print("""
Key Takeaways:
1. Positional encoding is ESSENTIAL because self-attention is permutation-invariant
2. Sin/cos waves provide unique, bounded encodings that capture relative positions
3. The model learns to use these patterns to understand sequence order
4. Different frequencies allow encoding both local and global position information
5. This is completely deterministic - no learning required for the encoding itself
6. The transformer learns HOW to use this information through attention weights
""")