# Learning matrix projections, what it means, how it looks

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
Exercise: Understanding Projections in Transformers (20-30 min)

Goal: Build intuition for why transformers use projections and how they work.
You'll create data, project it, and visualize what's happening.
"""

# Part 1: Create Synthetic "Token Embeddings" (5 min)
# Imagine we have 4 tokens with 6-dimensional embeddings
# Let's make them have clear patterns:

torch.manual_seed(42)

# Create 4 tokens, each 6-dimensional
tokens = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # Token 1: "cat" (animal features)
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Token 2: "dog" (similar to cat)
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # Token 3: "runs" (verb features)
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # Token 4: "quickly" (adverb)
    ]
)

# Let's say dimensions represent:
# [is_noun, is_verb, is_animal, is_action, is_modifier, is_subject]
#
# IMPORTANT: This is a SIMPLIFIED EXAMPLE for learning purposes!
# Real token embeddings don't have interpretable dimensions like this.
# In reality, each dimension is a learned combination of many features.

print("Original tokens shape:", tokens.shape)
print("\nToken patterns:")
print("cat:    ", tokens[0].tolist())
print("dog:    ", tokens[1].tolist())
print("runs:   ", tokens[2].tolist())
print("quickly:", tokens[3].tolist())


# Part 2: Manual Projection - Understanding the Math (10 min)
# Let's project from 6D to 2D to see what happens

# Create a projection matrix that extracts:
# - Dimension 0: "semantic role" (noun vs verb)
# - Dimension 1: "syntactic function" (subject vs modifier)

W_manual = torch.tensor(
    [
        [1.0, -1.0],  # is_noun contributes to semantic, opposes syntactic
        [-1.0, 0.0],  # is_verb opposes semantic
        [0.5, 0.5],  # is_animal contributes to both
        [-0.5, 0.0],  # is_action opposes semantic
        [0.0, 1.0],  # is_modifier contributes to syntactic
        [0.3, 0.7],  # is_subject contributes more to syntactic
    ]
)

# TODO 1: Project the tokens using matrix multiplication
projected = tokens @ W_manual
print("\nProjected shape:", projected.shape)
print("Projected tokens:\n", projected)

# TODO 2: Compute one projection manually to understand
# For token 0 ("cat"), compute the first output dimension:
proj_0_0 = (
    tokens[0, 0] * W_manual[0, 0]
    + tokens[0, 1] * W_manual[1, 0]
    + tokens[0, 2] * W_manual[2, 0]
    + tokens[0, 3] * W_manual[3, 0]
    + tokens[0, 4] * W_manual[4, 0]
    + tokens[0, 5] * W_manual[5, 0]
)
print(f"\nManual calculation for cat's first dimension: {proj_0_0}")

# What Real Embeddings Look Like:
"""
In actual transformers, token embeddings are dense vectors where:
- Each dimension captures abstract, learned features
- No single dimension means "is_noun" or "is_verb"
- The meaning is distributed across all dimensions

Example of real embedding values:
"cat" → [0.234, -0.891, 0.445, 0.122, -0.667, ...]
"dog" → [0.198, -0.823, 0.501, 0.089, -0.702, ...]

Similar words have similar vectors (cosine similarity), but you can't 
point to dimension 5 and say "this represents animal-ness".

The transformer LEARNS these representations during training!
"""

# Let's see what REAL embeddings might look like:
print("\n--- REAL vs SIMPLIFIED EMBEDDINGS ---")
print("Simplified (for learning):")
print("cat: [1.0, 0.0, 1.0, 0.0, 0.0, 1.0] <- each dim has clear meaning")
print("\nReal embeddings (example):")
real_cat = torch.randn(6) * 0.5  # Random values, typical magnitude
real_dog = real_cat + torch.randn(6) * 0.1  # Similar to cat but not identical
print(f"cat: {real_cat.tolist()}")
print(f"dog: {real_dog.tolist()}")
print(f"Cosine similarity: {torch.cosine_similarity(real_cat, real_dog, dim=0):.3f}")
print("\nNotice: In real embeddings, meaning is distributed across ALL dimensions!")

# Part 2: Manual Projection - Understanding the Math (10 min)


# Part 3: Learned Projections - Q, K, V (10 min)
# Now let's see how attention heads use different projections

embed_dim = 6
head_dim = 2

# Initialize three different projection matrices
W_Q = nn.Linear(embed_dim, head_dim, bias=False)
W_K = nn.Linear(embed_dim, head_dim, bias=False)
W_V = nn.Linear(embed_dim, head_dim, bias=False)

# TODO 3: Project tokens to Q, K, V spaces
Q = W_Q(tokens)
K = W_K(tokens)
V = W_V(tokens)

print("Q shape:", Q.shape)

# TODO 4: Compute attention scores
# Remember: attention = softmax(Q @ K^T / sqrt(d_k))
scores = Q @ K.T / np.sqrt(head_dim)
print("Scores shape:", scores.shape)
print("Scores:\n", scores)
# scores[i,j] = "How much should token i attend to token j?"
attention_weights = torch.softmax(scores, dim=1)
# dim=1 means: normalize each ROW on it's own
# dim=0 would mean COLUMN, "Of all tokens looking at 'cat', how much does each contribute?"
print("\nAttention pattern:")
print(attention_weights)


# Part 4: Visualization - Why Multiple Heads? (5 min)
def visualize_projections(tokens, projections, title):
    """Plot original vs projected space"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Original space (show first 2 dims only)
    ax1.scatter(tokens[:, 0], tokens[:, 1], s=100)
    for i, label in enumerate(["cat", "dog", "runs", "quickly"]):
        ax1.annotate(label, (tokens[i, 0], tokens[i, 1]))
    ax1.set_title("Original Space (first 2 of 6 dims)")
    ax1.set_xlabel("is_noun")
    ax1.set_ylabel("is_verb")

    # Projected space
    projections = projections.detach().numpy()
    ax2.scatter(projections[:, 0], projections[:, 1], s=100)
    for i, label in enumerate(["cat", "dog", "runs", "quickly"]):
        ax2.annotate(label, (projections[i, 0], projections[i, 1]))
    ax2.set_title(f"Projected Space (2D) - {title}")
    ax2.set_xlabel("Learned dim 0")
    ax2.set_ylabel("Learned dim 1")

    plt.tight_layout()
    plt.show()


# TODO 5: Visualize different projections
# Create 3 different projection matrices (like 3 attention heads)
# Show how each captures different relationships

visualize_projections(tokens, Q, "Q")
# visualize_projections(tokens, K, "K")
# visualize_projections(tokens, V, "V")


# After computing attention_weights, apply to V:
attended_output = attention_weights @ V
print("\nAttended output shape:", attended_output.shape)
print("This is what each token 'sees' after attention!")
# visualize_projections(tokens, attended_output, "Attended Output")


# Part 5: Think & Answer (5 min)
"""
Questions to ponder:

1. Why project from 6D to 2D instead of using all 6 dimensions?
   Hint: Think about computation and specialization.

2. How does projection help with finding patterns?
   Hint: Look at how similar tokens cluster in projected space.

3. Why does each head need different W_Q, W_K, W_V?
   Hint: What if all heads had the same projections?

Write your thoughts here:
YOUR_ANSWER = '''
1. Reducing computation, compressing information, forces to choose most important feats/dimensions
2. Dimensionality reduction, makes the selected dimensions much richer.
3. all heads would learn the same things.
'''
"""

# Bonus: Experiment with different projection matrices
# Try creating projections that:
# - Group nouns together
# - Separate actions from objects
# - Find subject-verb relationships
