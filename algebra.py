"""Prompt:
I'm trying to understand algebra required to understand machine learning and neural networks in depth, this includes things as vectors, eigen vectors, matrices, tensors, multiplying matrices and vectors, linear dependence and spans, eigendecomposition, dotproduct, transpose. 

I want to understand it from a basic math, but also from a meaning, specially meaning associated neural networks.  Include a few examples using multiplication, numpy, and pytorch, so that the relationship is understood.

Create a walkthrough exercise that could teach me all this concepts, make sure the exercise has todo's/exercises, visualizations, this should be enough for me to understand algebra related to machine learning, and shouldn't take me longer than 2 hours to execute and learn.

create algebra.py, make sure initial utils are defined, and each exercise section is a function, make only function 1/exercise 1 not commented, make subsequent ones commented, so I do that manually when going through it
"""

"""
Linear Algebra for Machine Learning - Interactive Walkthrough
============================================================

This file contains 7 exercises that will teach you the essential linear algebra
concepts needed for understanding neural networks and machine learning.

Estimated time: 2 hours

Instructions:
1. Run exercise_1() first (it's uncommented)
2. After completing each exercise, uncomment the next one
3. Each exercise builds on previous concepts
4. Pay attention to both mathematical concepts and ML applications

Requirements: numpy, torch, matplotlib
"""

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from typing import List  # noqa: E402

# Utility functions for visualization and helper operations
def plot_vectors_2d(vectors: List[np.ndarray], labels: List[str] = None, colors: List[str] = None):
    """Plot 2D vectors from origin."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'purple', 'orange']
    if labels is None:
        labels = [f'v{i+1}' for i in range(len(vectors))]
    
    for i, vec in enumerate(vectors):
        ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, 
                 color=colors[i % len(colors)], label=labels[i], width=0.005)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_aspect('equal')
    plt.title('Vector Visualization')
    plt.show()

def print_matrix_info(matrix: np.ndarray, name: str = "Matrix"):
    """Print useful information about a matrix."""
    print(f"\n{name}:")
    print(f"Shape: {matrix.shape}")
    print(f"Content:\n{matrix}")
    print(f"Dtype: {matrix.dtype}")

def create_neural_network_weights(input_size: int, hidden_size: int, output_size: int):
    """Create sample neural network weight matrices."""
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    return W1, W2

# Exercise 1: Vectors and Basic Operations (ACTIVE - UNCOMMENTED)
def exercise_1_vectors():
    """
    Exercise 1: Understanding Vectors
    
    Learning objectives:
    - What are vectors and how to create them
    - Vector operations: addition, scalar multiplication
    - Vector magnitude and normalization
    - Connection to neural network inputs/features
    """
    print("=" * 60)
    print("EXERCISE 1: VECTORS AND BASIC OPERATIONS")
    print("=" * 60)
    
    # 1.1: Creating vectors
    print("\n1.1 Creating Vectors")
    print("-" * 30)
    
    # In ML, vectors often represent features of data points
    # Example: A house with 3 features [bedrooms, bathrooms, square_feet]
    house_features = np.array([3, 2, 1500])
    print(f"House features vector: {house_features}")
    
    # In PyTorch (common in deep learning)
    house_features_torch = torch.tensor([3, 2, 1500], dtype=torch.float32)
    print(f"Same vector in PyTorch: {house_features_torch}")
    
    # TODO: Create your own vector representing a car with features:
    # [doors, cylinders, horsepower]
    # Example: 4 doors, 6 cylinders, 250 horsepower
    car_features = np.array([4, 6, 250])  # Replace with your vector
    print(f"Your car vector: {car_features}")
    
    # 1.2: Vector operations
    print("\n1.2 Vector Operations")
    print("-" * 30)
    
    # Vector addition - combining features
    v1 = np.array([1, 2])
    v2 = np.array([3, 1])
    v_sum = v1 + v2
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v_sum}")
    
    # Scalar multiplication - scaling features
    scaled_v1 = 2 * v1
    print(f"2 * v1 = {scaled_v1}")
    
    # Visualize these vectors
    plot_vectors_2d([v1, v2, v_sum, scaled_v1], 
                   ['v1', 'v2', 'v1+v2', '2*v1'],
                   ['red', 'blue', 'green', 'purple'])
    
    # TODO: Create two 2D vectors and visualize their sum
    your_v1 = np.array([1, 2])  # Create a 2D vector
    your_v2 = np.array([2, 3])  # Create another 2D vector
    # Uncomment the next line after creating your vectors:
    plot_vectors_2d([your_v1, your_v2, your_v1 + your_v2], ['your_v1', 'your_v2', 'sum'])
    
    # 1.3: Vector magnitude and normalization
    print("\n1.3 Vector Magnitude and Normalization")
    print("-" * 30)
    
    v = np.array([3, 4])
    magnitude = np.linalg.norm(v)
    normalized_v = v / magnitude
    
    print(f"Vector v = {v}")
    print(f"Magnitude ||v|| = {magnitude}")
    print(f"Normalized v = {normalized_v}")
    print(f"Magnitude of normalized v = {np.linalg.norm(normalized_v)}")
    
    # In ML: Feature normalization is crucial for neural networks
    # Example: normalizing pixel values from [0, 255] to [0, 1]
    pixel_values = np.array([255, 128, 64, 0])
    normalized_pixels = pixel_values / 255.0
    print(f"\nPixel normalization example:")
    print(f"Original pixels: {pixel_values}")
    print(f"Normalized pixels: {normalized_pixels}")
    
    # TODO: Calculate the magnitude of your car_features vector
    # and create a normalized version
    if car_features is not None:
        car_magnitude = None  # Calculate magnitude
        normalized_car = None  # Calculate normalized vector
        print(f"Car features magnitude: {car_magnitude}")
        print(f"Normalized car features: {normalized_car}")
    
    print("\n" + "="*60)
    print("Exercise 1 Complete!")
    print("Key takeaways:")
    print("- Vectors represent data points/features in ML")
    print("- Vector operations are fundamental to neural networks")
    print("- Normalization helps with training stability")
    print("- NumPy and PyTorch provide efficient vector operations")
    print("="*60)

# Call Exercise 1
exercise_1_vectors()

# Exercise 2: Matrices and Matrix Operations (COMMENTED - UNCOMMENT TO USE)
"""
def exercise_2_matrices():
    '''
    Exercise 2: Understanding Matrices
    
    Learning objectives:
    - What are matrices and how they relate to neural networks
    - Matrix operations: addition, multiplication
    - Matrix transpose
    - Connection to weight matrices in neural networks
    '''
    print("=" * 60)
    print("EXERCISE 2: MATRICES AND MATRIX OPERATIONS")
    print("=" * 60)
    
    # 2.1: Creating matrices
    print("\\n2.1 Creating Matrices")
    print("-" * 30)
    
    # In neural networks, weight matrices transform inputs
    # Example: 3 input features -> 2 hidden neurons
    W = np.array([[0.1, 0.2],    # weights for neuron 1
                  [0.3, 0.4],    # weights for neuron 2  
                  [0.5, 0.6]])   # weights for neuron 3
    
    print_matrix_info(W, "Weight Matrix W")
    
    # In PyTorch
    W_torch = torch.tensor(W, dtype=torch.float32)
    print(f"\\nSame matrix in PyTorch:\\n{W_torch}")
    
    # TODO: Create a weight matrix for 4 inputs -> 3 outputs
    # Use random values between -0.5 and 0.5
    your_weights = None  # Replace with your matrix
    
    # 2.2: Matrix operations
    print("\\n2.2 Matrix Operations")
    print("-" * 30)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Matrix addition
    C = A + B
    print_matrix_info(A, "Matrix A")
    print_matrix_info(B, "Matrix B")
    print_matrix_info(C, "A + B")
    
    # Matrix multiplication (crucial for neural networks!)
    D = np.dot(A, B)  # or A @ B
    print_matrix_info(D, "A @ B (matrix multiplication)")
    
    # Element-wise multiplication (Hadamard product)
    E = A * B
    print_matrix_info(E, "A * B (element-wise)")
    
    # TODO: Multiply your weight matrix by itself (if square)
    # or create appropriate matrices for multiplication
    
    # 2.3: Matrix transpose
    print("\\n2.3 Matrix Transpose")
    print("-" * 30)
    
    print_matrix_info(A, "Original A")
    print_matrix_info(A.T, "A transpose (A.T)")
    
    # In neural networks: W.T is used in backpropagation
    print("\\nNeural network connection:")
    print("Forward pass: output = input @ W")
    print("Backward pass: gradient flows through W.T")
    
    # 2.4: Identity matrix and matrix properties
    print("\\n2.4 Special Matrices")
    print("-" * 30)
    
    I = np.eye(3)  # 3x3 identity matrix
    print_matrix_info(I, "Identity Matrix")
    
    # Identity property: A @ I = A
    test_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    result = test_matrix @ I
    print(f"\\nTest: [[1,2,3], [4,5,6]] @ I = \\n{result}")
    
    print("\\n" + "="*60)
    print("Exercise 2 Complete!")
    print("Key takeaways:")
    print("- Matrices are fundamental to neural network computations")
    print("- Matrix multiplication transforms data through layers")
    print("- Transpose is crucial for backpropagation")
    print("- Understanding shapes is essential for debugging")
    print("="*60)
"""

# Exercise 3: Matrix-Vector Multiplication and Neural Networks (COMMENTED)
"""
def exercise_3_matrix_vector_multiplication():
    '''
    Exercise 3: Matrix-Vector Multiplication in Neural Networks
    
    Learning objectives:
    - How matrix-vector multiplication works
    - Connection to neural network forward pass
    - Understanding shapes and dimensions
    - Implementing a simple neural network layer
    '''
    print("=" * 60)
    print("EXERCISE 3: MATRIX-VECTOR MULTIPLICATION & NEURAL NETWORKS")
    print("=" * 60)
    
    # 3.1: Basic matrix-vector multiplication
    print("\\n3.1 Matrix-Vector Multiplication Mechanics")
    print("-" * 50)
    
    # Simple example
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    x = np.array([1, 0, 1])
    
    result = A @ x
    
    print_matrix_info(A, "Matrix A (2x3)")
    print(f"Vector x: {x} (shape: {x.shape})")
    print(f"Result A @ x: {result} (shape: {result.shape})")
    
    # Manual calculation to understand what's happening
    print("\\nManual calculation:")
    print(f"Row 1: [1,2,3] · [1,0,1] = 1*1 + 2*0 + 3*1 = {1*1 + 2*0 + 3*1}")
    print(f"Row 2: [4,5,6] · [1,0,1] = 4*1 + 5*0 + 6*1 = {4*1 + 5*0 + 6*1}")
    
    # TODO: Create your own 3x2 matrix and 2x1 vector
    # Multiply them and verify the result manually
    your_matrix = None  # 3x2 matrix
    your_vector = None  # 2x1 vector
    # your_result = your_matrix @ your_vector
    
    # 3.2: Neural network forward pass
    print("\\n3.2 Neural Network Forward Pass")
    print("-" * 50)
    
    # Simple neural network: 3 inputs -> 4 hidden -> 2 outputs
    np.random.seed(42)  # For reproducible results
    
    # Input (could be features of a data point)
    input_features = np.array([0.5, -0.2, 0.8])
    print(f"Input features: {input_features}")
    
    # First layer: 3 inputs -> 4 hidden neurons
    W1 = np.random.randn(3, 4) * 0.5
    b1 = np.random.randn(4) * 0.1
    
    hidden = W1.T @ input_features + b1  # Linear transformation
    hidden_activated = np.maximum(0, hidden)  # ReLU activation
    
    print(f"\\nFirst layer:")
    print(f"W1 shape: {W1.shape}")
    print(f"Hidden (before activation): {hidden}")
    print(f"Hidden (after ReLU): {hidden_activated}")
    
    # Second layer: 4 hidden -> 2 outputs
    W2 = np.random.randn(4, 2) * 0.5
    b2 = np.random.randn(2) * 0.1
    
    output = W2.T @ hidden_activated + b2
    
    print(f"\\nSecond layer:")
    print(f"W2 shape: {W2.shape}")
    print(f"Final output: {output}")
    
    # 3.3: Batch processing (multiple inputs at once)
    print("\\n3.3 Batch Processing")
    print("-" * 50)
    
    # Multiple data points (batch)
    batch_inputs = np.array([[0.5, -0.2, 0.8],
                            [0.1, 0.3, -0.4],
                            [-0.2, 0.7, 0.1]])
    
    print(f"Batch inputs shape: {batch_inputs.shape} (3 samples, 3 features each)")
    
    # Process entire batch at once
    hidden_batch = batch_inputs @ W1 + b1  # Broadcasting handles bias
    hidden_batch_activated = np.maximum(0, hidden_batch)
    output_batch = hidden_batch_activated @ W2 + b2
    
    print(f"Batch outputs shape: {output_batch.shape}")
    print(f"Batch outputs:\\n{output_batch}")
    
    # TODO: Create your own mini-batch of 2 samples with 4 features each
    # Process them through a 4->3->1 network
    
    # 3.4: PyTorch implementation
    print("\\n3.4 PyTorch Implementation")
    print("-" * 50)
    
    # Convert to PyTorch tensors
    input_torch = torch.tensor(input_features, dtype=torch.float32)
    W1_torch = torch.tensor(W1.T, dtype=torch.float32)  # Note: transpose for PyTorch convention
    b1_torch = torch.tensor(b1, dtype=torch.float32)
    
    # Forward pass in PyTorch
    hidden_torch = torch.relu(torch.matmul(input_torch, W1_torch) + b1_torch)
    print(f"PyTorch hidden layer: {hidden_torch}")
    
    # Using nn.Linear (the standard way)
    import torch.nn as nn
    
    layer1 = nn.Linear(3, 4)
    layer2 = nn.Linear(4, 2)
    
    with torch.no_grad():  # Disable gradient computation for this example
        out1 = torch.relu(layer1(input_torch))
        out2 = layer2(out1)
    
    print(f"Using nn.Linear: {out2}")
    
    print("\\n" + "="*60)
    print("Exercise 3 Complete!")
    print("Key takeaways:")
    print("- Matrix-vector multiplication is the core of neural networks")
    print("- Each layer transforms input through W @ x + b")
    print("- Batch processing uses matrix-matrix multiplication")
    print("- PyTorch abstracts these operations with nn.Linear")
    print("="*60)
"""

# Exercise 4: Dot Products and Similarity (COMMENTED)
"""
def exercise_4_dot_products():
    '''
    Exercise 4: Dot Products and Similarity
    
    Learning objectives:
    - Understanding dot products geometrically and algebraically
    - Connection to similarity and attention mechanisms
    - Cosine similarity
    - Applications in recommendation systems and transformers
    '''
    print("=" * 60)
    print("EXERCISE 4: DOT PRODUCTS AND SIMILARITY")
    print("=" * 60)
    
    # 4.1: Basic dot product
    print("\\n4.1 Understanding Dot Products")
    print("-" * 40)
    
    a = np.array([3, 4])
    b = np.array([1, 2])
    
    dot_product = np.dot(a, b)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Dot product a · b: {dot_product}")
    
    # Manual calculation
    manual_dot = a[0]*b[0] + a[1]*b[1]
    print(f"Manual calculation: {a[0]}*{b[0]} + {a[1]}*{b[1]} = {manual_dot}")
    
    # Geometric interpretation
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    cos_angle = dot_product / (mag_a * mag_b)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    print(f"\\nGeometric interpretation:")
    print(f"|a| = {mag_a:.3f}")
    print(f"|b| = {mag_b:.3f}")
    print(f"cos(θ) = {cos_angle:.3f}")
    print(f"Angle θ = {angle_deg:.1f} degrees")
    
    # Visualize
    plot_vectors_2d([a, b], ['a', 'b'])
    
    # TODO: Calculate dot product between [1, 0] and [0, 1]
    # What does this tell you about perpendicular vectors?
    
    # 4.2: Cosine similarity
    print("\\n4.2 Cosine Similarity")
    print("-" * 40)
    
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Example: Document similarity
    # Documents represented as word count vectors
    doc1 = np.array([2, 1, 0, 1])  # [cat, dog, fish, bird]
    doc2 = np.array([1, 2, 0, 0])  # [cat, dog, fish, bird]
    doc3 = np.array([0, 0, 3, 1])  # [cat, dog, fish, bird]
    
    sim_1_2 = cosine_similarity(doc1, doc2)
    sim_1_3 = cosine_similarity(doc1, doc3)
    sim_2_3 = cosine_similarity(doc2, doc3)
    
    print("Document similarity example:")
    print(f"Doc1 (cats & dogs): {doc1}")
    print(f"Doc2 (more dogs): {doc2}")
    print(f"Doc3 (fish & birds): {doc3}")
    print(f"\\nSimilarities:")
    print(f"Doc1 ↔ Doc2: {sim_1_2:.3f}")
    print(f"Doc1 ↔ Doc3: {sim_1_3:.3f}")
    print(f"Doc2 ↔ Doc3: {sim_2_3:.3f}")
    
    # TODO: Create word vectors for three sentences and compute similarities
    # Sentence 1: "I love machine learning"
    # Sentence 2: "Machine learning is awesome"  
    # Sentence 3: "I love pizza"
    # Use vocabulary: [I, love, machine, learning, is, awesome, pizza]
    
    # 4.3: Attention mechanism (simplified)
    print("\\n4.3 Attention Mechanism (Simplified)")
    print("-" * 40)
    
    # Simplified attention: how much should we focus on each word?
    query = np.array([1, 0, 1])      # What we're looking for
    key1 = np.array([1, 0, 0])      # First word representation
    key2 = np.array([0, 1, 0])      # Second word representation  
    key3 = np.array([1, 0, 1])      # Third word representation
    
    # Calculate attention scores (dot products)
    score1 = np.dot(query, key1)
    score2 = np.dot(query, key2)
    score3 = np.dot(query, key3)
    
    print(f"Query: {query}")
    print(f"Key1: {key1}, Score: {score1}")
    print(f"Key2: {key2}, Score: {score2}")
    print(f"Key3: {key3}, Score: {score3}")
    
    # Convert to attention weights (softmax)
    scores = np.array([score1, score2, score3])
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    
    print(f"\\nAttention weights: {attention_weights}")
    print(f"Sum of weights: {np.sum(attention_weights):.3f}")
    
    # 4.4: Recommendation system example
    print("\\n4.4 Recommendation System Example")
    print("-" * 40)
    
    # User preferences (ratings for different movie genres)
    user_preferences = np.array([4, 2, 5, 1, 3])  # [action, comedy, sci-fi, romance, horror]
    
    # Movie profiles
    movie1 = np.array([5, 1, 4, 0, 2])  # Action sci-fi
    movie2 = np.array([1, 5, 1, 4, 1])  # Romantic comedy
    movie3 = np.array([2, 1, 5, 0, 4])  # Sci-fi horror
    
    # Calculate similarity scores
    rec_score1 = cosine_similarity(user_preferences, movie1)
    rec_score2 = cosine_similarity(user_preferences, movie2)
    rec_score3 = cosine_similarity(user_preferences, movie3)
    
    print("Movie recommendation example:")
    print(f"User preferences: {user_preferences}")
    print(f"Movie 1 (Action Sci-fi): {rec_score1:.3f}")
    print(f"Movie 2 (Rom-com): {rec_score2:.3f}")
    print(f"Movie 3 (Sci-fi Horror): {rec_score3:.3f}")
    
    best_movie = np.argmax([rec_score1, rec_score2, rec_score3]) + 1
    print(f"\\nRecommended movie: Movie {best_movie}")
    
    print("\\n" + "="*60)
    print("Exercise 4 Complete!")
    print("Key takeaways:")
    print("- Dot products measure similarity between vectors")
    print("- Cosine similarity is scale-invariant")
    print("- Attention mechanisms use dot products for relevance")
    print("- Recommendation systems rely on similarity measures")
    print("="*60)
"""

# Exercise 5: Linear Dependence and Spans (COMMENTED)
"""
def exercise_5_linear_dependence():
    '''
    Exercise 5: Linear Dependence and Spans
    
    Learning objectives:
    - Understanding linear combinations
    - Linear independence and dependence
    - Vector spans and subspaces
    - Connection to feature representations and dimensionality
    '''
    print("=" * 60)
    print("EXERCISE 5: LINEAR DEPENDENCE AND SPANS")
    print("=" * 60)
    
    # 5.1: Linear combinations
    print("\\n5.1 Linear Combinations")
    print("-" * 40)
    
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    # Any 2D vector can be written as a linear combination of v1 and v2
    target = np.array([3, 2])
    
    # target = a*v1 + b*v2
    a, b = 3, 2
    combination = a * v1 + b * v2
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"Target = {target}")
    print(f"Linear combination: {a}*v1 + {b}*v2 = {combination}")
    
    # Visualize
    plot_vectors_2d([v1, v2, target, a*v1, b*v2], 
                   ['v1', 'v2', 'target', f'{a}*v1', f'{b}*v2'])
    
    # TODO: Express the vector [4, -1] as a linear combination of v1 and v2
    
    # 5.2: Linear independence
    print("\\n5.2 Linear Independence")
    print("-" * 40)
    
    # Independent vectors
    ind_v1 = np.array([1, 0])
    ind_v2 = np.array([0, 1])
    
    print("Independent vectors:")
    print(f"v1 = {ind_v1}")
    print(f"v2 = {ind_v2}")
    print("These vectors are linearly independent because:")
    print("c1*v1 + c2*v2 = [0,0] only when c1=0 and c2=0")
    
    # Dependent vectors
    dep_v1 = np.array([1, 2])
    dep_v2 = np.array([2, 4])  # This is 2 * dep_v1
    
    print(f"\\nDependent vectors:")
    print(f"v1 = {dep_v1}")
    print(f"v2 = {dep_v2}")
    print("These are linearly dependent because v2 = 2*v1")
    print("So: 2*v1 - 1*v2 = [0,0]")
    
    # Verify dependence
    dependence_check = 2 * dep_v1 - 1 * dep_v2
    print(f"Check: 2*v1 - 1*v2 = {dependence_check}")
    
    # Visualize dependent vectors
    plot_vectors_2d([dep_v1, dep_v2], ['v1', 'v2 (=2*v1)'])
    
    # TODO: Check if vectors [1,2,3] and [2,4,6] are linearly dependent
    
    # 5.3: Determining linear independence with matrices
    print("\\n5.3 Matrix Method for Linear Independence")
    print("-" * 50)
    
    # Create matrix with vectors as columns
    vectors = np.array([[1, 2],
                       [2, 3],
                       [3, 4]])
    
    print_matrix_info(vectors, "Matrix with vectors as columns")
    
    # Calculate rank
    rank = np.linalg.matrix_rank(vectors)
    print(f"Rank: {rank}")
    print(f"Number of vectors: {vectors.shape[1]}")
    
    if rank == vectors.shape[1]:
        print("Vectors are linearly independent")
    else:
        print("Vectors are linearly dependent")
    
    # 5.4: Span and basis
    print("\\n5.4 Span and Basis")
    print("-" * 40)
    
    # Standard basis for R²
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    print("Standard basis for R²:")
    print(f"e1 = {e1}")
    print(f"e2 = {e2}")
    print("These vectors span all of R² (any 2D vector can be expressed using them)")
    
    # Non-standard basis
    b1 = np.array([1, 1])
    b2 = np.array([1, -1])
    
    print(f"\\nAlternative basis:")
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    
    # Express a vector in this new basis
    target_vec = np.array([3, 1])
    # target_vec = c1*b1 + c2*b2
    # [3, 1] = c1*[1,1] + c2*[1,-1]
    # 3 = c1 + c2
    # 1 = c1 - c2
    # Solving: c1 = 2, c2 = 1
    
    c1, c2 = 2, 1
    reconstructed = c1 * b1 + c2 * b2
    print(f"\\nExpress {target_vec} in new basis:")
    print(f"{c1}*{b1} + {c2}*{b2} = {reconstructed}")
    
    # 5.5: Connection to neural networks
    print("\\n5.5 Connection to Neural Networks")
    print("-" * 40)
    
    print("In neural networks:")
    print("- Each layer learns a new representation (basis) for the data")
    print("- Features should be linearly independent for best performance")
    print("- Redundant features (linearly dependent) waste capacity")
    print("- Principal Component Analysis (PCA) finds the best basis")
    
    # Simple example: removing redundant features
    original_features = np.array([[1, 2, 2],    # Sample 1: [feature1, feature2, 2*feature1]
                                 [2, 3, 4],    # Sample 2
                                 [3, 1, 6]])   # Sample 3
    
    print(f"\\nOriginal features (with redundancy):")
    print(f"Shape: {original_features.shape}")
    print(original_features)
    
    # Remove the redundant third column (it's 2*first column)
    reduced_features = original_features[:, :2]
    print(f"\\nReduced features (redundancy removed):")
    print(f"Shape: {reduced_features.shape}")
    print(reduced_features)
    
    print("\\n" + "="*60)
    print("Exercise 5 Complete!")
    print("Key takeaways:")
    print("- Linear independence ensures no redundant information")
    print("- Basis vectors span the entire space efficiently")
    print("- Neural networks learn new representations/bases")
    print("- Removing linear dependence improves efficiency")
    print("="*60)
"""

# Exercise 6: Eigenvalues and Eigenvectors (COMMENTED)
"""
def exercise_6_eigendecomposition():
    '''
    Exercise 6: Eigenvalues and Eigenvectors
    
    Learning objectives:
    - Understanding eigenvalues and eigenvectors
    - Eigendecomposition of matrices
    - Principal Component Analysis (PCA)
    - Applications in dimensionality reduction and neural networks
    '''
    print("=" * 60)
    print("EXERCISE 6: EIGENVALUES AND EIGENVECTORS")
    print("=" * 60)
    
    # 6.1: Basic concept
    print("\\n6.1 Understanding Eigenvectors")
    print("-" * 40)
    
    # Simple 2x2 matrix
    A = np.array([[3, 1],
                  [0, 2]])
    
    print_matrix_info(A, "Matrix A")
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\\n{eigenvectors}")
    
    # Verify the eigenvalue equation: A * v = λ * v
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ vec
        lambda_v = val * vec
        print(f"\\nEigenvector {i+1}: {vec}")
        print(f"A * v = {Av}")
        print(f"λ * v = {lambda_v}")
        print(f"Close? {np.allclose(Av, lambda_v)}")
    
    # Visualize eigenvectors
    plot_vectors_2d([eigenvectors[:, 0], eigenvectors[:, 1]], 
                   [f'v1 (λ={eigenvalues[0]:.2f})', f'v2 (λ={eigenvalues[1]:.2f})'])
    
    # TODO: Find eigenvalues and eigenvectors of [[2, 1], [1, 2]]
    # What do you notice about the eigenvalues?
    
    # 6.2: Geometric interpretation
    print("\\n6.2 Geometric Interpretation")
    print("-" * 40)
    
    # Create some test vectors
    test_vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]]).T
    
    print("Transformation visualization:")
    print("Original vectors vs. transformed vectors")
    
    # Apply transformation
    transformed = A @ test_vectors
    
    # Plot original and transformed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original vectors
    for i in range(test_vectors.shape[1]):
        ax1.quiver(0, 0, test_vectors[0, i], test_vectors[1, i], 
                  angles='xy', scale_units='xy', scale=1, width=0.005)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Vectors')
    ax1.set_aspect('equal')
    
    # Transformed vectors
    for i in range(transformed.shape[1]):
        ax2.quiver(0, 0, transformed[0, i], transformed[1, i], 
                  angles='xy', scale_units='xy', scale=1, width=0.005)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Transformed Vectors')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 6.3: Principal Component Analysis (PCA)
    print("\\n6.3 Principal Component Analysis")
    print("-" * 40)
    
    # Generate sample data (2D with correlation)
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated data
    x1 = np.random.randn(n_samples)
    x2 = 0.5 * x1 + 0.5 * np.random.randn(n_samples)
    data = np.column_stack([x1, x2])
    
    print(f"Data shape: {data.shape}")
    print(f"Data preview:\\n{data[:5]}")
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print_matrix_info(cov_matrix, "Covariance Matrix")
    
    # Find principal components (eigenvectors of covariance matrix)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print(f"\\nPrincipal component eigenvalues: {eigenvals}")
    print(f"Variance explained: {eigenvals / np.sum(eigenvals) * 100}%")
    
    # Plot data and principal components
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.6)
    plt.quiver(0, 0, eigenvecs[0, 0]*np.sqrt(eigenvals[0]), 
              eigenvecs[1, 0]*np.sqrt(eigenvals[0]), 
              color='red', scale=1, scale_units='xy', angles='xy', width=0.01)
    plt.quiver(0, 0, eigenvecs[0, 1]*np.sqrt(eigenvals[1]), 
              eigenvecs[1, 1]*np.sqrt(eigenvals[1]), 
              color='blue', scale=1, scale_units='xy', angles='xy', width=0.01)
    plt.title('Data with Principal Components')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Transform data to principal component space
    data_pca = data_centered @ eigenvecs
    
    plt.subplot(1, 2, 2)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.6)
    plt.title('Data in Principal Component Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Dimensionality reduction: keep only first component
    reduced_data = data_pca[:, :1]
    print(f"\\nOriginal data shape: {data.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")
    print(f"Information retained: {eigenvals[0]/np.sum(eigenvals)*100:.1f}%")
    
    # 6.4: Connection to neural networks
    print("\\n6.4 Applications in Deep Learning")
    print("-" * 40)
    
    print("Eigendecomposition applications:")
    print("1. PCA for preprocessing: reduce input dimensions")
    print("2. Spectral normalization: stabilize GAN training")
    print("3. Eigenvalues of Hessian: analyze loss landscape")
    print("4. Graph neural networks: use graph Laplacian eigendecomposition")
    
    # Simple example: PCA preprocessing for neural network
    print("\\nExample: PCA preprocessing pipeline")
    
    # Simulate high-dimensional data
    high_dim_data = np.random.randn(50, 10)  # 50 samples, 10 features
    
    # Apply PCA to reduce to 3 dimensions
    data_centered_hd = high_dim_data - np.mean(high_dim_data, axis=0)
    cov_hd = np.cov(data_centered_hd.T)
    vals_hd, vecs_hd = np.linalg.eig(cov_hd)
    
    # Sort and take top 3 components
    idx_hd = np.argsort(vals_hd)[::-1]
    top_3_components = vecs_hd[:, idx_hd[:3]]
    
    # Transform data
    reduced_hd = data_centered_hd @ top_3_components
    
    print(f"Original shape: {high_dim_data.shape}")
    print(f"Reduced shape: {reduced_hd.shape}")
    print(f"Variance retained: {np.sum(vals_hd[idx_hd[:3]])/np.sum(vals_hd)*100:.1f}%")
    
    print("\\n" + "="*60)
    print("Exercise 6 Complete!")
    print("Key takeaways:")
    print("- Eigenvectors show directions of maximum variance")
    print("- PCA uses eigendecomposition for dimensionality reduction")
    print("- Eigenvalues indicate importance of each component")
    print("- Preprocessing with PCA can improve neural network performance")
    print("="*60)
"""

# Exercise 7: Tensors and Neural Network Applications (COMMENTED)
"""
def exercise_7_tensors():
    '''
    Exercise 7: Tensors and Advanced Applications
    
    Learning objectives:
    - Understanding tensors as multi-dimensional arrays
    - Tensor operations in deep learning
    - Convolution as tensor operation
    - Attention mechanisms with tensors
    - Practical PyTorch tensor manipulations
    '''
    print("=" * 60)
    print("EXERCISE 7: TENSORS AND NEURAL NETWORK APPLICATIONS")
    print("=" * 60)
    
    # 7.1: Understanding tensors
    print("\\n7.1 Understanding Tensors")
    print("-" * 40)
    
    # Scalar (0D tensor)
    scalar = np.array(5)
    print(f"Scalar (0D): {scalar}, shape: {scalar.shape}")
    
    # Vector (1D tensor)
    vector = np.array([1, 2, 3])
    print(f"Vector (1D): {vector}, shape: {vector.shape}")
    
    # Matrix (2D tensor)
    matrix = np.array([[1, 2], [3, 4]])
    print(f"Matrix (2D):\\n{matrix}, shape: {matrix.shape}")
    
    # 3D tensor (common in computer vision: height × width × channels)
    tensor_3d = np.random.randn(32, 32, 3)  # RGB image
    print(f"3D tensor (image): shape {tensor_3d.shape}")
    
    # 4D tensor (batch of images: batch × height × width × channels)
    tensor_4d = np.random.randn(16, 32, 32, 3)  # 16 RGB images
    print(f"4D tensor (batch of images): shape {tensor_4d.shape}")
    
    # TODO: Create a 5D tensor representing a batch of video frames
    # Dimensions: [batch, time, height, width, channels]
    # Use shape (8, 10, 64, 64, 3) for 8 videos, 10 frames each
    
    # 7.2: Tensor operations
    print("\\n7.2 Tensor Operations")
    print("-" * 40)
    
    # Element-wise operations
    A = np.random.randn(2, 3, 4)
    B = np.random.randn(2, 3, 4)
    
    C = A + B  # Element-wise addition
    D = A * B  # Element-wise multiplication
    
    print(f"Tensor A shape: {A.shape}")
    print(f"Tensor B shape: {B.shape}")
    print(f"A + B shape: {C.shape}")
    print(f"A * B shape: {D.shape}")
    
    # Broadcasting
    broadcast_tensor = A + np.array([1, 2, 3, 4])  # Broadcasts across last dimension
    print(f"Broadcasting result shape: {broadcast_tensor.shape}")
    
    # Tensor contraction (generalized matrix multiplication)
    # Example: batch matrix multiplication
    batch_A = np.random.randn(5, 3, 4)  # 5 matrices of size 3×4
    batch_B = np.random.randn(5, 4, 2)  # 5 matrices of size 4×2
    
    # Multiply corresponding matrices in each batch
    batch_result = np.matmul(batch_A, batch_B)  # or use @ operator
    print(f"\\nBatch matrix multiplication:")
    print(f"Batch A: {batch_A.shape}")
    print(f"Batch B: {batch_B.shape}")
    print(f"Result: {batch_result.shape}")
    
    # 7.3: Convolution as tensor operation
    print("\\n7.3 Convolution Operations")
    print("-" * 40)
    
    # Simple 2D convolution example
    def simple_conv2d(input_tensor, kernel):
        h_in, w_in = input_tensor.shape
        h_k, w_k = kernel.shape
        h_out = h_in - h_k + 1
        w_out = w_in - w_k + 1
        
        output = np.zeros((h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                output[i, j] = np.sum(input_tensor[i:i+h_k, j:j+w_k] * kernel)
        
        return output
    
    # Example: edge detection
    image = np.array([[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]])
    
    edge_kernel = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
    
    edge_result = simple_conv2d(image, edge_kernel)
    
    print("Original image:")
    print(image)
    print("\\nEdge detection kernel:")
    print(edge_kernel)
    print("\\nConvolution result:")
    print(edge_result)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im1 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im2 = axes[1].imshow(edge_kernel, cmap='RdBu')
    axes[1].set_title('Edge Detection Kernel')
    axes[1].axis('off')
    
    im3 = axes[2].imshow(edge_result, cmap='gray')
    axes[2].set_title('Convolution Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # TODO: Try different kernels (blur, sharpen) on the same image
    
    # 7.4: Attention mechanism with tensors
    print("\\n7.4 Attention Mechanism")
    print("-" * 40)
    
    # Simplified multi-head attention
    seq_len = 4
    d_model = 6
    
    # Input sequence (e.g., word embeddings)
    X = np.random.randn(seq_len, d_model)
    
    # Weight matrices for Query, Key, Value
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)
    
    # Compute Q, K, V
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    print(f"Input X shape: {X.shape}")
    print(f"Query Q shape: {Q.shape}")
    print(f"Key K shape: {K.shape}")
    print(f"Value V shape: {V.shape}")
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_model)
    print(f"\\nAttention scores shape: {scores.shape}")
    print(f"Attention scores:\\n{scores}")
    
    # Apply softmax to get attention weights
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attention_weights = softmax(scores)
    print(f"\\nAttention weights:\\n{attention_weights}")
    print(f"Row sums (should be 1): {np.sum(attention_weights, axis=1)}")
    
    # Apply attention to values
    output = attention_weights @ V
    print(f"\\nOutput shape: {output.shape}")
    
    # 7.5: PyTorch tensor operations
    print("\\n7.5 PyTorch Tensor Operations")
    print("-" * 40)
    
    # Create PyTorch tensors
    x_torch = torch.randn(3, 4)
    y_torch = torch.randn(4, 5)
    
    print(f"PyTorch tensor x: {x_torch.shape}")
    print(f"PyTorch tensor y: {y_torch.shape}")
    
    # Matrix multiplication
    z_torch = torch.matmul(x_torch, y_torch)
    print(f"Matrix multiplication result: {z_torch.shape}")
    
    # Reshape operations
    reshaped = x_torch.view(-1)  # Flatten
    print(f"Flattened tensor: {reshaped.shape}")
    
    reshaped_2 = x_torch.view(2, 6)  # Reshape to 2×6
    print(f"Reshaped to 2×6: {reshaped_2.shape}")
    
    # Transpose
    transposed = x_torch.t()
    print(f"Transposed: {transposed.shape}")
    
    # Broadcasting in PyTorch
    broadcast_result = x_torch + torch.tensor([1, 2, 3, 4])
    print(f"Broadcasting result: {broadcast_result.shape}")
    
    # GPU operations (if available)
    if torch.cuda.is_available():
        x_gpu = x_torch.cuda()
        print(f"Tensor on GPU: {x_gpu.device}")
    else:
        print("CUDA not available, tensors remain on CPU")
    
    # Automatic differentiation
    x_grad = torch.randn(2, 3, requires_grad=True)
    y_grad = x_grad.sum()
    y_grad.backward()
    print(f"\\nGradient of sum: {x_grad.grad}")
    
    # 7.6: Practical neural network example
    print("\\n7.6 Practical Example: Mini CNN")
    print("-" * 40)
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    class MiniCNN(nn.Module):
        def __init__(self):
            super(MiniCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
            self.fc = nn.Linear(8 * 6 * 6, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x
    
    # Create model and sample input
    model = MiniCNN()
    sample_input = torch.randn(1, 1, 10, 10)  # Batch=1, Channels=1, Height=10, Width=10
    
    print(f"Input shape: {sample_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\\nTotal parameters: {total_params}")
    
    print("\\n" + "="*60)
    print("Exercise 7 Complete!")
    print("Key takeaways:")
    print("- Tensors are the fundamental data structure in deep learning")
    print("- Higher-dimensional tensors enable batch processing")
    print("- Convolution is a tensor operation for spatial feature extraction")
    print("- Attention mechanisms use tensor operations for sequence modeling")
    print("- PyTorch provides efficient tensor operations with GPU support")
    print("="*60)
    
    print("\\n" + "="*60)
    print("CONGRATULATIONS! You've completed all 7 exercises!")
    print("="*60)
    print("You now understand the key linear algebra concepts for ML:")
    print("✓ Vectors and vector operations")
    print("✓ Matrices and matrix multiplication")
    print("✓ Dot products and similarity")
    print("✓ Linear independence and spans")
    print("✓ Eigendecomposition and PCA")
    print("✓ Tensors and deep learning operations")
    print("\\nNext steps:")
    print("- Practice implementing these concepts in your own projects")
    print("- Explore how these operations work in transformer models")
    print("- Try building neural networks using these mathematical foundations")
    print("="*60)
"""

print("\n" + "="*60)
print("WELCOME TO LINEAR ALGEBRA FOR MACHINE LEARNING!")
print("="*60)
print("This interactive walkthrough contains 7 exercises covering:")
print("1. Vectors and Basic Operations")
print("2. Matrices and Matrix Operations") 
print("3. Matrix-Vector Multiplication & Neural Networks")
print("4. Dot Products and Similarity")
print("5. Linear Dependence and Spans")
print("6. Eigenvalues and Eigenvectors")
print("7. Tensors and Advanced Applications")
print("\nInstructions:")
print("- Exercise 1 is ready to run!")
print("- After completing each exercise, uncomment the next one")
print("- Each exercise builds on previous concepts")
print("- Complete the TODO items for hands-on practice")
print("="*60)