# Mastering Transformers

> Mastering Transformers in 60 days, and becoming an AI Researcher.

[Goal and resources](https://docs.google.com/document/d/1W6N9xQ3Giz7lK243EkP5GBJQu9cSSzHr-P5A7oRb7Eo/edit?usp=sharing).

Each word here was thought, not generated.


### Table of Contents

- [Mastering Transformers](#mastering-transformers)
    - [Table of Contents](#table-of-contents)
    - [Transformers](#transformers)
      - [Encoder/Decoders](#encoderdecoders)
      - [Encoder/Decoder in Transformers](#encoderdecoder-in-transformers)
      - [Sequential Data Before](#sequential-data-before)
        - [LSTMs](#lstms)
        - [Seq2Seq](#seq2seq)
      - [What is Attention?](#what-is-attention)
        - [LSTMs Attention](#lstms-attention)
        - [Transformers Attention](#transformers-attention)
  - [Encoder Architecture](#encoder-architecture)
      - [Each Section Briefly Explained](#each-section-briefly-explained)
    - [Attention Details](#attention-details)
      - [Basics of Q,K,V Matrices](#basics-of-qkv-matrices)
      - [Formula Part 1](#formula-part-1)
      - [Formula Part 2](#formula-part-2)
    - [Residual Layers (Add \& Norm)](#residual-layers-add--norm)
      - [Add](#add)
      - [Norm](#norm)
    - [Position Wise FFN](#position-wise-ffn)
    - [FAQ](#faq)
  - [Decoder Architecture](#decoder-architecture)
  - [Positional Encoding \& Embeddings](#positional-encoding--embeddings)
  - [Training](#training)
  - [Inference](#inference)
  - [Interpretability and Visualizing](#interpretability-and-visualizing)
  - [Arguing Architectural Decisions](#arguing-architectural-decisions)
  - [Tasks](#tasks)


### Transformers
- A neural network architecture made for the processing sequential data. Much more than sequential now.
- Novel idea: a new architecture making an `attention mechanism` its main component.
<br>

#### Encoder/Decoders
- A neural networks design pattern, each briefly:
- Encoder: "Understand and compress"
- Decoder: "Generate and expand"
- Encoder-Decoder: "Transform from one domain/format to another"

<br>

#### Encoder/Decoder in Transformers
- `Decoder` is the one that outputs tokens, like GPT models. Token n pays attention to only previous tokens, so it learns what to generate then.
- `Encoder` is used for classification tasks where we input a sequence, and try to classify that input as x,y,z options. Also when we want to find a meaningful representation of the whole input sequence.
- `Encoder <> Decoder` = `Encoder` generates part of attention and get's a deep representation of the input, `Decoder` makes another part of attention, and uses it to map it to it's own learned representation and expected output. Used in Translation ("Hi", "Hola"), Summarization, Speech to text, Multimodality. `Attention` here is called `cross-attention`.
<br>

#### Sequential Data Before
##### LSTMs
- Processing of tokens it's sequential, n, n+1, n+2... and for processing n+2, n+1 has to be computed, and so on.
- Token n=10, has no direct access to token n=5, instead has access only to n=9 as a representation of everything before itself+itself, whereas in [attention](#what-is-attention), tokens can interact far apart individually directly, not just with n-1 as the result of everything before.
- This sequential nature, means information is lost in longer context, cause let's say at n=1000, it will have no relevance from n=0, and sometimes it should.
- This also means we have vanishing gradients, which makes earlier layers of the LSTM's to not be able to learn as much.
- Further expand: RNN, Encoder<>Decoder interactions. Encoder only, bidirectional

##### Seq2Seq

<br>


#### What is Attention?
It's a mechanism which neural networks use to **learn to focus** on the relevant parts of the input.
- `Attention` means computing similarities between input sequences, to figure out things token i has to attend, and things token !=i can provide where they are being attended. This for every token in the input sequence.
- What does it mean for a token to pay attention to other tokens?
  - It means the token understood how other tokens in the input sequence change its meaning, and updates itself accordingly.
  - analogy: the isolated understanding of a word given a text it's completely different as a part of the text.
- With this tokens updates, the model learns what things in the input to pay attention to.


##### LSTMs Attention
- Initially implemented on Machine Translation tasks, which means over an LSTM with Encoder<>Decoder architecture. https://arxiv.org/pdf/1409.0473
- So what it used is known as `cross-attention`, which means connecting Encoder<>Decoder through an attention layer. Encoder computes a part, Decoder uses it to compute the other part. This was not a replacement on the way LSTM interacts, but an extra computation between Encoder<>Decoder, definitely improved translation quality.
- The issue here is that due to sequentiallity, encoder/decoder are still computing sequentially, so a few things are fixed on top of LSTMs (mainly quality, due to tokens far away not being able to interact) but they are still **not scalable**.
- This doesn't let the architecture scale, we can have more GPU's, data, batch_size, layers, neurons, but the time of training a big model is gigantic.
- **Self-attention**: Encoder or Decoder with an attention computation inside between each alone was tested in LSTMs before Transformers as well. But again this is on top of LSTMs, not replacing, so quality increased, but **sequentiality remained**, as well as **scale**.


##### Transformers Attention
- **Self attention** as key in the architecture itself, the Encoder/Decoder themselves are a bunch of self-attention layers.
- When **Encoder<>Decoder**, cross attention is the same thing as in LSTMs.
- This `self-attention` layers at Encoder/Decoder have multiple heads, this just means multiple stacked sublayers of attention being computed in parallel, each one with different weights, which means each one learns different patterns in the data.

> [RWKV](https://wiki.rwkv.com/) is fixing LSTM's sequentiallity.

<br>

**New Things we can do with Transformers**
Is there something that was technically impossible to do before `Transformers` like the simplest/non-prod MVP was not doable?
- **No**,
- e.g. Text Generation, Multimodal AI, translation, was already possible.
- Transformers just made them to work really really well.
- Also, most capabilities on Transformers, emerge from Scale:
  - e.g. `In-context learning`, as the model pays attention in parallel to every other token, you can feed it examples, and it can do something it has never seen before in it's training data.


<br>
<br>

## Encoder Architecture

![encoder](images/encoder.png)

#### Each Section Briefly Explained

- **Input embedding:** input is tokenized (int) (each token/word is mapped to an int), then each token is expanded to an `embedding` vector that later learns to represent the token in more detail.
- **Positional encoding:** provide the attention layer with position of each token (in LSTM's this information is known by default cause the proces is sequential, not in Transformers attention), is not because row order can't be traced throughout matrix operations, but attention is computed in parallel and only from embeddings, we need to insert the position in the embedding, so attention can understand, noun comes before subject, or things like that. This is done via sin/cos waves patterns.
  > Self-attention itself is permutation-invariant; "sequence" behavior appears only after you add a positional encoding. Swap the encoding for e.g 2D grid for images, and the block serves other data types.
- **Multi-Head self-attention:** Each token looks at all other tokens to improve its context understanding moving the embedding vector to a different space with richer meaning, `multi-head`, means is done by multiple (smaller) matrices/heads, each one focusing on  different relationships, syntax/semantics/verbs. 
- **Add & Norm / Residual:** add the original input tokens to the Attention layer output, helps preventing `vanishing gradient` problem + normalization.
- **Feed forward pos wise:** Learns complex patterns out of Attention Layer, each token is processed individually, which means position is kept. This layer learns deeper individual tokens meanings. why? non-linearity the magic of Neural networks. Attention is linear operations, so in the semantic space it will be able to separate `animal | color`, this `FFN` understands `animal analogy` | `animal idiom` | `real animal`.
- Output **Linear+softmax:** You use your network with language understanding layers to apply a basic nn to classify inputs. e.g. Sentiment analysis. Output neurons is a prob distribution `softmax` between "positive" "negative" "neutral".

<br>

<br>

### Attention Details

> This is the most relevant piece.

$
\text{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$


**Prerequisites:**
- [Algebra](algebra.ipynb).
  - **Key higlight:** Matrices can represent 2 different things during Attention.
  - Sometimes e.g. `W_Q`, matrices in here are meant to represent a transformation, `X @ W_Q`, taking input X and transforming it, is direction/magnitude but in this case a complete different coordinate system as well `W_Q`.
  - Other times, e.g. `Q @ K^T`, Q,K represent a list of vectors, not transformations, and we are taking those vectors independently to compute similarities with each other (dot product).
- Statistics `notebook to be added soon`

<br>

#### Basics of Q,K,V Matrices
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What can I be matched against?"
- **Value (V):** "What actual information do I contain?"

`Q,K,V = X @ W_Q, X @ W_K, X @ W_V`.
`n @ m` in this case, means n projecting into m, or transforming n through m.
Prefix `W`, refers to weights (trainable parameters).

**X:**
- Refers to input sequence.
- Matrix with each row being the embedding vector of token i.

**Q:**
- `W_Q` (Query weights) get trained to create queries that find relevant information. So it develops like query templates.
- `X` is being projected and transformed into `W_Q`, which outputs `Q`. This `X @ W_Q` is basically using `W_Q` learned ability to ask questions, to make it specific to the input vector, so `Q`, is the specific query each token/vector should look for.
- `W_Q` = "what do I need to pay attention to". `Q` says, as `Q_$word` "I need to pay attention to: determiners, attributes, and predicates"

**K**: `W_K`, identity templates, `X @ W_K` projection is `K`, which uses `W_K` learned ability to determine searchable params/features to make it specific to `X`. `k_$word` then says, this are the things I can be matched against / searched for, which based on the ex above, it could be, "I can be searched when trying to find nouns, adjectives, connectors", etc.

**V**: "As `V_$word`, this is the information I'll contribute when someone pays attention to me". e.g. `W_V` learns to extract the most useful information from each token - semantic meaning, grammatical role, contextual features, etc.

<br>

#### Formula Part 1

$(Q K^\top)$    - `Q @ K^T`

- Here we already projected `X` into `W_Q`, `W_K`, giving us `Q`, `K`, which `Q` tells us for each token in `X`, this are the things I'm looking to pay attention to, and `K`  fo each token in `X`, this are the I can be searched for.
- `Q, K, V` matrices represent a list of vectors each, not transformations.
- `K^T` Transpose, just to match dimensions for dot product to work.
- `Q @ K.T`, computes similarity scores on queries/keys pairs.
- **Result:** A matrix known as `Attention Scores` where entry (i,j) = "How much should token i attend to token j?" 
  - **why/how?** because we used queries (things that eac token should look for), keys (identify things each token can be searched for), we know now based on similarity scores between things to search / things to be searched for, what things it should pay attention to.


During training the model discovers that:
- Nouns should attend to their modifiers
- Adjectives should attend to what they modify
- Verbs should attend to their arguments


**Walkthrough:**
```python
_input = "The car is red"

# Q_the = "As 'the', I need to find: the noun I'm modifying"
# Q_car = "As 'car', I need to find: my attributes, determiners, and related predicates" 

# K_the = "I can be matched as: determiner/article pattern"
# K_car = "I can be matched as: noun/subject/entity pattern"

# (used later)
# V_the = "I provide: definiteness, specificity signals"
# V_car = "I provide: vehicle semantics, subject-entity information"

similarity(Q_car, K_the) = HIGH   # car needs its determiner
similarity(Q_car, K_is)  = HIGH   # car needs its predicate  
similarity(Q_car, K_red) = HIGH   # car needs its attribute
similarity(Q_car, K_car) = LOW    # most tokens have low attention to themselves


attention_scores = Q @ K^T  # Shape: (4, 4)

attention_scores = [
  [0.1, 0.8, 0.3, 0.2],  # How much "The" wants to attend to [The, car, is, red]
  [0.2, 0.9, 0.7, 0.6],  # How much "car" wants to attend to [The, car, is, red]  
  [0.1, 0.8, 0.4, 0.5],  # How much "is" wants to attend to [The, car, is, red]
  [0.3, 0.9, 0.2, 0.1]   # How much "red" wants to attend to [The, car, is, red]
]
```


<br>


#### Formula Part 2

$
\mathrm{softmax}\left(\frac{Part 1}{\sqrt{d_k}}\right) V
$

<br>

2.1. **What the /√dk means?**
- Instead of the normal `dot_product`, we call this `scaled_dot_product`.
- Check [algebra.ipynb](algebra.ipynb) `Similarity operations` section.
- So `dk` is simply the size of the key vector, which in Attention is `dk = embedding_dim // num_heads`. check more detail about attention heads in the FAQ below.
- Why? check the [algebra.ipynb](algebra.ipynb) notebook, tldr, dot product of tons of dimensions, causes large similarity scores with high variance (which is just saying, range of distances from the avg is very large)
- [ ] Need a proof on why/how /d_k helps reduce variance, and why d_k is chosen.

<br>
  

2.2 **Softmax**
Turns a bunch of values into a % that sum up to 1 (weighted sum).
[Pytorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

Once we have `Part1/√dk`, we apply a `softmax` to get a weighted sum of % of attention each token should pay to each other, this are the `attention_weights`.

`attention_weights` **meaning:** as token "The", "I want 40% of *car* info, 25% of *red* info, 35% of *is* info.

- [ ] Include images of pattern generated by attention_viz_exercise.py 

<br>

2.3. **attention_weights @ V**
- Using `attention_weights` (knowing how much attention to get from every other token), retrieves that information from `V`.
  - `V` matrix provides: The actual information each token provides.
  - Again we are computing similarities, Matrices here are meant to represent a list of vectors, not transformations.
  - **Output:** The input tokens but after paying attention to every other relevant token in the input sequence, which moved them around the embedding space to have a much richer contextual representation.

<br>

**Walkthrough:**
```python
attention_weights = softmax(attention_scores)  # Each row sums to 1.0

attention_weights = [
  [0.1, 0.5, 0.2, 0.2],  # "The" attention distribution  
  [0.1, 0.4, 0.3, 0.2],  # "car" attention distribution
  [0.1, 0.4, 0.2, 0.3],  # "is" attention distribution  
  [0.2, 0.6, 0.1, 0.1]   # "red" attention distribution
]

# skipping /square root of dk

# attention_weights @ V

V = [
  [0.1, 0.2, 0.3, 0.4],  # V_the: determiner information
  [1.0, 0.8, 0.5, 0.2],  # V_car: vehicle/noun information  
  [0.3, 0.9, 0.1, 0.3],  # V_is: linking-verb information
  [0.7, 0.2, 0.8, 1.0]   # V_red: color/adjective information
]

# softmax(Q * K^T) * V
```

> **The final result is:** we mutate/move each token around the semantic space based on the other tokens it needs to attend to, updating each token to a deeper meaning of its representation in the whole input sequence.

<br>
<br>

### Residual Layers (Add & Norm)
![residual](images/residual.png)
#### Add
- `MultiHeadAttention(pos_enc_input) + pos_enc_input`
- The attention layer outputs the input tokens but with its learned changes, having modified vector space, we take that and add the initial positional_encoded_input.
- why?
  - vanishing gradient problem on deep neural networks.
  - but also don't make huge changes right away (stabilization)
  - helps preserve information, original input is not completely changed.
- so it's like adding the new insights into the initial token embedding
- worst case safety: if a layer learns nothing, output as close to original
- adding vectors means combining information?
  - it depends, if they have same semantic space, if not related/diff dim, no.
  - The model learns to make outputs that work well with the layer

#### Norm
- `nn.LayerNorm`, layer($add_result)
- After Multi-Head Attention, each token could be:
  ```python
  token_1 = [0.001, 0.002, 0.003]    # Tiny numbers
  token_2 = [1000, 2000, 3000]       # Huge numbers  
  token_3 = [-500, 800, -200]        # Mixed scales
  ```
- Unstable training, huge gradients some, other tiny ones, activations functions to extremes, model doesn't find good learning rates.
- Allows to stack multiple layers reliably

<br>

### Position Wise FFN
![ffn_pos_wise](images/ffn_pos_wise.png)
- So attention gave us a new representation for each token, having moved its embedding space position and having paid attention to each other token correspondingly. We found patterns across tokens and put the information in it.
- Attention gave linear relationships between tokens, **but**, the magic of nn's is in their non-linearity, ffn give us this non linearity, which means more complex patterns.
- The ffn now takes those vectors, and understand what each token (after attention modified them) means further.
- what complex meanings emerge after the vector modifications.
  - finding idioms, physical structures, attention heads give us semantic understanding, ffn finds patterns that emerge from semanticness.

**So, wtf is Linearity**
- A function is linear if: f(ax + by) = af(x) + bf(y)
  - predictable proportional behavior, scale input by 2, output scales by 2, add 2 inputs.
- Linear functions can scale (stretch/shrink), rotate, project, combine (weighted sums), cannot create curves, make "if-then" decisions, separate complex patterns.
- Basically all our attention operations are linear, `@` and `sqrt`
- Why is `FFN` non linear?
  - example
    ```python
    def ffn(x):
      x = linear1(x)        # Linear: expand dimensions
      x = GELU(x)           # NON-LINEAR: the magic happens here!
      x = linear2(x)        # Linear: compress back
      return x
    ```
- some used non linear functions are `GELU`, `RELU`, `Tanh`, `Sigmoid`.
- ok but why a non-linear func (curves is clear), but decision boundaries, regions, complex patterns, how does this translates?
  - **I think I got it**
    - try to think in a 512 dim space, it's hard, but try. In here geometric regions correspond to semantic spaces. Correspond to if/then decisions.
  - so if you make a line separation, there's just so many rules you can make, color | animals, but not deeper at idioms, or real reference to animals.
  - a subtle change in the inputs will have a much different representation, non-linear \:)
- `x^2`, or `e^x` are those non linear? yes, [use logic above to prove it] why don't we use them then?
  - `x^2` is always postive, no negative info, gradient issues, `dx/dd=2x`, at `x=0`, gradient=0, dead neurons, large x explodes, has no off states, only 0.
  - `e^x`, explodes, always positive, huge gradients
- but, we could in theory use them, and they could represent complex patterns, they are just not as good.

<br>
<br>

### FAQ
- **Single vs Multi Head Attention:** each head learns different patterns, and is the parallelization of training. Head 1: Learns syntax relationships (subject-verb), Head 2: Learns semantic relationships (adjective-noun) ... Head n.
  - During training each head focuses on smth different
  - Syntactic Heads, Semantic Heads, Positional Heads.
- **Self-Attention vs Cross Attention:**
  - Self: The sequence "attends to itself", each word attens all other words in input
  - Cross: Q comes from one sequence, K/V from another. Encoder <> Decoder architecture
    - The decoder (K/V) attends the encoder outputs (Q).
- How to determine `embedding_dim`, and `num_heads`?
  - **Embedding dim**
    - This is the big decision in the architecture
    - larger dim, more complex patterns, more expensive, slower training
    - input complexity (sentiment ~ 128-256, language modeling 512-1024, gpt-3/4 ~ 12288)
  - **num_heads** and **head_dim**
    - if embedding size = 512, num_heads=n, head_dim=512/n
    - more heads, more perspectives, but each head is simpler
    - fewer heads, less perspectives, but each head is richer
    - each head receives a *slice of each input token*.
- Is it actually slicing each embedding token?
  - Not really!!, `self.q = nn.Linear(embed_dim, head_dim)`, input dim is `embed_dim`, `W_Q` receives the whole thing, but we then are projecting `X @ W_Q` to get `Q`, and this one is actually of a smaller size, cause `head_dim` output is embed_dim/n_heads.
  - We take each token embedding and project it to W_Q, X @ W_Q, and that way we get Q, which has 1/8th dimensionality reduction!!
- Why not **having Q,K,V** of size `embed_dim*embed_dim`, instead of `embed_dim*1/n_heads`?
  - all heads would learn similar patterns, we want them to specialize contains x information, each head's projection learns different patterns of language
  - compute just grows absurdly, exponentially. (512*64, to 512*512)
  - It was tried is not better, MQA Google, GQA Llama 2
- What's the **loss function**, what's the source of truth?
  - `tokenizer` is the only source of truth we have, embeddings are params, attention Q,K,V and W_O are params, FFN are params.
  - **Encoder only** = we have labels, so is just a normal classifier output, prob of every option
    - What if we want to get a **sequence embeddings model**? (no classifier head)
    `MLM` and `NSP` (BERT), Contrastive Learning (compare 2 similar inputs, cosine similarity loss).
  - **Decoder only** = output log prob of the whole vocabulary (all possible tokens), how likely each token is to be the next one, we have the whole input sequence, so we can compare (self-supervised-learning).
  - **Encoder <> Decoder** = we have labels, supervised learning, for translation, summarization, so it's again log prob vs ground truth.
- **Why does this work at all?**
  - gradient flows through everything, all updates based on final loss
  - each component job emerges
    - Embeddings: Learn to represent tokens in a useful space
    - `W_Q`, `W_K`, `W_V`: Learn to find relevant relationships between tokens
    - FFN: Learn to transform and combine information
    - Classifier: Learn to map final representations to predictions
  - Scale (tokens, params, compute) + simple objective
  - Enough learning capacity (params), enough data, a clear objecive, gradient descent.
- How do we **merge multiple attention heads** outputs?
  - Example
    ```python
    head_1 = [0.2, 0.8, 0.1, 0.9]  # Maybe focuses on syntax
    head_2 = [0.7, 0.3, 0.6, 0.2]  # Maybe focuses on semantics  
    head_3 = [0.1, 0.5, 0.8, 0.4]  # Maybe focuses on position
    # concat = [head_1, head_2, head_3] # (1x12)
    # linear_layer_shape = 12x4
    # output of concat @ linear_layer = (1x4)
    # effectively combining the multiple heads outputs
    ```
  - concatenate, put outputs side by side, in a single vector.
  - add a linear layer that combines all, learns to take 30% from h1 (syntax), 50% h2 (semantics), and so on
  - Basically learns the optimal mixing rations depending on word/context, **mix the different types of attention learned**
- What do multiple **attention layers** do, how do they **stack** on each other?
  - 1 complete encoder block: (mha - residual - ffn - residual), gpt 3 has 96 blocks.
  - 1 attention layer, means the mha layer in the block.
  - how many to stack together?
    - empirical scalling laws, chinchilla scaling laws, common existing sizes
    - [ ] Further writing about this is required
    - bigger, more learnings, harder to train, if dataset is small, risk overfitting.
- **Dimensionality changes are confusing**:
  > This will not make it clearer, but it's an attempt. Maybe I'll understand it when I go through this later. 
  - Initializing variables
    - **batch_size**: inputs to be processed in paralle through the network (a single backpropagation turn is made for 1 batch). Let's assume is `1` for now.
    - **embedding_dimension**: key parameter of the architecture, let's set it to `128`.
    - **vocab_size:** tokenizer vocabulary size, lookup of input: integer. let's set it to `30000`
    - **max_length**: how many tokens can your model receive as input, bigger, more computation, let's set it to `100` (this also refers to input_sequence_length, we just pad/trim) 
    - **num_heads:** purpose of exercise. `8`.
  - input string sequence
  - **tokenizer** batch is converted to a list of integers of max_length `[1, 100]`
  - **Embedding** layer shape: `[30000, 128]` this is a lookup table, not a matmul, so output here is `[1, 100, 128]`, we just add the embedding dimension
  - **pos_encoding** layer shape: `[100, 1 (no related to batch), 128]`, this is just  modifying our token embeddings with a sum to give them a position wave signal. So output remains the same as above. `[1, 100, 128]`
  - **head_dim** = embedding_dim/num_heads = 128/8 = `16`
  - **Attention:**
    - `W_Q/W_K/W_V` shape: `[head_dim, embedding_dim]` `[16, 128]`
    - After taking `pos_encoding_output @ W_Q/W_K/W_V`, we get `Q,K,V`
      - Each operation being of shape: `[1, 100, 128] @ [16, 128].T`
    - So `Q,K,V` shape is: `[1, 100, 16]`
    - `K^T` = .transpose(1,2), switches e.g. from `[1, 100, 16]` to `[1, 16, 100]`
    - Q * K^T 
      - `[1, 100, 16]` * `[1, 16, 100]` = `[1, 100, 100]`
      - here we do `softmax` and get `attention_weights`
    - **attention_weights** `* V` shape is:  `[1, 100, 100]` * `[1, 100, 16]` = `[1, 100, 16]`
    - `W_O` = input is concat of each head output at `-1` dim = `[1, 100, 16+16+16+16+16+16+16+16]`
      - Final output is `[1, 100, 128]`
  - **residual layer** doens't change dimensions.
  - **FFN pos wise**: just a linear classifier, usually hidden layer has `*4` embedding_dim, then shrinks back to embedding_dim. Output is still `[1, 100, 128]`
  - **Final classifier** output: scales `*4` again, but output layer has the # of options to be classified, e.g. positive | negative, 2 options, making a probability distribution, you take the neuron with the highest score, that's your output.
  - Use `Transformer.get_dimensions()` to get the traces.

<br>

## Decoder Architecture
- Explain the differences at each step, in the diagram, in training, and so on.

<br>

## Positional Encoding & Embeddings

![positional](images/positional-encoding.png)

> so confusing

- [ ] A more in depth analysis of sin/cos + refresh on its meaning is needed
- [ ] I'm still not fully grasping this

<br>

Quoting my explanation above:
> Provide the attention layer with position of each token (in LSTM's this information is known by default cause the proces is sequential, not in Transformers attention). 
> 
> Is not because row order can't be traced through the operations, but attention is computed in parallel and only from embeddings data, we need to insert the position in the embedding, so attention can understand it, e.g. *noun comes before subject*, or things like that.



- **1st Q**: "wait, but can't the rows just be traced around in the multiple matrix operations, and just retrieve the index?
  - The answer is that the attention layer wasn't made with this in mind, each token embedding is processed in parallel, and only the embedding is taken, nothing else, so we gotta tell the layer through the embedding what's its position
  - Apparently was a simpler way of doing it, lol.
- **2nd Q**, "why at moment of processing each embedding in `Attention` we don't just, let the layer know about the index?" -> the answer is yes, modern architectures do this positional encoding differently, Tf-XL, T5, DeBERTa.
- **3rd Q**, "so how does a sin/cos wave tell the attention layer that this other token comes behind you, and this is what it means"
  - Part of the answer: 
    - Each position 0,1,...,n, get's aidfferent sin/cos pattern
    - Let's say n token is at idx 5, and m is idx 8, 3 positions apart would have a consistent math relationship as if they were index 0 and 3.
    - This would mean we are rather showing the model relative distances between tokens, so if abs `(token_n.idx8-token_m.idx11)` = `(token_n.idx5-token_m.idx8)`
- **4th Q:** Why is the embedding not broken? like, we randomly initialize it (learned params), and then we just add sin/cos waves and not breaking what it's learning. How can even the attention layer be able to separate what this mean at a position level.
- Isn't this changing any way on every training run? how's position kept.


<br>
<br>


## Training
- How are the transformer weights initialized? (optimal strategies)
- Key ideas on making training stable, 12+layers, not corrupted
- How does backpropagation work through each layer, specially the attention mechanism?
- What is the actual loss function and how does it drive learning?
- How does a single loss function coordinate learning across all these complex components?
- How do the weight matrices W_Q, W_K, W_V actually learn their "templates"?
- What drives them to specialize in query/key/value roles?
- How do they discover what patterns to look for?
- How do attention heads actually specialize during training?
- What forces different heads to learn different patterns?
- Can we predict or control what each head will learn?
- How does information actually flow through the transformer?
- What information is preserved vs. transformed at each layer?
- How do early vs. late layers differ in their function?
- What is the computational complexity and memory usage?
- Detailed analysis of O(n²) attention complexity
- Memory bottlenecks during training vs. inference


<br>

## Inference

<br>

<br>

## Interpretability and Visualizing
- Take a model, and view/interpret what each head is learning about language
- Tried inspectus and bertviz [_12_interpretability.py](_12_interpretability.py)
- What's Q "asking", K "providing", V "representing"?
- How's W_O mixing each head
- What complex patterns is FFN learning, can neurons be traced? 
- https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb

<br>

## Arguing Architectural Decisions
- Why do these specific architectural choices work so well?
- Why this particular combination of attention + FFN + residuals?
- What happens if you change the order of operations?
- How do you determine optimal model size for a given task?
- The relationship between data size, model parameters, and performance
- When do you get diminishing returns from scaling?

- What are the theoretical limitations of the attention mechanism?
- What types of patterns can/cannot be learned?
- Why does performance degrade with very long sequences?
- How does the transformer architecture relate to other computational models?
- Connections to database queries, memory systems, etc.
- What makes it fundamentally different from CNNs/RNNs?
<br>

## Tasks
- [ ] Dimensionality changes over the whole input output confuse me a lot.
- [ ] Transformer.params() tells how many params has at each layer.

<br>

- [ ] implement and reimplement
- [ ] Try different classifier heads
- [ ] how to train in different data, and make it bigger, require GPU, 100M params
- [ ] from scratch to a decoder only, from scratch as well + walk through each layer as above.
- [ ] implement again
- [ ] SFT on it