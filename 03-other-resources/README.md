### Other Resources

https://www.kaggle.com/code/auxeno/transformers-from-scratch-dl?scriptVersionId=137372275

- Before Transformers, you did LSTM (RNN), for processing **sequential data**.
- LSTMs forget on long sequences + training is sequential, not parallelizable
- Transformers solve this (training scales), attention pattern understands different portions
- Encoder <> Decoder, 1 learns to represent the inputs (vectors), 2 translates encoded data into output
  - Encoder Only = inputs into rich numerical implementation (embeddings), attend to tokens at n-i, and n+i, full ctx understanding in both directions. BERT. Vector models, e.g. `ada` are an example.
  - Decoder Only = Next Token Prediction, attends previous tokens, (masking), text generation, gpt
  - Encoder + Decoder = 1) generates embeddings, 2) autoregressively outputs tokens. Translation, Summarization.


### Encoder

![encoder](encoder.png)

- Input embedding: input is tokenized (each token/word is converted into vectors)
- Positional encoding: provide TF with position of each token (in LSTM's this information is known by default, not in TF)
- Multi-Head self-attention: each token looks at all other tokens to improve it's context understanding moving the vector to a different space, multi-head, means each head learns different relationships, syntax/semantics/verbs. In decoders, it'd only look backward words.
  - During training it learns where to look for information (attention pattern ~ grid K*Q*V)
- Add & Norm / Residual: add the original input tokens to the Attention layer output, helps preventing vanishing gradient problem.
- Feed forward: Learns patterns from attention layer, fluffy and cat are adjective/noun.
- Output Linear+softmax (it depends):
  - Wouldn't be needed for embedding models, as we already have the semantic understanding.
  - If the task is sentiment analysis, probabilities depend on # of options (positive/negative)
  - Linear layer: translates encoder understanding into the expected output
  - Softmax: simply converts into a weighted sum.