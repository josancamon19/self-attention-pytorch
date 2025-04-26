from datetime import date
import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import os
from io import open


class PositionalEncoding(nn.Module):
    """
    Gives the model a sense of *order*.

    Embeddings alone only tell the network **what** word it is looking at, not
    **where** it is in the sentence.  PositionalEncoding produces a vector
    “stamp” for every position (1st word, 2nd word, …) and simply *adds* that
    stamp to the normal token embedding.  Because the stamp is made from smooth
    sine/cosine waves of many different frequencies, every position ends up
    with a unique pattern that the model can later decode.

    Parameters
    ----------
    d_model : int
        Size of each embedding vector (must match the embedding layer so we can
        add them together).
    dropout : float, default 0.1
        Dropout applied right after the positional information is added.
    max_len : int, default 5000
        Longest sequence length we expect to feed the model.

    Example
    -------
    >>> pos_enc = PositionalEncoding(d_model=512)
    >>> out = pos_enc(token_embeddings)  # same shape as input, but now position-aware
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a (max_len x d_model) matrix with positional “wave” values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Waves of different frequencies: low freq in the first dims, high freq later
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices:  sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  indices:  cos

        # Shape becomes (max_len, 1, d_model) so we can broadcast over batches
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer → saved with the model but not a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional information to the token embeddings.

        Parameters
        ----------
        x : Tensor
            Shape ``[seq_len, batch_size, d_model]``.

        Returns
        -------
        Tensor
            Same shape as the input, but with positional data mixed in.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Transformer):
    def __init__(
        self,
        ntoken: int,
        ninp: int,
        nhead: int,
        nhid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        """
        ntoken : how many different words/characters the model knows (vocabulary size)
        ninp   : length of the vector that represents each token (embedding size)
        nhead  : how many parallel attention 'views' to look at the data (attention heads)
        nhid   : size of the hidden feed-forward layer inside every transformer block
        nlayers: how many transformer blocks to stack on top of each other
        dropout: random % of neurons to turn off while training to avoid over-fitting
        """
        super().__init__(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            num_encoder_layers=nlayers,
        )

        # will hold the causal mask that hides “future” tokens
        self.src_mask = None

        # adds information about token order
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # looks up an `ninp`-dimensional vector for every token id
        self.input_emb = nn.Embedding(ntoken, ninp)

        self.ninp = ninp  # keep around for scaling

        # turns encoder output back into vocabulary logits
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int):
        # lower-triangular 0/1 matrix → log() makes upper triangle ‑inf
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self):
        # small random weights & zero bias to start
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask: bool = True):
        # build (or clear) the causal mask
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                self.src_mask = self._generate_square_subsequent_mask(len(src)).to(
                    device
                )
        else:
            self.src_mask = None

        # token ids → embeddings, then scale
        src = self.input_emb(src) * math.sqrt(self.ninp)
        # add positional encodings
        src = self.pos_encoder(src)  # WHY?
        # run through the transformer encoder
        output = self.encoder(src, mask=self.src_mask)
        # project back to vocabulary size and return log-probs
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


### ------- TRAINING --------- ###


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        print(f"Tokenizing {path}")
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


# Set the random seed manually for reproducibility.
torch.manual_seed(42)
device = torch.device("mps")  # cpu, cuda?
corpus = Corpus("./wikitext-2")


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, 20)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

criterion = nn.NLLLoss()
bptt = 35  # sequence length
lr = 20
log_interval = 200
clip = 0.25
dry_run = True
epochs = 40


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


def train(model, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // bptt,
                    lr,
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()
        if dry_run:
            break


def evaluate(model, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def run():
    global lr  # Make sure we are modifying the global learning-rate variable

    model = TransformerModel(ntokens, 200, 2, 200, 2, 0.2).to(device)
    # Loop over epochs.
    best_val_loss = None

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train(model, epoch)

            # Evaluate on the validation set
            val_loss = evaluate(model, val_data)

            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open("model.pt", "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open("model.pt", "rb") as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model, test_data)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)
