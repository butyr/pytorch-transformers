# pytorch-transformers

This repository aims at providing the main variations of the transformer model in PyTorch. 
Currently it includes the initial model based on "Attention Is All You Need" 
([Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)) and the OpenAI GPT2 model based on 
[Radford et al. 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
and [Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).


## Installation

Install via pip:

```
pip install git+https://github.com/butyr/pytorch-transformer.git
```

## Usage

```python
from transformer import Transformer


model = Transformer(
        vocab_size=25_000,
        model_dim=512,
        hidden_dim=2048,
        nheads=8,
        max_len=512,
        depth=6,
    )

src_batch = # Tensor with shape (batch_size, src_sentence_length)
tgt_batch = # Tensor with shape (batch_size, tgt_sentence_length)

outputs = model(src_batch, tgt_batch)

```
