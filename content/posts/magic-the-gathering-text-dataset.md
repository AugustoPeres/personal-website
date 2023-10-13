+++
title = "Magic the Gathering text dataset"
author = ["Augusto"]
description = """
  Hosted through HuggingFace this dataset contains the names, types, and oracle texts of all magic the gathering cards up to 2022. It can easily be used for text generation tasks. Just use:
  
  ```python
from datasets import load_dataset

dataset = load_dataset('augustoperes/mtg_text')
  ```
  """
draft = false
tags = ["python", "machine learning", "text generation", "datasets"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++

# Dataset contents

This dataset contains the names, type lines and oracle texts of all
Magic the Gathering cards up to 2022. This consists of `20063` training
examples and `5016` validation examples.

# Dataset Usage

HuggingFace makes using this data super simple. First just make sure
that you have the HuggingFace's
[`datasets`](https://huggingface.co/docs/datasets/index) library
installed. To do that just type: `pip install datasets`. Once that is
done you can have immediate access to the data using:

  ```python
from datasets import load_dataset

dataset = load_dataset('augustoperes/mtg_text')
training_dataset = dataset['train']
validation_dataset = dataset['validation']

print(training_dataset[0])

# Outputs
# {'card_name': 'Recurring Insight',
#  'type_line': 'Sorcery',
#  'oracle_text': "Draw cards equal to the number of cards in target opponent's hand.\nRebound (If you cast this spell from your hand, exile it as it resolves. At the beginning of your next upkeep, you may cast this card from exile without paying its mana cost.)"}
  ```
  
# Extra

Suppose that you want to start training your pytorch models. The first
step is to get a tokenizer and tokenize the dataset:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(sample):
    sample["card_name"] = tokenizer(sample["card_name"])["input_ids"]
    sample["type_line"] = tokenizer(sample["type_line"])["input_ids"]
    sample["oracle_text"] = tokenizer(sample["oracle_text"])["input_ids"]
    return sample

tokenized_dataset = train_dataset.map(tokenize)
print(tokenized_dataset[0])

# Outputs
# {'card_name': [101, 10694, 12369, 102], 'type_line': [101, 2061, 19170, 2854, 102], 'oracle_text': [101, 4009, 5329, 5020, 2000, 1996, 2193, 1997, 5329, 1999, 4539, 7116, 1005, 1055, 2192, 1012, 27755, 1006, 2065, 2017, 3459, 2023, 6297, 2013, 2115, 2192, 1010, 8340, 2009, 2004, 2009, 10663, 2015, 1012, 2012, 1996, 2927, 1997, 2115, 2279, 2039, 20553, 2361, 1010, 2017, 2089, 3459, 2023, 4003, 2013, 8340, 2302, 7079, 2049, 24951, 3465, 1012, 1007, 102]}
```

Now, to use it with pytorch, you just need to create a
`dataloader`. However, because in a batch all the sequences must be of
the same length we need to add padding to the shorter
sequences. We can easily to this by using a custom `collate_fn` function:

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(sequences):
    # Pad the sequences to the maximum length in the batch
    card_names = [torch.tensor(sequence['card_name']) for sequence in sequences]
    type_line = [torch.tensor(sequence['type_line']) for sequence in sequences]
    oracle_text = [torch.tensor(sequence['oracle_text']) for sequence in sequences]

    padded_card_name = pad_sequence(card_names, batch_first=True, padding_value=0)
    padded_type_line = pad_sequence(type_line, batch_first=True, padding_value=0)
    padded_oracle_text = pad_sequence(oracle_text, batch_first=True, padding_value=0)
    
    return {'card_name': padded_card_name, 'type_line': padded_type_line, 'padded_oracle_text': padded_oracle_text}

loader = torch.utils.data.DataLoader(tokenized_dataset, collate_fn=collate_fn, batch_size=4)

for e in loader:
    print(e)
    break

# Outputs:
# {'card_name': tensor([[ 101, 10694, 12369, 102, 0],
#                       [ 101, 3704, 9881, 102, 0],
#                       [ 101, 22639, 20066, 7347, 102],
#                       [ 101, 25697, 1997, 6019, 102]]),
# 'type_line': tensor([[ 101, 2061, 19170, 2854, 102, 0, 0],
#                      [ 101, 6492, 1517, 4743, 102, 0, 0],
#                      [ 101, 6492, 1517, 22639, 102, 0, 0],
#                      [ 101, 4372, 14856, 21181, 1517, 15240, 102]]),
# 'padded_oracle_text': [omitted for readability])}
```

The dataloader is prepared to be given to your training functions.
