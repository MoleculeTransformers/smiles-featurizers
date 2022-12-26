# SMILES Featurizers

Extract SMILES embeddings from language models pre-trained with various objectives architectures.

## Getting Started

```bash
pip install smiles-featurizers
```

## Model List

Our released models are listed as following. You can import these models by using the `smiles-featurizers` package or using [HuggingFace's Transformers](https://github.com/huggingface/transformers).
| Model | Type |
|:-------------------------------|:--------:|
| [shahrukhx01/smole-bert](https://huggingface.co/shahrukhx01/smole-bert) | Bert|
| [shahrukhx01/smole-bert-mtr](https://huggingface.co/shahrukhx01/smole-bert-mtr) | Bert|
| [shahrukhx01/smole-bart](https://huggingface.co/shahrukhx01/smole-bart) | Bart|
| [shahrukhx01/muv2x-simcse-smole-bart](https://huggingface.co/shahrukhx01/muv2x-simcse-smole-bert) | Simcse|
| [shahrukhx01/siamese-smole-bert-muv-1x](https://huggingface.co/shahrukhx01/siamese-smole-bert-muv-1x) | SentenceTransformer|

## Use SMILES Featurizers

### Bert Featurizer

```python
from smiles_featurizers import BertFeaturizer
import torch

## set device
use_gpu = True if torch.cuda.is_available() else False

featurizer = BertFeaturizer("shahrukhx01/smole-bert", use_gpu=use_gpu)
embeddings = featurizer.embed(["CCC(C)(C)Br"])
```

### Bart (Encoder) Featurizer

```python
from smiles_featurizers import BartFeaturizer

featurizer = BartFeaturizer("shahrukhx01/smole-bart")
embeddings = featurizer.embed(["CCC(C)(C)Br"], embedder="encoder")
```

### Bart (Decoder) Featurizer

```python
from smiles_featurizers import BartFeaturizer

featurizer = BartFeaturizer("shahrukhx01/smole-bart")
embeddings = featurizer.embed(["CCC(C)(C)Br"], embedder="decoder")
```

### SimCSE Featurizer

```python
from smiles_featurizers import SimcseFeaturizer
import torch

## set device
device = "cuda" if torch.cuda.is_available() else "cpu"

featurizer = SimcseFeaturizer("shahrukhx01/muv2x-simcse-smole-bert", device=device)
embeddings = featurizer.embed(["CCC(C)(C)Br"])
```

### SentenceTransformer Featurizer

```python
from smiles_featurizers import SentenceTransformersFeaturizer
import torch

## set device
device = "cuda" if torch.cuda.is_available() else "cpu"

featurizer = SentenceTransformersFeaturizer("shahrukhx01/siamese-smole-bert-muv-1x", device=device)
embeddings = featurizer.embed(["CCC(C)(C)Br"])
```
