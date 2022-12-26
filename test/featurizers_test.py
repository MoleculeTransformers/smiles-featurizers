import pytest
from smiles_featurizers import (
    SimcseFeaturizer,
    BertFeaturizer,
    BartFeaturizer,
    SentenceTransformersFeaturizer,
)


def test_simcse_featurizer():
    featurizer = SimcseFeaturizer("shahrukhx01/muv2x-simcse-smole-bert")
    embeddings = featurizer.embed(["CCC(C)(C)Br"])
    assert embeddings.shape == (1, 512)


def test_bert_featurizer():
    featurizer = BertFeaturizer("shahrukhx01/smole-bert")
    embeddings = featurizer.embed(["CCC(C)(C)Br"])
    assert embeddings.shape == (1, 512)


def test_bart_featurizer():
    featurizer = BartFeaturizer("shahrukhx01/smole-bart")
    embeddings = featurizer.embed(["CCC(C)(C)Br"])
    assert embeddings.shape == (1, 256)


def test_sentence_transformer_featurizer():
    featurizer = SentenceTransformersFeaturizer("shahrukhx01/siamese-smole-bert-muv-1x")
    embeddings = featurizer.embed(["CCC(C)(C)Br"])
    assert embeddings.shape == (1, 512)
