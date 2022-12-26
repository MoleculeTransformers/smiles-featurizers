from typing import List, Union
from smiles_featurizers import BaseFeaturizer
from transformers import BartModel, BartTokenizer
from torch import Tensor
from numpy import ndarray


class BartFeaturizer(BaseFeaturizer):
    def __init__(self, model_name_or_path: str):
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        self.model = BartModel.from_pretrained(model_name_or_path)

    def embed(
        self, smiles: Union[str, List[str]], embedder: str = "encoder"
    ) -> Union[List[Tensor], ndarray, Tensor]:
        assert len(smiles) > 0, "SMILES can not be empty!"
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        if embedder == "encoder":
            embeddings = outputs.encoder_last_hidden_state.mean(dim=1)
        elif embedder == "decoder":
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            raise NotImplementedError

        return embeddings
