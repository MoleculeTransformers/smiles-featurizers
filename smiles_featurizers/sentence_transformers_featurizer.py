from typing import List, Union
from smiles_featurizers import BaseFeaturizer
from sentence_transformers import SentenceTransformer
from torch import Tensor
from numpy import ndarray


class SentenceTransformersFeaturizer(BaseFeaturizer):
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def embed(
        self, smiles: Union[str, List[str]], batch_size: int = 8
    ) -> Union[List[Tensor], ndarray, Tensor]:
        assert len(smiles) > 0, "SMILES can not be empty!"
        embeddings = self.model.encode(
            smiles, batch_size=batch_size, device=self.device
        )
        return embeddings
