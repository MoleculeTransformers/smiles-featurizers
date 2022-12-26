from typing import List, Union
from smiles_featurizers import BaseFeaturizer
from simcse import SimCSE
from torch import Tensor
from numpy import ndarray


class SimcseFeaturizer(BaseFeaturizer):
    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.device = device
        self.model = SimCSE(model_name_or_path, device=device)

    def embed(
        self, smiles: Union[str, List[str]], batch_size: int = 8, max_length: int = 512
    ) -> Union[List[Tensor], ndarray, Tensor]:
        assert len(smiles) > 0, "SMILES can not be empty!"
        embeddings = self.model.encode(
            smiles, batch_size=batch_size, max_length=max_length, device=self.device
        )
        return embeddings
