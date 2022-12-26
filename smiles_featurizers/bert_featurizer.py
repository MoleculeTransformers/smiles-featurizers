from typing import List, Union
from smiles_featurizers import BaseFeaturizer
from farm.infer import Inferencer
from torch import Tensor
from numpy import ndarray


class BertFeaturizer(BaseFeaturizer):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        max_length: int = 512,
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        use_gpu=False,
    ):
        self.model = Inferencer.load(
            model_name_or_path=model_name_or_path,
            task_type="embeddings",
            extraction_strategy=pooling_strategy,
            extraction_layer=emb_extraction_layer,
            gpu=use_gpu,
            batch_size=batch_size,
            max_seq_len=max_length,
            num_processes=0,
        )

    def embed(
        self, smiles: Union[str, List[str]]
    ) -> Union[List[Tensor], ndarray, Tensor]:
        assert len(smiles) > 0, "SMILES can not be empty!"

        if isinstance(smiles, str):
            smiles = [smiles]

        vectors = self.model.inference_from_dicts(
            dicts=[{"text": smile} for smile in smiles]
        )
        embeddings = Tensor([(vector["vec"]) for vector in vectors])
        return embeddings
