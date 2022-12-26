from abc import ABC, abstractmethod
from typing import List, Union


class BaseFeaturizer(ABC):
    @abstractmethod
    def embed(self, smiles: Union[str, List[str]]):
        pass
