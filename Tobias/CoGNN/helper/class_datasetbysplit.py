from typing import NamedTuple, List, Union
from torch_geometric.data import Data

class DatasetBySplit(NamedTuple):
    train: Union[Data, List[Data]]
    val: Union[Data, List[Data]]
    test: Union[Data, List[Data]]