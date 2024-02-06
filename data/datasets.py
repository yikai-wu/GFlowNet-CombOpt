from dataclasses import dataclass,field
from typing import List

@dataclass
class random_graph_dataset:
    dataset: str =  "random_graph_dataset"
    graph_type: str="reg"
    n: int=100
    d: int=3
    p: float=0.1
    random_seed: int=42
    num_graphs: int=10

@dataclass
class rb_graph_dataset:
    dataset: str =  "rb_graph_dataset"
    graph_type: str="small"
    random_seed: int=42
    num_graphs: int=10

@dataclass
class partition_graph_dataset:
    dataset: str = "partition_graph_dataset"
    edge_generation: str = "linear_hash"
    node_list: str="500,500,500"
    random_seed: int=42
    num_graphs: int=10
    m: int=7
    p: float=0.2