import os, sys, time
import numpy as np
from itertools import combinations
import networkx as nx
import random
import sys, os
import pathlib
from pathlib import Path
import argparse

from pop2 import color, coloring_preprocessing_fast

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="partition_100_7/test/GP_linear_hash_100,100,100_7")
    parser.add_argument('--total_no', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(seed=args.seed)
    for i in range(args.total_no):
        data_path = Path(__file__).parent.parent / "data"
        data_path_no = data_path / (args.path + "_{}.gpickle".format(i))
        g = nx.read_gpickle(data_path_no)
        g, _ = coloring_preprocessing_fast(g)
        color(g)

    