import os
import sys
import pickle
import random
import numpy as np
import shutil
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import multiprocessing
import networkx as nx


"""
python rbgraph_generator.py --num_graph 4000 --graph_type small --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --graph_type small --save_dir rb200-300/test  
"""

def random_partition_graph(node_num_list, p, permutation=False):
    # generate the random graph with len(node_num_list) partitions
    # return the adjacency matrix
    total_nodes = np.sum(np.array(node_num_list))
    adj_mat = np.random.binomial(1, np.sqrt(p), size=(total_nodes, total_nodes))
    adj_mat = adj_mat * adj_mat.T # make it symmetrical
    borders = [0]
    for num in node_num_list:
        borders.append(borders[-1]+num)
    for i in range(len(node_num_list)):
        s = borders[i]
        e = borders[i+1]
        adj_mat[s:e, s:e] = np.zeros((node_num_list[i], node_num_list[i]))
    return adj_mat

def random_partition_graph_diffp(node_num_list, p, permutation=False):
    # generate the random graph with len(node_num_list) partitions
    # return the adjacency matrix
    a, b = p.shape
    assert a == b
    assert a == len(node_num_list)
    total_nodes = np.sum(np.array(node_num_list))
    adj_mat = np.zeros((total_nodes, total_nodes))
    borders = [0]
    for num in node_num_list:
        borders.append(borders[-1]+num)
    for i in range(len(node_num_list)):
        si = borders[i]
        ei = borders[i+1]
        for j in range(i+1, len(node_num_list)):
            sj = borders[j]
            ej = borders[j+1]
            adj_mat[si:ei, sj:ej] = np.random.binomial(1, p[i,j], size=(node_num_list[i], node_num_list[j]))
            adj_mat[sj:ej, si:ei] = adj_mat[si:ei, sj:ej].T
    return adj_mat

def partition_graph_linearhash(node_num_list, m):
    # m = 1/p is the modular
    def simple_hash(x, m, a=587, b=1787):
        return (a * x + b) % m
    total_nodes = np.sum(np.array(node_num_list))
    adj_mat = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(i+1, total_nodes):
            if_edge = (simple_hash(i * total_nodes + j, m) == 0)
            if if_edge:
                adj_mat[i,j] = 1
                adj_mat[j,i] = 1
    borders = [0]
    for num in node_num_list:
        borders.append(borders[-1]+num)
    for i in range(len(node_num_list)):
        s = borders[i]
        e = borders[i+1]
        adj_mat[s:e, s:e] = np.zeros((node_num_list[i], node_num_list[i]))
    return adj_mat

def generate_partition_graph_dataset(data_config):
    adj_list = []
    # set random seed
    np.random.seed(data_config.random_seed)
    node_list_str = data_config.node_list.split(',')
    node_list = [int(n) for n in node_list_str]
    if data_config.edge_generation == "linear_hash":
        for i in range(data_config.num_graphs):
            adj_list.append(partition_graph_linearhash(node_list, data_config.m))
    elif data_config.edge_generation == "prob":
        for i in range(data_config.num_graphs):
            adj_list.append(random_partition_graph(node_list, data_config.p, permutation=False))
    elif data_config.edge_generatation == "prob_mat":
        pass
    else:
        raise NotImplementedError
    graph_list = [nx.from_numpy_matrix(A) for A in adj_list]
    return graph_list

def get_partition_instance(graph_type, node_list, m, p):
    node_list_str = node_list.split(',')
    node_lst = [int(n) for n in node_list_str]
    if graph_type == "linear_hash":
        A = partition_graph_linearhash(node_lst, m)
    elif graph_type == "prob":
        A = random_partition_graph(node_lst, p, permutation=False)
    elif graph_type == "prob_mat":
        pass
    else:
        raise NotImplementedError
    G = nx.from_numpy_matrix(A)
    return G

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graph', type=int, default=10)
    parser.add_argument('--graph_type', type=str, default="linear_hash")
    parser.add_argument('--node_list', type=str, default="100,100,100")
    parser.add_argument('-m', type=int, default=7)
    parser.add_argument('-p', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    if not os.path.isdir("{}".format(args.save_dir)):
        os.makedirs("{}".format(args.save_dir))
    print("Final Output: {}".format(args.save_dir))
    print("Generating graphs...")

    for num_g in tqdm(range(args.num_graph)):
        path = Path(f'{args.save_dir}')
        if args.graph_type == "linear_hash":
            stub = f"GP_{args.graph_type}_{args.node_list}_{args.m}_{num_g}"
        elif args.graph_type == "prob":
            stub = f"GP_{args.graph_type}_{args.node_list}_{args.p}_{num_g}"
        else:
            raise NotImplementedError
        g = get_partition_instance(args.graph_type, args.node_list, args.m, args.p)
        g.remove_nodes_from(list(nx.isolates(g)))
        output_file = path / (f"{stub}.gpickle")

        with open(output_file, 'wb') as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
        print(f"Generated graph {path}")