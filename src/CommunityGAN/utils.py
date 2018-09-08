import numpy as np
import random
import os
import datetime


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_embeddings(filename, n_node, n_embed):

    with open(filename, "r") as f:
        embedding_matrix = np.random.rand(n_node, n_embed)
        f.readline()  # skip the first line
        for line in f:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def read_embeddings_with_id_convert(filename, graph, n_embed):

    with open(filename, "r") as f:
        embedding_matrix = np.random.rand(graph.n_node, n_embed)
        f.readline()  # skip the first line
        for line in f:
            emd = line.split()
            embedding_matrix[graph.name2id[emd[0]], :] = str_list_to_float(emd[1:])
        return embedding_matrix


def agm(x):  # x is 1d-array
    agm_x = 1 - np.exp(-x)
    agm_x[np.isnan(agm_x)] = 0
    return np.clip(agm_x, 1e-6, 1)


def agm_softmax(x):  # x is 1d-array
    agm_x = 1 - np.exp(-x)
    agm_x[np.isnan(agm_x)] = 0
    agm_x = np.clip(agm_x, 1e-6, 1)
    return agm_x / agm_x.sum()


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines if not line[0].startswith('#')]
    return edges


def create_file_dir_in_config(config):
    for k, v in config.__dict__.items():
        if not k.startswith('_') and 'filename' in k:
            if not isinstance(v, list):
                v = [v]
            for path in v:
                dirname = os.path.dirname(path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)


def shuffle(*args):
    idx = list(range(len(args[0])))
    random.shuffle(idx)
    results = []
    for array in args:
        results.append([array[i] for i in idx])
    return tuple(results)


def genearate_tmp_filename(config):
    return ('tmp-' + str(hash(str(config.__dict__))) + str(datetime.datetime.now()) + '.pkl').replace(' ', '_').replace(':', '_')
