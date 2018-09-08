import numpy as np
import multiprocessing
import sys
import pickle
import utils
import random


def choice(samples, weight):
    s = np.sum(weight)
    target = random.random() * s
    for si, wi in zip(samples, weight):
        if target < wi:
            return si
        target -= wi
    return si


class Sampling(object):
    def __init__(self):
        super(Sampling, self).__init__()
        self.config = pickle.load(open(sys.argv[1], 'rb'))
        self.id2nid = pickle.load(open(self.config.cache_filename_prefix + '.neighbor.pkl', 'rb'))
        self.total_motifs = pickle.load(open(self.config.cache_filename_prefix + '.motifs.pkl', 'rb'))
        self.theta_g = pickle.load(open(self.config.cache_filename_prefix + '.theta.pkl', 'rb'))
        self.args = pickle.load(open(self.config.cache_filename_prefix + '.args.pkl', 'rb'))
        # print('load data done', datetime.datetime.now())

    def run(self):
        cores = self.config.num_threads
        motifs, paths = zip(*multiprocessing.Pool(cores).map(self.g_s, self.args))
        pickle.dump(motifs, open(self.config.cache_filename_prefix + '.motifs_sampled.pkl', 'wb'))
        pickle.dump(paths, open(self.config.cache_filename_prefix + '.paths.pkl', 'wb'))

    def g_s(self, args):  # for multiprocessing, pass multiple args in one tuple
        root, n_sample, only_neg = args
        motifs = []
        paths = []
        for i in range(2 * n_sample):
            if len(motifs) >= n_sample:
                break
            motif = [root]
            path = [root]
            for j in range(1, self.config.motif_size):
                v, p = self.g_v(motif)
                if v is None:
                    break
                motif.append(v)
                path.extend(p)
            if len(set(motif)) < self.config.motif_size:
                continue
            motif = tuple(sorted(motif))
            if only_neg and motif in self.total_motifs:
                continue
            motifs.append(motif)
            paths.append(path)
        return motifs, paths

    def g_v(self, roots):
        g_v_v = self.theta_g[roots[0]].copy()
        for nid in roots[1:]:
            g_v_v *= self.theta_g[nid]
        current_node = roots[-1]
        previous_nodes = set()
        path = []
        is_root = True
        while True:
            if is_root:
                node_neighbor = list({neighbor for root in roots for neighbor in self.id2nid[root]})
            else:
                node_neighbor = self.id2nid[current_node]
            if len(node_neighbor) == 0:  # the root node has no neighbor
                return None, None
            if is_root:
                tmp_g = g_v_v
            else:
                tmp_g = g_v_v * self.theta_g[current_node]
            relevance_probability = np.sum(self.theta_g[node_neighbor] * tmp_g, axis=1)
            relevance_probability = utils.agm(relevance_probability)
            next_node = choice(node_neighbor, relevance_probability)  # select next node
            if next_node in previous_nodes:  # terminating condition
                break
            previous_nodes.add(current_node)
            current_node = next_node
            path.append(current_node)
            is_root = False
        return current_node, path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python sampling.py config.pkl')
    s = Sampling()
    s.run()
