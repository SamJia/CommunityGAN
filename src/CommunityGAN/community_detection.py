import numpy as np
import utils
import scipy.sparse as sp
import math


class CommunityEval(object):
    def __init__(self, embed_filename, ground_truth_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.ground_truth_filename = ground_truth_filename  # each line: node_ids in the community(space seperated)
        self.n_node = n_node
        self.n_embed = n_embed

    def eval_community(self):
        self.load_ground_truth()
        self.load_embed()
        return self.f1_score()

    def load_ground_truth(self):
        row = []
        col = []
        data = []
        com_idx = 0
        for line in open(self.ground_truth_filename):
            if line.startswith('#'):
                continue
            line = utils.str_list_to_int(line.split())
            col.extend(line)
            row.extend([com_idx] * len(line))
            com_idx += 1
            data.extend([1] * len(line))
        self.ground_truth_m = sp.csr_matrix((data, (row, col)), shape=(com_idx, max(col) + 1), dtype=np.uint32)

    def load_embed(self):
        self.emb = utils.read_embeddings(self.embed_filename, n_node=self.n_node, n_embed=self.n_embed)
        epsilon = 1e-8  # ref to BIGCLAM
        threshold = math.sqrt(-math.log(1 - epsilon))  # ref to BIGCLAM
        self.emb = self.emb > threshold
        self.embed_m = sp.csr_matrix(self.emb.T, dtype=np.uint32)

    def f1_score(self):
        n = (self.ground_truth_m.dot(self.embed_m.T)).toarray().astype(float)  # cg * cd
        p = n / np.array(self.embed_m.sum(axis=1)).clip(min=1).reshape(-1)
        r = n / np.array(self.ground_truth_m.sum(axis=1)).clip(min=1).reshape(-1, 1)
        f1 = 2 * p * r / (p + r).clip(min=1e-10)

        f1_s1 = f1.max(axis=1).mean()
        f1_s2 = f1.max(axis=0).mean()
        f1_s = (f1_s1 + f1_s2) / 2
        return f1_s
