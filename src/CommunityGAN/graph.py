import os


class Graph(object):
    """Save the graph, and provide some functions to get nodes, edges"""

    def __init__(self, motif_size):
        super(Graph, self).__init__()
        self.name2id = {}  # V
        self.id2name = []  # V
        self.id2nid = []  # list of list, max: V * V
        self.id2nid_set = []  # list of list, max: V * V
        self.id2ncount = []  # V
        self.n_node = 0
        self.motif_size = motif_size

    def load_graph(self, filename):
        if not os.path.isfile(filename):
            raise Exception("No such file when load graph: %s" % filename)
        for line in open(filename, encoding='utf-8'):
            if line.startswith('#'):
                continue
            node1, node2 = line.split()
            # if node1 == node2:
            #     continue
            if node1 not in self.name2id:
                self.name2id[node1] = len(self.name2id)
                self.id2name.append(node1)
                self.id2nid.append([])
            if node2 not in self.name2id:
                self.name2id[node2] = len(self.name2id)
                self.id2name.append(node2)
                self.id2nid.append([])
            id1, id2 = self.name2id[node1], self.name2id[node2]
            self.id2nid[id1].append(id2)
            self.id2nid[id2].append(id1)
        self.id2nid_set = [set(nodes) for nodes in self.id2nid]
        self.id2ncount = [len(nid) for nid in self.id2nid]
        self.n_node = len(self.id2name)

    def get_motifs_with_one_more_node(self, motifs):
        motifs_next = set()
        for motif in motifs:
            nei = self.id2nid_set[motif[0]] - set(motif)
            for node in motif[1:]:
                nei = nei & self.id2nid_set[node]
            for node in nei:
                motifs_next.add(tuple(sorted(list(motif) + [node])))
        return motifs_next

    def get_motifs(self):
        self.motifs = set((node, ) for node in range(self.n_node))
        for i in range(self.motif_size - 1):
            print('getting motifs with size of %d' % (i + 2))
            self.motifs = self.get_motifs_with_one_more_node(self.motifs)
            print('totally %d motifs' % len(self.motifs))
        self.id2motifs = [[] for i in range(self.n_node)]
        for motif in self.motifs:
            for nid in motif:
                self.id2motifs[nid].append(motif)

    def graph_preparation(self):
        self.get_motifs()
