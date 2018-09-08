import os
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from config import Config
import generator
import discriminator
import graph
import utils
import community_detection as cd
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CommunityGAN(object):
    def __init__(self):
        for k, v in config.__dict__.items():
            print('%s:\t%s' % (k, str(v)))
        pickle.dump(config, open(config.cache_filename_prefix + '.config.pkl', 'wb'))

        print("reading graphs...")
        self.graph = graph.Graph(config.motif_size)
        self.graph.load_graph(config.train_filename)
        self.graph.graph_preparation()
        pickle.dump(self.graph, open(config.cache_filename_prefix + '.graph.pkl', 'wb'))
        pickle.dump(self.graph.id2nid, open(config.cache_filename_prefix + '.neighbor.pkl', 'wb'))
        pickle.dump(self.graph.motifs, open(config.cache_filename_prefix + '.motifs.pkl', 'wb'))
        # print(self.graph.n_node)

        print("reading initial embeddings...")
        self.node_embed_init_d = utils.read_embeddings_with_id_convert(filename=config.pretrain_emb_filename_d,
                                                                       graph=self.graph,
                                                                       n_embed=config.n_emb)
        self.node_embed_init_g = utils.read_embeddings_with_id_convert(filename=config.pretrain_emb_filename_g,
                                                                       graph=self.graph,
                                                                       n_embed=config.n_emb)

        print("building GAN model...")
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()

        print('preparing checkpoint')
        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        self.saver = tf.train.Saver()

        print('tensorflow initialization')
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

    def build_generator(self):
        """initializing the generator"""

        with tf.variable_scope("generator"):
            self.generator = generator.Generator(self.graph.n_node, self.node_embed_init_g, config)

    def build_discriminator(self):
        """initializing the discriminator"""

        with tf.variable_scope("discriminator"):
            self.discriminator = discriminator.Discriminator(self.graph.n_node, self.node_embed_init_d, config)

    def train(self):

        self.write_embeddings_to_file()
        self.evaluation(self, pre_train=True)

        print("start training...")
        for epoch in range(config.n_epochs):
            print("epoch %d" % epoch)

            self.train_d()
            self.train_g()

            self.write_embeddings_to_file()
            self.evaluation(self)

        print("training completes")

    def train_d(self):
        motifs = []
        labels = []
        for d_epoch in range(config.n_epochs_dis):

            # generate new subsets for the discriminator for every dis_interval iterations
            if d_epoch % config.dis_interval == 0:
                motifs, labels = self.prepare_data_for_d()

            # training
            train_size = len(motifs)
            start_list = list(range(0, train_size, config.batch_size_dis))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_dis
                self.sess.run(self.discriminator.d_updates,
                              feed_dict={self.discriminator.motifs: np.array(motifs[start:end]),
                                         self.discriminator.label: np.array(labels[start:end])})
                self.sess.run(self.discriminator.clip_op)

    def train_g(self):
        motifs = []
        reward = []
        for g_epoch in range(config.n_epochs_gen):

            # generate new subsets for the generator for every gen_interval iterations
            if g_epoch % config.gen_interval == 0:
                motifs, reward = self.prepare_data_for_g()

            # training
            train_size = len(motifs)
            start_list = list(range(0, train_size, config.batch_size_gen))
            np.random.shuffle(start_list)
            for start in start_list:
                end = start + config.batch_size_gen
                self.sess.run(self.generator.g_updates,
                              feed_dict={self.generator.motifs: np.array(motifs[start:end]),
                                         self.generator.reward: np.array(reward[start:end])})
                self.sess.run(self.generator.clip_op)

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator"""
        motifs = []
        labels = []
        g_s_args = []
        poss = []
        negs = []
        for i in range(self.graph.n_node):
            if np.random.rand() < config.update_ratio:
                pos = random.sample(self.graph.id2motifs[i], min(len(self.graph.id2motifs[i]), config.n_sample_dis))
                poss.append(pos)
                g_s_args.append((i, len(pos), True))

        negs, _ = self.sampling(g_s_args)
        for pos, neg in zip(poss, negs):
            if len(pos) != 0 and neg is not None:
                motifs.extend(pos)
                labels.extend([1] * len(pos))
                motifs.extend(neg)
                labels.extend([0] * len(neg))

        motifs, labels = utils.shuffle(motifs, labels)
        pickle.dump(motifs, open(config.cache_filename_prefix + '.motifs_ford.pkl', 'wb'))
        pickle.dump(labels, open(config.cache_filename_prefix + '.labels_ford.pkl', 'wb'))
        return motifs, labels

    def prepare_data_for_g(self):
        """sample subsets for the generator"""
        paths = []
        g_s_args = []
        for i in range(self.graph.n_node):
            if np.random.rand() < config.update_ratio:
                g_s_args.append((i, config.n_sample_gen, False))

        motifs, paths = self.sampling(g_s_args)
        motifs = [j for i in motifs for j in i]

        reward = []
        for i in range(0, len(motifs), 10000):
            reward.append(self.sess.run(self.discriminator.reward,
                                        feed_dict={self.discriminator.motifs: np.array(motifs[i: i + 10000])}))
        reward = np.concatenate(reward)
        motifs, reward = utils.shuffle(motifs, reward)
        return motifs, reward

    def sampling(self, args):
        self.theta_g = self.sess.run(self.generator.embedding_matrix)
        pickle.dump(args, open(config.cache_filename_prefix + '.args.pkl', 'wb'))
        pickle.dump(self.theta_g, open(config.cache_filename_prefix + '.theta.pkl', 'wb'))
        subprocess.call('python sampling.py %s' % (config.cache_filename_prefix + '.config.pkl'), shell=True)

        motifs = pickle.load(open(config.cache_filename_prefix + '.motifs_sampled.pkl', 'rb'))
        paths = pickle.load(open(config.cache_filename_prefix + '.paths.pkl', 'rb'))
        return motifs, paths

    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            embedding_list = embedding_matrix.tolist()
            embedding_str = [self.graph.id2name[idx] + "\t" + "\t".join([str(x) for x in emb]) + "\n"
                             for idx, emb in enumerate(embedding_list)]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.graph.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

    @staticmethod
    def evaluation(self, pre_train=False):
        results = []
        if config.app == 'community_detection':
            for i in range(2):
                ce = cd.CommunityEval(
                    config.emb_filenames[i], config.community_filename, self.graph.n_node, config.n_emb)
                result = ce.eval_community()
                results.append(config.modes[i] + ":" + str(result) + "\n")
        print(results)
        with open(config.result_filename, mode="a+") as f:
            if pre_train:
                f.write('==================================\n')
            f.writelines(results)


if __name__ == "__main__":
    config = Config()
    kwargs = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv), 2)}
    config.reset_config(**kwargs)
    utils.create_file_dir_in_config(config)
    community_gan = CommunityGAN()
    community_gan.train()
