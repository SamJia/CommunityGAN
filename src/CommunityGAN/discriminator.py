import tensorflow as tf


class Discriminator(object):
    def __init__(self, n_node, node_emd_init, config):
        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.motif_size = config.motif_size
        self.max_value = config.max_value

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)

        self.motifs = tf.placeholder(tf.int32, shape=[None, config.motif_size])
        self.label = tf.placeholder(tf.float32, shape=[None])

        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.motifs)  # Batch * motif_size * embedding_size
        self.score = tf.reduce_sum(tf.reduce_prod(self.node_embedding, axis=1), axis=1)
        self.p = 1 - tf.exp(-self.score)
        self.p = tf.clip_by_value(self.p, 1e-5, 1)

        self.loss = -(tf.reduce_sum(self.label * (self.p) + (1 - self.label) * (1 - self.p)))
        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        self.reward = 1 - self.p
        self.clip_op = tf.assign(self.embedding_matrix, tf.clip_by_value(self.embedding_matrix, 0, self.max_value))
