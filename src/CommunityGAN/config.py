class Config():
    def __init__(self):
        super(Config, self).__init__()

        self.modes = ["gen", "dis"]

        # training settings
        self.motif_size = 3  # number of nodes in a motif
        self.batch_size_gen = 64  # batch size for the generator
        self.batch_size_dis = 64  # batch size for the discriminator
        self.n_sample_gen = 5  # number of samples for the generator
        self.n_sample_dis = 5  # number of samples for the discriminator
        self.lr_gen = 1e-3  # learning rate for the generator
        self.lr_dis = 1e-3  # learning rate for the discriminator
        self.n_epochs = 10    # number of outer loops
        self.n_epochs_gen = 3  # number of inner loops for the generator
        self.n_epochs_dis = 3  # number of inner loops for the discriminator
        self.gen_interval = self.n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
        self.dis_interval = self.n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
        self.update_ratio = 1    # updating ratio when choose the trees
        self.max_value = 1000  # max value in embedding matrix

        # model saving
        self.load_model = False  # whether loading existing model for initialization
        self.save_steps = 10

        # other hyper-parameters
        self.n_emb = 100
        self.num_threads = 16
        self.window_size = 5

        # application and dataset settings
        self.app = "community_detection"
        self.dataset = "com-amazon"
        self.update_path()

    # change app and dataset
    def reset_config(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = type(self.__dict__[k])(v)
        self.update_path()

    # path settings
    def update_path(self):
        self.train_filename = "../../data/" + self.app + "/" + self.dataset + "_train.txt"
        self.pretrain_emb_filename_d = "../../pre_train/" + self.app + "/" + self.dataset + "_pre_train.emb"
        self.pretrain_emb_filename_g = "../../pre_train/" + self.app + "/" + self.dataset + "_pre_train.emb"
        self.community_filename = "../../data/" + self.app + "/" + self.dataset + ".sampled.cmty.txt"
        self.model_log = "../../log/"
        self.cache_filename_prefix = "../../cache/" + self.app + "/" + self.dataset
        self.emb_filenames = ["../../results/" + self.app + "/" + self.dataset + "_gen_.emb",
                              "../../results/" + self.app + "/" + self.dataset + "_dis_.emb"]
        self.result_filename = "../../results/" + self.app + "/" + self.dataset + ".txt"
