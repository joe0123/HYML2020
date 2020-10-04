class configurations(object):
    def __init__(self, args):
        self.batch_size = 96
        self.emb_dim = 256
        self.hidden_size = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.lr = 2e-4
        self.seq_len = 50
        self.total_steps = 10000
        self.summary_steps = 500    # Frquency to evaluate and store model
        self.load_model = False
        self.store_model_path = args[2]
        self.load_model_path = args[2]
        self.data_path = args[1]
        self.attention = True
        self.beam_width = 10
        self.ss_case = "sig"
