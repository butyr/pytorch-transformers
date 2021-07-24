class TransformerConfig:

    def __init__(
            self,
            random_seed=1234,
            nheads=8,
            key_dim=64,
            model_dim=512,
            hidden_dim=128,
            depth=6,
            max_len=5_000,
            epochs=10,
            eval_rate=1_000,
            train_batch_size=32,
            eval_batch_size=32,
            lr=3e-4,
            train_shuffle=True,
            num_workers=0,
    ):

        self.random_seed = random_seed
        self.nheads = nheads
        self.key_dim = key_dim
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.max_len = max_len
        self.epochs = epochs
        self.eval_rate = eval_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.lr = lr
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
