
GPT_CONFIG_124M = {
    "vocab_size" : 50257, # vocabulary size
    "batch_size" : 4, # batch size
    "ctx_len" : 32,     # context length
    "emb_dim" : 6,      # embedding dimesension
    "n_head" : 3,        # number of attention heads
    "n_layers" : 3,      # number of layers
    "learning_rate" : 1e-3, # learning rate
    "drop_rate" : 0.1,    # Dropout rate
    "qkv_bias" : False,  # qkv bias
    "eval_interval" : 100, # evaluation interval
    "max_iters" : 1000,
    "eval_iters" : 200
}