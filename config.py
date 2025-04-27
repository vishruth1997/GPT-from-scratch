# config.py

config = {
    "batch_size": 64,
    # "block_size": 128,       # context window
    "block_size": 768, 
    "max_iters": 10000,
    "eval_interval": 100,
    "learning_rate": 1e-3,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "eval_iters": 200,
    "n_embed": 256,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.3,
    "train_test_split": 0.9,
}
