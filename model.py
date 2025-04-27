# model.py

import torch
import torch.nn as nn
from config import config
import math

# Self-Attention Head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.scale = head_size ** -0.5  # for stability

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        # Apply attention
        v = self.value(x)
        out = att @ v  # (B, T, head_size)
        return out

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embed, block_size, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, n_embed, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

# Feedforward network
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual
        x = x + self.ffwd(self.ln2(x)) # residual
        return x

# Full GPT model
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, config["n_embed"])
        self.pos_embed = nn.Parameter(torch.zeros(1, config["block_size"], config["n_embed"]))
        self.blocks = nn.Sequential(*[
            TransformerBlock(config["n_embed"], config["n_heads"], config["block_size"], config["dropout"])
            for _ in range(config["n_layers"])
        ])
        self.ln_f = nn.LayerNorm(config["n_embed"])
        self.head = nn.Linear(config["n_embed"], vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)              # (B, T, n_embed)
        pos_emb = self.pos_embed[:, :T, :]           # (1, T, n_embed)
        x = tok_emb + pos_emb                        # (B, T, n_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                        # (B, T, vocab_size)
        return logits

    # def generate(self, idx, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -config["block_size"]:]
    #         logits = self(idx_cond)
    #         logits = logits[:, -1, :]
    #         probs = torch.softmax(logits, dim=-1)
    #         next_token = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, next_token), dim=1)
    #     return idx

    def generate(self, idx, max_new_tokens, stop_token_id=None, min_tokens=10):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -config["block_size"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if stop_token_id is not None and i >= min_tokens and next_token.item() == stop_token_id:
                break

        return idx