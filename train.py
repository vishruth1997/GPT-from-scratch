# train.py

import torch
import torch.nn as nn
from model import GPT
from dataset import BpeDataset
from config import config

# Load dataset
dataset = BpeDataset("data/input.txt")
vocab_size = dataset.vocab_size
print("Vocab size:", vocab_size)

# Initialize model
model = GPT(vocab_size).to(config["device"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

# Training loop
for step in range(config["max_iters"]):
    model.train()
    x_batch, y_batch = dataset.get_batch("train")

    logits = model(x_batch)
    loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    if step % config["eval_interval"] == 0 or step == config["max_iters"] - 1:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(config["eval_iters"]):
                x_val, y_val = dataset.get_batch("val")
                logits_val = model(x_val)
                val_loss = loss_fn(logits_val.view(-1, vocab_size), y_val.view(-1))
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "gpt_model.pth")
print("✅ Model saved as gpt_model.pth")
