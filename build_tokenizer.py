# build_tokenizer.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

# Read your training data
with open("data/input.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Save to a temp file for tokenizer training
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    f.write(data)

# Create and train a BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# ✅ Include [|Human|], [|AI|], [SEP] as special tokens
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[|Human|]", "[|AI|]"]
)
tokenizer.train(["data/corpus.txt"], trainer)

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

tokenizer.save("tokenizer.json")
print("✅ Tokenizer trained and saved as tokenizer.json")
