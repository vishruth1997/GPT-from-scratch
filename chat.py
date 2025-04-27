# chat.py

import torch
from model import GPT
from dataset import BpeDataset
from config import config

# Load dataset and tokenizer
dataset = BpeDataset("data/input.txt")
vocab_size = dataset.vocab_size

# Load trained model
model = GPT(vocab_size).to(config["device"])
model.load_state_dict(torch.load("gpt_model.pth", map_location=config["device"]))
model.eval()

# Pre-encode [SEP] token for stopping logic
sep_token_id = dataset.encode("[SEP]")[0]

print("ChatGPT BPE model loaded! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # 🔥 Use only current prompt — no full context
    prompt = f"[|Human|] {user_input} [|AI|]"
    prompt_ids = dataset.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long).to(config["device"])

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=200, stop_token_id=sep_token_id, min_tokens=10)

    full_output = dataset.decode(out[0].tolist())
    response = full_output[len(prompt):].split("[SEP]")[0].strip()

    if not response:
        response = "Sorry, I couldn't generate a response. Could you try rephrasing?"

    print("Bot:", response)


# chat.py

# import torch
# from model import GPT
# from dataset import BpeDataset
# from config import config

# # Load dataset and tokenizer
# dataset = BpeDataset("data/input.txt")
# vocab_size = dataset.vocab_size

# # Load trained model
# model = GPT(vocab_size).to(config["device"])
# model.load_state_dict(torch.load("gpt_model.pth", map_location=config["device"]))
# model.eval()

# # Pre-encode [SEP] token for stopping logic
# sep_token_id = dataset.encode("[SEP]")[0]

# print("ChatGPT BPE model loaded! Type 'quit' to exit.")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         break

#     # Few-shot prompt examples + user input
#     context = (
#         "[|Human|] I feel sick [|AI|] You may have a flu. Please rest and stay hydrated. [SEP]\n"
#         "[|Human|] I feel dizzy in the morning [|AI|] It could be low blood pressure. Try eating something salty and get your blood checked. [SEP]\n"
#         f"[|Human|] {user_input} [|AI|]"
#     )

#     prompt_ids = dataset.encode(context)
#     idx = torch.tensor([prompt_ids], dtype=torch.long).to(config["device"])

#     with torch.no_grad():
#         out = model.generate(idx, max_new_tokens=200, stop_token_id=sep_token_id, min_tokens=10)

#     full_output = dataset.decode(out[0].tolist())

#     # Only extract the response after user's input
#     try:
#         response = full_output.split(f"[|Human|] {user_input} [|AI|]")[1].split("[SEP]")[0].strip()
#     except IndexError:
#         response = "Sorry, I couldn't generate a response. Could you try rephrasing?"

#     print("Bot:", response)
