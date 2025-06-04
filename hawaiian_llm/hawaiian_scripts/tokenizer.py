import os
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer

# Part 1: Create a tokenizer

tokenizer_folder = "/home/nrao2/language_preservation/hawaiian_llm/tokenizer"
data_folder = "/home/nrao2/uci141/esolares/hawaiian_tesseract/anthony/ordered_texts/"

# Get list of files
corpus_files = [data_folder + f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

# Initialize the tokenizer
# Documentation: https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py
tokenizer = ByteLevelBPETokenizer()

# Count the number of words and unique words
# Create a dictionary of word frequencies
# Use to determine the number of unique tokens

# Train the tokenizer
tokenizer.train(
    files=corpus_files,
    vocab_size=50257,  # Adjust as needed
)

# Save the tokenizer to disk
if not os.path.exists(tokenizer_folder):
    os.makedirs(tokenizer_folder)
tokenizer.save_model(tokenizer_folder)

tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_folder)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(tokenizer_folder)

# # Part 2: Test the tokenizer

# Load the trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_folder)

# Add special tokens if needed
# tokenizer.add_special_tokens({"pad_token": "<pad>"})

# Test on sample Hawaiian text
sample_text = "Aloha pehea ʻoe?"
encoded = tokenizer(sample_text)
decoded = tokenizer.decode(encoded["input_ids"])

print("Encoded:", encoded)
print("Decoded:", decoded)
