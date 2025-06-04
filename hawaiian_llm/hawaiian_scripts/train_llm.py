# # Train GPT-2 Model

# Prepare the dataset

from datasets import load_dataset
import os
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

data_folder = "/home/nrao2/uci141/esolares/hawaiian_tesseract/anthony/ordered_texts/"
model_folder = "/home/nrao2/language_preservation/hawaiian_llm/llms/gpt2_hawaiian_v3/"
tokenizer_folder = "/home/nrao2/language_preservation/hawaiian_llm/tokenizer"

corpus_files = [data_folder + f for f in os.listdir(data_folder)
                 if os.path.isfile(os.path.join(data_folder, f))]


# First 80% of files are for training, remaining 20% are for testing
num_train = int(0.8 * len(corpus_files))
dataset = load_dataset('text', data_files={'train': corpus_files[:num_train], 'test': corpus_files[num_train:]})


# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_folder)


# Define the GPT-2 configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # Adjust as per your tokenizer
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    attn_pdrop=0.1,  # Dropout in attention layers
    resid_pdrop=0.1,  # Dropout in residual connections
    embd_pdrop=0.1,   # Dropout in embedding layers
)

# Reset weights and create the GPT2 model
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer)) # Resize token embeddings after ensuring the correct vocab size


# Prepare the dataset for training

# def tokenize_function(examples):
#     # Tokenize the text
#     tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
#     return tokenized

# tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def tokenize_function(examples):
    # return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    return tokenizer(
        [text + "<|endoftext|>" for text in examples["text"]],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up the Trainer
training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    
    save_strategy="epoch", # Saves model once per epoch
    save_total_limit=2, # Keeps only the last 2 checkpoints to save space
    
    fp16=True # Enable if you have a compatible GPU
)

# Data collator for language modeling (adds labels automatically to the training data,
# where labels are the words shifted one over)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # No masking for GPT
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save the trained model
model.save_pretrained(model_folder)
