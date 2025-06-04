# srun command for testing:
# srun -p gpu-shared -A TG-MCB180035 --pty -N 1 -n 4 --gpus=1 --cpus-per-task=4 --mem=5G -t 01:00:00 --export=ALL /bin/bash

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

llm_folder = "/home/nrao2/language_preservation/hawaiian_llm/llms/"
tokenizer_folder = "/home/nrao2/language_preservation/hawaiian_llm/tokenizer"

if len(sys.argv) < 2:
    print("Format: python3.11 test_llm.py <GPT-2 model>")
    sys.exit()

llm = sys.argv[1] # e.g. 'gpt2_hawaiian'

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained(llm_folder + llm)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_folder)

# print(tokenizer("Test sentence <|endoftext|>"))
# print(tokenizer.decode([3041]))

def print_generated_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    output = model.generate(
        input_ids=inputs["input_ids"],
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,  # Penalizes repetitive tokens
        attention_mask=inputs["attention_mask"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    print("Number of tokens generated:", len(output[0]))
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    
    
    

# Translation: Hi, how are you?
print_generated_text("Aloha, pehea ʻoe?")
