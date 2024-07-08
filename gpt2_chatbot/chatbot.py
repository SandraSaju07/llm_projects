import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name,pad_token_id=tokenizer.eos_token_id)

def generate_response(prompt):
    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response based on the prompt
    outputs = model.generate(inputs, max_length=150, num_beams=5, no_repeat_ngram_size =2, early_stopping=True)
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces = True)
    
    return ".".join(response.split(".")[:-1])+"."
