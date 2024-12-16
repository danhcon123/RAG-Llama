import tiktoken
from prompt import prompt_template
from src.helper import download_huggingface_embeddings


# Initialize tokenizer for your model

model=download_huggingface_embeddings()
encoding = tiktoken.get_encoding(model)  # Replace "gpt2" with your model's tokenizer if different

def count_tokens(text):
    return len(encoding.encode(text))

# Example usage
prompt = prompt_template.format(context=your_context, question=your_question)
total_tokens = count_tokens(prompt)
print(f"Total tokens in prompt: {total_tokens}")
