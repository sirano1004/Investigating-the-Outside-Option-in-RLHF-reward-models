import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Type hints for better readability and debugging
from typing import List

def generate_output(model, tokenizer, inputs: str, max_input: int, max_output: int, num_return_sequences: int = 1) -> List[str]:
    """
    Generates text from a Hugging Face Causal Language Model.

    This function prepares the input with the tokenizer and uses the model's
    'generate' method to create new text based on the provided input prompt.

    Args:
        model: The pre-trained Hugging Face model for text generation.
        tokenizer: The corresponding tokenizer for the model.
        inputs (str): The input text prompt to generate from.
        max_input (int): The maximum number of tokens for the input.
        max_output (int): The maximum number of tokens for the combined input and output.
        num_return_sequences (int): The number of independent sequences to generate.

    Returns:
        List[str]: A list of generated text sequences.
    """
    # Tokenize the input string and move to CUDA
    # `max_length` ensures the input fits within the model's context window.
    tokenized_inputs = tokenizer(inputs,
                                 return_tensors='pt',
                                 max_length=max_input,
                                 truncation=True,
                                 padding=True).to('cuda')

    # Generate new tokens based on the input prompt
    # `max_new_tokens` controls the length of the new generated text
    outputs = model.generate(
        **tokenized_inputs,
        max_new_tokens=max_output - tokenized_inputs.input_ids.shape[1],
        do_sample=True,
        num_return_sequences=num_return_sequences
    )

    # Decode the generated tokens back into a human-readable string
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def get_rewards(model, generated_output: List[str]) -> List[float]:
    """
    Calculates a 'positive' reward score for a list of text outputs.

    This function uses a sentiment analysis or reward model pipeline to score
    each text output and returns a normalized score for the 'positive' label.

    Args:
        model: The Hugging Face pipeline model for sentiment analysis or reward scoring.
        generated_output (List[str]): A list of text strings to be scored.

    Returns:
        List[float]: A list of normalized scores (from -1.0 to 1.0) for the positive label.
    """
    results = model(generated_output, top_k=None)
    pos_scores = []
    
    # Iterate through the results to find the 'positive' score
    for result in results:
        # Each result is a list of dictionaries with 'label' and 'score'
        for item in result:
            if item['label'].lower() == 'positive':
                # Normalize the score from the range [0.0, 1.0] to [-1.0, 1.0]
                pos_scores.append((item['score'] - 0.5) * 2)

    return pos_scores

