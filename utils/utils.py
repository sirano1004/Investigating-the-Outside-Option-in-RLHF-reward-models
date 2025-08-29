import torch
import torch.nn as nn

from transformers import AutoTokenizer

from typing import Dict, List
import numpy as np
from scipy import stats
import stats



def get_log_Z(model: nn.Module, tokenizer: AutoTokenizer, dataset: List[List[Dict]], device: str, prompt_length) -> torch.Tensor:
    # Combine prompts and responses into full chat histories
    full_chats = [item['first_responses'] for item in dataset]

    # Tokenize the full chat histories
    # We need to pad to the left for decoder-only models
    model_inputs = tokenizer(
        full_chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=(prompt_length + 1),
    )

    # Move tensors to the correct device
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)

    # Get model logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    log_z = outputs.view(-1)

    return log_z


def get_log_probs(model, tokenizer, dataset, response: str, device: str, prompt_length, max_length, offset = 0) -> torch.Tensor:
    # Combine prompts and responses into full chat histories
    selected_response = [item[response] for item in dataset]

    # Tokenize the full chat histories
    # We need to pad to the left for decoder-only models
    model_inputs = tokenizer(
        selected_response,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Create labels by cloning input_ids and masking the prompt part
    labels = model_inputs['input_ids'].clone()
    labels[:, :(prompt_length+offset+1)] = -100
    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100

    # Move tensors to the correct device
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    labels = labels.to(device)

    # Get model logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:,1:].contiguous()

    # Use CrossEntropyLoss to calculate the log probability of the sequence
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index = -100)
    log_probs = -loss_fct(shifted_logits.view(-1, model.config.vocab_size), shifted_labels.view(-1))

    log_probs = log_probs.view(len(dataset), -1).sum(dim=1)

    return log_probs


def get_log_probs_sft(model, tokenizer, dataset, device, prompt_length, max_length):

    # Tokenize the full chat histories
    # We need to pad to the left for decoder-only models
    model_inputs = tokenizer(
        dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Create labels by cloning input_ids and masking the prompt part
    labels = model_inputs['input_ids'].clone()
    # Fixed prompt length tokens are inputs + <bos> token
    labels[:, :(prompt_length+1)] = -100
    # Mask padding tokens in labels
    labels[labels == tokenizer.pad_token_id] = -100

    # Move tensors to the correct device
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    labels = labels.to(device)

    # Get model logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # The required shift for correct alignment
    logits_for_loss = logits[:, :-1, :].contiguous()
    labels_for_loss = labels[:, 1:].contiguous()

    # Use CrossEntropyLoss to calculate the log probability of the sequence
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    log_probs = loss_fct(
        logits_for_loss.view(-1, model.config.vocab_size),
        labels_for_loss.view(-1)
    )
    log_probs = log_probs.view(len(dataset), -1)
    # Count valid tokens (non-'-100') for normalization
    valid_tokens = (labels_for_loss != -100).sum(dim=1)

    return log_probs, valid_tokens


def describe_rewards(scores):
    # Convert to a NumPy array for efficient calculations
    scores_array = np.array(scores)
    n = len(scores_array)

    # 2. Calculate the Mean
    mean_score = np.mean(scores_array)

    # 3. Calculate the Standard Deviation
    # Using ddof=1 for the sample standard deviation is standard practice.
    std_dev = np.std(scores_array, ddof=1)

    # 4. Calculate the 95% Confidence Interval
    # Get the z-score for 95% confidence (which is 1.96)
    z_score = stats.norm.ppf(0.975)
    # Calculate the margin of error
    margin_of_error = z_score * (std_dev / math.sqrt(n))
    # Calculate the interval bounds
    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error

    # Print the results
    print(f"Mean: {mean_score:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"95% Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")