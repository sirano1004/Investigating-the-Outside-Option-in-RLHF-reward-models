import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
from data import CustomDataset
import os

class NewDPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                choices: torch.Tensor,
                log_z: torch.Tensor):
        
        r1 = self.beta * (log_probs_y1_policy - log_probs_y1_ref + log_z)
        r2 = self.beta * (log_probs_y2_policy - log_probs_y2_ref + log_z)
        logits = torch.stack([torch.zeros_like(r1), r1, r2], dim=1) # class 0=neither

        loss = F.cross_entropy(logits, choices)
        
        return loss
    

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                choices: torch.Tensor):
        
        r1 = self.beta * (log_probs_y1_policy - log_probs_y1_ref)
        r2 = self.beta * (log_probs_y2_policy - log_probs_y2_ref)
        logits = torch.stack([r1, r2], dim=1) # class 0=neither

        loss = F.cross_entropy(logits, choices)
        
        return loss


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


def get_reward(model: nn.Module, tokenizer: AutoTokenizer, dataset: List[List[Dict]], response: str, device: str, max_length) -> torch.Tensor:
    # Combine prompts and responses into full chat histories
    full_chats = [item[response] for item in dataset]

    # Tokenize the full chat histories
    # We need to pad to the left for decoder-only models
    model_inputs = tokenizer(
        full_chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
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

def precompute_reference_log_probs(ref_model: nn.Module, tokenizer: AutoTokenizer, dataset: CustomDataset, device: str, prompt_length, max_length) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterates through a dataset to pre-calculate the log probabilities for
    the first and second responses using the reference model.
    """
    ref_model.eval()
    

    print("Pre-computing log probabilities for first responses...")
    with torch.no_grad():
        log_probs_y1_ref = get_log_probs(ref_model, tokenizer, dataset, 'first_responses', device, prompt_length, max_length)
    
    print("Pre-computing log probabilities for second responses...")
    with torch.no_grad():
        log_probs_y2_ref = get_log_probs(ref_model, tokenizer, dataset, 'second_responses', device, prompt_length, max_length)
    
    return log_probs_y1_ref, log_probs_y2_ref

def precompute_reference_log_probs_batched(
    ref_model: torch.nn.Module,
    tokenizer,
    dataset_list: list,
    batch_size: int,
    device: str,
    prompt_length,
    max_length,
    offset=0
):
    """
    Calculates reference log probabilities for a DataFrame in batches to conserve memory.
    """

    # Create a simple dataloader. The lambda function ensures batches are lists of dicts.
    dataloader = DataLoader(dataset_list, batch_size=batch_size, collate_fn=lambda batch: batch)

    all_log_probs_y1 = []
    all_log_probs_y2 = []

    ref_model.eval()
    print("Pre-computing reference log probabilities in batches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            # Your get_log_probs function already expects a batch as a list of dicts
            log_probs_y1 = get_log_probs(ref_model, tokenizer, batch, 'first_responses', device, prompt_length, max_length, offset)
            log_probs_y2 = get_log_probs(ref_model, tokenizer, batch, 'second_responses', device, prompt_length, max_length, offset)

            # Move results to CPU to free up VRAM for the next batch
            all_log_probs_y1.append(log_probs_y1.cpu())
            all_log_probs_y2.append(log_probs_y2.cpu())

    # Concatenate all batch results into single tensors
    final_log_probs_y1 = torch.cat(all_log_probs_y1)
    final_log_probs_y2 = torch.cat(all_log_probs_y2)

    return final_log_probs_y1, final_log_probs_y2

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

# Assume get_log_probs function is defined as in the previous step
# def get_log_probs(model, tokenizer, dataset, device):
#     ...
