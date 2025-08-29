
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from utils.utils import get_log_probs

from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data: List[Dict], ref_log_probs_y1: torch.Tensor, ref_log_probs_y2: torch.Tensor):
        self.data = data
        self.ref_log_probs_y1 = ref_log_probs_y1
        self.ref_log_probs_y2 = ref_log_probs_y2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "first_responses": item["first_responses"],
            "second_responses": item["second_responses"],
            "choices": torch.tensor(item["choices"], dtype=torch.long),
            "ref_log_probs_y1": self.ref_log_probs_y1[idx],
            "ref_log_probs_y2": self.ref_log_probs_y2[idx],
        }
    
def transform_df_for_dpo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a pandas DataFrame with specific string formats into a DPO-ready dataset.

    Args:
        df: A DataFrame with the following columns:
            - 'first_response': A string representation of a model response dict.
            - 'second_response': A string representation of a model response dict.
            - 'preference_order': A string indicating preference (e.g., '1 > 2 > neither').

    Returns:
        A new DataFrame with three columns: 'prompt', 'chosen', and 'rejected'.
    """
    
    # Lists to store the processed data
    first_responses = []
    second_responses = []
    choices = []

    # Use tqdm for a progress bar, which is helpful for large datasets
    for _, row in df.iterrows():
        # Determine which response is chosen and which is rejected
        preference = row['pref']

        # Find the positions of '1' and '2' to determine the correct order
        pos_1 = preference.find('1')
        pos_2 = preference.find('2')

        if pos_1 < pos_2:
            choice = 0
        else:
            choice = 1
        
        # 2. Parse the chosen and rejected responses
        # The responses are strings of single dicts, e.g., "{'role': 'model', 'content': '...'}"
        first_response = row['first_response']
        second_response = row['second_response']
        
        # Append the cleaned data to our lists
        first_responses.append(first_response)
        second_responses.append(second_response)
        choices.append(choice)

    # Create the final, clean DataFrame
    dpo_df = pd.DataFrame({
        'first_responses': first_responses,
        'second_responses': second_responses,
        'choices': choices
    })
    
    return dpo_df.to_dict(orient='records')


def transform_df_for_new_dpo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a pandas DataFrame with specific string formats into a DPO-ready dataset.

    Args:
        df: A DataFrame with the following columns:
            - 'prompt': A string representation of a list of user message dicts.
            - 'first_response': A string representation of a model response dict.
            - 'second_response': A string representation of a model response dict.
            - 'preference_order': A string indicating preference (e.g., '1 > 2 > neither').

    Returns:
        A new DataFrame with three columns: 'prompt', 'chosen', and 'rejected'.
    """
    
    # Lists to store the processed data
    first_responses = []
    second_responses = []
    choices = []

    # Use tqdm for a progress bar, which is helpful for large datasets
    for _, row in df.iterrows():
        # Determine which response is chosen and which is rejected
        preference = row['pref'][0]
        
        choice = 0 if preference == 'n' else int(preference)

        # 2. Parse the chosen and rejected responses
        # The responses are strings of single dicts, e.g., "{'role': 'model', 'content': '...'}"
        first_response = row['first_response']
        second_response = row['second_response']
            
        first_responses.append(first_response)
        second_responses.append(second_response)
        choices.append(choice)

    # Create the final, clean DataFrame
    dpo_df = pd.DataFrame({
        'first_responses': first_responses,
        'second_responses': second_responses,
        'choices': choices
    })
    
    return dpo_df.to_dict(orient='records')


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