import torch
from utils.utils import get_log_probs, get_log_Z, get_log_probs_sft
from utils.reward_scoring import generate_output, get_rewards

def evaluate_sft(model, eval_dataloader, tokenizer, device, prompt_length, max_length):
    """
    Evaluates the model on the provided evaluation dataset.

    Returns:
        float: The average loss over the entire evaluation dataset.
    """
    # 1. Set the model to evaluation mode
    model.eval()
    
    total_loss = 0.0
    total_valid_tokens = 0

    # 2. Disable gradient calculations for efficiency
    with torch.no_grad():
        # Loop through all batches in the evaluation dataloader
        for batch in eval_dataloader:
            
            # The 'batch' from your dataloader is expected to be a list of strings
            
            # Calculate the summed negative log probabilities for the batch
            # Note: The output of get_log_probs_sft is effectively the summed loss per sequence
            batch_nll, batch_valid_tokens = get_log_probs_sft(model, tokenizer, batch, device, prompt_length, max_length)
                        
            # 3. Aggregate the loss
            total_loss += batch_nll.sum().item()

            total_valid_tokens += batch_valid_tokens.sum().item()

    # 4. Calculate the average loss over all batches
    average_loss = total_loss / total_valid_tokens if total_valid_tokens > 0 else 0.0
    
    # 5. Set the model back to training mode
    model.train()
    
    return average_loss


def evaluate_dpo(model, dataloader, loss_fn, tokenizer, device, prompt_length, max_length):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            log_probs_y1_policy = get_log_probs(model, tokenizer, batch, 'first_responses', device, prompt_length, max_length)
            log_probs_y2_policy = get_log_probs(model, tokenizer, batch, 'second_responses', device, prompt_length, max_length)

            # Stack the tensor items
            choices = torch.stack([item['choices'] for item in batch])
            ref_log_probs_y1 = torch.stack([item['ref_log_probs_y1'] for item in batch])
            ref_log_probs_y2 = torch.stack([item['ref_log_probs_y2'] for item in batch])

            loss = loss_fn(
                log_probs_y1_policy,
                log_probs_y2_policy,
                ref_log_probs_y1.to(device),
                ref_log_probs_y2.to(device),
                choices.to(device)
            )
            total_loss += loss.item()
    
    model.train()

    return total_loss / len(dataloader)

def evaluate_new_dpo(policy_model, Z_model, dataloader, loss_fn, tokenizer, device, prompt_length, max_length, offset):
    policy_model.eval()
    Z_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            log_probs_y1_policy = get_log_probs(policy_model, tokenizer, batch, 'first_responses', device, prompt_length, max_length, offset)
            log_probs_y2_policy = get_log_probs(policy_model, tokenizer, batch, 'second_responses', device, prompt_length, max_length, offset)

            log_Z = get_log_Z(Z_model, tokenizer, batch, device, prompt_length)

            # Stack the tensor items
            choices = torch.stack([item['choices'] for item in batch])
            ref_log_probs_y1 = torch.stack([item['ref_log_probs_y1'] for item in batch])
            ref_log_probs_y2 = torch.stack([item['ref_log_probs_y2'] for item in batch])

            loss = loss_fn(
                log_probs_y1_policy,
                log_probs_y2_policy,
                ref_log_probs_y1.to(device),
                ref_log_probs_y2.to(device),
                choices.to(device),
                log_Z
            )
            total_loss += loss.item()

    policy_model.train()
    Z_model.train()
    return total_loss / len(dataloader)

def evaluate_rewards(model, reward_model, tokenizer, dataloader, max_input, max_output):
    rewards = []
    for batch in dataloader:
        dataset =  [item['first_responses'] for item in batch] 
        generated_output = generate_output(model, tokenizer, dataset, max_input, max_output)
        rewards.extend(get_rewards(reward_model, generated_output))
    return rewards