from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
from peft import LoraConfig, PeftModel, get_peft_model
import os

def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation='eager'
    )
    return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

reward_model = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")

class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, path = None):
        super().__init__()
                
        # Load the Gemma Base Model
        base_model = get_model(model_name)
        if path:
            self.base_model = PeftModel.from_pretrained(base_model, path)
            hidden_size = self.base_model.config.hidden_size
            self.classifier_head = nn.Linear(hidden_size, num_labels)
            classifier_path = os.path.join(path, "classifier_head.pth")
            if os.path.exists(classifier_path):
                self.classifier_head.load_state_dict(torch.load(classifier_path))
            else:
                print(f"Warning: No classifier_head.pth found in {path}. Classifier head is not loaded.")
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.base_model = get_peft_model(base_model, lora_config)
            
            # Freeze all non-LoRA parameters
            for name, param in self.base_model.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                            
            hidden_size = self.base_model.config.hidden_size
            self.classifier_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden_state = last_hidden_state[:, -1, :]
        
        logits = self.classifier_head(last_token_hidden_state)
        return logits

    def save_pretrained(self, save_directory):
        """
        Saves the LoRA adapter weights and the classifier head to a directory.
        """
        print(f"Saving model weights to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        
        # 1. Save LoRA adapter weights
        self.base_model.save_pretrained(save_directory)
        
        # 2. Save the classifier head's state dictionary
        classifier_path = os.path.join(save_directory, "classifier_head.pth")
        torch.save(self.classifier_head.state_dict(), classifier_path)