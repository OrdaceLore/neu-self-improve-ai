"""
Qwen2.5-1.5B-Instruct model integration
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
import warnings


class QwenModel(nn.Module):
    """Wrapper for Qwen2.5-1.5B-Instruct model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load model and tokenizer
        self.model_name = config.model.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.training.mixed_precision else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Set model config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Freeze some layers if needed
        if hasattr(config, 'freeze_layers') and config.freeze_layers:
            self._freeze_layers()
    
    def _freeze_layers(self):
        """Freeze some layers for fine-tuning"""
        # Freeze embedding layers
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False
        
        # Freeze first few transformer layers
        num_freeze_layers = getattr(self.config, 'num_freeze_layers', 0)
        for i in range(min(num_freeze_layers, len(self.model.model.layers))):
            for param in self.model.model.layers[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            "attentions": outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_length: int = None, temperature: float = 1.0, 
                 top_p: float = 1.0, do_sample: bool = True,
                 pad_token_id: int = None, **kwargs) -> torch.Tensor:
        """Generate text using the model"""
        if max_length is None:
            max_length = self.config.model.max_length
        
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        
        # Generation parameters
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        return outputs
    
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get logits for input tokens"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs["logits"]
    
    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for input tokens"""
        logits = self.get_logits(input_ids, attention_mask)
        return torch.log_softmax(logits, dim=-1)
    
    def compute_perplexity(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute perplexity for input sequence"""
        outputs = self.forward(input_ids, attention_mask, labels=input_ids)
        loss = outputs["loss"]
        return torch.exp(loss)
    
    def save_pretrained(self, path: str) -> None:
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_pretrained(self, path: str) -> None:
        """Load model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


class ValueModel(nn.Module):
    """Value function model for A*-PO"""
    
    def __init__(self, base_model: QwenModel, hidden_size: int = 2048):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(base_model.model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for value estimation"""
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Pool hidden states
        pooled_output = self._pool_hidden_states(hidden_states, attention_mask)
        
        # Get value estimate
        value = self.value_head(pooled_output)
        
        return value.squeeze(-1)
    
    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool hidden states using attention mask"""
        # Mean pooling over sequence length
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden = hidden_states * mask_expanded
        
        # Sum over sequence length
        summed = masked_hidden.sum(dim=1)
        
        # Divide by number of non-padding tokens
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        pooled = summed / lengths
        
        return pooled


class PolicyModel(nn.Module):
    """Policy model wrapper for A*-PO"""
    
    def __init__(self, base_model: QwenModel):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of policy model"""
        return self.base_model.forward(input_ids, attention_mask, labels)
    
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Generate using policy model"""
        return self.base_model.generate(input_ids, attention_mask, **kwargs)
    
    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities from policy model"""
        return self.base_model.get_log_probs(input_ids, attention_mask)
    
    def save_pretrained(self, path: str) -> None:
        """Save policy model"""
        self.base_model.save_pretrained(path)
    
    def load_pretrained(self, path: str) -> None:
        """Load policy model"""
        self.base_model.load_pretrained(path)
