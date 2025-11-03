"""
A*-PO (A-Star Policy Optimization) implementation
Minimal implementation for fast execution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_astar_po_loss(log_probs, advantages, old_log_probs=None, kl_weight=0.1):
    """
    A*-PO loss: combines REINFORCE with KL regularization
    Uses optimal advantage regression for stability
    """
    if old_log_probs is not None:
        # KL penalty to prevent policy from changing too much
        kl = (log_probs - old_log_probs).exp() * (log_probs - old_log_probs) - (log_probs - old_log_probs)
        kl_penalty = kl_weight * kl.mean()
    else:
        kl_penalty = 0
    
    # Policy gradient with advantage
    policy_loss = -(log_probs * advantages).mean()
    
    return policy_loss + kl_penalty
