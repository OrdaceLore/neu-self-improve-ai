"""
A*-PO (A-Star Policy Optimization) implementation
Same as Week 6, reused for RAGEN
"""
import torch

def compute_astar_po_loss(log_probs, advantages, old_log_probs=None, kl_weight=0.1):
    """A*-PO loss with KL regularization"""
    if old_log_probs is not None:
        kl = (log_probs - old_log_probs).exp() * (log_probs - old_log_probs) - (log_probs - old_log_probs)
        kl_penalty = kl_weight * kl.mean()
    else:
        kl_penalty = 0
    policy_loss = -(log_probs * advantages).mean()
    return policy_loss + kl_penalty
