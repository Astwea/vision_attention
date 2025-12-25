"""Probabilistic router for attention-based resource allocation.

This module implements probabilistic routing where attention scores are converted
to probabilities, and patches are randomly sampled to use Large MLP (fine-grained)
or Small MLP (coarse-grained) based on these probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class GumbelSoftmaxRouter(nn.Module):
    """
    Probabilistic router using Gumbel-Softmax for differentiable sampling.
    
    Converts attention scores to probabilities and uses Gumbel-Softmax
    to sample routing decisions in a differentiable way.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        hard: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            hard: If True, use hard sampling in forward, soft in backward
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.eps = eps
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with probabilistic routing.
        
        Args:
            attention_scores: Attention scores (B, N_patches)
            return_stats: If True, return routing statistics
            inference_mode: Inference mode ("sampling", "expectation", "top_p", "hard_threshold")
            
        Returns:
            routing_mask: Binary mask (B, N_patches), 1 = Large MLP, 0 = Small MLP
            stats: Dictionary with routing statistics
        """
        B, N = attention_scores.shape
        
        # Convert attention scores to probabilities
        probs = torch.sigmoid(attention_scores)  # (B, N_patches) in (0, 1)
        
        if self.training and inference_mode == "sampling":
            # Training: Use Gumbel-Softmax sampling
            routing_mask = self._gumbel_softmax_sample(probs)
        else:
            # Inference: Use deterministic strategy
            if inference_mode == "expectation":
                # Use probabilities directly (soft routing)
                routing_mask = probs
            elif inference_mode == "top_p":
                # Select top-p% patches (default p=0.3)
                top_p = 0.3
                k = max(1, int(N * top_p))
                _, top_indices = torch.topk(probs, k, dim=1)
                routing_mask = torch.zeros_like(probs)
                routing_mask.scatter_(1, top_indices, 1.0)
            elif inference_mode == "hard_threshold":
                # Hard threshold at 0.5
                routing_mask = (probs > 0.5).float()
            else:  # "sampling" in eval mode
                # Still use sampling but deterministic with fixed seed
                routing_mask = self._gumbel_softmax_sample(probs)
        
        stats = {}
        if return_stats:
            stats['routing_ratio'] = routing_mask.mean().item()
            stats['probs_mean'] = probs.mean().item()
            stats['probs_std'] = probs.std().item()
            stats['probs'] = probs.detach().cpu()
        
        return routing_mask, stats
    
    def _gumbel_softmax_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.
        
        Args:
            probs: Probabilities (B, N_patches)
            
        Returns:
            Binary mask (B, N_patches)
        """
        # Sample Gumbel noise
        uniform = torch.rand_like(probs)
        gumbel_noise = -torch.log(-torch.log(uniform + self.eps) + self.eps)
        
        # Convert to logits
        logits = torch.log(probs + self.eps) - torch.log(1 - probs + self.eps)
        logits = logits + gumbel_noise
        
        # Gumbel-Softmax
        if self.hard:
            # Hard sampling: argmax in forward, softmax in backward
            y_soft = F.softmax(logits / self.temperature, dim=-1)
            y_hard = (y_soft > 0.5).float()  # Binary decision
            # Straight-through estimator
            routing_mask = y_hard - y_soft.detach() + y_soft
        else:
            # Soft sampling
            routing_mask = F.softmax(logits / self.temperature, dim=-1)
        
        return routing_mask


class StraightThroughRouter(nn.Module):
    """
    Probabilistic router using Straight-Through Estimator.
    
    Uses hard random sampling in forward pass, soft gradients in backward pass.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        threshold: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            temperature: Temperature for soft gradient
            threshold: Threshold for hard decision
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.eps = eps
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Straight-Through probabilistic routing.
        
        Args:
            attention_scores: Attention scores (B, N_patches)
            return_stats: If True, return routing statistics
            inference_mode: Inference mode ("sampling", "expectation", "top_p", "hard_threshold")
            
        Returns:
            routing_mask: Binary mask (B, N_patches), 1 = Large MLP, 0 = Small MLP
            stats: Dictionary with routing statistics
        """
        B, N = attention_scores.shape
        
        # Convert attention scores to probabilities
        probs = torch.sigmoid(attention_scores)  # (B, N_patches) in (0, 1)
        
        if self.training and inference_mode == "sampling":
            # Training: Hard random sampling with soft gradients
            # Forward: Hard sampling
            uniform = torch.rand_like(probs)
            hard_mask = (uniform < probs).float()
            
            # Backward: Soft gradient through sigmoid
            soft_mask = torch.sigmoid((attention_scores - self.threshold) / self.temperature)
            
            # Straight-through estimator
            routing_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # Inference: Use deterministic strategy
            if inference_mode == "expectation":
                routing_mask = probs
            elif inference_mode == "top_p":
                top_p = 0.3
                k = max(1, int(N * top_p))
                _, top_indices = torch.topk(probs, k, dim=1)
                routing_mask = torch.zeros_like(probs)
                routing_mask.scatter_(1, top_indices, 1.0)
            elif inference_mode == "hard_threshold":
                routing_mask = (probs > 0.5).float()
            else:  # "sampling" in eval mode
                # Deterministic sampling based on probabilities
                uniform = torch.rand_like(probs)
                routing_mask = (uniform < probs).float()
        
        stats = {}
        if return_stats:
            stats['routing_ratio'] = routing_mask.mean().item()
            stats['probs_mean'] = probs.mean().item()
            stats['probs_std'] = probs.std().item()
            stats['probs'] = probs.detach().cpu()
        
        return routing_mask, stats


class ProbabilisticRouter(nn.Module):
    """
    Unified probabilistic router that can use either Gumbel-Softmax or Straight-Through.
    """
    
    def __init__(
        self,
        mode: str = "gumbel_softmax",
        temperature: float = 1.0,
        threshold: float = 0.5,
        hard: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            mode: Router mode ("gumbel_softmax" or "straight_through")
            temperature: Temperature parameter
            threshold: Threshold for Straight-Through (if used)
            hard: Use hard sampling for Gumbel-Softmax (if used)
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.mode = mode
        
        if mode == "gumbel_softmax":
            self.router = GumbelSoftmaxRouter(
                temperature=temperature,
                hard=hard,
                eps=eps,
            )
        elif mode == "straight_through":
            self.router = StraightThroughRouter(
                temperature=temperature,
                threshold=threshold,
                eps=eps,
            )
        else:
            raise ValueError(f"Unknown router mode: {mode}")
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        return_stats: bool = False,
        inference_mode: str = "sampling",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass.
        
        Args:
            attention_scores: Attention scores (B, N_patches)
            return_stats: If True, return routing statistics
            inference_mode: Inference mode ("sampling", "expectation", "top_p", "hard_threshold")
            
        Returns:
            routing_mask: Binary mask (B, N_patches)
            stats: Dictionary with routing statistics
        """
        return self.router(attention_scores, return_stats, inference_mode)

