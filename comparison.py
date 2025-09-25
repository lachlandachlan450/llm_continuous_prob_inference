import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from lpe.methods import QLD, ITGIS, MHIS
from lpe.method_utils import *
from lpe.utils import Transformer
from lpe.utils import datasets as lpe_datasets



def COLD(
    model,
    input_dists,
    target: int,
    temp: float = 1.0,
    n_samples: int = 2**13,
    n_iterations: int = 300, 
    stepsize: float = 0.1,
    noise_schedule: Optional[List[float]] = None,
    noise_schedule_iters: Optional[List[int]] = None,
    fluency_weight: float = 0.5,
    target_weight: float = 0.5,
    batch_size: int = 64, 
    device: Optional[str] = None
) -> float:
    if device is None:
        device = model.device
        
    if noise_schedule is None:
        noise_schedule = [0.5, 0.3, 0.1, 0.05, 0.01]
    if noise_schedule_iters is None:
        noise_schedule_iters = [0, 50, 100, 150, 180]  #try edit and figure difference
    
    vocab_size = model.embed.d_vocab
    
    # Step 1: Optimize particles to find good sampling distribution
    all_importance_estimates = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    adaptive_stepsize = stepsize / np.sqrt(batch_size)
    
    for batch_idx in tqdm(range(n_batches), desc="COLD batches"):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
        # Sample contexts
        contexts = []
        for _ in range(current_batch_size):
            context = []
            for dist in input_dists:
                if hasattr(dist, 'sample'):
                    token = dist.sample()
                else:
                    token = torch.multinomial(dist, 1).item()
                context.append(token)
            contexts.append(torch.tensor(context, device=device))
        
        contexts = torch.stack(contexts) 
        
        # Initialize particles
        particles = torch.randn(current_batch_size, vocab_size, device=device) * 0.1
        
        # Get model predictions for this context
        with torch.no_grad():
            # Debug the contexts shape
            if batch_idx == 0:
                print(f"Debug - contexts.shape: {contexts.shape}")
            
            x = model.embed(contexts)  # Should be [current_batch_size, seq_len, d_model]
            x = x + model.pos_embed(contexts)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x)
            
            # Debug intermediate shapes
            if batch_idx == 0:
                print(f"Debug - x.shape after ln_final: {x.shape}")
            
            last_token_repr = x[:, -1, :]  # Shape: [current_batch_size, d_model]
            
            if batch_idx == 0:
                print(f"Debug - last_token_repr.shape: {last_token_repr.shape}")
            
            model_logits = model.unembed(last_token_repr)  # Should be [current_batch_size, vocab_size]
            
            if batch_idx == 0:
                print(f"Debug - model_logits.shape before processing: {model_logits.shape}")
            
            if model_logits.dim() == 3:
                print(f"Debug - model_logits is 3D, shape: {model_logits.shape}")
                # The unembed layer is returning [1, batch_size, vocab_size] instead of [batch_size, vocab_size]
                # We need to squeeze the first dimension and transpose
                model_logits = model_logits.squeeze(0)  # Remove the spurious batch dimension
                print(f"Debug - model_logits after squeeze: {model_logits.shape}")
                
                # Now model_logits should be [batch_size, vocab_size] = [32, 48262]
                # But if it's still [32, 48262] where 32 is treated as sequence length, we might need to transpose
                if model_logits.shape[0] == current_batch_size and model_logits.shape[1] == vocab_size:
                    # This is correct: [batch_size, vocab_size]
                    pass
                elif model_logits.shape[0] == vocab_size and model_logits.shape[1] == current_batch_size:
                    # Need to transpose: [vocab_size, batch_size] -> [batch_size, vocab_size]  
                    model_logits = model_logits.transpose(0, 1)
                    print(f"Debug - model_logits after transpose: {model_logits.shape}")
                elif model_logits.shape[1] == vocab_size:
                    # It's [32, 48262] which is [batch_size, vocab_size] - correct!
                    pass
                else:
                    print(f"Debug - unexpected model_logits shape: {model_logits.shape}")
                
            if batch_idx == 0:
                print(f"Debug - model_logits.shape after processing: {model_logits.shape}")
                
            model_probs = F.softmax(model_logits / temp, dim=-1)  # Should be [current_batch_size, vocab_size]
        
        def get_noise_std(iteration: int) -> float:
            for i, iter_thresh in enumerate(noise_schedule_iters):
                if iteration < iter_thresh:
                    return noise_schedule[i-1] if i > 0 else noise_schedule[0]
            return noise_schedule[-1]
        
        def energy_function(particles: torch.Tensor) -> torch.Tensor:
            particle_probs = F.softmax(particles / temp, dim=-1)
            
            # Energy should encourage high target probability while staying somewhat reasonable
            target_logprobs = F.log_softmax(particles / temp, dim=-1)[:, target]
            target_energy = -target_logprobs  # Lower energy when target prob is higher
            
            # Regularization to prevent completely unrealistic distributions
            fluency_penalty = F.kl_div(
                particle_probs.log(), 
                model_probs, 
                reduction='none'
            ).sum(-1)
            
            return target_energy + fluency_weight * fluency_penalty
        
        # Optimize particles to find good sampling distribution
        for iteration in range(n_iterations):
            noise_std = get_noise_std(iteration)
            
            particles = particles.detach().requires_grad_(True)
            
            energy = energy_function(particles)
            energy_mean = energy.mean()
            
            energy_mean.backward()
            gradients = particles.grad.clone()
            
            with torch.no_grad():
                noise = torch.randn_like(particles) * noise_std
                particles = particles - adaptive_stepsize * gradients + noise
                particles = torch.clamp(particles, -5*temp, 5*temp)
            
            particles.grad = None
            
            if iteration % 20 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Step 2: Sample from optimized distribution and compute estimates
        with torch.no_grad():
            # Get final optimized distribution q(x) 
            q_logits = particles.detach()
            q_probs = F.softmax(q_logits / temp, dim=-1)
            
            # Debug shapes
            if batch_idx == 0:
                print(f"Debug - current_batch_size: {current_batch_size}")
                print(f"Debug - model_probs.shape: {model_probs.shape}")
                print(f"Debug - q_probs.shape: {q_probs.shape}")
            
            # For each context in this batch, sample multiple times from the optimized distribution
            n_samples_per_context = 10  # Sample multiple times per optimized distribution
            batch_estimates = []
            
            for context_idx in range(current_batch_size):
                context_estimates = []
                
                for _ in range(n_samples_per_context):
                    # Sample a token from the optimized distribution q
                    sampled_token = torch.multinomial(q_probs[context_idx], 1).item()
                    
                    # Get the original model probability for this context-token pair
                    p_prob = model_probs[context_idx, sampled_token].item()
                    q_prob = q_probs[context_idx, sampled_token].item()
                    
                    if q_prob > 1e-10:  # Avoid division by zero
                        # Importance sampling: weight by p/q
                        importance_weight = p_prob / q_prob
                        
                        # Indicator: 1 if we sampled the target, 0 otherwise  
                        indicator = 1.0 if sampled_token == target else 0.0
                        
                        # The estimate for this sample
                        sample_estimate = importance_weight * indicator
                        context_estimates.append(sample_estimate)
                    else:
                        context_estimates.append(0.0)
                
                # Average estimates for this context
                if context_estimates:
                    batch_estimates.append(np.mean(context_estimates))
            
            all_importance_estimates.extend(batch_estimates)
        
        # Cleanup
        del particles, contexts, model_probs, model_logits
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Final importance sampling estimate
    if all_importance_estimates:
        return np.mean(all_importance_estimates)
    else:
        return 0.0
        
def run_lpe_with_cold():
    """Example usage following the LPE framework pattern"""
    
    model_name = "gelu-1l"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer.from_pretrained(model_name).to(device)

    dist_name = "camel"
    gt_freqs = load_ground_truth(model_name, [dist_name], device=device)[dist_name]
    gt_probs = gt_freqs / gt_freqs.sum()
    targets = pick_random_tokens(gt_freqs, 16, 1e-9, 1e-5)
    acts = gen_activ_samples(model, dist_name, n_samples=2**13, show_progress=True)

    
    methods = ["QLD", "COLD", "ITGIS", "MHIS"]
    estimates = {}
    orig_dists = distribution_registry[dist_name](model.tokenizer, device=model.device).input_dists(n_reps=N_REPS_DICT[dist_name])
    
    for method in methods:
        print(f"Computing estimates for {method}")
        estimates[method] = {}
        for target in tqdm(list(targets)):
            if method == "QLD":
                estimates[method][target] = QLD(model.unembed.W_U, acts, target)
            elif method == "ITGIS":
                estimates[method][target] = ITGIS(model, orig_dists, target, temp=RECOMMENDED_TEMPS[model_name]["ITGIS"][dist_name], n_samples=2**13)
            elif method == "MHIS":
                estimates[method][target] = MHIS(model, orig_dists, target, temp=RECOMMENDED_TEMPS[model_name]["MHIS"][dist_name], n_samples=2**13, burn_in=2**10)
            elif method == "COLD":
                estimates[method][target] = COLD(
                    model, 
                    orig_dists, 
                    target, 
                    temp=RECOMMENDED_TEMPS[model_name].get("COLD", {}).get(dist_name, 1.0),
                    n_samples=2**13, 
                    n_iterations=300,  
                    batch_size=32  
                )

 
    plt.figure(figsize=(12, 6))
    colors = {'QLD': 'orange', 'ITGIS': 'red', 'COLD': 'blue', 'MHIS': 'purple'}
    
    for method in methods:
        estimates_for_method = [estimates[method][target] for target in targets]
        plt.scatter(gt_probs[targets].cpu().numpy(), estimates_for_method, label=method, color=colors[method])


        zero_targets = list(filter(lambda target: estimates[method][target] == 0, targets))
        if zero_targets:
            plt.scatter(gt_probs[zero_targets].cpu().numpy(), [1e-9]*len(zero_targets), color=colors[method], marker='x')

    plt.plot([1e-9, 1e-5], [1e-9, 1e-5], label='ground truth', color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Ground Truth Probability')
    plt.ylabel('Estimate')
    plt.title(f"Estimates for {dist_name} on {model_name}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_lpe_with_cold()
