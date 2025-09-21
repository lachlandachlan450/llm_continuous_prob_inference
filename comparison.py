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
    n_samples: int = 2**11, 
    n_iterations: int = 1000,  
    stepsize: float = 0.1,
    noise_schedule: Optional[List[float]] = None,
    noise_schedule_iters: Optional[List[int]] = None,
    fluency_weight: float = 0.5,
    target_weight: float = 0.5,
    topk: int = 5,
    batch_size: int = 64,
    device: Optional[str] = None
) -> float:
    """
    COLD-style continuous space sampling for low probability estimation.
    Memory-efficient version with batching and cleanup.
    """
    if device is None:
        device = model.device
        
    if noise_schedule is None:
        noise_schedule = [1.0, 0.5, 0.1, 0.05, 0.01]
    if noise_schedule_iters is None:
        noise_schedule_iters = [0, 40, 80, 120, 160]  
    
    vocab_size = model.embed.d_vocab
    
  
    all_estimates = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="COLD batches"):
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        
  
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
        
      
        particles = torch.randn(current_batch_size, vocab_size, device=device) * temp
        
      
        with torch.no_grad():
            x = model.embed(contexts)
            x = x + model.pos_embed(contexts)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x)
            model_logits = model.unembed(x[:, -1, :])
            model_probs = F.softmax(model_logits / temp, dim=-1)
        
        def get_noise_std(iteration: int) -> float:
            for i, iter_thresh in enumerate(noise_schedule_iters):
                if iteration < iter_thresh:
                    return noise_schedule[i-1] if i > 0 else noise_schedule[0]
            return noise_schedule[-1]
        
        def energy_function(particles: torch.Tensor) -> torch.Tensor:
            """Simplified energy function using precomputed model probs"""
         
            particle_probs = F.softmax(particles / temp, dim=-1)
            
        
            fluency_loss = F.kl_div(
                particle_probs.log(), 
                model_probs, 
                reduction='none'
            ).sum(-1)
            
           
            target_logprobs = F.log_softmax(particles, dim=-1)[:, target]
            target_loss = -target_logprobs
            
            return fluency_weight * fluency_loss + target_weight * target_loss
        
       
        for iteration in range(n_iterations):
            noise_std = get_noise_std(iteration)
            
        
            particles = particles.detach().requires_grad_(True)
            
            energy = energy_function(particles)
            energy_total = energy.sum()
            
  
            energy_total.backward()
            gradients = particles.grad.clone()
            
 
            with torch.no_grad():
                noise = torch.randn_like(particles) * noise_std
                particles = particles - stepsize * gradients + noise
            
    
            particles.grad = None
            

            if iteration % 20 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        

        with torch.no_grad():
            final_probs = F.softmax(particles / temp, dim=-1)
            target_probs = final_probs[:, target]
            batch_estimate = target_probs.mean().item()
            all_estimates.append(batch_estimate)
        

        del particles, contexts, model_probs, model_logits
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    

    return np.mean(all_estimates)



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
                    n_samples=512,  
                    n_iterations=100,
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
