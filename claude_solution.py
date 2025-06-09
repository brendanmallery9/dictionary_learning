import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

'''
def solve_for_sparse_coefficients(
    model,
    base_point_tensor,
    mapping_batch,
    device,
    tolerance,
    sparse_reg):

    max_iter = 5000
    epsilon = 1e-9
    lr = 1e-2
    no_heads = model.no_heads
    batch_size = mapping_batch.shape[0]

    total_start = time.time()

    # Model forward pass stays on GPU
    with torch.no_grad():
        output = model(base_point_tensor)  # (no_heads, supp_size, dim)
    
    # === Move everything to CPU for optimization ===
    lambda_batch = torch.full((batch_size, no_heads), 1.0 / no_heads, device="cpu", requires_grad=True)
    mapping_batch_cpu = mapping_batch.detach().cpu()
    output_cpu = output.detach().cpu()

    cpu_start = time.time()

    for i in range(max_iter):
        with torch.no_grad():
            weights = 1 / (lambda_batch + epsilon)

        loss = adaptive_LASSO_loss(lambda_batch, weights, mapping_batch_cpu, output_cpu, sparse_reg)
        loss.backward()

        grad_norm = lambda_batch.grad.norm()
        if grad_norm < tolerance:
            print(f'Coefficient converged at iteration {i}')
            break

        with torch.no_grad():
            lambda_batch -= lr * lambda_batch.grad
            lambda_batch = project_to_simplex_batch(lambda_batch)

        lambda_batch.requires_grad_()
        lambda_batch.grad = None

    cpu_end = time.time()

    # === Move result back to original device ===
    lambda_batch_gpu = lambda_batch.detach().to(device)

    total_end = time.time()

    print(f"Optimization time on CPU: {cpu_end - cpu_start:.4f} sec")
    print(f"Total time including transfers: {total_end - total_start:.4f} sec")
    return lambda_batch_gpu
'''

def adaptive_LASSO_loss_optimized(lambda_batch, weights, mapping_batch, model_output, reg):
    """Optimized version with reduced memory allocations and vectorized ops"""
    # Pre-compute einsum once
    cvx_combo = torch.einsum('bh, shd -> bsd', lambda_batch, model_output)
    
    # Compute both terms efficiently
    l2_loss = 0.5 * torch.sum((cvx_combo - mapping_batch) ** 2)
    l1_penalty = reg * torch.sum(lambda_batch * weights)
    
    return l2_loss + l1_penalty

def project_to_simplex_batch_optimized(v):
    """Optimized simplex projection with fewer operations"""
    batch_size, n_dim = v.shape
    
    # Sort once
    v_sorted, _ = torch.sort(v, dim=1, descending=True)
    
    # Vectorized cumsum and threshold computation
    cumsum_v = torch.cumsum(v_sorted, dim=1)
    arange_tensor = torch.arange(1, n_dim + 1, device=v.device, dtype=v.dtype)
    
    # Broadcasting for efficiency
    thresholds = (cumsum_v - 1) / arange_tensor.view(1, -1)
    valid_mask = v_sorted > thresholds
    
    # Find rho efficiently using torch.sum instead of explicit indexing
    rho = torch.sum(valid_mask, dim=1) - 1
    
    # Extract theta values
    theta = thresholds[torch.arange(batch_size), rho].unsqueeze(1)
    
    return torch.clamp(v - theta, min=0.0)


def identify_smallest_k_coefficients(coefficients_batch, k):
    """
    Identify the k smallest coefficients for each sample in the batch.
    
    Args:
        coefficients_batch: (batch_size, no_heads)
        k: number of smallest coefficients to identify
    
    Returns:
        mask: (batch_size, no_heads) boolean tensor, True where coefficients should be zeroed
    """
    batch_size, no_heads = coefficients_batch.shape
    
    # Get the k smallest values for each sample (column index)
    _, smallest_indices = torch.topk(coefficients_batch, k, dim=1, largest=False)
    
    # Create mask empty mask
    mask = torch.zeros_like(coefficients_batch, dtype=torch.bool)
    row_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1), row index
    mask[row_indices, smallest_indices] = True #populate mask 
    return mask



def solve_reduced_sparse_problem(
    model,
    base_point_tensor,
    mapping_batch,
    coefficients_batch,
    active_heads,
    device,
    simplex_constraint,
    sparse_reg,
    precompute_gram,
    tolerance=1e-6):
    
    use_weighted_l1=True
    max_iter=600
    epsilon = 1e-7
    gram_matrix_reg = 1e-7
    batch_size = mapping_batch.shape[0]
    n_active_heads = active_heads.sum().item()
    
    # Model forward pass - get only active heads
    with torch.no_grad():
        full_output = model(base_point_tensor)  # (supp_size, no_heads, dim)
        output = full_output[:, active_heads, :]  # (supp_size, n_active_heads, dim)
    # Move to CPU for optimization
    coefficients_batch = coefficients_batch.clone().detach().cpu().requires_grad_(True)
    mapping_batch_cpu = mapping_batch.detach().cpu().float()
    output_cpu = output.detach().cpu().float()

    # Precompute Gram matrix for the reduced problem
    gram_matrix = None
    output_batch_prod = None
    output_reshaped = None
    
    if use_weighted_l1 and output_cpu.shape[0] * output_cpu.shape[2] > n_active_heads**2:
        # Compute reduced Gram matrix
        supp_size, n_active_heads, dim = output_cpu.shape
        output_reshaped = output_cpu.permute(1, 0, 2).reshape(n_active_heads, supp_size * dim).T
        gram_matrix = torch.mm(output_reshaped.T, output_reshaped) + gram_matrix_reg * torch.eye(n_active_heads)
        # Precompute mapping_batch * output interaction
        mapping_batch_flat = mapping_batch_cpu.reshape(batch_size, supp_size * dim)
        output_batch_prod = torch.mm(mapping_batch_flat, output_reshaped)
    # Use L-BFGS for optimization
    #if use_weighted_l1:
    #    optimizer = torch.optim.LBFGS([coefficients_batch], lr=1.0, max_iter=20, 
    #                              line_search_fn='strong_wolfe')
    #else:
    #    optimizer = torch.optim.AdamW([coefficients_batch], lr=0.01, weight_decay=0.0)
    optimizer = torch.optim.AdamW([coefficients_batch], lr=0.01, weight_decay=0.0)

    def closure():
        optimizer.zero_grad()
        
        #if gram_matrix is not None:
        if use_weighted_l1==True:
            # Use precomputed Gram matrix (reduced version)
            quadratic_terms = torch.sum(coefficients_batch * torch.mm(coefficients_batch, gram_matrix), dim=1)
            quadratic_term = 0.5 * torch.sum(quadratic_terms)
            
            linear_term = -torch.sum(coefficients_batch * output_batch_prod)
            data_norm = 0.5 * torch.sum(mapping_batch_cpu ** 2)
            l2_loss = quadratic_term + linear_term + data_norm
        else:
            quadratic_term = torch.einsum('bh,shd->bsd', coefficients_batch, output_cpu)

            # Compute L2 reconstruction loss
            residual = mapping_batch_cpu - quadratic_term
            l2_loss = 0.5 * torch.sum(residual ** 2)

            
        # Choose L1 regularization type
        if use_weighted_l1==True:
            # Weighted L1: weights inversely proportional to current coefficient magnitude
            weights = 1 / torch.clamp(torch.abs(coefficients_batch), min=epsilon)
            l1_penalty = sparse_reg * torch.sum(torch.abs(coefficients_batch) * weights)
        else:
            # Standard L1 regularization
            l1_penalty = sparse_reg * torch.sum(torch.abs(coefficients_batch))
        
        total_loss = l2_loss + l1_penalty
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([coefficients_batch], max_norm=10.0)
        return total_loss
    
    # Optimization loop
    prev_loss = float('inf')
    for i in range(max_iter):
        iter=i
        loss_val = optimizer.step(closure)
        # Project to simplex after each step
        if simplex_constraint==True:
            with torch.no_grad():
                coefficients_batch.data = project_to_simplex_batch_optimized(coefficients_batch.data)
        else:
            with torch.no_grad():
                coefficients_batch.data.clamp_(min=0.0)
        # Check convergence
        if abs(prev_loss - loss_val) < tolerance:
            break
        prev_loss = loss_val.item()
    
    #print('basis pursuit ended at {}'.format(iter))
    
    # Move result back to original device
    #result = coefficients_batch.detach().to(device)
        # Create full coefficient tensor with zeros for inactive heads
    no_heads = active_heads.shape[0]  # Total number of heads

    full_coefficients = torch.zeros(batch_size, no_heads, device=device)
    full_coefficients[:, active_heads] = coefficients_batch.detach().to(device)

    return full_coefficients

def unique_rows_with_inverse(mask_tensor):
    """
    Emulates torch.unique(mask_tensor, dim=0, return_inverse=True)
    using row hashing. Compatible with MPS.
    """
    # Flatten boolean rows to bytes
    flat = mask_tensor.float().to('cpu').numpy().astype(int)
    hashes = [tuple(row) for row in flat]

    # Map each unique row to an index
    unique_rows = list(dict.fromkeys(hashes))  # Preserves order
    row_to_index = {row: idx for idx, row in enumerate(unique_rows)}
    
    inverse_indices = torch.tensor([row_to_index[row] for row in hashes], dtype=torch.long, device=mask_tensor.device)
    unique_tensor = torch.tensor(unique_rows, dtype=mask_tensor.dtype, device=mask_tensor.device)
    
    return unique_tensor, inverse_indices

def solve_cpu_threaded(
    model, base_point_tensor, mapping_batch, coefficients_batch, 
    unique_masks, inverse_indices, solutions, device, simplex_constraint, sparse_reg, 
    precompute_gram, tolerance, n_groups, max_workers=None):
    """
    Threaded processing - works well for both CPU and GPU.
    """
    
    if max_workers is None:
        # For GPU, you might want fewer workers to avoid memory contention
        if device.type == 'cuda':
            max_workers = min(n_groups, 4)  # Conservative for GPU
        else:
            max_workers = min(n_groups, mp.cpu_count())
    
    def solve_single_group_cpu(group_idx):
        group_mask = (inverse_indices == group_idx)
        group_indices = torch.where(group_mask)[0]
        
        if len(group_indices) == 0:
            return group_idx, None, None
            
        current_sparsity_mask = unique_masks[group_idx]
        active_heads = ~current_sparsity_mask
        
        group_mapping_batch = mapping_batch[group_indices]
        group_coefficients = coefficients_batch[group_indices][:, active_heads]
        
        group_solutions = solve_reduced_sparse_problem(
            model, base_point_tensor, group_mapping_batch, group_coefficients,
            active_heads, device,simplex_constraint, sparse_reg, precompute_gram, tolerance)
        
        return group_idx, group_indices, group_solutions
    
    # Process groups in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(solve_single_group_cpu, range(n_groups)))
    
    # Collect results
    for group_idx, group_indices, group_solutions in results:
        if group_indices is not None:
            solutions[group_indices] = group_solutions
    
    return solutions

def solve_grouped_sparse_coefficients(
    model,
    base_point_tensor,
    mapping_batch,
    coefficients_batch,
    sparsity_mask,
    device,
    simplex_constraint,
    sparse_reg,
    precompute_gram,
    tolerance):
    
    """
    Solve sparse coefficients by grouping samples with identical sparsity patterns.
    """
    parallel=False
    batch_size, no_heads = coefficients_batch.shape
    
    unique_masks, inverse_indices = unique_rows_with_inverse(sparsity_mask)

    #unique_masks gives a list of the types of sparsity patterns
    #sparsity pattern is a boolean tensor, true indicates the entry will be zero'd out
    #inverse_indices gives the inverse mapping from type of mask to position in original batch
    #e.g. inverse_indices is 1 in all the rows with the first type of mask

    n_groups = len(unique_masks)
    
  #  print(f"Batch size: {batch_size}, Unique sparsity patterns: {n_groups}")
    
    # Initialize solution tensor
    solutions = torch.ones_like(coefficients_batch).to(device)
    solutions = solutions / solutions.sum(dim=1, keepdim=True)
    #print('number of groups:',n_groups)
    # Process each group
    if parallel==False:

        for group_idx in range(n_groups):
            # Get samples belonging to this group
            group_mask = (inverse_indices == group_idx)
            #finds the zero'd out indices
            group_indices = torch.where(group_mask)[0]
            
            #if len(group_indices) == 0:
            #    continue
                
            # Get the sparsity pattern for this group
            current_sparsity_mask = unique_masks[group_idx]  # (no_heads,)
            active_heads = ~current_sparsity_mask  # Which heads are active (not zeroed)
            #print('active_heads', active_heads)
            #n_active_heads = active_heads.sum().item()
            #if n_active_heads == 0:
                # All coefficients are zeroed - skip this group
            #    continue
            # Extract data for this group
            group_mapping_batch = mapping_batch[group_indices]  # (group_size, supp_size, dim)

            group_coefficients = coefficients_batch[group_indices][:, active_heads]  # (group_size, n_active_heads)
            # Solve the reduced problem for this group
            # l1 minimization with just active model heads, identified by group_mapping_batch
            group_solutions = solve_reduced_sparse_problem(
                    model, base_point_tensor, group_mapping_batch, group_coefficients,
                    active_heads, device,simplex_constraint, sparse_reg, precompute_gram, tolerance)
            solutions[group_indices] = group_solutions.clone()

    else:
        solutions=solve_cpu_threaded(
                        model, base_point_tensor, mapping_batch, coefficients_batch, 
                        unique_masks, inverse_indices, solutions, device,simplex_constraint, sparse_reg, 
                        precompute_gram, tolerance, n_groups, max_workers=None)
        # Place solutions back into the full tensor

    return solutions





def solve_for_sparse_coefficients_k_sparse(
    model,
    base_point_tensor,
    mapping_batch,
    coefficients_batch,
    device,
    simplex_constraint,
    sparse_reg,
    precompute_gram,
    k_sparse,
    tolerance=1e-6):
    
    """
    K-sparse version of the optimized solver using efficient batch grouping.
    Forces the k smallest coefficients to zero for each sample.
    
    Args:
        k_sparse: number of smallest coefficients to force to zero
    """
    
    # First, identify which coefficients to zero out
    with torch.no_grad():
        sparsity_mask = identify_smallest_k_coefficients(coefficients_batch, k_sparse)
    # Group samples by their sparsity pattern
    solutions = solve_grouped_sparse_coefficients(
        model, base_point_tensor, mapping_batch, coefficients_batch, 
        sparsity_mask, device,simplex_constraint, sparse_reg, precompute_gram, tolerance
    )
    return solutions


def solve_for_sparse_coefficients(
    model,
    base_point_tensor,
    mapping_batch,
    coefficients_batch,
    device,
    simplex_constraint,
    sparse_reg,
    precompute_gram,
    tolerance):
    
    #max_iter=2000
    max_iter = 400
    epsilon = 1e-7
    gram_matrix_reg = 1e-7
    no_heads = model.no_heads
    batch_size = mapping_batch.shape[0]

    with torch.no_grad():
        output = model(base_point_tensor)  # (supp_size, no_heads, dim)
    
    coefficients_batch = coefficients_batch.clone().detach().cpu().requires_grad_(True)
    mapping_batch_cpu = mapping_batch.detach().cpu().float()
    output_cpu = output.detach().cpu().float()

    gram_matrix = None
    output_batch_prod = None

    if precompute_gram and output_cpu.shape[0] * output_cpu.shape[2] > no_heads**2:
        supp_size, no_heads, dim = output_cpu.shape
        output_reshaped = output_cpu.permute(1, 0, 2).reshape(no_heads, supp_size * dim).T
        gram_matrix = torch.mm(output_reshaped.T, output_reshaped)
        gram_matrix += gram_matrix_reg * torch.eye(no_heads)
        mapping_batch_flat = mapping_batch_cpu.reshape(batch_size, supp_size * dim)
        output_batch_prod = torch.mm(mapping_batch_flat, output_reshaped)

    optimizer = torch.optim.LBFGS([coefficients_batch], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        if gram_matrix is not None:
            quad = torch.sum(coefficients_batch * torch.mm(coefficients_batch, gram_matrix), dim=1)
            quadratic_term = 0.5 * torch.sum(quad)
            linear_term = -torch.sum(coefficients_batch * output_batch_prod)
            data_norm = 0.5 * torch.sum(mapping_batch_cpu ** 2)
            l2_loss = quadratic_term + linear_term + data_norm
        else:
            # fallback (not implemented in your snippet)
            raise NotImplementedError("Direct computation path not implemented.")

        # Add L1 penalty with clipped weights for stability
        weights = 1 / torch.clamp(torch.abs(coefficients_batch), min=epsilon)
        l1_penalty = sparse_reg * torch.sum(torch.abs(coefficients_batch) * weights)

        total_loss = l2_loss + l1_penalty

        if torch.isnan(total_loss):
            print("NaN in loss. Skipping update.")
            total_loss = torch.tensor(1e10, requires_grad=True)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([coefficients_batch], max_norm=10.0)
        return total_loss

    prev_loss = float('inf')
    for i in range(max_iter):
        loss_val = optimizer.step(closure)
        if simplex_constraint==True:
            with torch.no_grad():
                coefficients_batch.data = project_to_simplex_batch_optimized(coefficients_batch.data)
        else:
            with torch.no_grad():
                coefficients_batch.data.clamp_(min=0.0)

        if abs(prev_loss - loss_val) < tolerance:
            break
        prev_loss = loss_val.item()
    return coefficients_batch.detach().to(device)



'''
def solve_for_sparse_coefficients_optimized(
    model,
    base_point_tensor,
    mapping_batch,
    coefficients_batch,
    device,
    sparse_reg,
    precompute_gram,
    tolerance=1e-6):
    
    """
    Optimized solver with multiple acceleration techniques:
    1. L-BFGS optimizer instead of SGD
    2. Precomputed Gram matrix when beneficial (optional)
    3. Adaptive learning rate
    4. Early stopping with better convergence criteria
    5. Fallback to direct computation when gram_matrix=False
    """
    
    max_iter = 2000  # Reduced since L-BFGS converges faster
    epsilon = 1e-7
    gram_matrix_reg = 1e-6
    no_heads = model.no_heads
    batch_size = mapping_batch.shape[0]
    
    total_start = time.time()
    
    # Model forward pass
    with torch.no_grad():
        output = model(base_point_tensor)  # (supp_size, no_heads, dim)
    
    # Move to CPU for optimization
    coefficients_batch = coefficients_batch.clone().detach().cpu().requires_grad_(True)
    mapping_batch_cpu = mapping_batch.detach().cpu().float()
    output_cpu = output.detach().cpu().float()
    
    # Precompute Gram matrix if it's computationally beneficial
    gram_matrix = None
    output_batch_prod = None
    output_reshaped = None
    
    if precompute_gram and output_cpu.shape[0] * output_cpu.shape[2] > no_heads**2:
        # output_cpu shape: (supp_size, no_heads, dim)
        # Reshape for efficient computation: (supp_size*dim, no_heads)
        supp_size, no_heads, dim = output_cpu.shape
        output_reshaped = output_cpu.permute(1, 0, 2).reshape(no_heads, supp_size * dim).T  # (supp_size*dim, no_heads)
        gram_matrix = torch.mm(output_reshaped.T, output_reshaped) + gram_matrix_reg * torch.eye(no_heads)  # (no_heads, no_heads)
        
        # Precompute mapping_batch * output interaction
        mapping_batch_flat = mapping_batch_cpu.reshape(batch_size, supp_size * dim)  # (batch_size, supp_size*dim)
        output_batch_prod = torch.mm(mapping_batch_flat, output_reshaped)  # (batch_size, no_heads)
    
    cpu_start = time.time()
    
    # Use L-BFGS for better convergence
    optimizer = torch.optim.LBFGS([coefficients_batch], lr=1.0, max_iter=20, 
                                    line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        
        if gram_matrix is not None:
            # Use precomputed Gram matrix for faster loss computation
            # coefficients_batch: (batch_size, no_heads), gram_matrix: (no_heads, no_heads)
            # Compute quadratic form: sum over batch of lambda_b^T * G * lambda_b
            quadratic_terms = torch.sum(coefficients_batch * torch.mm(coefficients_batch, gram_matrix), dim=1)  # (batch_size,)
            quadratic_term = 0.5 * torch.sum(quadratic_terms)
            
            linear_term = -torch.sum(coefficients_batch * output_batch_prod)
            data_norm = 0.5 * torch.sum(mapping_batch_cpu ** 2)
            l2_loss = quadratic_term + linear_term + data_norm
            
    prev_loss = float('inf')
    for i in range(max_iter):
        loss_val = optimizer.step(closure)
        
        # Project to simplex after each step
        with torch.no_grad():
            coefficients_batch.data = project_to_simplex_batch_optimized(coefficients_batch.data)
        
        # Check convergence
        if abs(prev_loss - loss_val) < tolerance:
            #print(f'L-BFGS converged at iteration {i}')
            break
        prev_loss = loss_val.item()
    
    
    # Move result back to original device
    coefficients_batch_result = coefficients_batch.detach().to(device)
    
    #print(f"Optimization time on CPU: {cpu_end - cpu_start:.4f} sec")
    #print(f"Total time including transfers: {total_end - total_start:.4f} sec")
    #print(f"Used {'Gram matrix' if gram_matrix is not None else 'direct'} computation")
    
    return coefficients_batch_result
'''