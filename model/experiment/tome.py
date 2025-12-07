
import torch

def bipartite_soft_matching(metric, r):
    """
    Performs bipartite soft matching and finds connected components in the bipartite graph,
    assigning labels based on connections between A and B. Used for TOME.
    Args:
        metric (torch.Tensor): A similarity matrix of shape [num_tokens, feature_dim].
        r (int): The number of most similar edges to keep.
    Returns:
        labels (torch.Tensor): A tensor of shape [num_tokens] where labels[i] indicates the component id of node i.
    """
    num_tokens = metric.shape[0]
    device = metric.device
    if num_tokens % 2 == 1: # Handle odd number of tokens by padding
        # Pad with a zero vector and assign it a unique label
        metric = torch.cat([metric, torch.zeros(1, metric.shape[1], device=device)], dim=0)
        num_tokens += 1
        padded = True
    else:
        padded = False

    A_indices = torch.arange(0, num_tokens, 2, device=device)
    B_indices = torch.arange(1, num_tokens, 2, device=device)

    # Calculate similarity between A and B sets
    A_to_B_sim = torch.matmul(metric[A_indices], metric[B_indices].T) 

    # Find the most similar B for each A
    most_similar_B = A_to_B_sim.argmax(dim=-1)  
    similarity_values = A_to_B_sim.max(dim=-1)[0]

    # Keep the top-r most similar pairs
    top_r_similarities, top_r_indices = torch.topk(similarity_values, min(r, len(similarity_values)), largest=True)
    
    A_top_r = A_indices[top_r_indices]
    B_top_r = B_indices[most_similar_B[top_r_indices]]

    # Initialize labels: each token is its own component initially
    labels = torch.arange(num_tokens if not padded else num_tokens - 1, device=device) 
    
    # Merge components: Assign labels of selected A points to their matched B points
    labels[A_top_r] = labels[B_top_r]
    
    if padded:
        labels = labels[:-1] # Remove the label for the padded token
    
    return labels

def merge_connected_components(X, sizes, labels):
    """
    Merge tokens within each connected component using the labels.
    Args:
        X : Input tensor of tokens to merge (shape [num_tokens, dim]).
        sizes : Tensor of token sizes with shape [num_tokens]. (Used for weighted merging)
        labels : Tensor of component labels with shape [num_tokens].
    Returns:
        new_X : Merged tensor of tokens using the sizes for weighted merging.
        new_sizes : Updated sizes of the merged tokens.
        mask : Mask indicating which tokens were kept.
    """
    device = X.device
    num_tokens, dim = X.shape
    # Handle potential negative labels from padding logic if any (though not expected here)
    valid_label_mask = labels >= 0 
    if not valid_label_mask.all():
        # This part handles if labels were -1 for some reason, though standard TOME shouldn't produce that
        # For simplicity in this context, we might skip or handle differently, 
        # but let's assume labels are valid indices or IDs for merging.
        # If needed, adjust logic here. For standard TOME output, this should be fine.
        pass

    # Use unique labels to identify components
    unique_labels, inverse_indices = torch.unique(labels, sorted=True, return_inverse=True)
    num_components = unique_labels.size(0)

    # Calculate component sizes by summing the 'sizes' of tokens in each component
    component_sizes = torch.zeros(num_components, device=device, dtype=sizes.dtype).scatter_add(
        0, inverse_indices, sizes
    )

    # Calculate weighted sum for merging
    weighted_X = X * sizes.unsqueeze(-1)
    component_weighted_sum = torch.zeros(num_components, dim, device=device, dtype=weighted_X.dtype).scatter_add(
        0, inverse_indices.unsqueeze(-1).expand(-1, dim), weighted_X
    )

    # Compute merged token representations (weighted average)
    # Add a small epsilon to component_sizes to avoid division by zero if any component has size 0 (unlikely)
    merged_tokens = component_weighted_sum / (component_sizes.unsqueeze(-1) + 1e-8)

    # Prepare outputs
    new_X = torch.zeros_like(X) 
    new_sizes = torch.zeros_like(sizes) 
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device) 

    # Scatter the merged results back to the original token positions
    # Only the representative token (identified by unique_labels) in each component gets the merged value
    new_X[unique_labels] = merged_tokens.to(new_X.dtype)
    new_sizes[unique_labels] = component_sizes
    mask[unique_labels] = True 

    return new_X, new_sizes, mask


class TOME:
    def __init__(self, r):
        """
        Args:
            r (int): The number of most similar edges to keep for merging.
        """
        self.r = r

    def fit(self, X_clustering, X_merge, sizes):
        """
        Applies TOME token merging by finding connected components based on the r most similar edges.
        Args:
            X_clustering (torch.Tensor): Input tensor used to calculate similarity (shape is [num_tokens, feature_dim]).
            X_merge (torch.Tensor): Input tensor of tokens that will be merged (shape is [num_tokens, feature_dim]).
            sizes (torch.Tensor): Tensor containing the sizes of tokens (shape is [num_tokens]).
        Returns:
            new_X (torch.Tensor): Tensor with the merged tokens.
            mask (torch.Tensor): Mask indicating which tokens were kept and which were discarded.
            new_sizes (torch.Tensor): Updated sizes of the tokens.
        """
        num_tokens = X_clustering.shape[0]
        # Ensure r is within valid bounds
        self.r = max(min(self.r, num_tokens // 3), 0) 
        
        # Normalize the token representations for similarity calculation (cosine similarity)
        X_clustering_norm = X_clustering / (X_clustering.norm(dim=-1, keepdim=True) + 1e-6) # Add eps to avoid div by zero
        
        # Build the bipartite graph and assign labels based on connections
        labels = bipartite_soft_matching(X_clustering_norm, self.r)
        
        # Merge tokens within each connected component using the labels
        merged_X, new_sizes, mask = merge_connected_components(X_merge, sizes, labels)
        
        return merged_X, mask, new_sizes

def reduce_tokens_per_frame_tome(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduces the number of tokens per frame by half using the ToMe (Token Merging) algorithm.
    Assumes the input tensor shape is [frame_number, token_per_frame, channel].

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [frame_number, token_per_frame, channel].

    Returns:
        torch.Tensor: Output tensor with half the tokens per frame, shape [frame_number, token_per_frame//2, channel].
    """
    if input_tensor.dim() != 3:
        raise ValueError("Input tensor must have 3 dimensions: [frame_number, token_per_frame, channel]")
    
    batch_size, num_tokens, channels = input_tensor.shape

    if num_tokens % 3 != 0:
         raise ValueError("Number of tokens per frame must be even for this simple halving implementation.")

    # Determine the number of tokens to merge (keep half)
    r = num_tokens // 3

    # Initialize ToMe model
    tome_model = TOME(r=r)

    # Prepare output list
    output_frames = []

    # Process each frame independently
    for i in range(batch_size):
        frame_features = input_tensor[i]  # Shape: [num_tokens, channels]
        
        # For clustering, we use the features themselves
        X_clustering = frame_features 
        # The features to be merged are the same
        X_merge = frame_features 
        # Initialize sizes as ones (uniform weighting)
        sizes = torch.ones(num_tokens, device=frame_features.device, dtype=frame_features.dtype) 

        # Apply TOME
        try:
            merged_features, mask, new_sizes = tome_model.fit(X_clustering, X_merge, sizes)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            # In case of error, return the original frame (or handle as needed)
            output_frames.append(frame_features)
            continue
            
        # Select only the merged/kept tokens using the mask
        reduced_frame = merged_features[mask] # Shape: [num_tokens - r, channels] = [num_tokens//2, channels]
        
        # Optional: Check if the output size is as expected
        # assert reduced_frame.shape[0] == num_tokens // 2, f"Frame {i} reduction failed: expected {num_tokens//2}, got {reduced_frame.shape[0]}"
        
        output_frames.append(reduced_frame)

    # Stack the reduced frames back into a batch tensor
    # All frames should now have `num_tokens // 2` tokens
    try:
        output_tensor = torch.stack(output_frames, dim=0) # Shape: [batch_size, num_tokens//2, channels]
    except RuntimeError as e: # Catch if frames have different numbers of tokens after reduction (shouldn't happen with TOME)
        print(f"Error stacking frames: {e}. Frames might have different token counts after reduction.")
        raise e

    return output_tensor


