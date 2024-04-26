import torch

def first_inf_indices(tensor):
    """
    # Example tensor
    tensor = torch.tensor([[1,2,4], [1, 2, float('-inf')], [0, float('-inf'), float('-inf')], [float('-inf'), float('-inf'), float('-inf')]])

    # Find the first index of -inf in each row
    indices = first_inf_indices(tensor)
    print(indices.tolist())  # Output: [3, 2, 1, 0]
    """
    # Check if tensor contains -inf
    inf_mask = tensor == -1
    # Find the indices where -inf occurs
    inf_indices = torch.argmax(inf_mask.int(), dim=1)
    # Replace indices where -inf doesn't occur with seq_len
    inf_indices.masked_fill_(inf_mask.sum(dim=1) == 0, tensor.size(1))
    return inf_indices


def vectorized_update_mem_2d(
    mem: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    """
    Update a 2D tensor 'mem' by appending values from 'update' tensor to each 'row' of 'mem'.
    This function operates in a queue-like fashion for each row.
    If a row has zeros (indicated by non_zero_counts), the first zero is replaced by the corresponding value from 'update'.
    If a row is full, it is shifted one place to the left, and the corresponding value from 'update' is added to the end.

    Parameters:
    mem (torch.Tensor): A 2D tensor of shape (n, seq_len) representing the memory to be updated.
    non_zero_counts (torch.Tensor): A 1D tensor of shape (n,) containing the count of non-zero 'elements' in each 'row' of 'mem'.
                                    Each element should be an integer.
    update (torch.Tensor): A 1D tensor of shape (n,) containing the values to be appended to each 'row' of 'mem'.

    Returns:
    torch.Tensor: The updated memory tensor of shape (n, seq_len).

    Note:
    - 'n' is the number of sequences, and 'seq_len' is the sequence length.
    - This function assumes that 'mem' has been appropriately padded with -infs where necessary.
    - The function operates in-place on 'mem', and the updated 'mem' tensor is also returned.
    # Example usage
    n = 3
    seq_len = 5
    mem = torch.rand(n, seq_len)  # Example 2D tensor
    print(mem)
    non_inf_counts = torch.tensor([2, 3, 5])  # Counts of non-inf elements in each row
    update = torch.rand(n)  # Update tensor
    print(update)
    updated_mem = vectorized_update_mem_2d(mem, non_inf_counts, update)
    print(updated_mem)
    """
    non_inf_counts = first_inf_indices(mem)
    n, seq_len = mem.shape

    # Mask for rows that are full and need shifting
    full_rows_mask = non_inf_counts >= seq_len

    # Shift the full rows
    mem[full_rows_mask] = torch.roll(mem[full_rows_mask], -1, dims=1)

    # Determine insertion index for each row
    insertion_index = non_inf_counts.clone()
    insertion_index[full_rows_mask] = seq_len - 1  # If row is full, set index to last position

    # Create a mask for insertion points
    insertion_mask = torch.zeros(n, seq_len, dtype=torch.bool, device=mem.device)
    insertion_mask.scatter_(1, insertion_index.unsqueeze(1), True)

    # Insert new values
    mem = torch.where(insertion_mask, update.unsqueeze(1), mem)

    return mem
