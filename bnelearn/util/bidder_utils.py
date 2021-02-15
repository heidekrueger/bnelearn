def get_welfare(allocations, valuations):
    """
    Args:
        allocations ([torch.Tensor])
        valuations ([torch.Tensor])

    Returns:
        [torch.Tensor] 
    """
    assert allocations.dim() == 2 # batch_size x items
    item_dimension = valuations.dim() - 1
    welfare = (valuations * allocations).sum(dim=item_dimension)
    return welfare