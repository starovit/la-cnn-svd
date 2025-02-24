import torch
import torch.nn as nn
import numpy as np

def apply_svd_to_conv_layer(conv_layer, rank=10):
    """
    Applies fixed-rank SVD to decompose a Conv2d layer into two layers.
    """
    if not isinstance(conv_layer, nn.Conv2d):
        return conv_layer

    weight = conv_layer.weight.data.cpu().numpy()  # shape: (N, C, d, d)
    out_channels, in_channels, d1, d2 = weight.shape
    assert d1 == d2, "Only square kernels are supported."
    d = d1

    stride = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride, conv_layer.stride)
    padding = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding, conv_layer.padding)

    # Reshape and perform SVD
    weight_transposed = np.transpose(weight, (1, 2, 3, 0))
    weight_matrix = weight_transposed.reshape(in_channels * d, d * out_channels)
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    effective_rank = min(rank, U.shape[1])

    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vt_r = Vt[:effective_rank, :]
    sqrt_S_r = np.sqrt(S_r)
    lower_factor = U_r * sqrt_S_r[np.newaxis, :]
    upper_factor = Vt_r * sqrt_S_r[:, np.newaxis]

    lower_tensor = lower_factor.reshape(in_channels, d, 1, effective_rank)
    lower_tensor = np.transpose(lower_tensor, (3, 0, 1, 2))
    upper_tensor = upper_factor.reshape(effective_rank, d, out_channels)
    upper_tensor = np.transpose(upper_tensor, (2, 0, 1))
    upper_tensor = upper_tensor[:, :, np.newaxis, :]

    lower_tensor = torch.tensor(lower_tensor, dtype=conv_layer.weight.dtype, device=conv_layer.weight.device)
    upper_tensor = torch.tensor(upper_tensor, dtype=conv_layer.weight.dtype, device=conv_layer.weight.device)

    lower_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=effective_rank,
        kernel_size=(d, 1),
        stride=1,
        padding=(padding[0], 0),
        bias=False
    )
    lower_conv.weight.data = lower_tensor

    upper_conv = nn.Conv2d(
        in_channels=effective_rank,
        out_channels=conv_layer.out_channels,
        kernel_size=(1, d),
        stride=stride,
        padding=(0, padding[1]),
        bias=True
    )
    upper_conv.weight.data = upper_tensor
    if conv_layer.bias is not None:
        upper_conv.bias.data = conv_layer.bias.data

    return nn.Sequential(lower_conv, upper_conv)

def compress_model_with_svd(model, rank=10):
    """
    Iterates over model.features and replaces every Conv2d layer with its fixed-rank SVD-decomposed version.
    """
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            decomposed = apply_svd_to_conv_layer(module, rank)
            setattr(model.features, name, decomposed)
    return model

def apply_svd_to_conv_layer_by_energy(conv_layer, energy_threshold=0.9):
    """
    Applies SVD to a Conv2d layer and selects the effective rank dynamically so that the
    cumulative energy (sum of squared singular values) is at least energy_threshold.
    """
    if not isinstance(conv_layer, nn.Conv2d):
        return conv_layer

    weight = conv_layer.weight.data.cpu().numpy()  # shape: (N, C, d, d)
    out_channels, in_channels, d1, d2 = weight.shape
    assert d1 == d2, "Only square kernels supported."
    d = d1

    stride = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride, conv_layer.stride)
    padding = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding, conv_layer.padding)

    weight_transposed = np.transpose(weight, (1, 2, 3, 0))
    weight_matrix = weight_transposed.reshape(in_channels * d, d * out_channels)
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

    # Compute energy and determine effective rank
    energy_vals = S**2
    total_energy = np.sum(energy_vals)
    cumulative_energy = np.cumsum(energy_vals)
    effective_rank = int(np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1)
    print(f"Layer shape {weight.shape}: Energy threshold {energy_threshold} -> Effective rank = {effective_rank}")

    U_r = U[:, :effective_rank]
    S_r = S[:effective_rank]
    Vt_r = Vt[:effective_rank, :]
    sqrt_S_r = np.sqrt(S_r)
    lower_factor = U_r * sqrt_S_r[np.newaxis, :]
    upper_factor = Vt_r * sqrt_S_r[:, np.newaxis]

    lower_tensor = lower_factor.reshape(in_channels, d, 1, effective_rank)
    lower_tensor = np.transpose(lower_tensor, (3, 0, 1, 2))
    upper_tensor = upper_factor.reshape(effective_rank, d, out_channels)
    upper_tensor = np.transpose(upper_tensor, (2, 0, 1))
    upper_tensor = upper_tensor[:, :, np.newaxis, :]

    lower_tensor = torch.tensor(lower_tensor, dtype=conv_layer.weight.dtype, device=conv_layer.weight.device)
    upper_tensor = torch.tensor(upper_tensor, dtype=conv_layer.weight.dtype, device=conv_layer.weight.device)

    lower_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=effective_rank,
        kernel_size=(d, 1),
        stride=1,
        padding=(padding[0], 0),
        bias=False
    )
    lower_conv.weight.data = lower_tensor

    upper_conv = nn.Conv2d(
        in_channels=effective_rank,
        out_channels=conv_layer.out_channels,
        kernel_size=(1, d),
        stride=stride,
        padding=(0, padding[1]),
        bias=True
    )
    upper_conv.weight.data = upper_tensor
    if conv_layer.bias is not None:
        upper_conv.bias.data = conv_layer.bias.data

    return nn.Sequential(lower_conv, upper_conv)

def compress_model_with_svd_by_energy(model, energy_threshold=0.9):
    """
    Iterates over model.features and replaces each Conv2d layer with an SVD-decomposed version
    where the effective rank is chosen dynamically to preserve at least energy_threshold of the energy.
    """
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            decomposed = apply_svd_to_conv_layer_by_energy(module, energy_threshold)
            setattr(model.features, name, decomposed)
    return model

def compress_model_with_svd_by_energy_skip_n(model, energy_threshold=0.9, skip_count=1):
    """
    Similar to compress_model_with_svd_by_energy but skips compression for the first `skip_count`
    Conv2d layers in model.features.
    """
    conv_count = 0
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            conv_count += 1
            if conv_count <= skip_count:
                print(f"Skipping compression for layer {name} (frozen first {skip_count} conv layer{'s' if skip_count > 1 else ''})")
                continue
            decomposed = apply_svd_to_conv_layer_by_energy(module, energy_threshold)
            setattr(model.features, name, decomposed)
    return model
