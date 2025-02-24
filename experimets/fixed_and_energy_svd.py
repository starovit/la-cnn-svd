import copy
import numpy as np
from src.models import count_parameters, test_model
from src.compress import compress_model_with_svd, compress_model_with_svd_by_energy, compress_model_with_svd_by_energy_skip_n

original_model = copy.deepcopy(model)
original_params = count_parameters(original_model)
original_accuracy = test_model(original_model)

# Fixed-rank experiments
rank_values = list(range(20, 220, 20))
fixed_param_counts = []
fixed_accuracies = []
for rank in rank_values:
    print(f"Fixed Rank = {rank}")
    reduced_model = copy.deepcopy(model)
    reduced_model = compress_model_with_svd(reduced_model, rank)
    fixed_param_counts.append(count_parameters(reduced_model))
    fixed_accuracies.append(test_model(reduced_model))
fixed_compression_percentages = [100 * (1 - (p / original_params)) for p in fixed_param_counts]

# Energy-based experiments
energy_thresholds = np.concatenate((np.arange(0.2, 0.8, 0.2),  np.arange(0.8, 1, 0.02)), axis=None)
skip_options = [0, 1, 2, 3]
energy_results = []
for threshold in energy_thresholds:
    for skip in skip_options:
        print(f"Energy-based compression with threshold={threshold} and skipping first {skip} conv layer{'s' if skip > 1 else ''}")
        reduced_model_energy = copy.deepcopy(model)
        reduced_model_energy = compress_model_with_svd_by_energy_skip_n(reduced_model_energy, energy_threshold=threshold, skip_count=skip)
        param_count = count_parameters(reduced_model_energy)
        accuracy = test_model(reduced_model_energy)
        compression_pct = 100 * (1 - (param_count / original_params))
        energy_results.append({
            'energy_threshold': threshold,
            'skip_layers': skip,
            'compression_pct': compression_pct,
            'accuracy': accuracy * 100  # as percentage
        })