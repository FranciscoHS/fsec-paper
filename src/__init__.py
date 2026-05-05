"""
Steering Activation Plateaus - Core modules

Usage:
    from src.model import load_model, get_layer0_output, forward_from_layer1
    from src.paths import slerp_path, optimize_path
    from src.analysis import compute_velocity_profile, compute_top_singular_values
    from src.visualization import plot_velocity_profiles
"""

from .model import load_model, get_layer0_output, forward_from_layer1, get_logits_at_point
from .paths import slerp_path, linear_path, optimize_path, compute_path_length
from .analysis import (
    compute_velocity_profile,
    peak_to_average_ratio,
    compute_entropy,
    get_top_k_predictions,
    compute_directional_sensitivity,
    sample_random_sensitivities,
    compute_top_singular_values,
    analyze_path_semantics,
)
from .visualization import (
    plot_velocity_profiles,
    plot_path_lengths,
    plot_singular_values,
    plot_optimization_history,
    plot_sensitivity_histogram,
)
