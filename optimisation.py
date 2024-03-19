import numpy as np
from itertools import combinations

# Generate synthetic data
np.random.seed(42)

# Define weights for the metric
weight_confidence = 0.6
weight_overlap = 0.4


# Define a function to calculate the metric
def calculate_metric(confidences, errors, weight_confidence, weight_overlap):
    metric = sum(
        weight_confidence * confidences - weight_overlap * overlap(confidences, errors)
    )
    return metric


# Define a function to calculate the overlap between intervals
def overlap(confidences, errors):
    overlaps = []
    for pair in combinations(range(len(confidences)), 2):
        i, j = pair
        overlap_area = max(
            0,
            min(confidences[i] + errors[i], confidences[j] + errors[j])
            - max(confidences[i] - errors[i], confidences[j] - errors[j]),
        )
        union_area = min(confidences[i] + errors[i], confidences[j] + errors[j]) - max(
            confidences[i] - errors[i], confidences[j] - errors[j]
        )
        overlaps.append(overlap_area / union_area)
    return sum(overlaps)


# Perform a grid search
best_metric = float("-inf")
best_config = None

for num_intervals in range(2, 5):
    # Perform grid search with different configurations
    confidences_grid = np.linspace(0.2, 0.8, num_intervals)
    errors_grid = np.random.rand(num_intervals) * 0.1
    print(confidences_grid)
    # Calculate the metric for the current configuration
    current_metric = calculate_metric(
        confidences_grid, errors_grid, weight_confidence, weight_overlap
    )

    # Update best configuration if the metric is improved
    if current_metric > best_metric:
        best_metric = current_metric
        best_config = (confidences_grid, errors_grid)

# Output the best configuration
print("Best Configuration:")
print("Confidences:", best_config[0])
print("Errors:", best_config[1])
print("Best Metric:", best_metric)
