import numpy as np
from itertools import combinations
from scipy.optimize import minimize

# Generate synthetic data
np.random.seed(42)
num_intervals = 10
confidences = np.random.rand(num_intervals)
errors = np.random.rand(num_intervals) * 0.1

# Define weights for the objective function
weight_confidence = 0.6
weight_overlap = 0.4


# Define the objective function to be minimized
def objective_function(x):
    # Extract selected confidences and errors
    selected_confidences = confidences[x == 1]
    selected_errors = errors[x == 1]

    # Calculate confidence and overlap
    confidence = np.sum(selected_confidences)
    overlap_area = overlap(selected_confidences, selected_errors)

    # Objective function to minimize (negative because scipy performs minimization)
    return -(weight_confidence * confidence - weight_overlap * overlap_area)


# Define a function to calculate the overlap between selected intervals
def overlap(selected_confidences, selected_errors):
    overlaps = []
    for pair in combinations(range(len(selected_confidences)), 2):
        i, j = pair
        overlap_area = max(
            0,
            min(
                selected_confidences[i] + selected_errors[i],
                selected_confidences[j] + selected_errors[j],
            )
            - max(
                selected_confidences[i] - selected_errors[i],
                selected_confidences[j] - selected_errors[j],
            ),
        )
        union_area = min(
            selected_confidences[i] + selected_errors[i],
            selected_confidences[j] + selected_errors[j],
        ) - max(
            selected_confidences[i] - selected_errors[i],
            selected_confidences[j] - selected_errors[j],
        )
        overlaps.append(overlap_area / union_area)
    return sum(overlaps)


# Initialize the binary selection array
initial_selection = np.ones(num_intervals, dtype=int)

# Define constraints for the optimization
constraints = {
    "type": "eq",
    "fun": lambda x: np.sum(x) - 3,
}  # Example constraint: Select exactly 3 intervals

# Perform the optimization
result = minimize(
    objective_function,
    initial_selection,
    constraints=constraints,
    bounds=[(0, 1)] * num_intervals,
    method="SLSQP",
)

# Extract the optimal selection
optimal_selection = result.x.astype(int)

# Output the results
print("Optimal Selection:")
print("Confidences:", confidences[optimal_selection == 1])
print("Errors:", errors[optimal_selection == 1])
print("Objective Function Value:", -result.fun)
