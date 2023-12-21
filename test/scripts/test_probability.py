import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_union_probability(p_A, p_B):
    """
    Calculate the probability of the union of two independent events A and B.

    Parameters:
    p_A (float): Probability of event A
    p_B (float): Probability of event B

    Returns:
    float: Probability of the union of A and B
    """
    p_A_and_B = p_A * p_B
    p_A_or_B = p_A + p_B - p_A_and_B
    return p_A_or_B

# Generate a range of probabilities from 0 to 1
probabilities = np.linspace(0, 1, 100)

# Prepare data for graphing
data = []
for p_A in probabilities:
    for p_B in probabilities:
        p_A_or_B = calculate_union_probability(p_A, p_B)
        data.append([p_A, p_B, p_A_or_B])

# Convert the data into a Pandas DataFrame
data_df = pd.DataFrame(data, columns=['p_A', 'p_B', 'p_A_or_B'])

# Pivot the DataFrame for the heatmap
pivot_df = data_df.pivot(index='p_B', columns='p_A', values='p_A_or_B')

# Define labels for the x and y axes with 2 decimal places
label_interval = 5  # Adjust the interval as needed
xtick_labels = [f"{x:.2f}" if i % label_interval == 0 else "" for i, x in enumerate(pivot_df.columns)]
ytick_labels = [f"{y:.2f}" if i % label_interval == 0 else "" for i, y in enumerate(pivot_df.index)]

# Plot the data
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=False, cmap='coolwarm', xticklabels=xtick_labels, yticklabels=ytick_labels)
plt.title("Probability of P(A∪B) for Various Values of P(A) and P(B)")
plt.xlabel("Probability of B (p_B)")
plt.ylabel("Probability of A (p_A)")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)   # Set y-axis labels straight
plt.savefig("probability_union.png")
