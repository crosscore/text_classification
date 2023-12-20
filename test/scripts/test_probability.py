# Define the probabilities of A and B
p_A = 0.9  # P(A)
p_B = 0.8  # P(B)

# Calculate P(A∩B) for independent events
p_A_and_B = p_A * p_B

# Calculate P(A∪B) using the formula
p_A_or_B = p_A + p_B - p_A_and_B

# Output the result
print(f"The probability of P(A∪B) when A and B are independent is: {p_A_or_B}")
