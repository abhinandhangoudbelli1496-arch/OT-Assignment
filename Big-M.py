import numpy as np

# ---------------------------------------------------------
# BIG-M SIMPLEX METHOD
# ---------------------------------------------------------

# Big-M value
M = 1000000

# Variables:
# X, Y, S1, S2, S3, A1
#
# Constraint 1:
# X + Y >= 4
# X + Y - S1 + A1 = 4
#
# Constraint 2:
# 2X + Y <= 8
# 2X + Y + S2 = 8
#
# Constraint 3:
# X + 2Y <= 10
# X + 2Y + S3 = 10

# Columns:
# X, Y, S1, S2, S3, A1, RHS

tableau = np.array([
    [1, 1, -1, 0, 0, 1, 4],
    [2, 1,  0, 1, 0, 0, 8],
    [1, 2,  0, 0, 1, 0, 10]
], dtype=float)

# Objective:
# Max Z = 3X + 5Y - M*A1

C = np.array([
    3, 5, 0, 0, 0, -M
], dtype=float)

# Initial basic variables:
# A1, S2, S3
basis = [5, 3, 4]

# ---------------------------------------------------------
# SIMPLEX ITERATIONS
# ---------------------------------------------------------

for iteration in range(100):

    # Calculate Zj
    cb = C[basis]
    zj = cb @ tableau[:, :-1]

    # Calculate Cj - Zj
    cj_zj = C - zj

    # Check optimality
    if np.max(cj_zj) <= 0:
        break

    # Entering variable
    entering = np.argmax(cj_zj)

    # Ratio test
    ratios = []

    for i in range(len(tableau)):
        if tableau[i, entering] > 0:
            ratio = tableau[i, -1] / tableau[i, entering]
            ratios.append((ratio, i))

    # Leaving variable
    leaving = min(ratios)[1]

    # Pivot element
    pivot = tableau[leaving, entering]

    # Make pivot element 1
    tableau[leaving] = tableau[leaving] / pivot

    # Make other entries in pivot column zero
    for i in range(len(tableau)):
        if i != leaving:
            tableau[i] = (
                tableau[i]
                - tableau[i, entering] * tableau[leaving]
            )

    # Update basis
    basis[leaving] = entering


# ---------------------------------------------------------
# FIND FINAL SOLUTION
# ---------------------------------------------------------

solution = np.zeros(6)

for i, variable in enumerate(basis):
    solution[variable] = tableau[i, -1]

X = solution[0]
Y = solution[1]

maximum_profit = 3 * X + 5 * Y

print("BIG-M SIMPLEX METHOD")
print("--------------------")

print("Optimal X =", X)
print("Optimal Y =", Y)

print("Maximum Profit =", maximum_profit)
