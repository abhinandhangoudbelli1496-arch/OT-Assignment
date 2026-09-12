import numpy as np

# ---------------------------------------------------------
# VOGEL'S APPROXIMATION METHOD (VAM)
# ---------------------------------------------------------

# Transportation cost matrix
cost = np.array([
    [2, 12, 4, 14],
    [8,  5, 8,  7],
    [6, 10, 13, 10]
], dtype=float)

# Supply and demand
supply = [40, 30, 50]
demand = [30, 25, 35, 30]

rows = len(supply)
cols = len(demand)

# Allocation matrix
allocation = np.zeros((rows, cols))

# Copies of supply and demand
s = supply.copy()
d = demand.copy()

# Active rows and columns
active_rows = [True] * rows
active_cols = [True] * cols

# ---------------------------------------------------------
# VAM ITERATIONS
# ---------------------------------------------------------

while any(active_rows) and any(active_cols):

    # Calculate row penalties
    row_penalty = []

    for i in range(rows):

        if not active_rows[i]:
            row_penalty.append(-1)
            continue

        values = [
            cost[i][j]
            for j in range(cols)
            if active_cols[j]
        ]

        values.sort()

        if len(values) >= 2:
            penalty = values[1] - values[0]
        else:
            penalty = values[0]

        row_penalty.append(penalty)

    # Calculate column penalties
    col_penalty = []

    for j in range(cols):

        if not active_cols[j]:
            col_penalty.append(-1)
            continue

        values = [
            cost[i][j]
            for i in range(rows)
            if active_rows[i]
        ]

        values.sort()

        if len(values) >= 2:
            penalty = values[1] - values[0]
        else:
            penalty = values[0]

        col_penalty.append(penalty)

    # Select row or column having maximum penalty

    if max(row_penalty) >= max(col_penalty):

        i = np.argmax(row_penalty)

        # Select minimum cost cell in that row
        j = min(
            [j for j in range(cols) if active_cols[j]],
            key=lambda j: cost[i][j]
        )

    else:

        j = np.argmax(col_penalty)

        # Select minimum cost cell in that column
        i = min(
            [i for i in range(rows) if active_rows[i]],
            key=lambda i: cost[i][j]
        )

    # Allocate maximum possible quantity
    quantity = min(s[i], d[j])

    allocation[i][j] = quantity

    # Update supply and demand
    s[i] -= quantity
    d[j] -= quantity

    # Close exhausted row or column
    if s[i] == 0:
        active_rows[i] = False

    if d[j] == 0:
        active_cols[j] = False


# ---------------------------------------------------------
# CALCULATE INITIAL COST
# ---------------------------------------------------------

initial_cost = np.sum(allocation * cost)

print("INITIAL SOLUTION USING VAM")
print("--------------------------")

print(allocation)

print("\nInitial Transportation Cost =", initial_cost)
