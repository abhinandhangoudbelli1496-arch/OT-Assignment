import numpy as np

# ---------------------------------------------------------
# MODI METHOD
# ---------------------------------------------------------

cost = np.array([
    [2, 12, 4, 14],
    [8,  5, 8,  7],
    [6, 10, 13, 10]
], dtype=float)

# Initial solution obtained from VAM
allocation = np.array([
    [0,  0, 35,  5],
    [0, 25,  0,  5],
    [30, 0,  0, 20]
], dtype=float)

rows, cols = cost.shape


# ---------------------------------------------------------
# FIND CLOSED LOOP
# ---------------------------------------------------------

def find_cycle(path, basic, start, move_row):

    i, j = path[-1]

    # If we returned to starting cell
    if len(path) >= 4:

        if move_row and i == start[0]:
            return path + [start]

        if not move_row and j == start[1]:
            return path + [start]

    # Move horizontally
    if move_row:

        candidates = [
            (i, new_j)
            for new_j in range(cols)
            if basic[i][new_j]
            and (i, new_j) not in path
        ]

    # Move vertically
    else:

        candidates = [
            (new_i, j)
            for new_i in range(rows)
            if basic[new_i][j]
            and (new_i, j) not in path
        ]

    for cell in candidates:

        result = find_cycle(
            path + [cell],
            basic,
            start,
            not move_row
        )

        if result is not None:
            return result

    return None


# ---------------------------------------------------------
# MODI ITERATIONS
# ---------------------------------------------------------

for iteration in range(100):

    basic = allocation > 0

    # Potentials
    u = [None] * rows
    v = [None] * cols

    # Set first u value to zero
    u[0] = 0

    changed = True

    while changed:

        changed = False

        for i in range(rows):
            for j in range(cols):

                if basic[i][j]:

                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                        changed = True

                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i][j] - v[j]
                        changed = True

    # -----------------------------------------------------
    # Calculate opportunity costs
    # -----------------------------------------------------

    delta = np.full((rows, cols), np.nan)

    for i in range(rows):
        for j in range(cols):

            if not basic[i][j]:
                delta[i][j] = cost[i][j] - u[i] - v[j]

    # Check optimality
    if np.nanmin(delta) >= 0:
        print("Optimal solution reached.")
        break

    # Select most negative opportunity cost
    entering = np.unravel_index(
        np.nanargmin(delta),
        delta.shape
    )

    start = entering

    # Find closed loop
    cycle = find_cycle(
        [start],
        basic,
        start,
        True
    )

    if cycle is None:
        cycle = find_cycle(
            [start],
            basic,
            start,
            False
        )

    cycle = cycle[:-1]

    # Cells with minus signs
    minus_cells = cycle[1::2]

    # Find theta
    theta = min(
        allocation[i][j]
        for i, j in minus_cells
    )

    # Update allocation
    for k, (i, j) in enumerate(cycle):

        if k % 2 == 0:
            allocation[i][j] += theta
        else:
            allocation[i][j] -= theta

    # Remove very small values
    allocation[
        np.abs(allocation) < 1e-9
    ] = 0


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

minimum_cost = np.sum(allocation * cost)

print("\nOPTIMAL TRANSPORTATION PLAN")
print("---------------------------")

print(allocation)

print("\nMinimum Transportation Cost =", minimum_cost)
