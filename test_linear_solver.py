import torch
from linear_solver import solve_linear_equations
from matrices_vectors import matrix_vector_product, matrix_sum, scalar_matrix

def run_test(A, b, test_name):
    print(f"--- Testing: {test_name} ---")
    x = solve_linear_equations(A, b)
    
    if x is None:
        print("Result: No Solution (Inconsistent)\n")
    else:
        print("Result: Solution Vector x:")
        print(x)
        
        # TODO: Calculate Error = Ax - b (it should be close to zero)
        # Hint: Use our custom matrix_vector_product to compute the linear combination: Ax
        # Hint: Use our custom scalar_matrix and matrix_sum to get the error vector: Ax-b
        # Hint: Get the L2 Norm (Euclidean distance) of the error vector (you may use torch.norm)
        # Hint: Report the L2 Norm result


# 1. Define Test Data (4 equations, 4 unknowns)
# Case 1: Consistent System (Unique solution or Infinite)
A_consistent = torch.tensor([
    [1, 2, 1, -1],
    [3, 2, 4,  4],
    [4, 4, 3,  4],
    [2, 0, 1,  5]
], dtype=torch.float32)

b_consistent = torch.tensor([[5], [16], [22], [15]], dtype=torch.float32)

# Case 2: Inconsistent System
# (Row 3 is a multiple of Row 1, but b is not)
A_inconsistent = torch.tensor([
    [1, 1, 1, 1],
    [2, 3, 1, 4],
    [1, 1, 1, 1], # Same coefficients as row 1
    [0, 1, 2, 3]
], dtype=torch.float32)

b_inconsistent = torch.tensor([[10], [20], [5], [15]], dtype=torch.float32) # b[2] contradicts b[0]

# BONUS: System Robustness & Scalability Challenge
# Consider to define the following cases and report the results.
#
# 1. Minimal Case (2x2 or 3x3):
#    - Test a simple, manually verifiable system to ensure core logic is sound.
#
# 2. Non-Square (Rectangular) Matrices:
#    - Underdetermined: "Fat" matrix (e.g., 3x5). Does it handle free variables?
#    - Overdetermined: "Tall" matrix (e.g., 6x3). Test for both consistent 
#      and inconsistent cases.
#
# 3. Large Scale (100x100+):
#    - Generate a large matrix A and vector x_true. 
#
# 4. Report Requirements:
#    - List the dimensions and properties of your test data.
#    - Provide the final verification error ||Ax - b|| for each case.

if __name__ == "__main__":
    run_test(A_consistent, b_consistent, "Consistent Case")
    run_test(A_inconsistent, b_inconsistent, "Inconsistent Case")

