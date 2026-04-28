import torch
from linear_solver import (
    solve_linear_equations, 
    solve_linear_equations_by_inverse, 
    test_invertibility, 
    test_span, 
    test_linear_dependence
)
from matrices_vectors import matrix_vector_product, matrix_sum, scalar_matrix


def run_test(matrix_M, vector_b, test_name):
    print(f"--- Testing Vector: {test_name} ---")
    
    if test_span(matrix_M, vector_b):
        print("Result: IN SPAN.")
        # TODO: Solve the solution by using solve_linear_equations_by_inverse
        x_inverse = solve_linear_equations_by_inverse(matrix_M, vector_b)
        x_gauss = solve_linear_equations(matrix_M, vector_b)
        # TODO: If the solution is not None, calculate and print the error: ||x_gauss - x_inverse||, 
        # where x_gauss is the solution from solve_linear_equations, and x_inverse is the solution 
        # from solve_linear_equations
        if x_inverse is not None and x_gauss is not None:
            # 計算 x_gauss + (-1 * x_inverse)
            neg_x_inverse = scalar_matrix(-1.0, x_inverse)
            diff_method = matrix_sum(x_gauss, neg_x_inverse)
            
            method_error = torch.norm(diff_method).item()
            print(f"Error ||x_gauss - x_inverse||: {method_error:.6e}")
        # TODO: If the solution is None, print a FAIL message:
        else:
            print("Inverse Method: FAILED (Matrix is singular/not invertible)")

        # TODO: Perform the Ax-b error check as in previous assignment, print the error
        # (Should done in HW1, not necessary to explain the codes)
        x = solve_linear_equations(matrix_M, vector_b)
        Ax = matrix_vector_product(matrix_M, x)
        neg_b = scalar_matrix(-1.0, vector_b)
        error_vec = matrix_sum(Ax, neg_b)
        error_norm = torch.norm(error_vec).item()
        print(f"L2 Norm of Error (||Ax - b||): {error_norm:.6e}")
    else:
        print("Result: NOT IN SPAN.")
    print("")

# --- 1. Define Matrix A---
A = torch.tensor([
    [1.0, 2.0,  1.0, -1.0],
    [3.0, 2.0,  4.0,  4.0],
    [4.0, 4.0,  5.0,  3.0], 
    [2.0, 0.0,  1.0,  5.0]
], dtype=torch.float32)

# --- 2. Define Matrix B ---
B = torch.tensor([
    [1.0, 2.0,  1.0, -1.0],
    [3.0, 5.0,  4.0,  4.0], 
    [4.0, 4.0,  9.0,  3.0], 
    [2.0, 0.0,  1.0,  8.0]  
], dtype=torch.float32)

# --- 3. The 6 Test Vectors---
test_b_vectors = [
    (torch.tensor([[3.0], [13.0], [16.0], [8.0]]),  "Vector_Alpha"),
    (torch.tensor([[1.0], [0.0], [0.0], [0.0]]),    "Vector_Beta"),
    (torch.tensor([[5.0], [18.0], [23.0], [7.0]]),  "Vector_Gamma"),
    (torch.tensor([[2.0], [1.0], [3.0], [4.0]]),    "Vector_Delta"),
    (torch.tensor([[0.0], [5.0], [5.0], [1.0]]),    "Vector_Epsilon"),
    (torch.tensor([[8.0], [4.0], [12.0], [1.0]]),   "Vector_Zeta")
]

if __name__ == "__main__":
    print("Linear Algebra Lab: Gaussian vs. Inverse Method")
    print("=" * 60)

    # Part 1: Matrix Property Check
    # TODO: 
    # 1. Include results in the report.
    # 2. Provide an analysis of the relation between the linear dependence and invertiblity 
    print(f"Matrix A (Dependent)  - Is Dependent: {test_linear_dependence(A)}")
    print(f"Matrix A (Dependent)  - Is Invertible: {test_invertibility(A)}")
    print("-" * 60)
    print(f"Matrix B (Independent)- Is Dependent: {test_linear_dependence(B)}")
    print(f"Matrix B (Independent)- Is Invertible: {test_invertibility(B)}")
    print("=" * 60)

    # Part 2: Test Scenario 1
    # TODO:
    # 1. Include results in the report.
    # 2. Provide an analysis of the relation between the span and invertibility 
    # 3. Provide an analysis of the relation between solve_linear_equations and 
    # solve_linear_equations_by_inverse
    print("\n[SCENARIO 1] Testing with Dependent Matrix A")
    for vec, name in test_b_vectors:
        run_test(A, vec, name)
    print("-" * 50)


    # Part 3: Test Scenario 2
    # TODO:
    # 1. Include results in the report.
    # 2. Provide an analysis of the relation between the span and invertibility 
    # 3. Provide an analysis of the relation between solve_linear_equations and 
    # solve_linear_equations_by_inverse
    print("\n[SCENARIO 2] Testing with Independent Matrix B")
    for vec, name in test_b_vectors:
        run_test(B, vec, name)
