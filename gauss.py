import torch
from matrices_vectors import scalar_matrix, matrix_sum

# --- Elementary Row Operations (EROs) ---

def row_interchange(R, i, j):
    """Interchanges row i and row j of matrix R."""
    # We use clone() to ensure we don't have reference issues during swapping
    temp = R[i].clone()
    # TODO: Perform row interchange
    return R

def row_scaling(R, i, s):
    """Scales row i of matrix R by a scalar s."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function to finish this row operation
    # Hint: To use the scalar_matrix, a row of 1D tensor should be reshaped to a row 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    return R

def row_addition(R, i, j, s):
    """Adds s times row i to row j."""
    cols = R.shape[1]
    # TODO: Use our custom scalar_matrix function and matrix_sum to finish this row operation
    # Hint: To use the scalar_matrix and matrix_sum, a row of 1D tensor should be reshaped to a row 
    # 2D tensor.
    # For example, R[i] should be reshaped as (1, cols) to fit the function requirments
    # Hint: You may use API flatten() to convert a 2D tensor back to 1D tensor
    return R

# --- Gaussian Elimination using EROs ---

def gauss_elimination(A):
    """
    Transforms matrix A into RREF using the encapsulated row operations.
    """
    R = A.clone().to(torch.float32)
    rows, cols = R.shape
    
    pivot_row = 0
    pivot_col = 0

    # Set zero thresh (if entry is larger than this value, treat it as zero. Otherwise, treat it as nonzero)
    zero_thresh = 1e-6

    # TODO: Implement step 1 to step 4 in the lecture slides (the forward phase to Row Echelon Form) 
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions

    # TODO: Implement step 5 and step 6 in the lecture slides (the backward phase to Reduced Row Echelon Form) 
    # Hint: You should use our custom row_interchange, row_scaling, row_addition funtions
                
    return R
