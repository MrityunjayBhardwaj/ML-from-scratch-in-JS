def zeros_matrix(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def copy_matrix(M):
    """
    Creates and returns a copy of a matrix.
        :param M: The matrix to be copied
        :return: A copy of the given matrix
    """
    # Section 1: Get matrix dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 2: Create a new matrix of zeros
    MC = zeros_matrix(rows, cols)

    # Section 3: Copy values of M into the copy
    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


def determinant_fast(A):
    # Section 1: Establish n parameter and copy A
    n = len(A)
    AM = copy_matrix(A)
 
    # Section 2: Row ops on A to get in upper triangle form
    for fd in range(n): # A) fd stands for focus diagonal
        for i in range(fd+1,n): # B) only use rows below fd row
            if AM[fd][fd] == 0: # C) if diagonal is zero ...
                AM[fd][fd] == 1.0e-18 # change to ~zero
            # D) cr stands for "current row"
            crScaler = AM[i][fd] / AM[fd][fd] 
            # E) cr - crScaler * fdRow, one element at a time
            for j in range(n): 
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
     
    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(n):
        # ... product of diagonals is determinant
        product *= AM[i][i] 
 
    return product

import numpy as np

A = [[1,2,3,4],[8,5,6,7],[9,12,10,11],[13,14,16,15]]
Det = determinant_fast(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det,9))
print("The Numpy Determinant of A is", round(npDet,9))
print()


print('eigen Decomp of A: ',np.linalg.eig(A));

#  Eigen Decomposition Implementation:-
# Reference: https://medium.com/@louisdevitry/intuitive-tutorial-on-eigenvalue-decomposition-in-numpy-af0062a4929b

def shifting_redirection(M, eigenvalue, eigenvector):
    """
    Apply shifting redirection to the matrix to compute next eigenpair: M = M-lambda v
    """
    return(M-eigenvalue*np.matmul(eigenvector.T, eigenvector))

def power_method(M, epsilon = 0.0001, max_iter = 10000):
    """
    This function computes the principal component of M by using the power method with parameters:
    - epsilon: (float) Termination criterion to stop the power method when changes in the solution is marginale
    - max_iter: (int) Hard termination criterion
    Notes:
    - I added another condition based on the dot product of two consecutive solutions
    """
    # Initialization
    x = [None]*int(max_iter)
    x[0] = np.random.rand(M.shape[0])
    x[1] = np.matmul(M, x[0])
    count = 0
    
    # Compute eigenvector
    while((np.linalg.norm(x[count] - x[count-1]) > epsilon) & (count < max_iter)):
        # Actual computations
        x[count+1] = np.matmul(M, x[count])/np.linalg.norm(np.matmul(M, x[count]))
        count += 1
        
    # Compute eigenvalue
    eigenvalue = np.matmul(np.matmul(x[count].T, M), x[count])
    
    return(x[count], eigenvalue)

def eigenpairs(M, epsilon = 0.00001, max_iter = 10e2, plot = True):
    # Initialization
    eigenvectors = [None]*M.shape[0]
    eigenvalues = [None]*M.shape[0]
    
    for i in range(0, M.shape[0]):
        # Actual computing
        eigenvectors[i], eigenvalues[i] = power_method(M, epsilon, max_iter, iteration = i+1) 
        M = shifting_redirection(M, eigenvalues[i], eigenvectors[i])

    return(eigenvectors, eigenvalues)
