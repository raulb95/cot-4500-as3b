
# QUESTION 1
# Augmented matrix
A = [[2, -1, 1, 6],
     [1, 3, 1, 0],
     [-1, 5, 4, -3]]


# Gaussian elimination
for i in range(len(A)):
    
    max_row = i
    for j in range(i+1, len(A)):
        if abs(A[j][i]) > abs(A[max_row][i]):
            max_row = j
    
    A[i], A[max_row] = A[max_row], A[i]
    
    pivot = A[i][i]
    for j in range(i, len(A[i])):
        A[i][j] /= pivot
    
    for j in range(len(A)):
        if j != i:
            factor = A[j][i]
            for k in range(i, len(A[i])):
                A[j][k] -= factor * A[i][k]

x = [int(row[-1]) for row in A]

print("Question 1")
print("The solved system of equations is: ", x)


# QUESTION 2
import numpy as np

def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = matrix[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]
    
    return L, U

def determinant(matrix):
    L, U = lu_decomposition(matrix)
    deter = np.prod(np.diag(U))
    return deter

A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

# Calculate deter, L & U
deter = determinant(A)
L, U = lu_decomposition(A)

print("Question 2")
print("a:")
print(deter)
print("b:")
print(L)
print("c:")
print(U)


# Question 3
print("Question 3")
A = np.array([[ 9, 0, 5, 2, 1],
            [ 3, 9, 1, 2, 1],
             [0, 1, 7, 2, 3],
             [4, 2, 3, 12, 2],
             [3, 2, 4, 0, 8]])

def dd(A):
    D = np.diag(np.abs(A))
    S = np.sum(np.abs(A), axis=1) - D
    if np.all(D > S):
        print("The matrix is diagonally dominant.")
    else:
        print("The matrix is not diagonally dominant.")
    return
dd(A)


# Question 4
print("Question 4")
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

def is_posi_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return ("Matrix is a positive definite.")
        except np.linalg.LinAlgError:
            return ("Matrix is not a positive definite.")
    else:
        return ("Matrix is not a positive definite.")
print(is_posi_def(A))
