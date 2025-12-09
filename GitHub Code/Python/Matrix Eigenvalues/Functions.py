import numpy as np

# =======================================================================================================
#                                       General Matrices Functions
# =======================================================================================================

def det2(mat):
    """Finds the determinant of a 2x2 matrix."""
    return mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]
    
def find_det(mat):
    """Finds the determinant of an NxN matrix. Using Laplace / Cofactor Expansion."""
    # Check the shape of mat
    shape = mat.shape[0]
    det_sum = 0
    # If matrix shape is (2x2), call det2() and return output
    if shape == 2:
        return det2(mat)
    elif shape < 2:
        raise ValueError("Matrix is too small! Minimum size matrix is 2x2.")
    # If matrix shape isn't (2x2), generate minor matrices and find their determinants.
    else:
        sign = np.array([1,-1]*(round(shape/2)+1)) # Cofactor signs
        for i in range(shape):
            det_sum += sign[i]*mat[0,i]*find_det(np.concatenate((mat[1:shape,0:i],mat[1:shape,i+1:shape]),axis=1))
            
    return det_sum

def column(mat, i):
    """Returns a specified column of a matrix as a row. Index from 0 -> (n-1)."""
    if i >= mat.shape[1] or i<0:
        raise IndexError(f"Index of {i} greater/less than the number of columns {mat.shape[1]}! Cannot find a column to return. Index between 0 and (n-1).")
    else:
        col = [row[i] for row in mat]
        return np.array(col, dtype=np.float64)   

# =======================================================================================================
#                                       Functions for QU Method
# =======================================================================================================

def find_fs(mat):
    """Takes a matrix and returns a new matrix with a new set of orthogonal columns, fs. Based on the Gram-Schmidt process."""
    # Create a list to append the f columns to
    fs = []
    # Calculate the f columns
    for i in range(mat.shape[1]):
        c_k = column(mat,i)
        f_k = c_k
        for j in range(i):
            f_k -= ((c_k @ fs[j])/(np.linalg.norm(fs[j])**2))*fs[j]

        fs.append(f_k)
    
    return np.array(fs).transpose()

def find_Q(mat):
    """Change the form of the f columns to q_k = f_k / |f_k|. Returns a matrix with columns q."""
    qs = []
    for i in range(mat.shape[1]):
        fs = find_fs(mat)
        f_k = column(fs,i)
        q_k = f_k / np.linalg.norm(f_k)
        qs.append(q_k)
    
    return np.array(qs).transpose()

def find_U(mat):
    """Returns the upper triangular matrix for the QU algorithm."""
    fs = find_fs(mat)
    Q = find_Q(mat)
    n = fs.shape[1]
    U = np.zeros((n,n))
    # Add the diagonal elements first
    for j in range(n):
        for i in range(n):
            if i==j:
                U[j,i] = np.linalg.norm(column(fs,i))
            else:
                pass
    # Fill in the values for the upper triangle
    for j in range(n):
        for i in range(j):
            U[i,j] = column(mat,j) @ column(Q,i)
    
    return U
    
def find_eigenvals(mat):
    """Finds the eigenvalues for a given invertible 2x2 matrix, using the QU algorithm."""

    # Check if the given matrix is invertible!!!
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix is not square, so is not invertible! Inputted matrix shape: {mat.shape[0],mat.shape[1]}")
    else:
        det = find_det(mat)
        if det == 0:
            raise ValueError("Determinant is 0 so matrix is not invertible!")
        else:
            # QU Algorithm
            A = mat
            Q = find_Q(mat)
            U = find_U(mat)

            iteration = 0
            while iteration < 1000:

                # Save the eigenvalues for the previous matrix
                eigvals_old = []
                for i in range(mat.shape[0]):
                    eigvals_old.append(A[i,i])
                eigvals_old = np.array(eigvals_old)

                # Calculate new matrix
                A = U @ Q
                U = find_U(A)
                Q = find_Q(A)

                # Save the eigenvalues for the new matrix
                eigvals_new = []
                for i in range(mat.shape[0]):
                    eigvals_new.append(A[i,i])
                eigvals_new = np.array(eigvals_new)

                # Compare the change in eigenvalues
                eigvals_diff = abs(eigvals_new - eigvals_old)
                sum_eigvals_diff = sum(eigvals_diff)
                
                # If the eigenvalue difference is negligible, break the while loop.
                if abs(sum_eigvals_diff) < 1e-9:
                    break

                iteration += 1
                
            if iteration == 1000 :
                raise StopIteration(f"Eigenvalues failed to converge after {iteration} iterations.")
        
    return A, iteration

# =======================================================================================================
#                                       Functions for Coupled Oscillators
# =======================================================================================================

def find_freq(mat):
    """Uses the equation for coupled harmonic oscillators: [mat][A] = [-w^2][A] to find the fundamental frequencies of the system."""
    # Find the eigenvalues of the given matrix
    matrix, iteration = find_eigenvals(mat)
    eigvals = []
    for i in range(matrix.shape[0]):
        eigvals.append(matrix[i,i])
    eigvals = np.array(eigvals)

    freqs = (-eigvals)**0.5
    
    return freqs

def form_mat(k,m1,m2):
    """Forms a 2x2 matrix: 
    [[-2k/m1 , k/m1]
     [k/m2 , -2k/m2]]"""
    
    mat = np.array([[-2*k/m1 , k/m1],[k/m2 , -2*k/m2]])
    return mat

def form_mat3x3(k,m1,m2,m3):
    """Returns a 3x3 matrix for a 3-particle coupled oscillator system."""
    mat = np.array([[-2*k/m1, k/m1, 0],[k/m2, -2*k/m2, k/m2],[0, k/m3, -2*k/m3]])
    return mat

def hand_check(k,m):
    """Returns the eigenvalues of a matrix using the characteristic equation method in the context of a coupled oscillator with 2 masses, 3 springs."""
    lam1 = -(2*k)/m + k/m
    lam2 = -(2*k)/m - k/m
    return lam1, lam2