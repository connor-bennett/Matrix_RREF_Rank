# Connor Bennett
# program 11
# Math for computing Ranks Matrices and converts to Reduced Row Echelon Form
# 3/14/23
import numpy as np

TOL = 1.0e-14  # tolerance


# exchange rows i and j in matrix M
def swap(m, i, j):
    if i == j:
        return    # nothing to do
    t = m[i, ].copy()
    m[i, ] = m[j, ]
    m[j, ] = t


# multiply row i by the number a
def mult(m, i, a):
    m[i, ] = a * m[i, ]


# multiple row i by number a and add to row j and replace row j
#  a*r(i) + r(j) -> r(j)
def er3(m, i, a, j):
    m[j, ] = a * m[i, ] + m[j, ]


def findFirstNonZero(m, arow):
    mrows, ncols = m.shape
    for icol in range(ncols):
        for irow in range(arow, mrows):
            if m[irow, icol] != 0:
                return irow, icol
    return None


def rref(M):
    """
    input - matrix m augmented matrix for system of equations
    return - matrix m in reduced row echelon form
    """
    mrows, mcols = M.shape
    for irow in range(mrows):
        r = findFirstNonZero(M, irow)
        if r is None:
            # we are done
            break
        else:
            arow, acol = r  # unpack row and col
            swap(M, arow, irow)
            mult(M, irow, 1 / M[irow, acol])  # make leading 1 at M[irow,acol]
            for jrow in range(irow):
                # zero out entries above leading 1
                er3(M, irow, -M[jrow, acol], jrow)
            for jrow in range(irow + 1, mrows):
                # zero out entries below leading 1
                er3(M, irow, -M[jrow, acol], jrow)

    return M


def rank(m):
    """
    input - matrix m is augmented matrix in reduced row echeclon format (RREF)
    return - rank of matrix m
    if M is inconsistent, then return 0
    M is inconsistent if there is a row that has a leading 1 in the
    """
    nrows, ncols = m.shape
    my_rank = 0
    col1 = 0  # start column for search

    # we assume M is in RREF form.  So we can just count the number of rows
    # that have a leading 1 in a column except for the last column

    # since the pattern of leading 1's go from left to right, we don't have to
    # search the entire row, just the portion starting from the column of last 1

    for irow in range(nrows):
        for icol in range(col1, ncols):
            if abs(m[irow, icol]) < TOL:
                continue
            elif abs(m[irow, icol] - 1.0) < TOL and icol < ncols - 1:
                my_rank = my_rank + 1
                col1 = icol + 1
                break
            else:
                return 0

    return my_rank


# Swaps the rows and cols
def swap(m, row1, row2):
    ncols = m.shape[1]
    for col in range(ncols):
        t = m[row1, col]
        m[row1, col] = m[row2, col]
        m[row2, col] = t


if __name__ == "__main__":
    print("Test 2")
    M = np.matrix([[1., 1., 1., 5.], [1., -1., 2., 11.], [1., -1., -2., -5.]])
    # save a copy of M
    R = M.copy()
    mrows, mcols = M.shape
    print(M)
    M = rref(M)
    print("Row echelon form")
    print(M)
    r = rank(M)
    print('rank =', r)
    # answer is in last column of M
    # verify answer by doing R[coefficients] X M[last column] == R[constants]
    print("verify answer")
    print(np.matmul(R[:, 0:mcols - 1], M[:, mcols - 1]))
    print("expected answer")
    print(R[:, mcols - 1])
