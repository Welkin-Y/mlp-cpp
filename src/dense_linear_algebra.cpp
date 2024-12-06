#include "dense_linear_algebra.h"

namespace BasicDenseLinearAlgebra {
//=============================================================================
/// Linear solver: Takes matrix and rhs
/// vector and returns the solution of the linear system.
//============================================================================
template <typename T>
Vector<T> LULinearSolver<T>::lu_solve(const SquareMatrix<T> &matrix,
                                      const Vector<T> &rhs) {
  // factorise
  factorise(matrix);

  // Get result via backsubstitution
  Vector<T> result = backsub(rhs);

  return result;
}

//=============================================================================
/// LU decompose the matrix.
//=============================================================================
template <typename T>
void LULinearSolver<T>::factorise(const SquareMatrix<T> &matrix) {
  // Set the number of unknowns
  const unsigned n = matrix.n();

  // Allocate storage for the LU factors and the permutation index
  // set entries to zero.
  LU_factors.resize(n * n, 0.0);
  Index.resize(n, 0);

  // Now we know that memory has been allocated, copy over
  // the matrix values
  unsigned count = 0;
  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = 0; j < n; j++) {
      LU_factors[count] = matrix(i, j);
      ++count;
    }
  }

  // Loop over columns
  for (unsigned j = 0; j < n; j++) {
    // Initialise imax, the row with the largest entry in the present column
    unsigned imax = 0;

    // Do rows up to diagonal
    for (unsigned i = 0; i < j; i++) {
      double sum = LU_factors[n * i + j];
      for (unsigned k = 0; k < i; k++) {
        sum -= LU_factors[n * i + k] * LU_factors[n * k + j];
      }
      LU_factors[n * i + j] = sum;
    }

    // Initialise search for largest pivot element
    double largest_entry = 0.0;

    // Do rows below diagonal -- here we still have to pivot!
    for (unsigned i = j; i < n; i++) {
      double sum = LU_factors[n * i + j];
      for (unsigned k = 0; k < j; k++) {
        sum -= LU_factors[n * i + k] * LU_factors[n * k + j];
      }
      LU_factors[n * i + j] = sum;

      // New largest entry found in a row below the diagonal?
      double tmp = std::fabs(sum);
      if (tmp >= largest_entry) {
        largest_entry = tmp;
        imax = i;
      }
    }

    // Test to see if we need to interchange rows; if so, do it!
    if (j != imax) {
      for (unsigned k = 0; k < n; k++) {
        double tmp = LU_factors[n * imax + k];
        LU_factors[n * imax + k] = LU_factors[n * j + k];
        LU_factors[n * j + k] = tmp;
      }
    }

    // Record the index (renumbering rows of the orignal linear
    // system to reflect pivoting)
    Index[j] = imax;

    // Divide by pivot element
    if (j != n - 1) {
      double pivot = LU_factors[n * j + j];
      if (pivot == 0.0) {
        std::string error_message =
            "Singular matrix: zero pivot in row " + std::to_string(j);
        throw LinearSolverError(error_message.c_str());
      }
      double tmp = 1.0 / pivot;
      for (unsigned i = j + 1; i < n; i++) {
        LU_factors[n * i + j] *= tmp;
      }
    }

  } // End of loop over columns
}

//=============================================================================
/// Do the backsubstitution for the DenseLU solver.
//=============================================================================
template <typename T>
Vector<T> LULinearSolver<T>::backsub(const Vector<T> &rhs, T zero) {
  // Initially copy the rhs vector into the result vector
  const unsigned n = rhs.n();
  Vector<T> result(n);
  for (unsigned i = 0; i < n; ++i) {
    result[i] = rhs[i];
  }

  // Loop over all rows for forward substitution
  unsigned k = 0;
  for (unsigned i = 0; i < n; i++) {
    unsigned ip = Index[i];
    double sum = result[ip];
    result[ip] = result[i];
    if (k != 0) {
      for (unsigned j = k - 1; j < i; j++) {
        sum -= LU_factors[n * i + j] * result[j];
      }
    } else if (sum != zero) {
      k = i + 1;
    }
    result[i] = sum;
  }

  // Now do the back substitution
  // Note: this has to be an int to avoid wrapping around!
  for (int i = n - 1; i >= 0; i--) {
    double sum = result[i];
    for (unsigned j = i + 1; j < n; j++) {
      sum -= LU_factors[n * i + j] * result[j];
    }
    result[i] = sum / LU_factors[n * i + i];
  }

  return result;
}
//==========================================================================
/// Helper function to get the max. error of the solution of Ax=b, defined
/// as max_i |(A_ij x_j - b_i)|
//==========================================================================
template <typename T>
T max_error(const SquareMatrix<T> &matrix, const Vector<T> &rhs,
            const Vector<T> &soln) {
  unsigned n = rhs.n();
  double max_error = 0.0;
  for (unsigned i = 0; i < n; i++) {
    double error = rhs[i];
    for (unsigned j = 0; j < n; j++) {
      error -= matrix(i, j) * soln[j];
    }
    if (fabs(error) > max_error)
      max_error = fabs(error);
  }
  return max_error;
}
} // namespace BasicDenseLinearAlgebra
