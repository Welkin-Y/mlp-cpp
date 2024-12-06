#pragma once
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ############################################################
/// Helper file with functions/classes for basic linear algebra
// ############################################################

//============================================================
/// Namespace for basic dense linear algebra
//============================================================
namespace BasicDenseLinearAlgebra {

template <typename T> class Vector {
public:
  Vector(size_t const &n = 0)
      : N(n) { // Resize storage and initialise entries to zero
    Vector_storage.resize(N, 0.0);
  }
  ~Vector() = default;
  Vector(Vector const &other)
      : N(other.N), Vector_storage(other.Vector_storage) {}
  Vector(Vector &&other) noexcept
      : N(other.N), Vector_storage(std::move(other.Vector_storage)) {
    other.N = 0;
  }
  Vector &operator=(Vector const &other) {
    if (this != &other) {
      N = other.N;
      Vector_storage = other.Vector_storage;
    }
    return *this;
  }
  Vector &operator=(Vector &&other) noexcept {
    if (this != &other) {
      N = other.N;
      Vector_storage = std::move(other.Vector_storage);
      other.N = 0;
    }
    return *this;
  }

  /// Size of vector
  size_t n() const { return N; }

  /// Const access to i-th entry
  T operator[](const unsigned &i) const {
#ifdef RANGE_CHECKING
    assert(i < N);
#endif
    return Vector_storage[i];
  }

  /// Read/write access to i-th entry
  T &operator[](const unsigned &i) {
#ifdef RANGE_CHECKING
    assert(i < N);
#endif
    return Vector_storage[i];
  }

  /// Resize (and zero the entries)
  void resize(const unsigned &n) {
    N = n;
    Vector_storage.resize(n, 0.0);
  }

  /// Output to std::cout
  void output() const { output(std::cout); }

  /// Output (specify filename)
  void output(std::string filename) const {
    std::ofstream outfile(filename.c_str());
    output(outfile);
    outfile.close();
  }

  /// Output (specify stream)
  void output(std::ostream &outfile) const {
    for (unsigned i = 0; i < N; i++) {
      outfile << i << " " << Vector_storage[i] << " " << std::endl;
    }
  }

  /// Input from file (specify filename)
  void read(std::string filename) {
    std::ifstream infile(filename.c_str());
    read(infile);
    infile.close();
  }

  /// Input from file (specify stream)
  void read(std::ifstream &infile) {
    unsigned i_read = 0;
    for (unsigned i = 0; i < N; i++) {
      infile >> i_read;
      if (i != i_read) {
        std::stringstream str;
        str << "\n\nERROR: Row index in matrix is i = " << i
            << " but data is provided for " << i_read << std::endl;
        throw std::runtime_error(str.str().c_str());
      }
      infile >> Vector_storage[i];
    }
  }

protected:
  size_t N;
  std::vector<T> Vector_storage;
};

//===================================================
/// Class for a dense general (not necessarily
/// square) matrix of doubles
//===================================================
template <typename T> class Matrix {
public:
  Matrix(size_t const &n, size_t const &m) : N(n), M(m) {
    Matrix_storage.resize(N * M, 0.0);
  }
  ~Matrix() = default;
  Matrix(const Matrix &other)
      : N(other.N), M(other.M), Matrix_storage(other.Matrix_storage) {}
  Matrix(Matrix &&other) noexcept
      : N(other.N), M(other.M),
        Matrix_storage(std::move(other.Matrix_storage)) {
    other.N = 0;
    other.M = 0;
  }
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      N = other.N;
      M = other.M;
      Matrix_storage = std::move(other.Matrix_storage);
      other.N = 0;
      other.M = 0;
    }
    return *this;
  }

  /// Number of rows
  size_t n() const { return N; }

  /// Number of columns
  size_t m() const { return M; }

  /// Const access to (i,j)-th entry
  T operator()(const unsigned &i, const unsigned &j) const {
#ifdef RANGE_CHECKING
    assert(i < N);
    assert(j < M);
#endif
    return Matrix_storage[i * M + j];
  }

  /// Read/write access to (i,j)-th entry
  T &operator()(const unsigned &i, const unsigned &j) {
#ifdef RANGE_CHECKING
    assert(i < N);
    assert(j < M);
#endif
    return Matrix_storage[i * M + j];
  }

  /// Output to std::cout
  void output() const { output(std::cout); }

  /// Output to file (specify filename)
  void output(std::string filename) const {
    std::ofstream outfile(filename.c_str());
    output(outfile);
    outfile.close();
  }

  /// Output to file (specify stream)
  void output(std::ostream &outfile) const {
    for (unsigned i = 0; i < N; i++) {
      for (unsigned j = 0; j < M; j++) {
        outfile << i << " " << j << " " << Matrix_storage[i * M + j] << " "
                << std::endl;
      }
    }
  }

  /// Input from file (specify filename)
  void read(std::string filename) {
    std::ifstream infile(filename.c_str());
    read(infile);
    infile.close();
  }

  /// Input from file (specify stream)
  void read(std::ifstream &infile) {
    unsigned i_read = 0;
    unsigned j_read = 0;
    for (unsigned i = 0; i < N; i++) {
      for (unsigned j = 0; j < M; j++) {
        infile >> i_read;
        if (i != i_read) {
          std::stringstream str;
          str << "\n\nERROR: Row index in matrix is i = " << i
              << " but data is provided for " << i_read << std::endl;
          throw std::runtime_error(str.str().c_str());
        }
        infile >> j_read;
        if (j != j_read) {
          std::stringstream str;
          str << "\n\nERROR: Column index in matrix is j = " << i
              << " but data is provided for " << j_read << std::endl;
          throw std::runtime_error(str.str().c_str());
        }
        infile >> Matrix_storage[i * M + j];
      }
    }
  }

private:
  /// Number of rows
  size_t N;

  /// Number of columns
  size_t M;

  /// Entries are flat packed, row by row:
  /// a(i,j) = a_flat_packed(i*M+j) (row by row)
  std::vector<T> Matrix_storage;
};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//===================================================
/// Class for a dense square matrix of doubles
//===================================================
template <typename T> class SquareMatrix : public Matrix<T> {
public:
  /// Constructor: Pass size
  SquareMatrix(const unsigned &n) : Matrix<T>(n, n) {}
};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//===================================================
/// Class for a vector of doubles
//===================================================

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//===================================================
/// Error for linear solvers
//===================================================
class LinearSolverError : public std::runtime_error {
public:
  /// Issue runtime error, outputting generic message
  LinearSolverError() : runtime_error("Error in linear solver!") {}

  /// Runtime error with context specific message
  LinearSolverError(std::string msg) : runtime_error(msg.c_str()) {}
};

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

//=============================================================================
/// Dense LU decomposition-based solver
//============================================================================
template <typename T> class LULinearSolver {
public:
  /// Constructor
  LULinearSolver() {}

  /// Destructor
  ~LULinearSolver() {}

  /// Do the linear solve: Takes matrix and rhs
  /// vector and returns the solution of the linear system.
  /// (Not const because it updates some internal storage for
  /// LU factors etc.)
  Vector<T> lu_solve(SquareMatrix<T> const &matrix, Vector<T> const &rhs);

private:
  /// Perform the LU decomposition of the matrix
  void factorise(SquareMatrix<T> const &matrix);

  /// Do the backsubstitution step to solve the system LU result = rhs
  Vector<T> backsub(Vector<T> const &rhs);

  /// Storage for the index of permutations in the LU solve
  /// (used to handle pivoting)
  std::vector<size_t> Index;

  /// Storage for the LU decomposition (flat-packed into nxn vector)
  std::vector<T> LU_factors;
};

} // namespace BasicDenseLinearAlgebra
