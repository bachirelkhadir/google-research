import logging
import cvxpy
import numpy as np


def _mult_poly_matrix_poly(p, mat_y):
  """Multiplies the polynomial matrix mat_y by the polynomial p entry-wise.

  Args:
    p: list of size d1+1 representation the polynomial sum p[i] t^i.
    mat_y: (m, m, d2+1) tensor representing a polynomial
      matrix Y_ij(t) = sum mat_y[i, j, k] t^k.

  Returns:
    (m, m, d1+d2+1) tensor representing the polynomial matrix p(t)*Y(t).
  """

  mult_op = lambda q: np.convolve(p, q)
  p_times_y = np.apply_along_axis(mult_op, 2, mat_y)
  return p_times_y


def _make_zero(p):
  """Returns the constraints p_i == 0.

  Args:
    p: list of cvxpy expressions.

  Returns:
    A list of cvxpy constraints [pi == 0 for pi in p].
  """

  return [pi == 0 for pi in p]


def _lambda(m, d, Q):
  """Returns the mxm polynomial matrix of degree d whose Gram matrix is Q.

  Args:
    m: size of the polynomial matrix to be returned.
    d: degreen of the polynomial matrix to be returned.
    Q: (m*d/2, m*d/2)  gram matrix of the polynomial matrix to be returned.

  Returns:
    (m, m, d+1) tensor representing the polynomial whose gram matrix is Q.
    i.e. $$Y_ij(t) == sum_{r, s s.t. r+s == k}  Q_{y_i t^r, y_j t^s} t^k$$.
  """

  d_2 = int(d / 2)
  def y_i_j(i, j):
    poly = list(np.zeros((d + 1, 1)))
    for k in range(d_2 + 1):
      for l in range(d_2 + 1):
        poly[k + l] += Q[i + k * m, j + l * m]
    return poly

  mat_y = [[y_i_j(i, j) for j in range(m)] for i in range(m)]
  mat_y = np.array(mat_y)
  return mat_y


def _alpha(m, d, Q):
  """Returns t*Lambda(Q) if d odd, Lambda(Q) o.w.

  Args:
    m: size of the polynomial matrix to be returned.
    d: degreen of the polynomial matrix to be returned.
    Q: gram matrix of the polynomial matrix.

  Returns:
    t*Lambda(Q) if d odd, Lambda(Q) o.w.
  """

  if d % 2 == 1:
    w1 = np.array([0, 1])  # t
  else:
    w1 = np.array([1])  # 1
  mat_y = _lambda(m, d + 1 - len(w1), Q)
  return _mult_poly_matrix_poly(w1, mat_y)


def _beta(m, d, Q):
  """Returns (1-t)*Lambda(Q) if d odd, t(1-t)*Lambda(Q) o.w.

  Args:
    m: size of the polynomial matrix to be returned.
    d: degreen of the polynomial matrix to be returned.
    Q: gram matrix of the polynomial matrix.

  Returns:
    (1-t)*Lambda(Q) if d odd, t(1-t)*Lambda(Q) o.w.
  """

  if d % 2 == 1:
    w2 = np.array([1, -1])  # 1 - t
  else:
    w2 = np.array([0, 1, -1])  # t - t^2
  mat_y = _lambda(m, d + 1 - len(w2), Q)
  return _mult_poly_matrix_poly(w2, mat_y)


def _make_matrix_psd(Q):
    """Returns the contraints Q is psd."""
    # The >> operator doesn't work with MOSEK
    # so Q >> 0 won't work
    return Q >> 0
    #return cvxpy.lambda_min(Q)>0


def make_poly_matrix_psd_on_0_1(mat_x):
  """Returns the constraint X(t) psd on [0, 1].

  Args:
    mat_x: (m, m, d+1) tensor representing a mxm polynomial matrix of degree d.

  Returns:
    A list of cvxpy constraints imposing that X(t) psd on [0, 1].
  """

  m, m2, d = len(mat_x), len(mat_x[0]), len(mat_x[0][0]) - 1

  # square matrix
  assert m == m2

  # build constraints: X == alpha(Q1) + beta(Q2) with Q1, Q2 >> 0
  d_2 = int(d / 2)
  size_Q1 = m * (d_2 + 1)
  size_Q2 = m * d_2 if d % 2 == 0 else m * (d_2 + 1)

  Q1 = cvxpy.Variable( (size_Q1, size_Q1) )
  Q2 = cvxpy.Variable( (size_Q2, size_Q2) )
  logging.info('Making X(t) psd requires matrice Q1 and Q2 of size %d, %d resp.',
               size_Q1, size_Q2)

  diff = mat_x - _alpha(m, d, Q1) - _beta(m, d, Q2)
  diff = diff.reshape(-1)

  const = _make_zero(diff)
  const += [_make_matrix_psd(Q1), _make_matrix_psd(Q2), Q1.T == Q1, Q2.T == Q2]

  return const