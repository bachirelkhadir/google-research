import numpy as np
import cvxpy


def integ_poly_0_1(p):
  """Return the integral of p(t) between 0 and 1."""

  return np.array(p).dot(1 / np.linspace(1, len(p), len(p)))


def fit_polynomial_with_regularization(x, y, deg=3, alpha=.01):
  """Fits a polynomial  to data `(x, y)`.
  
  Finds a polynomial `p` that minimizes the fitting error 
  |y - p(x)|_2 + alpha |p|_2,
  where p(x) = sum_i p_i x^i.
  
  Args:
    x: [N] ndarray of  input data. Must be increasing.
    y: [N] ndarray, same size as `x`.
    deg: int, degree of each polynomial piece of `p`.
    alpha: float, Regularizer.

  Returns:
     [deg+1] ndarray representing the polynomial `p`.
     
  """

  # coefficients of the polynomial of p.
  p = cvxpy.Variable(deg+1, name='p')

  # convert to numpy format because it is easier to work with.
  numpy_p = np.array([p[i] for i in range(deg+1)])

  regularizer = alpha * cvxpy.norm(p, 1)

  # compute p(px)
  px = eval_poly_from_coefficients(numpy_p, x)

  # fitting value of the current part of p,
  # equal to sqrt(sum |p(x_i) - y_i|^2), where the sum
  # is over data (x_i, y_i) in the current piece.
  fitting_value = cvxpy.norm(cvxpy.vstack(px - y), 1)
    
  min_loss = cvxpy.Minimize(fitting_value + regularizer)
  prob = cvxpy.Problem(min_loss)
  prob.solve(verbose=True, solver=cvxpy.MOSEK)

  return np.array(p.value).squeeze()


def spline_regression(x, y, num_parts, deg=3, alpha=.01, smoothness=1):
  """Fits splines with `num_parts` to data `(x, y)`.

  Finds a piecewise polynomial function `p` of degree `deg` with `num_parts`
  pieces that minimizes the fitting error sum |y_i - p(x_i)| + alpha |p|_1.

  Args:
    x: [N] ndarray of  input data. Must be increasing.
    y: [N] ndarray, same size as `x`.
    num_parts: int, Number of pieces of the piecewise polynomial function `p`.
    deg: int, degree of each polynomial piece of `p`.
    alpha: float, Regularizer.
    smoothness: int, the desired degree of smoothness of `p`, e.g.
      `smoothness==0` corresponds to a continuous `p`.

  Returns:
     [num_parts, deg+1] ndarray representing the piecewise polynomial `p`.
     Entry (i, j)  contains j^th coefficient of the i^th piece of `p`.
  """

  # coefficients of the polynomial of p.
  p = cvxpy.Variable((num_parts, deg + 1), name='p')

  # convert to numpy format because it is easier to work with.
  numpy_p = np.array([[p[i, j] for j in range(deg+1)] \
                    for i in range(num_parts)])

  regularizer = alpha * cvxpy.norm(p, 1)

  num_points_per_part = int(len(x) / num_parts)

  smoothness_constraints = []

  # cuttoff values
  t = []

  fitting_value = 0
  # split the data into equal `num_parts` pieces
  for i in range(num_parts):

    # the part of the data that the current piece fits
    sub_x = x[num_points_per_part * i:num_points_per_part * (i + 1)]
    sub_y = y[num_points_per_part * i:num_points_per_part * (i + 1)]

    # compute p(sub_x)
    # pow_x = np.array([sub_x**k for k in range(deg + 1)])
    # sub_p = polyval(sub_xnumpy_p[i, :].dot(pow_x)
    sub_p = eval_poly_from_coefficients(numpy_p[i], sub_x)

    # fitting value of the current part of p,
    # equal to sqrt(sum |p(x_i) - y_i|^2), where the sum
    # is over data (x_i, y_i) in the current piece.
    fitting_value += cvxpy.norm(cvxpy.vstack(sub_p - sub_y), 1)

    # glue things together by ensuring smoothness of the p at x1
    if i > 0:
      x1 = x[num_points_per_part * i]
      # computes the derivatives p'(x1) for the left and from the right of x1

      # x_deriv is the 2D matrix  k!/(k-j)! x1^(k-j) indexed by (j, k)
      x1_deriv = np.array(
          [[np.prod(range(k - j, k)) * x1**(k - j)
            for k in range(deg + 1)]
           for j in range(smoothness + 1)]).T

      p_deriv_left = numpy_p[i - 1].dot(x1_deriv)
      p_deriv_right = numpy_p[i].dot(x1_deriv)

      smoothness_constraints += [
          cvxpy.vstack(p_deriv_left - p_deriv_right) == 0
      ]
      t.append(x1)
  min_loss = cvxpy.Minimize(fitting_value + regularizer)
  prob = cvxpy.Problem(min_loss, smoothness_constraints)
  prob.solve(verbose=False)

  return _piecewise_polynomial_as_function(p.value, t)


def _piecewise_polynomial_as_function(p, t):
  """Returns the piecewise polynomial `p` as a function.

  Args:
    p: [N, d+1] array of coefficients of p.
    t: [N] array of cuttoffs.

  Returns:
    The function f s.t. f(x) = p_i(x) if t[i] < x < t[i+1].
  """

  def evaluate_p_at(x):
    """Returns p(x)."""

    pieces = [x < t[0]] + [(x >= ti)  & (x < ti_plusone) \
             for ti, ti_plusone in zip(t[:-1], t[1:])] +\
              [x >= t[-1]]

    # pylint: disable=unused-variable
    func_list = [
        lambda u, pi=pi: eval_poly_from_coefficients(pi, u) for pi in p
    ]

    return np.piecewise(x, pieces, func_list)

  return evaluate_p_at


def eval_poly_from_coefficients(coefficients, x):
  """Evaluates the polynomial whose coefficients are `coefficients` at `x`."""
  return coefficients.dot([x**i for i in range(len(coefficients))])