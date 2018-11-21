import logging

import cvxpy
import numpy as np
from numpy import array
import sympy
from sympy.core import S

import constraint_maker
import polynomial_tools


# Sympy helper functions


def _replace(sympy_exp, original, new):
  """Replaces every occurrence of `original` with `new` in `sympy_exp`."""

  exp = sympy.Matrix(sympy_exp)
  for old_value, new_values in zip(original, new):
    exp = exp.replace(old_value, new_values)
  return exp


def _iterate_over_monomials(variables, degree):
  """Iterate recursively over monomials of degree `degree` in `variables`."""

  if not variables:
    return set([S.One])
  else:
    # The recursion works as follows:
    # All monomials in variables x_1, ..., x_n
    # x_1^i * m, where m is a monomial in x_2, ..., x_n
    x, tail = variables[0], variables[1:]
    monoms = _iterate_over_monomials(tail, degree)
    for i in range(1, degree + 1):
      monoms |= set(
          [x**i * m for m in _iterate_over_monomials(tail, degree - i)])
    return monoms


def _extract_coeffs_from_expression(sym_t, expr, up_to=0):
  """Extract the coefficients of monomials in `sym_t` in `expr`."""

  c = list(sympy.Poly(expr, sym_t).all_coeffs()[::-1])
  # pad with zeros if degree of expr < up_to
  c += [0] * (up_to - len(c))
  return c


def sym_to_vec_coeffs(sym_t, exp, up_to=0):
  """Converts a vector-valued sympy expression to a matrix of coefficients."""

  exp = exp.tolist()
  X = [[
      _extract_coeffs_from_expression(sym_t, exp_ij, up_to) for exp_ij in exp_i
  ] for exp_i in exp]
  return np.array(X)


def _create_cvxpy_expr_from_string(name):
  logging.debug('Creating cvxpy variable %s', name)
  globals()[name] = cvxpy.Variable(name=name)    
    
def _sympy_expr_to_cvxpy(s):
  return eval(repr(s))


def _get_cvxpy_value_from_variable_name(v):
  return eval(v).value


def _build_matrix_from_scalar_and_vector(scalar, vector):
  """Returns the symbolic matrix [[scalar, vector^T], [vector, scalar*I]]."""

  dim = len(vector)
  matrix = np.array([[S.Zero] * (dim + 1)] * (dim + 1))
  np.fill_diagonal(matrix, scalar)
  matrix[0, 1:] = list(vector)
  matrix[1:, 0] = list(vector)
  return sympy.Matrix(matrix)


def _create_vector_valued_polynomial(dimension,
                                     degree,
                                     variables,
                                     name):
  """Create a symbolic vector valued polynomial."""

  coefficients_polynomial = []
  coefficients_names = []

  monomials = sorted(list(_iterate_over_monomials(variables, degree)), key=lambda x: x.sort_key())
  for d in range(dimension):
    names = [name % (d, i) for i in range(len(monomials))]
    coefficients_names += names
    coefficients_polynomial.append(sympy.symbols(names))

  polynomial = [(sympy.Matrix(coeff_i).T * sympy.Matrix(monomials))[0, 0]
                for coeff_i in coefficients_polynomial]

  return polynomial, coefficients_names


def _replace_symbolic_variables_with_value(expr, names_of_variables_to_replace):
  """Replace symbolic variables in `expr` with their optimal value."""

  for v in names_of_variables_to_replace:
    sv = sympy.Symbol(v)
    vv = _get_cvxpy_value_from_variable_name(v)
    for i in range(len(expr)):
      expr[i] = expr[i].subs(sv, vv)

  return expr


def learn_polynomial_vf_with_contraction(coeff_p,
                                         deg,
                                         tau,
                                         alpha=1e-4,
                                         make_zero_at_end_demo=False,
                                         solver=cvxpy.MOSEK,
                                         verbose=False):
  """Learns a contracting polynomial vector field from demonstration.

  Solves the current problem for a polynomial f: R^dim --> R^dim of degree
  `deg`:

  min  int_0^1 |f(p(t)) - p_dot(t)| dt + alpha |f|_1
  s.t. -tau I - J_f (p(t)) - J_f(p(t))^T  >= 0 forall t in [0, 1]

  Args:
    coeff_p: [dim, deg_p] ndarray, matrix of coefficients of the demonstration,
      given by a one dimensional curve p(t).
    deg: int, degree of the vector field to be learned.
    tau: float, Amount of contraction.
    alpha: float, Regularizer.
    make_zero_at_end_demo: bool, If true, adds the constraint f(p(1)) == 1.
    solver: SDP solver to use to solve the resulting TV-SDP.

  Returns:
    The optimal value and the optima l vector field obtained (opt_v, opt_f).
    opt_f is a sympy expression in the variable x_0, ..., x_{dim-1}.
  """

  deg_f = deg
  dim = len(coeff_p)
  deg_p = len(coeff_p[0]) - 1
  deg_f_p = deg_p * deg_f
  deg_J_f_p = deg_p * (deg_f - 1)
  deg_s = deg_f_p

  logging.info('degree of p(t) = %d, degree of f(x) = %d, dim = %d, tau = %s.',
               deg_p, deg_f, dim, tau)

  logging.info('Creating symbolic variables')
  sym_t = sympy.Symbol('t')
  sym_z = sympy.symbols('x_0:%d' % dim)
  sym_p = list(coeff_p.dot([sym_t**i for i in range(deg_p + 1)]))
  sym_p_dot = sympy.Matrix(sym_p).diff(sym_t)

  # create a polynomial f: R^dim --> R^dim of degree deg in variables sym_z
  sym_f, f_coefficients_names = _create_vector_valued_polynomial(
      dim, deg_f, sym_z, name='f%d_%d')
        
  # create a polynomial s: [0, 1] --> R of degree deg_s in variables sym_t
  sym_s, s_coefficients_names = _create_vector_valued_polynomial(
      1, deg_s, [sym_t], name='s%d_%d')
  sym_s = sym_s[0]

  logging.info('Compute the jacobian of f(x).')
  sym_Jf = sympy.Matrix(sym_f).jacobian(sym_z)

  logging.info('Compute f(p(t)), Jf(p(t)) and f(p(t)) - pdot(t).')
  sym_f_o_p = _replace(sym_f, sym_z, sym_p)
  sym_Jf_o_p = _replace(sym_Jf, sym_z, sym_p)
  sym_tracking_error = sym_f_o_p - sym_p_dot

  logging.info('Declare f(p(1))')
  sym_f_o_p_at_1 = list(sym_f_o_p.subs(sym_t, 1))

  logging.info('Declare X_error = [[s error^T], [error sI]].')
  sym_X_error = _build_matrix_from_scalar_and_vector(sym_s, sym_tracking_error)

  logging.info('Declare X_contraction = -Jf - Jf.T - tau * I.')
  Id = sympy.eye(dim)
  if tau is not None:
      sym_X_contraction = -sym_Jf_o_p - sym_Jf_o_p.T - tau * Id

  logging.info('End of symbolic declarations,'
               'converting from sympy polynomials to arrays of coefficient.')
  if tau is not None:
      X_contraction_coefficients = sym_to_vec_coeffs(sym_t, sym_X_contraction,
                                                 deg_J_f_p + 1)
  X_error_coefficients = sym_to_vec_coeffs(sym_t, sym_X_error, deg_s + 1)
  s_coefficients = sym_to_vec_coeffs(sym_t, sympy.Matrix([[sym_s]]))

  logging.info('Creating cvx vars for f(x) and s(t).')
  for v in f_coefficients_names + s_coefficients_names:
    _create_cvxpy_expr_from_string(name=v)

  logging.info('Converting sympy expressions to cvxpy.')
  if tau is not None:
      cvx_X_contraction = _sympy_expr_to_cvxpy(X_contraction_coefficients)
  cvx_X_error = _sympy_expr_to_cvxpy(X_error_coefficients)
  cvx_s = _sympy_expr_to_cvxpy(s_coefficients)[0, 0]
  cvx_f_o_p_at_1 = _sympy_expr_to_cvxpy(sym_f_o_p_at_1)
  cvx_coeff_f = list(map(eval, f_coefficients_names))

  logging.info('Declare constraints')
  constraints = []
  
  # X_contraction >= 0 on [0, 1]
  if tau is not None:  
      constraints += constraint_maker.make_poly_matrix_psd_on_0_1(cvx_X_contraction)
  # X_error  >= 0 on [0, 1]
  constraints += constraint_maker.make_poly_matrix_psd_on_0_1(cvx_X_error)
  # f(p(1)) == 0
  if make_zero_at_end_demo:
    constraints += [ci == 0 for ci in cvx_f_o_p_at_1]

  logging.info('Declare objective = int_0^1 s(t) dt + alpha*norm(f).')
  norm_f = cvxpy.norm(cvxpy.vstack(cvx_coeff_f), 1)
  arc_length = polynomial_tools.integ_poly_0_1(cvx_s)
  objective = cvxpy.Minimize(arc_length + alpha * norm_f)

  logging.info('Solving...')
  cvx_problem = cvxpy.Problem(objective, constraints)
  cvx_problem.solve(solver=solver, verbose=verbose)

  logging.info('Problem status is %s, tracking error is %.2f.'
               'Converting f back to sympy.',
               cvx_problem.status, cvx_problem.value)

  # replace symbolic coefficients in opt_f with their value
  opt_f = _replace_symbolic_variables_with_value(sym_f, f_coefficients_names)
  
  logging.info('Done.')
  
  return (cvx_problem.value, opt_f)
      


