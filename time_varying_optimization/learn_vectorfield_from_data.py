# Sample run: python learn_vectorfield_from_data.py --dataset="lasa_dataset/CShape.mat" --matlab_export_file="matlab_code.m"

import argparse
import logging
import learning_contracting_polynomials
import polynomial_tools
from sympy import octave_code
import numpy as np
import os
import scipy.io as sio

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='CShape.mat',
        help='Dataset File.')

    parser.add_argument(
        '--deg_p',
        type=int,
        default=4,
        help='Degree of the interpolating polynomial.')

    parser.add_argument(
        '--alpha_f',
        type=float,
        default=1e-4,
        help='Regularization coefficients for fitting p(t).')

    parser.add_argument(
        '--deg_f',
        type=int,
        default=4,
        help='Degree of the vectorfield')

    parser.add_argument(
        '--alpha_p',
        type=float,
        default=1e-4,
        help='Regularization coefficients for fitting f(x).')
    parser.add_argument(
        '--tau',
        type=str,
        default=None,
        help='Amount of contraction (None for no contraction).')

    parser.add_argument(
        '--matlab_export_file',
        type=str,
        default=None,
        help='Name of the file to export matlab code to. None for no export.')

    args = parser.parse_args()
    return args

def load_data(data_file):
    data = sio.loadmat(data_file)
    demos = data['demos'][0]

    x, t, x_dot, x_acc, dt  = map(lambda u: u, demos[0][0][0])
    t = t[0]
    t /= t[-1]
    x = x
    return t, x


def fit_polynomial_to_data(t, x, deg_p, alpha=1e-3):
    poly_fit_fct = polynomial_tools.fit_polynomial_with_regularization
    coefficients_p = [poly_fit_fct(t, xi, deg=deg_p, alpha=1e-3)
                      for xi in x]

    logging.debug('Coefficients of polynomial fit: %s',
                  coefficients_p)
    return np.array(coefficients_p)


def fit_vectorfield_to_poly_path(coefficients_p, params):
    # dim = len(coefficients_p)
    # logging.debug('Dimension = %d', dim)
    learning_fct = learning_contracting_polynomials.learn_polynomial_vf_with_contraction
    opt_value, opt_vf = learning_fct(coefficients_p, **params)
    # sym_x = sympy.symbols('x_0:%d' % dim)
    # vector_field_f = lambdify(sym_x, opt_vf)
    return opt_value, opt_vf


def export_vf_to_matlab_function(function_name, vf, scale, translation, header=''):
    template_code = """
% {header}
function Xdot = {function_name}(X)
    scale = {scale};
    translation = {translation};
    x_0 = (X(:, 1) - translation(1)) / scale;
    x_1 = (X(:, 2) - translation(2)) / scale;

    fx_0 = {fx_0};
    fx_1 = {fx_1};

    Xdot = [fx_0 fx_1] * scale;
end
"""
    code = template_code.format(scale=scale,
                                translation=translation,
                                fx_0=octave_code(vf[0]),
                                fx_1=octave_code(vf[1]),
                                function_name=function_name,
                                header=header)
    return code


def main():
    args = parse_arguments()
    dataset = args.dataset
    deg_p = args.deg_p
    deg_f = args.deg_f
    alpha_p = args.alpha_p
    alpha_f = args.alpha_f
    tau = float(args.tau) if args.tau else None

    msg = 'Fitting a polynomial of degree {deg_p} to {dataset} and learning a vectorfield of degree {deg_f} with tau = {tau}, alpha_p={alpha_p}, alpha_f={alpha_f}.'.format(**locals())
    print(msg)

    step = 10

    t, real_path = load_data(dataset)
    t = t[::step]
    real_path = real_path[:, ::step]

    scale = 1000.
    translation = np.mean(real_path, axis=1)

    scaled_path = (real_path - translation[:, None]) / scale

    p = fit_polynomial_to_data(t, scaled_path, deg_p=deg_p, alpha=alpha_p)
    params = {
        "deg": deg_f,
        "tau": .1,
        "alpha": alpha_f,
        "make_zero_at_end_demo": True,
        "verbose": False
    }

    opt_value, opt_vf = fit_vectorfield_to_poly_path(p, params)
    print('Optimal value: %.2f' % opt_value)
    print('Optimal vectorfield f(x_0, x_1) = %s' % opt_vf)

    if args.matlab_export_file:
        print('Generating matlab code.')
        function_name = os.path.basename(args.matlab_export_file)[:-2]
        code = export_vf_to_matlab_function(function_name,
                                            opt_vf, 
                                            scale, 
                                            translation,
                                            msg)
        print('Wrinting matlab code to', args.matlab_export_file)
        open(args.matlab_export_file, 'w').write(code)


if __name__ =='__main__':
    main()
