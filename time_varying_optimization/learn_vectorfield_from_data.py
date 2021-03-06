"""Script to learn a vector field from dataset.


python learn_vectorfield_from_data.py --dataset="lasa_dataset/CShape.mat" --matlab_export_file="matlab_code.m" --deg_p=2 --deg_f=2 --alpha_p=1e-2 --alpha_f=1e-2 --tau=0.1

The script starts by fitting a polynomial p(t) of degree deg_p to the data by minimizing the following objective:
tracking error + alpha * norm(p)

Then it fits a `tau`-contracting vectorfield f(x) of degree `deg_f` by minimizing the following objective:
tracking error + alpha * norm(f)

Finally, the script produces a matlab file called `matlab_export_file` containing a matlab function in the format you asked for.

"""


import argparse
import logging
import learning_contracting_polynomials
import polynomial_tools
from sympy import octave_code
import numpy as np
import matplotlib.pyplot as plt
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
        '--make_zero_at_end',
        type=bool,
        default=True,
        help='Import the constraint f(p(T)) = 0.')

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
    x = x
    return t, x, x_dot


def fit_polynomial_to_data(t, x, dx, deg_p, alpha=1e-3, verbose=False):
    poly_fit_fct = polynomial_tools.fit_polynomial_with_regularization
    coefficients_p = [poly_fit_fct(t, xi, dxi, deg=deg_p, alpha=alpha, verbose=verbose)
                      for xi, dxi in zip(x, dx)]

    logging.debug('Coefficients of polynomial fit: %s',
                  coefficients_p)
    return np.array(coefficients_p)


def fit_vectorfield_to_poly_path(coefficients_p, params):
    learning_fct = learning_contracting_polynomials.learn_polynomial_vf_with_contraction
    opt_value, opt_vf = learning_fct(coefficients_p, **params)
    return opt_value, opt_vf


def export_vf_to_matlab_function(function_name, vf, scale, time_scale, translation, header=''):
    template_code = """
% {header}
function Xdot = {function_name}(X)
    scale = {scale};
    time_scale = {time_scale};
    translation = {translation};
    x_0 = (X(:, 1) - translation(1)) / scale;
    x_1 = (X(:, 2) - translation(2)) / scale;

    fx_0 = {fx_0};
    fx_1 = {fx_1};

    Xdot = [fx_0 fx_1] * scale / time_scale;
end
"""
    code = template_code.format(scale=scale,
                                time_scale=time_scale,
                                translation=translation,
                                fx_0=octave_code(vf[0]),
                                fx_1=octave_code(vf[1]),
                                function_name=function_name,
                                header=header)
    return code


def learn_and_output(dataset, matlab_export_file, deg_p, deg_f, 
                     alpha_p, alpha_f, tau, make_zero_at_end):
    msg = 'Fitting a polynomial of degree {deg_p} to {dataset} and learning a vectorfield of degree {deg_f} with tau = {tau}, alpha_p={alpha_p}, alpha_f={alpha_f}. make_zero_at_end={make_zero_at_end}'.format(**locals())
    print(msg)
    step = 10

    t, real_path, real_path_dot = load_data(dataset)
    time_scale = t[-1]
    t = t[::step] / time_scale
    real_path = real_path[:, ::step]
    real_path_dot = real_path_dot[:, ::step]
    
    scale = 1000.
    translation = np.mean(real_path, axis=1)

    scaled_path = (real_path - translation[:, None]) / scale
    scaled_path_dot = real_path_dot / scale * time_scale
    
    verbose = logging.getLogger().getEffectiveLevel() <= logging.INFO

    p = fit_polynomial_to_data(t, scaled_path, scaled_path_dot, 
                               deg_p=deg_p, alpha=alpha_p,
                               verbose=verbose)
    
    params = {
        "deg": deg_f,
        "tau": .1,
        "alpha": alpha_f,
        "make_zero_at_end_demo": make_zero_at_end,
        "verbose": False
    }

    opt_value, opt_vf = fit_vectorfield_to_poly_path(p, params)
    print('Optimal value: %.2f' % opt_value)
    print('Optimal vectorfield f(x_0, x_1) = %s' % opt_vf)

    if matlab_export_file:
        print('Generating matlab code.')
        function_name = os.path.basename(matlab_export_file)[:-2]
        code = export_vf_to_matlab_function(function_name,
                                            opt_vf, 
                                            scale,
                                            time_scale, 
                                            translation,
                                            msg)
        print('Wrinting matlab code to', matlab_export_file)
        open(matlab_export_file, 'w').write(code)
        
    return (opt_value, opt_vf)


def visualize_vf(f, limits=[-.1, .1, -.1, .1]):
    nx, ny = 100, 100
    xx = np.linspace(*limits[:2], nx)
    yy = np.linspace(*limits[2:], ny)
    X, Y = np.meshgrid(xx, yy)
    XY = np.array([X.flatten(), Y.flatten()]).T
    fXY = f(XY)
    Ex, Ey = fXY[:, 0], fXY[:, 1]
    Ex, Ey = Ex.reshape(nx, ny), Ey.reshape(nx, ny)
    plt.streamplot(xx, yy, Ex, Ey,linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)
    plt.xlim(limits[:2])
    plt.ylim(limits[2:])
    
    
if __name__ =='__main__':
    args = parse_arguments()
    dataset = args.dataset
    matlab_export_file = args.matlab_export_file
    deg_p = args.deg_p
    deg_f = args.deg_f
    alpha_p = args.alpha_p
    alpha_f = args.alpha_f
    make_zero_at_end = args.make_zero_at_end
    tau = float(args.tau) if args.tau else None
    
    learn_and_output(dataset, deg_p, deg_f, alpha_p, alpha_f, tau, make_zero_at_end)
