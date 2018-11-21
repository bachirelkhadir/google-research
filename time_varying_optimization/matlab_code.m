
% Fitting a polynomial of degree 2 to lasa_dataset/CShape.mat and learning a vectorfield of degree 2 with tau = 0.1, alpha_p=0.01, alpha_f=0.01.
function Xdot = matlab_code(X)
    scale = 1000.0;
    translation = [-17.31737402  16.49598385];
    x_0 = (X(:, 1) - translation(1)) / scale;
    x_1 = (X(:, 2) - translation(2)) / scale;

    fx_0 = -3.98774946185103e-9*x_0.^2 - 6.27934155221722e-10*x_0.*x_1 - 0.0500003977770037*x_0 + 1.76723065196076e-10*x_1.^2 - 0.564032891205858*x_1 - 0.0125065997492261;
    fx_1 = 3.36563625258278e-9*x_0.^2 - 3.48389080878787e-9*x_0.*x_1 + 0.564032593399715*x_0 + 1.62216808945821e-9*x_1.^2 - 0.0500009267571564*x_1 - 0.0189373470474994;

    Xdot = [fx_0 fx_1] * scale;
end
