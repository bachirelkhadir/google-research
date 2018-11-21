
% Fitting a polynomial of degree 4 to lasa_dataset/CShape.mat and learning a vectorfield of degree 4 with tau = None, alpha_p=0.0001, alpha_f=0.0001.
function Xdot = matlab_code(X)
    scale = 1000.0;
    translation = [-17.31737402  16.49598385];
    x_0 = (X(:, 1) - translation(1)) / scale;
    x_1 = (X(:, 2) - translation(2)) / scale;

    fx_0 = -0.000204640716469697*x_0.^4 - 5.96344157380683e-5*x_0.^3.*x_1 - 8.76352278200918e-5*x_0.^3 - 1.57065852229631e-5*x_0.^2.*x_1.^2 + 0.000493304340291923*x_0.^2.*x_1 - 0.00922917972876929*x_0.^2 + 1.24969575833683e-5*x_0.*x_1.^3 + 1.23890129660652e-5*x_0.*x_1.^2 + 0.189962978091729*x_0.*x_1 - 1.60727852315499*x_0 + 1.61469388531273e-5*x_1.^4 + 0.000363405324261876*x_1.^3 + 0.0497694986836169*x_1.^2 - 4.63217661470223*x_1 - 0.0230103660925393;
    fx_1 = 0.000108006062477885*x_0.^4 - 1.66103568269637e-5*x_0.^3.*x_1 + 0.000887971168459636*x_0.^3 - 5.61297935379643e-5*x_0.^2.*x_1.^2 + 0.000167052364137464*x_0.^2.*x_1 - 0.00457230829791837*x_0.^2 - 2.79142445680633e-5*x_0.*x_1.^3 + 0.000346371993319082*x_0.*x_1.^2 + 0.0842563706367637*x_0.*x_1 + 2.46814915817516*x_0 + 3.25859453118893e-6*x_1.^4 - 0.000927685320966981*x_1.^3 + 0.0272985320104774*x_1.^2 - 0.804649696628092*x_1 - 0.049503995134469;

    Xdot = [fx_0 fx_1] * scale;
end