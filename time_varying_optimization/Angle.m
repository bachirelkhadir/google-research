
% Fitting a polynomial of degree 4 to lasa_dataset/Angle.mat and learning a vectorfield of degree 4 with tau = 0.0, alpha_p=1e-08, alpha_f=1e-08. make_zero_at_end=False
function Xdot = Angle(X)
    scale = 1000.0;
    time_scale = 2.451473384159569;
    translation = [-21.47615921  17.53288585];
    x_0 = (X(:, 1) - translation(1)) / scale;
    x_1 = (X(:, 2) - translation(2)) / scale;

    fx_0 = 132236.338943025*x_0.^4 + 110080.359288411*x_0.^3.*x_1 - 34181.9740874657*x_0.^3 + 25740.6852365444*x_0.^2.*x_1.^2 + 9.69268095459483*x_0.^2.*x_1 - 89.0938920636763*x_0.^2 - 7395.92573694926*x_0.*x_1.^3 + 1532.88885678874*x_0.*x_1.^2 - 480.794839360226*x_0.*x_1 + 6.34800355619336*x_0 + 455.941494976966*x_1.^4 + 3257.42350564328*x_1.^3 + 55.872359305657*x_1.^2 + 0.241356873711538*x_1 + 0.0421688907708813;
    fx_1 = 134203.537838083*x_0.^4 + 114233.401573667*x_0.^3.*x_1 + 20276.2459287119*x_0.^3 + 100779.73928263*x_0.^2.*x_1.^2 - 6925.14585223624*x_0.^2.*x_1 - 168.336489771709*x_0.^2 - 22992.0089883945*x_0.*x_1.^3 + 568.913888399177*x_0.*x_1.^2 + 24.2338798028622*x_0.*x_1 - 11.5234438443863*x_0 + 1507.14868252311*x_1.^4 - 1589.54579812072*x_1.^3 - 119.928582015165*x_1.^2 - 0.962773843933519*x_1 + 0.0426347347211049;

    Xdot = [fx_0 fx_1] * scale / time_scale;
end
