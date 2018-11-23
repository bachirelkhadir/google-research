function [xt, t_tol, y_tol, i_tol]= integrator(f, x0, T, tolerance)
% Inputs:
% f is a dynamical system - 
%       xdot = f(x) where x, xdot are d-dimensional column vectors
%       Infact, f can be vectorized and can accept d x N inputs and return d x N
%       outputs.
%       For a gradient system, use: f = @(t, state) -g(state')';
% x0 is a column vector
% T is a vector of timepoints where you want the solution.
% 
% Returns: 
%   xt: length(x0) x length(T)
%   You can compute xtdot: length(x0) x length(T) by calling xtdot = f(xt)
%
% Example: 
% A = [0.1 0; 0 0.1]; 
% f = @(x) -A*x; % a linear dynamical system in R^2
% xt  = integrator(f, randn(2,1), 0:100); %% size: 101 x 2
%

    function [value, isterminal, direction] = event_function(t, y)
        value(1) = double(norm(y) > tolerance);
        isterminal(1) = (tolerance>0);
        direction = [];
    end

options=odeset('AbsTol',1e-4,'RelTol',1e-2, 'Events', @event_function);

tic; [~, xt, t_tol, y_tol, i_tol] = ode45(@(t, x) f(x), T, x0, options); toc

end