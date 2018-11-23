function [results, check] = lfd_evaluation(f, demos, isdmp)
%% Dynamical System: xdot = f(x), xdot and x are n-dimensional
% f is (row) vectorized: Xdot  = f(X), Xdot, X are T x n matrices
% demos: training demonstration data (lasa handwriting)
%       demos.pos, demos.vel are n x T 
%
% results: structure with the following fields
%
%  results.trajectory_error
%  results.velocity_error
%  results.distance_to_goal
%  results.fraction_trajectories_converged
%  results.dtwd
% 
if nargin<3
   isdmp = 0; 
end

%% How well does the dynamical system represent the demonstrations?
 
for i=1:length(demos)
   fprintf(1,'%d...', i);
   T = demos{i}.t;
   if ~isdmp
       tic; X = integrator(@(x)f(x')', demos{i}.pos(:, 1), T, 0); results.integration_speed(i) = toc;
       V = f(X);
   else
       tic; [X, V, t]=integrator_dmp(f, demos{i}.pos(:, 1), T(end), demos{i}.dt, 0); results.integration_speed(i) = toc;
   end
   results.trajectory_error(i) = mean(sqrt(sum((X-demos{i}.pos').^2, 2))); %%norm(X - demos{i}.pos','fro');
   results.velocity_error(i) = mean(sqrt(sum((V-demos{i}.vel').^2, 2))); 
   results.distance_to_goal(i) =  norm(X(end,:) - demos{i}.pos(:, end)');
   results.dtwd(i) = Compute_DTWD(demos{i}.pos', X);
   
   check{i} = X;
   results
end



%% How long does it take to hit the target
radius = 1.0; %% simulation runs till 100*T until trajectory enters a 
%ball of radius 1mm around the equilibrium.
fprintf(1,'\n');
for i=1:length(demos)
   fprintf(1,'%d...', i);
   T = demos{i}.t;
   if ~isdmp
       tic; [X, t_tol] = integrator(@(x)f(x')', demos{i}.pos(:, 1), [0, 30*T(end)],radius);
       V = f(X);
       results.dtwd_at_30T(i) = Compute_DTWD(demos{i}.pos', X);
   else
       [X, V, t, t_tol]=integrator_dmp(f, demos{i}.pos(:, 1), 30*T(end), demos{i}.dt, radius);
       results.dtwd_at_30T(i) = Compute_DTWD(demos{i}.pos', X);
   end
   if ~isempty(t_tol)
    results.duration_to_goal(i) = t_tol(1);
   else
      results.duration_to_goal(i) = -1; %% trajectory never comes close -- unstable system?
   end
   results
end
%%return

%% Stability
fprintf(1,'\n');
x = demos{1}.pos';
xmin = floor(min(x(:,1)));
xmax = ceil(max(x(:, 1)));
ymin = floor(min(x(:,2)));
ymax = ceil(max(x(:, 2)));
hx = (xmax - xmin)/3;
hy = (ymax - ymin)/3;
[xx, yy] = meshgrid(xmin:hx:xmax, ymin:hy:ymax);
X0 = [xx(:) yy(:)];

results.grid = X0;
for i = 1:size(X0, 1)
    fprintf(1,'%d...', i);
    x0 = X0(i, :)';
    if ~isdmp
        tic; [x, t_tol] = integrator(@(x)f(x')', x0, [0, 30*T(end)], radius);
        results.grid_integration_speed(i) = toc;
    else
       tic; [x, ~, ~, t_tol]=integrator_dmp(f, x0, 30*T(end), demos{1}.dt, radius); 
       results.grid_integration_speed(i) = toc;
    end 
    
    if ~isempty(t_tol)
        results.grid_duration(i) = t_tol;
    else
        results.grid_duration(i) = -1;
    end
    for j=1:length(demos)
        results.grid_dtwd(i,j) = Compute_DTWD(x, demos{j}.pos');
    end
    xT = x(end, :);
    results.grid_distance_to_goal(i) = norm(xT);
    
    results
end
fprintf(1,'\n');
end
