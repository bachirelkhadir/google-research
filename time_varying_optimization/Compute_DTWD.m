function DTWD = Compute_DTWD(traj1,traj2,w)
% Computes Dynamic Time Wrapping Distance between two trajectories

% NOTE: Trajectories should be of the dimension Txd where d is the
% dimension of the signal and T is the number of samples

if nargin<3
   w = Inf; % no restrictions on window size
end

l_1 = size(traj1,1); % length of first trajectory
l_2 = size(traj2,1); % length of second trajectory
w = max([w,abs(l_1-l_2)]); % window size

DTW = zeros(l_1+1,l_2+1) + Inf;
DTW(1,1) = 0;

for ind_i = 2:1:l_1+1
   for ind_j = max(2,ind_i - w):1:min(l_2+1,ind_i+w) 
       cost = norm(traj1(ind_i-1,:)-traj2(ind_j-1,:));
       DTW(ind_i,ind_j) = cost + min([DTW(ind_i-1,ind_j),DTW(ind_i,ind_j-1),DTW(ind_i-1,ind_j-1)]);      
   end
end

DTWD = DTW(l_1+1,l_2+1); 
end

