
function packed = lsim_pack_para( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM , gmm_para_lsim , Y_in )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Converts the cell-array LSIM parameter API into the flat packed arrays used
% by forward_backward_lsim_kernel. Diagonal and full covariance are both
% supported and detected automatically.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Inputs (cell API)
%   pi_0_lsim{zee,1}                     M(zee) x 1 initial state probabilities
%   coupling_tetha_IM                    C x C, sum(coupling_tetha_IM) = ones(1,C)
%   transition_matrices_IM{c,zee}        M(c) x M(zee), rows sum to 1
%   gmm_para_lsim{zee,1}.gmm_para(i).P        K(zee) x 1 mixture weights
%   gmm_para_lsim{zee,1}.gmm_para(i).mu(k).x  d(zee) x 1
%   gmm_para_lsim{zee,1}.gmm_para(i).sigma(k).x  d(zee) x 1 (diagonal) or d(zee) x d(zee) (full)
%   Y_in{zee,1}                          d(zee) x T observations, NaN = missing

% Output: struct with the flat packed representation
%   pi_0                    sum(M) x 1                     (NOT regularized; the kernel does that)
%   transitions_matrices    sum(M) x sum(M)
%   coupling_tetha_IM       C x C                          (passed through unchanged)
%   P_all                   C x max(M) x max(K)
%   mu_all                  sum(d) x max(M) x max(K)
%   sigma_all               sum(d) x max(M) x max(K)              when sigma_diag = 1
%                           sum(d) x max(d) x max(M) x max(K)     when sigma_diag = 0
%   sigma_diag              1 for diagonal covariance, 0 for full
%   channel_time_series     sum(d) x T
%   channel_num_states, dim_observation, num_gmm_component        C x 1
%   state_numbers_index, dimension_numbers_index                  (C+1) x 1 block offsets
%   C, T

C = size(Y_in , 1);

dim_observation = zeros(C , 1);
channel_num_states = zeros(C , 1);
num_gmm_component =  zeros(C , 1);

for zee = 1:C

    dim_observation(zee,1)  = size( Y_in{zee,1} ,1);
    channel_num_states(zee,1)  = size( pi_0_lsim{zee,1} ,1);
    num_gmm_component(zee,1)  = length(gmm_para_lsim{zee,1}.gmm_para(1).P);

end

state_numbers_index =  ( [0;cumsum(channel_num_states(:))] );
dimension_numbers_index =  ( [0;cumsum(dim_observation(:))] );

P_all = zeros( C , max(channel_num_states) , max(num_gmm_component)  );
mu_all = zeros(  sum(dim_observation)  , max(channel_num_states) , max(num_gmm_component)   );

% detect diagonal vs full covariance on a channel with d>1, because for d==1 a
% full covariance is also 1x1 and would be misread as diagonal
zee_temp = find(dim_observation>1);

if isempty(zee_temp)
    zee_temp = 1;
else
    zee_temp = zee_temp(1);
end

if size(gmm_para_lsim{zee_temp,1}.gmm_para(1).sigma(1).x,2)==1
    sigma_all =  ones(  sum(dim_observation)  , max(channel_num_states) , max(num_gmm_component) );
    sigma_diag = 1;
else
    sigma_all =  ones(  sum(dim_observation)  , max(dim_observation) , max(channel_num_states) , max(num_gmm_component) );
    sigma_diag = 0;
end

transitions_matrices = ones( sum(channel_num_states) , sum(channel_num_states)  );
pi_0 = zeros( sum(channel_num_states) , 1  );

for zee = 1:C

    gmm_para = gmm_para_lsim{zee,1}.gmm_para ;

    for i=1:channel_num_states(zee)

        for k=1:num_gmm_component(zee)

            P_all(zee,i,k) =  gmm_para(i).P(k,1);
            if sigma_diag
                sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) = gmm_para(i).sigma(k).x ;
            else
                sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1), 1:dim_observation(zee), i , k) = gmm_para(i).sigma(k).x ;
            end
            mu_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) =  gmm_para(i).mu(k).x;

        end

    end

    % initial probabilities: regularization is applied inside the kernel, not here
    pi_0(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1) = pi_0_lsim{zee,1};

    for c=1:C
        % Transition probabilities  M(c) x M(zee)
        transitions_matrices( state_numbers_index(c)+1:state_numbers_index(c+1) , state_numbers_index(zee)+1:state_numbers_index(zee+1) ) = transition_matrices_IM{c,zee}  ;
    end

end

%Y is PxT, P is observation dimension & T is observations length
channel_time_series = zeros(sum(dim_observation) , size(Y_in{1,1},2));

for zee=1:C
    channel_time_series(dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1),:)= Y_in{zee,1};
end

packed.pi_0 = pi_0;
packed.transitions_matrices = transitions_matrices;
packed.coupling_tetha_IM = coupling_tetha_IM;
packed.P_all = P_all;
packed.mu_all = mu_all;
packed.sigma_all = sigma_all;
packed.sigma_diag = sigma_diag;
packed.channel_time_series = channel_time_series;
packed.channel_num_states = channel_num_states;
packed.dim_observation = dim_observation;
packed.num_gmm_component = num_gmm_component;
packed.state_numbers_index = state_numbers_index;
packed.dimension_numbers_index = dimension_numbers_index;
packed.C = C;
packed.T = size( channel_time_series , 2);

end
