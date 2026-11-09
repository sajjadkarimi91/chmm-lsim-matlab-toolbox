

function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SED forward-backward, cell-array API.
%
% Packs the cell parameters and calls forward_backward_lsim_kernel, which is the
% single implementation shared with forward_backward_lsim_sed and em_lsim.
% Diagonal and full covariance are both supported.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Y_in is observations, a cell of size Cx1 where Y_in{c,1} is the c'th subsystem
% observations (d(c) x T, NaN = missing). C is the total number of subsystems.

%Transition probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transition_matrices_IM is a cell array CxC which cell (i,j) contain
% parameters P(v^j_t|v^i_t-1) (similar to Rabiner) (i->j)

%each cell (i,j) contain a matrix M(i)xM(j) which shows the transition
%probabilities subsystem i on subsystem j according to Rabiner notations  A=transition_matrices_IM{i,j} is transition matrix A(n_j , n_i)= P(v^j_t = n_j | v^i_t = n_i ) & sum_n_j (A(i , j)) = 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Coupling probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupling matrice is a CxC  which  (c,zee) contain parameters
% coupling_tetha_IM(c,zee), so sum(coupling_tetha_IM)=ones(1,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initials hidden states probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pi_0_lsim{j} contain a vector with length M(j) which contain initials hidden
% states probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%emmision probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gmm_para_lsim{j}.gmm_para(n_j).P is  mixture wieght (probability) of the
% k'th GMM component related to hidden state (v^j_t = n_j)

% gmm_para_lsim{j}.gmm_para(n_j).mu(k).x is  mean vector of the k'th GMM component related to hidden state (v^j_t = n_j)

% gmm_para_lsim{j}.gmm_para(n_j).sigma(k).x is the covariance of the k'th GMM
% component related to hidden state (v^j_t = n_j); a d(j)x1 vector for diagonal
% covariance or a d(j)xd(j) matrix for full covariance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fb_constants is optional, see lsim_fb_constants. Defaults to raw POISE.

if nargin < 6
    fb_constants = lsim_fb_constants();
end

packed = lsim_pack_para( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM , gmm_para_lsim , Y_in );

[P_O_model ,alpha , beta , alpha_T , b_c_ot_nc , P_observ_cond_to_state , P_observ_cond_to_state_comp] = ...
    forward_backward_lsim_kernel( packed.pi_0 , packed.transitions_matrices , packed.coupling_tetha_IM , packed.P_all , packed.mu_all , packed.sigma_all , ...
    packed.channel_time_series , packed.state_numbers_index , packed.channel_num_states , packed.dimension_numbers_index , packed.num_gmm_component , packed.sigma_diag , fb_constants );


if(nargout>1)

    C = packed.C;
    state_numbers_index = packed.state_numbers_index;
    num_gmm_component = packed.num_gmm_component;

    alpha_out = cell(C,1);
    beta_out = cell(C,1);
    alpha_T_out = cell(C,1);
    b_c_ot_nc_out = cell(C,1);
    P_observ_cond_to_state_out = cell(C,1);
    P_observ_cond_to_state_comp_out = cell(C,1);

    for zee = 1:C

        zee_index = state_numbers_index(zee)+1:state_numbers_index(zee+1);

        alpha_out{zee,1} =  alpha(zee_index , :);
        beta_out{zee,1} =  beta(zee_index , :);
        alpha_T_out{zee,1} =  alpha_T(zee_index , :);
        b_c_ot_nc_out{zee,1} =  b_c_ot_nc(zee_index , :);

        P_observ_cond_to_state_out{zee,1} =  P_observ_cond_to_state(zee_index , :);
        P_observ_cond_to_state_comp_out{zee,1} = P_observ_cond_to_state_comp(zee_index , 1:num_gmm_component(zee) , :);

    end

end

end
