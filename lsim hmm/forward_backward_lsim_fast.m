
function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim_fast( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVI (fast) forward-backward, cell-array API.
%
% Thin wrapper over forward_backward_lsim_kernel with variant 'fast', the single
% implementation shared with forward_backward_lsim, forward_backward_lsim_sed,
% em_lsim and em_lsim_fast. The backward message uses fixed coupling weights
% raised to the power w_opt = sum_w( svi_eps + coupling_tetha_IM(zee,w) ) instead
% of the SED pairwise conditionals, which also means the sum(M)^2 x T arrays the
% SED variant needs are never allocated here.
%
% As a result of the unification this function now also supports full
% covariance matrices, using the same lsim_fb_constants as every other LSIM
% forward-backward and EM entry point.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See forward_backward_lsim for the full description of the arguments.

if nargin < 6
    fb_constants = lsim_fb_constants('fast');
end

if ~strcmp(fb_constants.variant , 'fast')
    error('forward_backward_lsim_fast: fb_constants.variant must be ''fast''')
end

if nargout>1

    [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out] = ...
        forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

else

    P_O_model = forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

end

end
