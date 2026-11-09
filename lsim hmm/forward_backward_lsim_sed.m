

function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim_sed( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SED forward-backward. Kept as an alias of forward_backward_lsim.
%
% This file used to hold a separate implementation that solved the beta-mixing
% weights via the Sherman-Morrison identity, while forward_backward_lsim solved
% them by an expanded closed form. The two are the same expression:
%
%   Sherman-Morrison:  x = A^-1 f - A^-1 u (u' A^-1 f)/(1 + u' A^-1 u)
%                      with A = diag(f-h), u = sqrt(h)*ones(C,1)
%                       => x = ( diag(f-h) + h*ones(C,C) ) \ f
%   closed form:       delta(w) = ( f(w) - g )/( f(w) - h ) ,
%                      g = h*sum_v f(v)/(f(v)-h) / (1 + h*sum_v 1/(f(v)-h))
%
% Expanding the first yields the second exactly (verified numerically to ~1e-15
% for C = 2,3,5 in both the f>h and f<h regimes). The two files differed only in
% their regularization constants and in the fact that this version supported
% diagonal covariance only; there is now a single set of constants
% (lsim_fb_constants) shared by every LSIM forward-backward and EM entry point,
% and this function also supports full covariance as a result of the
% unification.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See forward_backward_lsim for the full description of the arguments.

if nargin < 6
    fb_constants = lsim_fb_constants('sed');
end

if ~strcmp(fb_constants.variant , 'sed')
    error('forward_backward_lsim_sed: fb_constants.variant must be ''sed''')
end

if nargout>1

    [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out] = ...
        forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

else

    P_O_model = forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

end

end
