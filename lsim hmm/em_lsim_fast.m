

function [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = em_lsim_fast( channels_observations , channel_num_states , num_gmm_component , max_itration , extra, init_model)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM for LSIM with the SVI (fast) forward-backward.
%
% This is em_lsim with a different forward-backward variant and nothing else:
% same initialization, same M-step, same AIC/BIC, same convergence and
% best-iterate logic. It used to be a full copy of em_lsim carrying its own
% private kernel; the two main bodies were verified identical apart from the
% kernel call, so this now simply delegates.
%
% The fast variant replaces the SED pairwise conditionals with fixed coupling
% weights raised to the power w_opt = sum_w( svi_eps + coupling_tetha_IM(zee,w) ),
% which also means the sum(M)^2 x T arrays the SED variant needs are never
% allocated.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See em_lsim for the full description of the arguments and outputs.
%
% extra.fb_constants defaults to lsim_fb_constants('fast') and may be set to
% override individual constants, e.g. lsim_fb_constants('fast','pi0_reg',1e-6).

if nargin < 5 || isempty(extra)
    extra = struct();
end

if ~isfield(extra,'fb_constants')
    extra.fb_constants = lsim_fb_constants('fast');
end

if ~strcmp(extra.fb_constants.variant , 'fast')
    error('em_lsim_fast: extra.fb_constants.variant must be ''fast''')
end

if nargin == 6
    [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra, init_model);
else
    [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
end

end
