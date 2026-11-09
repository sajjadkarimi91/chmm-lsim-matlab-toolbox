function [pi_0_lsim, coupling_tetha_IM, transition_matrices_IM, ...
    gmm_para_lsim, AIC, log_likelyhood, BIC, pi_steady] = ...
    em_lsim_poise(channels_observations, channel_num_states, ...
    num_gmm_component, max_itration, extra, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM for LSIM using the POISE forward-backward kernel.
%
% This is a thin wrapper over em_lsim, so SED and POISE share exactly the
% same initialization, M-step, AIC/BIC calculation, convergence rule and
% best-iteration bookkeeping. Only the E-step recursion differs.
%
% Raw POISE is the default -- the refinement is OFF unless it is asked for:
%
%   em_lsim_poise(Y, M, K, max_iter, extra)            % raw POISE
%
% It is enabled only by passing the explicit final logical `true`:
%
%   em_lsim_poise(Y, M, K, max_iter, extra, true)
%
% A supplied initialization model remains supported. When both an initial
% model and the refinement switch are supplied, use:
%
%   em_lsim_poise(Y, M, K, max_iter, extra, init_model, true)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 5 || isempty(extra)
    extra = struct();
end

init_model = [];
% Default: raw POISE (refinement off), the same as lsim_fb_constants('poise').
refinement = false;

if numel(varargin) > 2
    error('em_lsim_poise: expected at most init_model and refinement inputs')
elseif numel(varargin) == 1
    if islogical(varargin{1}) && isscalar(varargin{1})
        refinement = varargin{1};
    else
        init_model = varargin{1};
    end
elseif numel(varargin) == 2
    init_model = varargin{1};
    refinement = varargin{2};
end

if ~(islogical(refinement) && isscalar(refinement))
    error('em_lsim_poise: refinement must be a scalar logical')
end

if isfield(extra, 'fb_constants')
    fb_constants = extra.fb_constants;
    fb_constants.variant = 'poise';
    fb_constants.poise_refinement = refinement;
else
    fb_constants = lsim_fb_constants('poise', ...
        'poise_refinement', refinement);
end
extra.fb_constants = fb_constants;

if isempty(init_model)
    [pi_0_lsim, coupling_tetha_IM, transition_matrices_IM, ...
        gmm_para_lsim, AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim(channels_observations, channel_num_states, ...
        num_gmm_component, max_itration, extra);
else
    [pi_0_lsim, coupling_tetha_IM, transition_matrices_IM, ...
        gmm_para_lsim, AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim(channels_observations, channel_num_states, ...
        num_gmm_component, max_itration, extra, init_model);
end

end
