function [P_O_model, alpha_out, beta_out, alpha_T_out, b_c_ot_nc_out, ...
    P_observ_cond_to_state_out, P_observ_cond_to_state_comp_out] = ...
    forward_backward_lsim_poise(pi_0_lsim, coupling_tetha_IM, ...
    transition_matrices_IM, gmm_para_lsim, Y_in, fb_constants, refinement)

% POISE forward-backward inference with optional refinement.
%
% The first five inputs and seven outputs follow forward_backward_lsim_sed.
% Refinement is OFF by default -- it is whatever fb_constants.poise_refinement
% says, and lsim_fb_constants defaults that to false (raw POISE is the faster
% recursion, and the refinement's weight max(0,0.75-theta_out) is zero for any
% channel whose total outgoing strength is at least 0.75). It can be set
% explicitly either as the sixth input
%
%   forward_backward_lsim_poise(pi_0, theta, A, gmm, Y, true)
%
% or after an optional numerical-constants struct:
%
%   forward_backward_lsim_poise(pi_0, theta, A, gmm, Y, fb_constants, false)
%
% Raw POISE uses the power-outside recursion. For channel xi,
%
%   theta_out = sum_c theta(xi,c)
%   rho_c     = theta(xi,c)/theta_out
%   q_t(m)    = sum_c rho_c sum_n A{xi,c}(m,n) beta_{t+1}^c(n)
%   beta_t(m) proportional to b_t(m) q_t(m)^theta_out.
%
% The one-shot observation-conditioned refinement uses frozen raw smoothed
% marginals. Its only weight rule is
%
%   w1 = max(0, 0.75-theta_out),  w2 = 1-w1.

if nargin < 6 || isempty(fb_constants)
    fb_constants = lsim_fb_constants('poise');
end

if (islogical(fb_constants) || (isnumeric(fb_constants) && isscalar(fb_constants))) && nargin < 7
    refinement = logical(fb_constants);
    fb_constants = lsim_fb_constants('poise');
elseif nargin < 7 || isempty(refinement)
    % One source of truth for the default: the constants struct, which now
    % defaults poise_refinement to false. This function used to hardcode `true`
    % here regardless of what fb_constants said, so the two disagreed.
    if isstruct(fb_constants) && isfield(fb_constants, 'poise_refinement')
        refinement = logical(fb_constants.poise_refinement);
    else
        refinement = false;
    end
end

if ~(islogical(refinement) && isscalar(refinement))
    error('forward_backward_lsim_poise: refinement must be a scalar logical.')
end

if ~isstruct(fb_constants) || ~isfield(fb_constants, 'pi0_reg')
    error('forward_backward_lsim_poise: invalid fb_constants input.')
end

packed = lsim_pack_para(pi_0_lsim, coupling_tetha_IM, ...
    transition_matrices_IM, gmm_para_lsim, Y_in);

C = packed.C;
T = packed.T;
state_numbers = packed.channel_num_states;
state_index = packed.state_numbers_index;
num_states = state_index(end);

if packed.sigma_diag
    [P_observ_cond_to_state, P_observ_cond_to_state_comp] = ...
        gmm_pdf_fast(packed.P_all, packed.mu_all, packed.sigma_all, ...
        packed.channel_time_series, state_numbers, ...
        packed.dimension_numbers_index, packed.num_gmm_component);
else
    [P_observ_cond_to_state, P_observ_cond_to_state_comp] = ...
        gmm_pdf(packed.P_all, packed.mu_all, packed.sigma_all, ...
        packed.channel_time_series, state_numbers, ...
        packed.dimension_numbers_index, packed.num_gmm_component);
end

alpha = zeros(num_states, T);
beta_raw = zeros(num_states, T);
alpha_T_raw = zeros(num_states, T);
b_c_ot_nc = zeros(num_states, T);
P_ot_c_cond_past = zeros(C, T);

mat_mult = zeros(num_states, C);
block_sum = zeros(num_states, C);
incoming_weights = zeros(num_states, C);
rho_out = zeros(num_states, C);
theta_out_channel = sum(coupling_tetha_IM, 2);
theta_out = zeros(num_states, 1);
linear_index = zeros(num_states, 1);
cursor = 0;

pi_0 = packed.pi_0;
for xi = 1:C
    ix = state_index(xi)+1:state_index(xi+1);
    if fb_constants.pi0_reg > 0
        pi_0(ix) = pi_0(ix) + fb_constants.pi0_reg;
        pi_0(ix) = pi_0(ix)/sum(pi_0(ix));
    end

    n = numel(ix);
    linear_index(cursor+1:cursor+n) = ((xi-1)*num_states + ix)';
    cursor = cursor+n;
    incoming_weights(ix, :) = repmat(coupling_tetha_IM(:, xi)', n, 1);
    theta_out(ix) = theta_out_channel(xi);
    if theta_out_channel(xi) > 0
        rho_out(ix, :) = repmat(...
            coupling_tetha_IM(xi, :)/theta_out_channel(xi), n, 1);
    end
end
block_sum(linear_index) = 1;

% Forward recursion shared by SED and POISE.
alpha(:, 1) = pi_0;
P_ot_c_cond_past(:, 1) = block_sum'*(alpha(:, 1).*P_observ_cond_to_state(:, 1));
b_c_ot_nc(:, 1) = P_observ_cond_to_state(:, 1)./(block_sum*P_ot_c_cond_past(:, 1));

for t = 2:T
    mat_mult(linear_index) = alpha(:, t-1).*b_c_ot_nc(:, t-1);
    predicted = packed.transitions_matrices'*mat_mult;
    alpha(:, t) = sum(predicted.*incoming_weights, 2);
    P_ot_c_cond_past(:, t) = block_sum'*(alpha(:, t).*P_observ_cond_to_state(:, t));
    b_c_ot_nc(:, t) = P_observ_cond_to_state(:, t)./(block_sum*P_ot_c_cond_past(:, t));
end

% Raw POISE backward recursion: power outside the outgoing mixture.
beta_raw(:, T) = b_c_ot_nc(:, T);
alpha_T_raw(:, T) = alpha(:, T).*beta_raw(:, T);

for t = T-1:-1:1
    mat_mult(linear_index) = beta_raw(:, t+1);
    child_messages = packed.transitions_matrices*mat_mult;
    q = sum(child_messages.*rho_out, 2);
    beta_t = b_c_ot_nc(:, t).*(q.^theta_out);
    beta_t(theta_out == 0) = b_c_ot_nc(theta_out == 0, t);

    [alpha_T_raw(:, t), normalizer] = normalize_blocks(alpha(:, t).*beta_t, block_sum);
    beta_raw(:, t) = beta_t./normalizer;
end

alpha_T = alpha_T_raw;

% One-shot refinement from frozen raw marginals.
if refinement && T > 1
    w1_channel = max(0, 0.75-theta_out_channel);
    w1 = zeros(num_states, 1);
    for xi = 1:C
        ix = state_index(xi)+1:state_index(xi+1);
        w1(ix) = w1_channel(xi);
    end

    for t = 2:T
        mat_mult(linear_index) = alpha_T_raw(:, t-1);
        propagated = packed.transitions_matrices'*mat_mult;
        propagated = sum(propagated.*incoming_weights, 2).*b_c_ot_nc(:, t);
        propagated = normalize_blocks(propagated, block_sum);
        alpha_T(:, t) = w1.*propagated + (1-w1).*alpha_T_raw(:, t);
    end
end

P_O_model = sum(log(P_ot_c_cond_past(:)));

if nargout > 1
    beta_effective = zeros(size(alpha_T));
    positive_alpha = alpha > 0;
    beta_effective(positive_alpha) = alpha_T(positive_alpha)./alpha(positive_alpha);

    alpha_out = cell(C, 1);
    beta_out = cell(C, 1);
    alpha_T_out = cell(C, 1);
    b_c_ot_nc_out = cell(C, 1);
    P_observ_cond_to_state_out = cell(C, 1);
    P_observ_cond_to_state_comp_out = cell(C, 1);

    for xi = 1:C
        ix = state_index(xi)+1:state_index(xi+1);
        alpha_out{xi, 1} = alpha(ix, :);
        beta_out{xi, 1} = beta_effective(ix, :);
        alpha_T_out{xi, 1} = alpha_T(ix, :);
        b_c_ot_nc_out{xi, 1} = b_c_ot_nc(ix, :);
        P_observ_cond_to_state_out{xi, 1} = P_observ_cond_to_state(ix, :);
        P_observ_cond_to_state_comp_out{xi, 1} = ...
            P_observ_cond_to_state_comp(ix, 1:packed.num_gmm_component(xi), :);
    end
end

end

function [normalized, normalizer] = normalize_blocks(values, block_sum)

normalizer_channel = block_sum'*values;
if any(~isfinite(normalizer_channel)) || any(normalizer_channel <= 0)
    error('forward_backward_lsim_poise: nonpositive or nonfinite channel normalizer.')
end
normalizer = block_sum*normalizer_channel;
normalized = values./normalizer;

end
