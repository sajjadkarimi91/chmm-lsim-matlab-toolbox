%DEMO_RUN_LSIM  LSIM inference and EM learning in a few steps.
%
%   1. Build a random 3-channel LSIM and simulate one observation sequence.
%   2. Smooth it with the three LSIM recursions (POISE, the default; SED; fast/SVI)
%      and compare each with exact inference on the equivalent CHMM.
%   3. Learn an LSIM from the data by EM (POISE E-step, the default).

clc; clear; close all
addpath(genpath(fileparts(mfilename('fullpath'))));   % this toolbox
rng(1)

%% 1. A random LSIM and one simulated sequence
C = 3;                              % channels
channel_num_states = [3 2 4];       % hidden states per channel
channel_dim_observ = [2 1 3];       % observation dimension per channel
T = 400;                            % samples

[pi_0, coupling, transitions, gmm_para, transition_chmm] = ...
    random_lsim(channel_num_states, channel_dim_observ);
observations = generate_chmm_time_series(T, channel_num_states, channel_dim_observ, ...
    gmm_para, transition_chmm, pi_0);
observations{2}(1, [5 50 200]) = NaN;   % missing samples are allowed (NaN)

%% 2. Smoothed marginals: approximate recursions vs exact inference
[~, ~, ~, exact_marginals] = forward_backward_chmm(pi_0, transition_chmm, gmm_para, observations);

fprintf('Mean Hellinger distance to the exact smoothed marginals:\n');
for variant = {'poise', 'sed', 'fast'}
    fbc = lsim_fb_constants(variant{1});
    tic
    [log_lik, ~, ~, marginals] = forward_backward_lsim(pi_0, coupling, transitions, ...
        gmm_para, observations, fbc);
    elapsed = toc;
    h = 0;
    for c = 1:C
        h = h + mean(sqrt(sum((sqrt(marginals{c}) - sqrt(exact_marginals{c})).^2, 1) / 2)) / C;
    end
    fprintf('  %-6s H = %.4f   (%.3f s, approximate log-likelihood %.2f)\n', ...
        upper(variant{1}), h, elapsed, log_lik);
end

%% 3. EM learning (POISE E-step by default; lsim_fb_constants('sed') or 'fast' via extra.fb_constants)
extra = struct('plot', 0, 'check_convergence', 0, 'sigma_diag', 1);
num_gmm_component = ones(1, C);
max_iteration = 50;
[pi_hat, coupling_hat, transitions_hat, gmm_hat, AIC, log_likelihood, BIC] = ...
    em_lsim(observations, channel_num_states, num_gmm_component, max_iteration, extra);
fprintf('\nEM: log-likelihood %.2f after 1 iteration, %.2f after %d (AIC %.1f, BIC %.1f)\n', ...
    log_likelihood(1), log_likelihood(end), max_iteration, AIC(end), BIC(end));
disp('Learned coupling matrix (columns sum to one):'); disp(round(coupling_hat, 3));

%% helper ------------------------------------------------------------------------------
function [pi_0, coupling, transitions, gmm_para, transition_chmm] = random_lsim(M, d)
% Random LSIM parameters, plus the joint-state transitions of the equivalent CHMM
% (needed only by generate_chmm_time_series and exact inference).
C = numel(M);
coupling = rand(C);
coupling = coupling ./ sum(coupling, 1);          % theta(c, zee): influence of c on zee
index_matrix = zeros(C, prod(M));                 % Cartesian product of channel states
for i = 1:C
    after = M; after(1:i) = [];
    before = M; before(i:end) = [];
    index_matrix(i, :) = repmat(kron(1:M(i), ones(1, prod(after))), 1, prod(before));
end
weights = [flip(cumprod(M(C:-1:2))) 1]';
for zee = 1:C
    p = rand(M(zee), 1);
    pi_0{zee, 1} = p / sum(p);
    for c = 1:C
        a = rand(M(c), M(zee)).^2;
        transitions{c, zee} = a ./ sum(a, 2);     % P(x^zee_t | x^c_{t-1})
    end
    for j = 1:size(index_matrix, 2)
        row = weights' * (index_matrix(:, j) - 1) + 1;
        for i = 1:M(zee)
            transition_chmm{zee, 1}(row, i) = sum(arrayfun(@(c) ...
                coupling(c, zee) * transitions{c, zee}(index_matrix(c, j), i), 1:C));
        end
    end
    for i = 1:M(zee)
        gmm_para{zee, 1}.gmm_para(i).P = 1;
        gmm_para{zee, 1}.gmm_para(i).mu(1).x = 1.5 * i + randn(d(zee), 1);
        gmm_para{zee, 1}.gmm_para(i).sigma(1).x = 0.5 + rand(d(zee), 1);   % diagonal
    end
end
end
