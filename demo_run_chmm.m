%DEMO_RUN_CHMM  Full coupled HMM (CHMM) versus LSIM on the same data.
%
%   A CHMM conditions every channel's next state on the JOINT state of all
%   channels, so its parameters and its exact inference grow with prod(M).
%   An LSIM replaces the joint conditional by a theta-weighted mixture of
%   pairwise transitions, so it grows with sum(M)^2. This demo simulates data
%   from a small 2-channel LSIM and fits both models by EM.

clc; clear; close all
addpath(genpath(fileparts(mfilename('fullpath'))));   % this toolbox
rng(2)

C = 2; channel_num_states = [3 3]; channel_dim_observ = [1 2]; T = 1000;
[pi_0, coupling, transitions, gmm_para, transition_chmm] = ...
    random_lsim(channel_num_states, channel_dim_observ);
observations = generate_chmm_time_series(T, channel_num_states, channel_dim_observ, ...
    gmm_para, transition_chmm, pi_0);

extra = struct('plot', 0, 'check_convergence', 0, 'sigma_diag', 1);
num_gmm_component = ones(1, C);
max_iteration = 50;

% CHMM: exact joint-state EM
[pi_chmm, transition_chmm_hat, gmm_chmm, log_lik_chmm, AIC_chmm, BIC_chmm] = ...
    em_chmm(observations, channel_num_states, num_gmm_component, max_iteration, extra);

% LSIM: EM with the POISE E-step (the default)
[pi_lsim, coupling_hat, transitions_hat, gmm_lsim, AIC_lsim, log_lik_lsim, BIC_lsim] = ...
    em_lsim(observations, channel_num_states, num_gmm_component, max_iteration, extra);

fprintf('            final log-likelihood       AIC          BIC\n');
fprintf('  CHMM      %18.2f %12.1f %12.1f\n', log_lik_chmm(end), AIC_chmm(end), BIC_chmm(end));
fprintf('  LSIM      %18.2f %12.1f %12.1f\n', log_lik_lsim(end), AIC_lsim(end), BIC_lsim(end));

%% helper (same generator as demo_run_lsim) -------------------------------------------
function [pi_0, coupling, transitions, gmm_para, transition_chmm] = random_lsim(M, d)
C = numel(M);
coupling = rand(C);
coupling = coupling ./ sum(coupling, 1);
index_matrix = zeros(C, prod(M));
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
        transitions{c, zee} = a ./ sum(a, 2);
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
        gmm_para{zee, 1}.gmm_para(i).sigma(1).x = 0.5 + rand(d(zee), 1);
    end
end
end
