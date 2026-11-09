
function fb_constants = lsim_fb_constants( varargin )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The common numerical constants of every LSIM forward-backward and EM entry
% point: forward_backward_lsim, forward_backward_lsim_sed,
% forward_backward_lsim_fast, forward_backward_lsim_poise, em_lsim,
% em_lsim_fast and em_lsim_poise all use this one set.
%
% There is deliberately a single set of constants for the whole toolbox. The
% toolbox used to carry one copy of the forward-backward recursion per entry
% point, and those copies had drifted to slightly different regularization
% constants; the values below are the least intrusive of each, and the recursion
% is now shared. Do not reintroduce per-function constant sets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Usage
%   lsim_fb_constants()                        % POISE, raw (the default)
%   lsim_fb_constants('sed')                   % SED variant
%   lsim_fb_constants('fast')                  % fast/SVI variant
%   lsim_fb_constants('poise')                 % POISE, raw (refinement off)
%   lsim_fb_constants('poise','poise_refinement',true)  % POISE + refinement
%   lsim_fb_constants('fast','pi0_reg',1e-6)   % variant then name/value overrides
%
% Overrides exist for experiments and for regression tests that need to
% reproduce a historical setting; production code should just call
% lsim_fb_constants() or lsim_fb_constants('<variant>').

% Fields
%   variant          'sed'  -> pairwise-conditional backward message, delta
%                              weights solved per channel from
%                              x = ( diag(f-h) + h*ones(C,C) ) \ f
%                    'fast' -> SVI backward message with fixed coupling weights
%                              raised to the power
%                              w_opt = sum_w( svi_eps + coupling_tetha_IM(zee,w) )
%                    'poise' -> power-corrected outgoing-influence backward
%                               recursion; see forward_backward_lsim_poise
%   beta_floor       SED only: lower clamp on the mixed backward message
%   pi0_reg          additive regularization of pi_0, per channel, then
%                    renormalized. pi0_reg <= 0 leaves pi_0 exactly as given
%   f_reg            SED only: additive regularization of f in the delta solver
%   svi_eps          fast only: guard used both in the coupling weight
%                    denominator and in the exponent w_opt
%   poise_refinement poise only: apply the one-shot observation-conditioned
%                    refinement. Defaults to FALSE: raw POISE is the faster
%                    recursion and the refinement only does anything for a
%                    channel whose total outgoing strength is below 0.75
%                    (its weight is max(0, 0.75-theta_out)), so paying for it
%                    by default buys nothing on a strongly-coupled model. Pass
%                    'poise_refinement',true to enable it. The Python
%                    package (chmm_lsim.FbConstants) has the same defaults.

% The SED delta weights are not row-normalized. An earlier version of this
% kernel offered that as an option (mirroring one of the two historical
% implementations being unified), but scaling a delta row by a constant
% cancels out of beta/alpha_T further down (both are divided by the
% per-channel normalizer a few lines later in the kernel), so normalizing
% has no effect on the result at any beta_floor actually used. Removed.

fb_constants.variant         = 'poise';
fb_constants.beta_floor      = 10^-9;
fb_constants.pi0_reg         = 10^-10;
fb_constants.f_reg           = 10^-6;
fb_constants.svi_eps         = 10^-8;
fb_constants.poise_refinement = false;

args = varargin;

% optional leading variant name
if ~isempty(args) && ischar(args{1}) && ~any(strcmp(args{1} , fieldnames(fb_constants)))
    fb_constants.variant = args{1};
    args = args(2:end);
end

if rem(length(args),2)~=0
    error('lsim_fb_constants: overrides must be name/value pairs')
end

valid_names = fieldnames(fb_constants);

for i = 1:2:length(args)

    if ~ischar(args{i}) || ~any(strcmp(args{i} , valid_names))
        error('lsim_fb_constants: unknown option ''%s''', num2str(args{i}))
    end

    fb_constants.(args{i}) = args{i+1};

end

if ~any(strcmp(fb_constants.variant , {'sed','fast','poise'}))
    error('lsim_fb_constants: variant must be ''sed'', ''fast'' or ''poise'', got ''%s''', num2str(fb_constants.variant))
end

if ~(islogical(fb_constants.poise_refinement) && isscalar(fb_constants.poise_refinement))
    error('lsim_fb_constants: poise_refinement must be a scalar logical')
end

end
