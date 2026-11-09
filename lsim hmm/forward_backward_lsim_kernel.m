
function [P_O_model ,alpha , beta , alpha_T , b_c_ot_nc , P_observ_cond_to_state , P_observ_cond_to_state_comp]...
    = forward_backward_lsim_kernel( pi_0_lsim , transitions_matrices , coupling_tetha_IM , P_all , mu_all  , sigma_all ,  channel_time_series , state_numbers_index , channel_num_states ,  dimension_numbers_index , num_gmm_component , sigma_diag , fb_constants )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LSIM forward-backward, single implementation on packed arrays.
%
% This is the one kernel behind forward_backward_lsim, forward_backward_lsim_sed,
% forward_backward_lsim_fast, em_lsim and em_lsim_fast. Those entry points used to
% carry separate copies of the same recursion; within a variant the copies were
% identical apart from the constants in lsim_fb_constants and some dead leftover
% allocations. Both diagonal and full covariance are supported.
%
% Three variants, selected by fb_constants.variant:
%   'sed'   pairwise-conditional backward message, delta weights solved per
%           channel from x = ( diag(f-h) + h*ones(C,C) ) \ f
%   'fast'  SVI backward message with fixed coupling weights, raised to the
%           power w_opt = sum_w( svi_eps + coupling_tetha_IM(zee,w) )
%   'poise' power-corrected outgoing-influence recursion, optionally followed
%           by the one-shot observation-conditioned refinement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Inputs are the flat packed representation produced by lsim_pack_para:
%   pi_0_lsim              sum(M) x 1, raw (regularized below with fb_constants.pi0_reg)
%   transitions_matrices   sum(M) x sum(M), block (c,zee) is M(c) x M(zee) with rows summing to 1
%                          A(n_c , n_zee) = P( v^zee_t = n_zee | v^c_t-1 = n_c )
%   coupling_tetha_IM      C x C, sum(coupling_tetha_IM) = ones(1,C)
%   P_all                  C x max(M) x max(K)
%   mu_all                 sum(d) x max(M) x max(K)
%   sigma_all              sum(d) x max(M) x max(K)           when sigma_diag = 1
%                          sum(d) x max(d) x max(M) x max(K)  when sigma_diag = 0
%   channel_time_series    sum(d) x T, NaN = missing
%   state_numbers_index    [0;cumsum(M)]
%   dimension_numbers_index [0;cumsum(d)]
%   fb_constants           optional, see lsim_fb_constants (defaults used if omitted)

if nargin < 13 || isempty(fb_constants)
    fb_constants = lsim_fb_constants();
end

flag_sed = strcmp(fb_constants.variant , 'sed');
flag_poise = strcmp(fb_constants.variant , 'poise');

C =  ( length( channel_num_states ));
T =  ( size( channel_time_series , 2) );

P_ot_c_cond_past = zeros( C , T  );
b_c_ot_nc  = zeros( state_numbers_index(end) , T );

alpha = zeros( state_numbers_index(end) , T  );
alpha_T  = zeros( state_numbers_index(end) , T  );
beta  = zeros( state_numbers_index(end) , T );
a_alpha_b = zeros(state_numbers_index(end) , C , T-1 );


% emission probabilities
if sigma_diag
    [P_observ_cond_to_state  ,   P_observ_cond_to_state_comp ]  = ...
        gmm_pdf_fast( P_all , mu_all  , sigma_all  , channel_time_series   ,  channel_num_states ,  dimension_numbers_index , num_gmm_component  );
else
    [P_observ_cond_to_state  ,   P_observ_cond_to_state_comp ]  = ...
        gmm_pdf( P_all , mu_all  , sigma_all  , channel_time_series   ,  channel_num_states ,  dimension_numbers_index , num_gmm_component  );
end


mat_mult = zeros( state_numbers_index(end) , C  );
cuopling_repmat = zeros( state_numbers_index(end) , C  );

mat_mult_one = mat_mult;
vec_mat_mult = [];

if flag_sed

    cuopling_repmat_beta = zeros( state_numbers_index(end) , C  );
    coupling_transmat = zeros( state_numbers_index(end) , state_numbers_index(end)  );
    diag_zero = ones( state_numbers_index(end) , state_numbers_index(end)  );

    % the SED backward message needs the pairwise conditionals; the fast variant
    % does not, so these sum(M)^2 x T arrays are only allocated when needed
    P_vz_tm1_vw_t_O = zeros(state_numbers_index(end) , state_numbers_index(end)  , T );
    P_vz_tm1_vw_t_O_cond = zeros(state_numbers_index(end) , state_numbers_index(end)  , T );

elseif flag_poise

    weights_subsystems_poise = zeros( state_numbers_index(end) , C  );
    theta_out_poise = zeros( state_numbers_index(end) , 1 );

else

    weights_subsystems_svi = zeros( state_numbers_index(end) , C  );
    cuopling_repmat_pow = zeros( state_numbers_index(end) , C  );

end


for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    % pi0_reg <= 0 leaves pi_0 exactly as given
    if fb_constants.pi0_reg > 0
        pi_0_lsim(zee_index) = (fb_constants.pi0_reg+pi_0_lsim(zee_index))/sum(fb_constants.pi0_reg+pi_0_lsim(zee_index));
    end

    vec_mat_mult = cat(1,vec_mat_mult, ((zee-1)*state_numbers_index(end) + ( state_numbers_index(zee)+1:state_numbers_index(zee+1) ) )' );

    zee_coupling = coupling_tetha_IM(:,zee)';
    cuopling_repmat( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

    if flag_sed

        diag_zero( zee_index ,  zee_index )= 0 ;

        zee_coupling = coupling_tetha_IM(zee,:);
        cuopling_repmat_beta( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

    elseif flag_poise

        theta_out_channel = sum(coupling_tetha_IM(zee,:));
        theta_out_poise(zee_index) = theta_out_channel;
        if theta_out_channel > 0
            zee_coupling = coupling_tetha_IM(zee,:)/theta_out_channel;
            weights_subsystems_poise( zee_index ,: ) = ...
                repmat( zee_coupling ,channel_num_states(zee),1);
        end

    else

        zee_coupling = coupling_tetha_IM(zee,:)/(fb_constants.svi_eps+sum(coupling_tetha_IM(zee,:)));
        weights_subsystems_svi( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

        zee_coupling = fb_constants.svi_eps+coupling_tetha_IM(zee,:);
        cuopling_repmat_pow( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

    end

end


if flag_sed

    for zee = 1:C
        zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );
        coupling_transmat( : ,  zee_index ) = repmat(cuopling_repmat_beta(:,zee) , 1 , length(zee_index) ).* transitions_matrices(: , zee_index);
    end

elseif ~flag_poise

    % inert unless coupling_tetha_IM itself contains NaN, since the denominator
    % above carries the svi_eps guard; kept to match the former em_lsim_fast kernel
    weights_subsystems_svi(isnan(weights_subsystems_svi)) = 1/C;

    w_opt_svi = sum(cuopling_repmat_pow,2);

end


mat_mult_one(vec_mat_mult)=1;

alpha( :  , 1) = pi_0_lsim(:);

P_ot_c_cond_past(: , 1) = mat_mult_one'*(alpha(: , 1).* P_observ_cond_to_state(:,1));
b_c_ot_nc( : , 1) = P_observ_cond_to_state(:,1) ./ (mat_mult_one*P_ot_c_cond_past(: , 1));

%Compute Forward variables & Scaling coefficients
for t = (2:T)

    alpha_tm1_tm1 = alpha(: , t-1).*b_c_ot_nc(:,t-1);

    mat_mult(vec_mat_mult) = alpha_tm1_tm1;

    a_alpha_b( : , : ,t-1)= transitions_matrices'*mat_mult ;%compute from t=1 to t=T-1
    a_alpha_b_coupled = a_alpha_b(  : , : ,t-1).*cuopling_repmat;
    alpha(: , t) = sum( a_alpha_b_coupled , 2);

    P_ot_c_cond_past(: , t) = mat_mult_one'*(alpha(: , t).* P_observ_cond_to_state(:,t));

    emmbed_P_ot_c_cond_past = (mat_mult_one*P_ot_c_cond_past(: , t));
    b_c_ot_nc( : , t) = P_observ_cond_to_state(:,t) ./ emmbed_P_ot_c_cond_past;

    if flag_sed
        P_vz_tm1_vw_t_O_cond(:,:,t) =  ( coupling_transmat + (diag_zero.*repmat( alpha_tm1_tm1' , state_numbers_index(end)  , 1 )) * coupling_transmat );
        P_vz_tm1_vw_t_O(:,:,t) = P_vz_tm1_vw_t_O_cond(:,:,t).*(repmat( alpha_tm1_tm1 , 1 , state_numbers_index(end) )) ;
    end

end


beta(: , T) = b_c_ot_nc(: , T);
alpha_T(: , T) =  alpha(: , T).*beta(: , T);


if flag_sed

    flag_nan = sum(sum(coupling_tetha_IM,2)<=10^(-10))>0;

    channel_numbers_index = [0,C:C:C^2];

    %Compute Backward variables
    for t = (T-1:-1:1)

        mat_mult(vec_mat_mult) = beta(: , t+1);
        beta_C = P_vz_tm1_vw_t_O_cond(:,:,t+1)*mat_mult ;%compute from t=1 to t=T-1
        %weighted average beta

        if(C>1)

            temp = P_vz_tm1_vw_t_O(:,:,t+1);
            % alpha(:,t+1) can be exactly 0 for a state with no incoming mass
            % (sparse/left-to-right transitions). temp(:,j) is then exactly 0
            % too for that j (both are sums of the same nonnegative products),
            % so the true contribution to f is 0, not 0/0=NaN. Divide only
            % where the denominator is nonzero and leave those columns at 0.
            alpha_next_row = alpha(: , t+1)';
            denom_2d_f = repmat(alpha_next_row, state_numbers_index(end),1 );
            temp_2d_f = zeros(size(temp));
            nz_cols = denom_2d_f(1,:) ~= 0;
            temp_2d_f(:,nz_cols) = (temp(:,nz_cols).^2)./denom_2d_f(:,nz_cols);
            temp_aplha =  alpha(: , t).* b_c_ot_nc( : , t);

            h_zee_temp =  cumsum(temp_aplha.^2);
            h_zee_all = h_zee_temp(state_numbers_index(2:end)) - [0;h_zee_temp(state_numbers_index(2:end-1))] ;
            h_zee_all = repelem(h_zee_all,C,1);

            f_zee_temp = cumsum(temp_2d_f,1);
            f_zee_temp = f_zee_temp(state_numbers_index(2:end),:) - [zeros(1,state_numbers_index(end));f_zee_temp(state_numbers_index(2:end-1),:)] ;
            f_zee_temp = cumsum(f_zee_temp,2);

            f_zee_all = ( f_zee_temp(:,state_numbers_index(2:end)) - [zeros(C,1),f_zee_temp(:,state_numbers_index(2:end-1))] )';
            f_zee_all = f_zee_all(:)+fb_constants.f_reg;

            % x = ( diag(f-h) + h*ones(C,C) ) \ f , solved in closed form.
            % Equivalent to the Sherman-Morrison form of the former
            % forward_backward_lsim_sed.m (verified identical to ~1e-15).
            nume_temp = cumsum(f_zee_all./(f_zee_all-h_zee_all));
            nume_temp = nume_temp(channel_numbers_index(2:end)) - [0;nume_temp(channel_numbers_index(2:end-1))] ;
            nume_temp = h_zee_all.*repelem(nume_temp,C,1);

            denume_temp = cumsum(1./(f_zee_all-h_zee_all));
            denume_temp = denume_temp(channel_numbers_index(2:end)) - [0;denume_temp(channel_numbers_index(2:end-1))] ;
            denume_temp =1+ h_zee_all.*repelem(denume_temp,C,1);

            g_zee_all = nume_temp./denume_temp;

            d_hat = (f_zee_all-g_zee_all)./(f_zee_all-h_zee_all);

            temp_delta = reshape(d_hat ,C,C)';

            % No row normalization here: scaling a delta row by any constant
            % cancels out of beta/alpha_T further down (both are divided by
            % the per-channel normalizer a few lines below), so it has no
            % effect on the result at any beta_floor actually used.

            mat_mult(vec_mat_mult) = 1;
            weights_subsystems = mat_mult * temp_delta;

        else
            mat_mult(vec_mat_mult) = 1;
            weights_subsystems = mat_mult;
        end

        beta(: , t) = b_c_ot_nc( : , t).*( max(fb_constants.beta_floor, sum( beta_C.*weights_subsystems  , 2) ) ) ;

        if(flag_nan)
            beta(isnan(beta(: , t)) , t)=  b_c_ot_nc( isnan(beta(: , t)) , t);
        end

        alpha_T( : , t) = beta(: , t).*alpha(: , t);

        temp_sum = cumsum(alpha_T( : , t));
        normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
        normalizer_vec = mat_mult_one*normalizer(:);

        beta( : , t) = beta( : , t)./normalizer_vec;
        alpha_T( : , t) = alpha_T( : , t)./normalizer_vec;

    end

elseif flag_poise

    % Raw POISE backward recursion: power outside the outgoing mixture.
    for t = (T-1:-1:1)

        mat_mult(vec_mat_mult) = beta(: , t+1);
        beta_C_poise = transitions_matrices * mat_mult;
        q_poise = sum(beta_C_poise.*weights_subsystems_poise, 2);

        beta(:,t) = b_c_ot_nc(:,t).*(q_poise.^theta_out_poise);
        zero_out = theta_out_poise == 0;
        beta(zero_out,t) = b_c_ot_nc(zero_out,t);

        alpha_T(:,t) = beta(:,t).*alpha(:,t);
        temp_sum = cumsum(alpha_T(:,t));
        normalizer = temp_sum(state_numbers_index(2:end)) - ...
            [0;temp_sum(state_numbers_index(2:end-1))];
        % Same guard as forward_backward_lsim_poise.m's normalize_blocks: a
        % nonpositive or nonfinite channel normalizer is an error, not something
        % to divide by. Without it this branch returned NaN alpha_T/beta for a
        % model whose forward likelihood is perfectly finite, while the
        % standalone POISE entry point errored on the very same input.
        local_check_poise_normalizer(normalizer, t, 'backward recursion');
        normalizer_vec = mat_mult_one*normalizer(:);
        beta(:,t) = beta(:,t)./normalizer_vec;
        alpha_T(:,t) = alpha_T(:,t)./normalizer_vec;

    end

    % The refinement is one-shot: every propagated marginal is formed from
    % the frozen raw POISE marginals, never from an already refined slice.
    if fb_constants.poise_refinement && T > 1
        alpha_T_raw = alpha_T;
        w1_poise = max(0, 0.75-theta_out_poise);

        for t = 2:T
            mat_mult(vec_mat_mult) = alpha_T_raw(:,t-1);
            propagated_C = transitions_matrices'*mat_mult;
            propagated = b_c_ot_nc(:,t).*sum(propagated_C.*cuopling_repmat,2);

            temp_sum = cumsum(propagated);
            normalizer = temp_sum(state_numbers_index(2:end)) - ...
                [0;temp_sum(state_numbers_index(2:end-1))];
            local_check_poise_normalizer(normalizer, t, 'refinement');
            propagated = propagated./(mat_mult_one*normalizer(:));

            alpha_T(:,t) = w1_poise.*propagated + ...
                (1-w1_poise).*alpha_T_raw(:,t);
        end

        % EM's transition statistic uses beta. Reconstruct the effective
        % message so its one-slice marginal is exactly the refined alpha_T.
        beta(:) = 0;
        positive_alpha = alpha > 0;
        beta(positive_alpha) = alpha_T(positive_alpha)./alpha(positive_alpha);
    end

else

    %Compute Backward variables
    for t = (T-1:-1:1)

        mat_mult(vec_mat_mult) = beta(: , t+1);
        beta_C_svi = transitions_matrices *mat_mult ;%compute from t=1 to t=T-1

        beta(: , t) = b_c_ot_nc( : , t).*(  sum( beta_C_svi.*weights_subsystems_svi , 2) ).^w_opt_svi;

        beta(isnan(beta(: , t)) , t)=  b_c_ot_nc( isnan(beta(: , t)) , t);

        % note the operation order differs from the SED branch above: alpha_T is
        % recomputed from the already normalized beta rather than divided by the
        % normalizer. Mathematically the same, kept as-is for bit exactness.
        alpha_T_temp = beta(: , t).*alpha(: , t);
        temp_sum = cumsum(alpha_T_temp);
        normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
        normalizer_vec = mat_mult_one*normalizer(:);
        beta( : , t) = beta( : , t)./normalizer_vec;

        alpha_T( : , t) = beta(: , t).*alpha(: , t);

    end

end


P_O_model = sum( log(P_ot_c_cond_past(:)) );

if(isnan(P_O_model))
    error('failed to compute log likelyhood')
end

end


function local_check_poise_normalizer(normalizer, t, stage)

% The POISE branch's port of forward_backward_lsim_poise.m's normalize_blocks
% guard. Unconditional: a channel's total smoothed mass must be finite and
% positive, and a zero one makes every state of that channel 0/0 -- which, with
% the refinement on, then spreads to every channel and timestep.

bad = find(~isfinite(normalizer) | normalizer <= 0, 1);
if ~isempty(bad)
    error('forward_backward_lsim_kernel:poiseNormalizer', ...
        'POISE %s: channel %d normalizer at t=%d is nonpositive or nonfinite.', ...
        stage, bad, t);
end

end
