


function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim_svi( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVI forward-backward
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Y is observations which is a cell size Cx1  where Y{c,1} is the c'th subsystem observations

% C is the total number of subsystems

%Transition probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transitions_matrices is a cell array CxC which cell (i,j) contain
% parameters P(v^j_t|v^i_t-1) (similar to Rabiner) (i->j)

%each cell (i,j) contain a matrix M(i)xM(j) which shows the transition
%probabilities subsystem i on subsystem j according to Rabiner notations  A=Transitions_matrices{i,j} is transition matrix A(n_j , n_i)= P(v^j_t = n_j | v^i_t = n_i ) & sum_n_j (A(i , j)) = 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Coupling probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coupling matrice is a CxC  which  (c,zee) contain
% parameters Coupling_Tetha(c,zee)
%So sum(Coupling_Tetha)=ones(1,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Initials hidden states probabilities of CHMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PI_0{j} contain a vector with length M(j) which contain initials hidden
% states probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%emmision probabilities of CHMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHMM_GMM_Param{j}.gmm_para(n_j).P is  mixture wieght (probability) of the
% k'th GMM component related to hidden state (v^j_t = n_j)

% CHMM_GMM_Param{j}.gmm_para(n_j).mu(k).x is  mean vector of the k'th GMM component related to hidden state (v^j_t = n_j)

% CHMM_GMM_Param{j}.gmm_para(n_j).sigma(k).x is  Covariance matrix of the k'th GMM component related to hidden state (v^j_t = n_j)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initializing all parameters
C = size(Y_in , 1);

dim_observation = zeros(C , 1);
state_numbers = zeros(C , 1);
num_gmm_component =  zeros(C , 1);

for zee = 1:C

    dim_observation(zee,1)  = size( Y_in{zee,1} ,1);
    state_numbers(zee,1)  = size( pi_0_lsim{zee,1} ,1);
    num_gmm_component(zee,1)  = length(gmm_para_lsim{zee,1}.gmm_para(1).P);

end

state_numbers_index =  ( [0;cumsum(state_numbers(:))] );
dimension_numbers_index =  ( [0;cumsum(dim_observation(:))] );

% MU_all is N*C x Dimension
P_all = zeros( C , max(state_numbers) , max(num_gmm_component)  );
mu_all = zeros(  sum(dim_observation)  , max(state_numbers) , max(num_gmm_component)   );
sigma_all =  ones(  sum(dim_observation)  , max(state_numbers) , max(num_gmm_component) );

transitions_matrices = ones( sum(state_numbers) , sum(state_numbers)  );
pi_0 = zeros( sum(state_numbers) , 1  );

for zee = 1:C

    gmm_para = gmm_para_lsim{zee,1}.gmm_para ;

    for i=1:state_numbers(zee)

        for k=1:num_gmm_component(zee)

            P_all(zee,i,k) =  gmm_para(i).P(k,1);
            sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) = gmm_para(i).sigma(k).x ;
            mu_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) =  gmm_para(i).mu(k).x;

        end

    end

    temp_PI_0 = 0.001+pi_0_lsim{zee,1};
    temp_PI_0 = temp_PI_0/sum(temp_PI_0);
    % initial probabilities
    pi_0(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1) = temp_PI_0;

    for c=1:C
        % Transition probabilities  M(c) x M(zee)
        transitions_matrices( state_numbers_index(c)+1:state_numbers_index(c+1) , state_numbers_index(zee)+1:state_numbers_index(zee+1) ) = transition_matrices_IM{c,zee}  ;
    end

end




%Computing Alpha^c_(t|t-1) for all subsystems

channel_time_series = zeros(sum(dim_observation) , size(Y_in{1,1},2));
%Y is PxT, P is observation dimension & T is observations length
for zee=1:C
    channel_time_series(dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1),:)= Y_in{zee,1};
end

T =  ( size( channel_time_series , 2) );

P_ot_c_cond_past = zeros( C , T  );


alpha = zeros( state_numbers_index(end) , T  );
alpha_T  = zeros( state_numbers_index(end) , T  );
beta  = zeros( state_numbers_index(end) , T );

b_c_ot_nc  = zeros( state_numbers_index(end) , T );
a_alpha_b = zeros(state_numbers_index(end) , C , T-1 );


[P_observ_cond_to_state  ,   P_observ_cond_to_state_comp ]  = ...
    gmm_pdf_fast( P_all , mu_all  , sigma_all  , channel_time_series   ,  state_numbers ,  dimension_numbers_index , num_gmm_component  );


% zee_index{zee,1} = cell{C,1};
mat_mult = zeros( state_numbers_index(end) , C  );
cuopling_repmat = zeros( state_numbers_index(end) , C  );
cuopling_repmat_zero = zeros( state_numbers_index(end) , C  );
cuopling_repmat_beta = zeros( state_numbers_index(end) , C  );
cuopling_repmat_wopt = zeros( state_numbers_index(end) , 1  );

mat_mult_one = mat_mult;
vec_mat_mult = [];
digonal_transmat = zeros( state_numbers_index(end) , state_numbers_index(end)  );

for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    vec_mat_mult = cat(1,vec_mat_mult, ((zee-1)*state_numbers_index(end) + ( state_numbers_index(zee)+1:state_numbers_index(zee+1) ) )' );

    zee_coupling = coupling_tetha_IM(:,zee)';
    cuopling_repmat( zee_index ,: ) = repmat( zee_coupling ,state_numbers(zee),1);

    zee_coupling(zee)=0;
    cuopling_repmat_zero( zee_index ,: ) = repmat(zee_coupling ,state_numbers(zee),1);
    digonal_transmat( zee_index ,  zee_index ) = coupling_tetha_IM( zee , zee)* transitions_matrices(zee_index , zee_index);

    %     zee_coupling = 10^-8+coupling_tetha_convex_comb(zee,:);
    zee_coupling = coupling_tetha_IM(zee,:)/(10^-8+sum(coupling_tetha_IM(zee,:)));
    cuopling_repmat_beta( zee_index ,: ) = repmat( zee_coupling ,state_numbers(zee),1);
    cuopling_repmat_wopt( zee_index) = sum(coupling_tetha_IM(zee,:));

end


mat_mult_one(vec_mat_mult)=1;

alpha( :  , 1) = pi_0(:);
P_ot_c_cond_past(: , 1) = mat_mult_one'*(alpha(: , 1).* P_observ_cond_to_state(:,1));
b_c_ot_nc( : , 1) = P_observ_cond_to_state(:,1) ./ (mat_mult_one*P_ot_c_cond_past(: , 1));

%Compute Forward variables & Scaling coefficients
for t = (2:T)

    mat_mult(vec_mat_mult)=(alpha(: , t-1).*b_c_ot_nc(:,t-1));

    a_alpha_b( : , : ,t-1)= transitions_matrices'*mat_mult ;%compute from t=1 to t=T-1
    alpha(: , t) = sum( a_alpha_b(  : , : ,t-1).*cuopling_repmat , 2);

    P_ot_c_cond_past(: , t) = mat_mult_one'*(alpha(: , t).* P_observ_cond_to_state(:,t));
    b_c_ot_nc( : , t) = P_observ_cond_to_state(:,t) ./ (mat_mult_one*P_ot_c_cond_past(: , t));


end



beta(: , T) = b_c_ot_nc(: , T);
alpha_T(: , T) =  alpha(: , T).*beta(: , T);


%Compute Backward variables
for t = (T-1:-1:1)

    mat_mult(vec_mat_mult) = beta(: , t+1);

    beta_C = transitions_matrices *mat_mult ;%compute from t=1 to t=T-1

    temp_beta = b_c_ot_nc( : , t).*( sum( beta_C.*cuopling_repmat_beta , 2) );
    temp_beta(isnan(temp_beta))=  b_c_ot_nc( isnan(temp_beta) , t);

    beta(: , t) = temp_beta;
    alpha_T( : , t) = beta(: , t).*alpha(: , t);

    temp_sum = cumsum(alpha_T( : , t));
    normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
    normalizer_vec = mat_mult_one*normalizer(:);

    beta( : , t) = beta( : , t)./normalizer_vec;
    alpha_T( : , t) = alpha_T( : , t)./normalizer_vec;

end

% % % % flag_nan = sum(sum(coupling_tetha_IM,2)<=10^(-10))>0;
% % % % 
% % % % %Compute Backward variables
% % % % for t = (T-1:-1:1)
% % % % 
% % % %     mat_mult(vec_mat_mult) = beta(: , t+1);
% % % %     beta_C = transitions_matrices *mat_mult ;%compute from t=1 to t=T-1
% % % % 
% % % %     %weighted average beta
% % % %     beta(: , t) = b_c_ot_nc( : , t).*(  sum( beta_C.*cuopling_repmat_beta , 2) ) ;
% % % % 
% % % %     if(flag_nan)
% % % %         beta(isnan(beta(: , t)) , t)=  b_c_ot_nc( isnan(beta(: , t)) , t);
% % % %     end
% % % % 
% % % %     w_opt = cuopling_repmat_wopt;
% % % %     w_opt2 = max(0,C/2-w_opt);
% % % % 
% % % %     beta(: , t) = (w_opt2.*b_c_ot_nc( : , t) + w_opt.*beta(: , t));
% % % %     alpha_T( : , t) = beta(: , t).*alpha(: , t);
% % % % 
% % % %     temp_sum = cumsum(alpha_T( : , t));
% % % %     normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
% % % %     normalizer_vec = mat_mult_one*normalizer(:);
% % % % 
% % % %     beta( : , t) = beta( : , t)./normalizer_vec;
% % % %     alpha_T( : , t) = alpha_T( : , t)./normalizer_vec;
% % % % 
% % % % end


P_O_model = sum( log(P_ot_c_cond_past(:)) );



if(nargout>1)

    alpha_out = cell(C,1);
    beta_out = cell(C,1);
    alpha_T_out = cell(C,1);
    b_c_ot_nc_out = cell(C,1);
    P_observ_cond_to_state_out = cell(C,1);
    P_observ_cond_to_state_comp_out = cell(C,1);



    for zee = 1:C

        alpha_out{zee,1} =  alpha(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);
        beta_out{zee,1} =  beta(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);

        alpha_out{zee,1} =  alpha(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);
        beta_out{zee,1} =  beta(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);

        alpha_T_out{zee,1} =  alpha_T(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);
        b_c_ot_nc_out{zee,1} =  b_c_ot_nc(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);

        P_observ_cond_to_state_out{zee,1} =  P_observ_cond_to_state(state_numbers_index(zee)+1:state_numbers_index(zee+1) , :);
        P_observ_cond_to_state_comp_out{zee,1} = P_observ_cond_to_state_comp(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1:num_gmm_component(zee) , :);

    end


end

end



