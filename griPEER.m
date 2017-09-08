function  out =  griPEER(y, X, Z, Q, varargin)

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
% This function solves the griPEER optimization problem of the form:
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% argmin_{Beta, b}   { -2loglik(X*Beta + Z*b) + 
%           tau||Beta(2:end)||^2 + lambda1*b'*Q*b + lambda2||b||^2 +  }   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% where:
% loglik is the loglikelihood function of member of one-parameter 
% exponential family of distributions
%-------------------------------------------
% y is n-dimensional vector of observations
%-------------------------------------------
% X is n by m design matrix of confounding variables (with ones in first
% colum corresponding to the intercept)
%-------------------------------------------
% Beta is m-dimensional vector of interest
%-------------------------------------------
% Z is n by p design matrix of penalized variables
%-------------------------------------------
% b is p-dimensional vector of interest
%-------------------------------------------
% Q is p by b symmetric, positive-semidefiniete matrix, which imposes the
% penalty on coefficients in vector b. If Q is not of that form but has
% only non-negative entries and zeros on diagonal, then Q is labeled as
% adjacency matrix and normalized Laplacian is used instead.
%-------------------------------------------
% tau is an optional, user-specified parameter providing additional penalty
% on the confounding variables (set as zero by default)
%-------------------------------------------
% lambda1 and lambda2 are positive regularization parameters which are 
% automaticly adjusted by taking the advantage of the connection of 
% considered problem with generlized linear mixed model formulation.
%-------------------------------------------

%%%%%%%%%%%%%%%%%%         OPTIONAL INPUTS           %%%%%%%%%%%%%%%%%%%%%
%-------------------------------------------
%'mode'  --- possible values 1 (Q is not extendend to m+p by m+p matrix by 
% introducing zero entries) or 2(Q is extendend to m+p by m+p matrix by 
% introducing zero entries). When mode == 2, the entire m+p by m+p matrix
% is ridgified to remove nonsingularity and variables in B (beyond first 
% column of ones) are treated as random effects when GLMM framework is
% used. By default 'mode' = 1. This parameter can not be set as 2, if tau 
% was set as a positive number;
%-------------------------------------------
%'nboot' --- number of bootstrap samples produced to calcuate confidence
% interval. By default 'nboot' = 500;
%-------------------------------------------
%'ridConst' --- constatnt by which identity matrix is multiplied and then
% added to Q, if the last matrix was recognized as singular; By default 
% 'ridConst' = 0.001;
%-------------------------------------------
%'alpha' --- statistical significance level.By default 'alpha' = 0.05
%-------------------------------------------
%'UseParallel' --- logical value indicating if parallel computatiation
%should be used when bootstrap is employed. By default 'UseParallel' = true
%-------------------------------------------
% 'IncludeRidge' --- logical value indicating if logical Ridge estimate and
% inference should be also derived
%-------------------------------------------
%'FitMethod', {'Laplace','MPL','REMPL', 'ApproximateLaplace'}
%'Optimizer', {'quasinewton','fminsearch','fminunc'}
%'bootType',  {'norm', 'per', 'cper', 'bca', 'stud'}
%'ciType',    {'both', 'asymptotic'}

%%%%%%%%%%%%%%%%%%%%%%%%         OUTPUTS           %%%%%%%%%%%%%%%%%%%%%%%%
% out.estim        
% out.ciAsmp     
% out.estimBeta 
% out.ciAsmpBeta   
% out.statSigAsmp
% out.statSigAsmpBeta 
% out.lambdaPEER
% out.ciBoot 
% out.ciBootBeta 
% out.statSig     
% out.statSigBeta 
% %----------------------
% out.estimR          
% out.ciAsmpR          
% out.estimBetaR       
% out.ciAsmpBetaR      
% out.statSigAsmpR     
% out.statSigAsmpBetaR 
% out.lambdaRidge 
% out.ciBootR      
% out.ciBootBetaR 
% out.statSigR    
% out.statSigBetaR 

%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------

%% Checks
%=============================================
if any(abs(mean(Z,1)>1e-10))
   error('Columns in design matrix Z should be centered to 0 means') 
end

%=============================================
%---------------------------------------------
%=============================================

if size(X,2) ==0
    error('It should be at least one column in design matrix X')
end

%=============================================
%---------------------------------------------
%=============================================

if any(abs(mean(X(:,2:end),1) > 1e-10))
    error('Besides first column (corresponding to the intercept), all columns in design matrix X should be centered to 0 means') 
end

%=============================================
%---------------------------------------------
%=============================================

if max(max(abs(bsxfun( @minus, sqrt(sum([X(:,2:end), Z].^2,1)), sqrt(sum([X(:,2:end), Z].^2,1))'))))>1e-10
    error('Columns in X (beyond the first column) and columns in Z should have the same Euclidean norms')
end

%=============================================
%---------------------------------------------
%=============================================

if length(unique(X(:,1))) == 1 && X(1,1) ==1
else
    error('First column of design matrix X should represent the intercept and consist of ones')
end

%=============================================
%---------------------------------------------
%=============================================

if all(abs(diag(Q)) == 0) && isempty(find(Q<0, 1)) % program detected that Q is not Lap but Adj
    disp('Provided prior information was recognized as adjacency matrix, Laplacian will be used instead.')
    D      = diag(sum(Q,1));
    Q      = (D^(-.5))*(D - Q)*(D^(-.5));
end

%=============================================
%---------------------------------------------
%=============================================
warn_id   = 'stats:classreg:regr:lmeutils:StandardGeneralizedLinearMixedModel:BadFinalPLSolution';
warning('off',warn_id)

%% Additional parameters (AP)
AP               = inputParser;
defaultTau       = 0;
defaultMode      = 1;
defaultNboot     = 500;
defaultAlpha     = 0.05;
defaultParallel  = true;
defaultType      = 'per';
defaultCItype    = 'asymptotic';
defaultFamily    = 'binomial';
defaultLambdaR   = 'auto';
defaultInLambs   = [0.0001; 0.00001];
defaultGridd     = [0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001;   0.001, 0.0001, 0.00001, 0.001, 0.0001, 0.00001, 0.001, 0.0001];
defaultMaxIter   = 30;
defaultStopCrit  = 1e-6;

%------------------------------------

addRequired(AP, 'y');
addRequired(AP, 'X');
addRequired(AP, 'Z');
addRequired(AP, 'Q');
addOptional(AP, 'mode', defaultMode, @(x)ismember(x,[1,2]));
addOptional(AP, 'tau', defaultTau, @(x)(x>=0) );
addOptional(AP, 'nboot', defaultNboot, @isnumeric);
addOptional(AP, 'alpha', defaultAlpha, @isnumeric);
addOptional(AP, 'UseParallel', defaultParallel, @islogical);
addOptional(AP, 'bootType', defaultType, @(x) any(validatestring(x,{'norm', 'per', 'cper', 'bca', 'stud'} ) ) );
addOptional(AP, 'ciType', defaultCItype, @(x) any(validatestring(x,{'both', 'asymptotic'} ) ) );
addOptional(AP, 'family', defaultFamily, @(x) any(validatestring(x,{'binomial', 'poisson'} ) ) );
addOptional(AP, 'lambdaR', defaultLambdaR, @(x) (x>=0) );
addOptional(AP, 'initLambs', defaultInLambs, @(x) all(x>0) );
addOptional(AP, 'lambsGrid', defaultGridd, @(x) all(all(x>0)) );
addOptional(AP, 'maxIter', defaultMaxIter, @(x) (x>0) );
addOptional(AP, 'stopCrit', defaultStopCrit, @(x) (x>0) );


%-------------------------------------
 parse(AP, y, X, Z, Q, varargin{:}) %-
%-------------------------------------

mode        = AP.Results.mode;
tau         = AP.Results.tau;
nboot       = AP.Results.nboot;
alpha       = AP.Results.alpha; 
type        = AP.Results.bootType;
ciType      = AP.Results.ciType;
family      = AP.Results.family;
lambdaR     = AP.Results.lambdaR;
intitLambs  = AP.Results.initLambs;
lambsGrid   = AP.Results.lambsGrid;
maxIter     = AP.Results.maxIter;
stopCrit    = AP.Results.stopCrit;
options     = statset('UseParallel', AP.Results.UseParallel);  

%% More checks
%=============================================
if and(strcmp(lambdaR, 'auto'), ~isequal(size(intitLambs),[2,1])   )
    error('intitLambs should be a vector of length 2, since lambdaQ and lambdaR are chosen to be automatically adjusted.')
end
%=============================================
%---------------------------------------------
%=============================================
if and(  and(~strcmp(lambdaR, 'auto'), length(intitLambs)>1),    ~isequal(defaultInLambs, intitLambs)  )
    warning('only first entry of intitLambs was used,  since lambdaR was provided.')
end
%=============================================
%---------------------------------------------
%=============================================
if and(strcmp(lambdaR, 'auto'), size(lambsGrid,1)<2)
    error('It should be two rows in lambsGrid matrix, if lambdaQ and lambdaR are automatically adjusted.')
end
%=============================================
%--------------------------------------------- 
%=============================================
if strcmp(lambdaR, 'auto')
    gridd    = lambsGrid(1:2,:);
else
    gridd    = unique(lambsGrid(1,:));
end
%=============================================
%---------------------------------------------
%=============================================
if and(and(size(lambsGrid,1)>1, ~strcmp(lambdaR, 'auto')),  ~isequal(defaultGridd, lambsGrid) )        
    warning('Only one row of lambsGrid was taken, since lambdaR was provided.')
end
%=============================================
%---------------------------------------------
%=============================================
if and(size(lambsGrid,1)>2, strcmp(lambdaR, 'auto'))               
    warning('Only two first rows of lambsGrid were taken (first gives grid for lambdaQ and second gives grid for lambdaR)')
end
%=============================================

%-----------------------------------------------------------------------------------------------------------
%-----------------------------------------------------------------------------------------------------------
%-----------------------------------------------------------------------------------------------------------

%% Objects
Xr            =  X(:, 2:end);
[n,p]         =  size(Z);
m             =  size(X,2);                               % this value might be changed if mode==2
im            =  m;                                       % initial m, will not be changed
ip            =  p;                                       % initial p, will not be changed
colsNorm      =  mean(sqrt(sum([X(:,2:end), Z].^2,1)));   % columns must have identical Euclidean norm, colsNorm tracks this value
max_lam_iter  =  maxIter;

%------------ Options for optimization software ------------------
opts                          = optimoptions('fsolve','Display','off');
opts.Algorithm                = 'trust-region';
opts.FunctionTolerance        = 1e-10;
opts.OptimalityTolerance      = 1e-10;
opts.SpecifyObjectiveGradient = true;
 

%opts = optimset(optimset('fsolve'), 'TolFun', 1.0e-12, 'TolX',1.0e-12, 'UseParallel', usePar);

%% Checks for the situation with lambdaR defined by the user as 0 
%-----------------------------------------------
if and(lambdaR==0, cond(Q) > 1e8)
   error('When lambdaR is fixed as zero, conditional number of Q is assumed to be not larger than 1e8. Consider the modiffication of Q by adding the identity matrix with small multiplicative constant.')
end


%% Defining the way of treating fixed effects
% if mode==1 and tau ==0, all variables in X are treated as fixed effects, 
% if mode ==2, only intercept is treated as fixed effect, Q is extendend to
% (p+m-1)x(p+m-1) matrix by zeros and entire such matrix is ridgified

if mode == 2 && tau > 0
    error('mode must be set as 1 (default), if tau>0')
end

if im == 1 && tau > 0
    error('tau>0 is not used, when matrix X contains only one column of ones corresponding to intercept')
end

if mode == 2 && size(X,2)>1 && tau == 0   
    Z  =  [X(:,2:end), Z];
    X  =  X(:,1);
    Q  =  [zeros(m-1, m+p-1);[zeros(p, m-1), Q]];
    p  =  size(Z,2);
    m  =  size(X,2);
end

%% Family of distrubution
if strcmp(family, 'binomial')
    psiPrim                =  @(x) exp(x)./(1 + exp(x));
    psiBis                 =  @(x) exp(x)./( (1+exp(x)).^2);
    glm_chosen_family      =  @(y, X) glm_logistic(y, X, 'nointercept');
else
    psiPrim                =  @(x) exp(x);
    psiBis                 =  @(x) exp(x);
    glm_chosen_family      =  @(y, X) glm_poisson(y, X, 'nointercept');
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ______________________________________________________
%|                                                      |
%|---------  Condition: tau == 0 versus tau >0  --------|
%|______________________________________________________|
%

lamQ_prev  = intitLambs(1);
if strcmp(lambdaR, 'auto')
    lamR_prev  = intitLambs(2);
else
    lamR_prev  = lambdaR;
end
stopp      = 0;
countt     = 1;
Lambdas    = zeros(2, max_lam_iter);
bs         = zeros(ip, max_lam_iter);
Gradients  = zeros(2, max_lam_iter);
Hessians   = zeros(4, max_lam_iter);
Ispos      = zeros(1, max_lam_iter);

%--------- CONDITION: tau == 0
if tau == 0   
    %-------------------------------------------------  
    % WHILE: START
    while  stopp==0
    Qt_prev       =  lamQ_prev*Q + lamR_prev*eye(p);
    L             =  chol(Qt_prev);
    ZL            =  Z* (L^(-1));  
    model         =  glm_chosen_family(y, [X, ZL]);
    fit           =  penalized(model, @p_ridge, 'penaltywt', [zeros(m,1); ones(p,1)], 'lambda', 1/n, 'standardize', false);   
    Beta          =  fit.beta(1:m); 
    b             =  L^(-1)* (fit.beta((m+1):end));
    theta         =  X*Beta + Z*b;
    W_k           =  diag( psiBis(theta).^(-1) ); 
    W_k_inv       =  diag( psiBis(theta) );
    y_k           =  W_k*(y - psiPrim(theta) ) + theta;
    P             =  eye(n) - X*(X'*W_k_inv*X)^(-1)*X'*W_k_inv;
    PZ            =  P*Z;   
    yt_k          =  P*y_k;   
    Omega_k       =  PZ'* W_k_inv *PZ;
    q_k           =  PZ'* W_k_inv *yt_k;
    
    % finding the update of lambda
    %-------------------------------------------------
    %_________________________________________________
    fun           =  @(x)log(  det( (abs(x(1))*Q + abs(x(2))*eye(p) + Omega_k)*(abs(x(1))*Q + abs(x(2))*eye(p))^(-1) )  ) - q_k'*( abs(x(1))*Q + abs(x(2))*eye(p) + Omega_k )^(-1)*q_k;
    val_prop      =  zeros(1,size(gridd, 2));
    lambda_prop   =  zeros(2,size(gridd, 2));
    
    % looking at the border
    gridd2        =  unique(gridd(1,:));
    val_prop2     =  zeros(1,size(gridd2, 2));
    lambda_prop2  =  zeros(2,size(gridd2, 2));
    
    % looking over the grid of starting points
    if strcmp(lambdaR, 'auto')      
        parfor gg = 1:size(gridd, 2)            
            lambda_prop(:, gg)  =  abs(fsolve(@(x) fbnd_o(Q, Omega_k, q_k, p, x), gridd(:,gg), opts));
            val_prop(gg)        =  fun(lambda_prop(:, gg));
        end
        parfor gg = 1:length(gridd2)            
            lambda_prop2(:, gg)  =  [0; abs(fsolve(@(x) fbnd_zero_lambQ(Omega_k, q_k, p, x), gridd2(:,gg), opts))];
            val_prop2(gg)        =  fun(lambda_prop2(:, gg));
        end
        lambda_prop3       =  [lambda_prop, lambda_prop2];
        val_prop3          =  [val_prop, val_prop2];
    else
        parfor gg = 1:size(gridd, 2)
            lambda_prop(:, gg)  =  [abs(fsolve(@(x) fbnd_o_fix_lambR(Q, Omega_k, q_k, p, lamR_prev, x), gridd(:,gg), opts)); lamR_prev];
            val_prop(gg)        =  fun(lambda_prop(:, gg));
        end
        lambda_prop3       =  lambda_prop;
        val_prop3          =  val_prop;
    end
    [~, minIdx]        =  min(val_prop3);
    lambda_k           =  lambda_prop3(:, minIdx);
    Lambdas(:, countt) =  lambda_k;
    estim              =  [Beta; b];
    bs(:, countt)      =  estim((im+1):end);
    %------------------
    if strcmp(lambdaR, 'auto')
        [F, H] = fbnd_o(Q, Omega_k, q_k, p, lambda_k);
    else
        [F, H] = fbnd_o_fix_lambR(Q, Omega_k, q_k, p, lamR_prev, lambda_k(1));
    end
    Gradients(:, countt) =  F;
    Hessians(:, countt)  =  H(:);
    %------------------
    if and( H(1,1) >= 0,  det(H)>=0 )
        Ispos(countt) = 1;
    end    
    %------------------
    
    %_________________________________________________
    
    %-------------------------------------------------
    
    % checking stop conditions
    lambda_prev = [lamQ_prev; lamR_prev];
    if (norm(lambda_k - lambda_prev)/norm(lambda_prev)) < stopCrit || countt >= max_lam_iter
        stopp = 1;
    else
        countt = countt + 1;
    end
    %-------------------------------------------------
    lamQ_prev     =  lambda_k(1);
    lamR_prev     =  lambda_k(2);
    end
    
    % final estimate
    lamQ_final    =  lamQ_prev;
    lamR_final    =  lamR_prev;
    Qt_final      =  lamQ_final*Q + lamR_final*eye(p);
    L_final       =  chol(Qt_final);
    L_final_inv   =  L_final^(-1);
    ZL            =  Z*L_final_inv;
    model         =  glm_chosen_family(y, [X, ZL]);
    fit           =  penalized(model, @p_ridge, 'penaltywt', [zeros(m,1); ones(p,1)], 'lambda', 1/n, 'standardize', false);   
    Estimate      =  [fit.beta( 1:m ); L_final_inv * (fit.beta( (m+1):end) )];
    Beta_est      =  Estimate( 1:im ); 
    b_est         =  Estimate( (im+1):end );
    
    % confidence interval
    theta_final   =  [X,Z]*Estimate;
    Psi           =  diag(psiBis(theta_final));
    XZ            =  [X, Z];
    Qt_final_ext  =  blkdiag(zeros(m,m), Qt_final);
    estimVar      =  (XZ'*Psi*XZ + Qt_final_ext)^(-1)*XZ'*Psi*XZ*(XZ'*Psi*XZ + Qt_final_ext)^(-1);
    deltaa        =  norminv(1-alpha/2) * sqrt(diag(estimVar));
    estimBand     =  [ [Beta_est; b_est] - deltaa, [Beta_est; b_est] + deltaa ];
    
    % bootstrap confidence interval (optionally)
    if strcmp(ciType, 'both')
        DATASET       = [y, X, Z];
        if strcmp(family, 'binomial')
            boot          = @(Dataset) boot_griPEER_binomial(L_final_inv, m, colsNorm, Dataset);
        else
            boot          = @(Dataset) boot_griPEER_poisson(L_final_inv, m, colsNorm, Dataset);
        end
        CI_boot       = ( bootci(nboot,{boot, DATASET}, 'Options', options, 'type', type, 'alpha', alpha) )'; 
    end
  
%--------- CONDITION: tau > 0
else
    %-------------------------------------------------       
    % while loop
    while  stopp==0
    Qt_prev       =  [[tau*eye(m-1), zeros(m-1,p)]; [zeros(p, m-1), lamQ_prev*Q + lamR_prev*eye(p)]];
    L             =  chol(Qt_prev);
    XZL           =  [Xr, Z]*( L^(-1) );
    model         =  glm_chosen_family(y, [ones(n,1), XZL]);
    fit           =  penalized(model, @p_ridge, 'penaltywt', [zeros(1,1); ones(p+m-1,1)], 'lambda', 1/n, 'standardize', false);
    int           =  fit.beta(1); 
    Beta_b        =  L^(-1)* (fit.beta(2:end));
    theta         =  int + [Xr, Z]*Beta_b;
    W_k           =  ( diag(psiBis(theta)) )^(-1);  
    W_k_inv       =  W_k^(-1);
    y_k           =  W_k*(y - psiPrim(theta)) + theta;
    P             =  eye(n) - ones(n,1) * (ones(n,1)'*W_k_inv*ones(n,1))^(-1) * ones(n,1)'*W_k_inv;
    PZ            =  P*Z;   
    yt_k          =  P*y_k;
    Omega_k       =  PZ'*(W_k + Xr*Xr'/tau)^(-1)*PZ;
    q_k           =  PZ'*(W_k + Xr*Xr'/tau)^(-1)*yt_k;
    
    % finding the update of lambda
    %-------------------------------------------------
    %_________________________________________________

    fun          =  @(x)log(  det( (abs(x(1))*Q + abs(x(2))*eye(p) + Omega_k)*(abs(x(1))*Q + abs(x(2))*eye(p))^(-1) )  ) - q_k'*( abs(x(1))*Q + abs(x(2))*eye(p) + Omega_k )^(-1)*q_k;
    lambda_prop  =  zeros(2,size(gridd, 2));
    val_prop     =  zeros(1,size(gridd, 2));
    
    % looking at the border
    gridd2        =  unique(gridd(1,:));
    val_prop2     =  zeros(1,size(gridd2, 2));
    lambda_prop2  =  zeros(2,size(gridd2, 2));
    
    % looking over the grid of starting points
    if strcmp(lambdaR, 'auto')      
        parfor gg = 1:size(gridd, 2)
            lambda_prop(:, gg)  =  abs(fsolve(@(x) fbnd_o(Q, Omega_k, q_k, p, x), gridd(:,gg), opts));
            val_prop(gg)        =  fun(lambda_prop(:, gg));
        end
        parfor gg = 1:length(gridd2)            
            lambda_prop2(:, gg)  =  [0; abs(fsolve(@(x) fbnd_zero_lambQ(Omega_k, q_k, p, x), gridd2(:,gg), opts))];
            val_prop2(gg)        =  fun(lambda_prop2(:, gg));
        end
        lambda_prop3       =  [lambda_prop, lambda_prop2];
        val_prop3          =  [val_prop, val_prop2];
    else
        parfor gg = 1:size(gridd, 2)
            lambda_prop(:, gg)  =  [abs(fsolve(@(x) fbnd_o_fix_lambR(Q, Omega_k, q_k, p, lamR_prev, x), gridd(:,gg), opts)); lamR_prev];
            val_prop(gg)        =  fun(lambda_prop(:, gg));
        end
        lambda_prop3       =  lambda_prop;
        val_prop3          =  val_prop;    
    end
    [~, minIdx]        =  min(val_prop3);
    lambda_k           =  lambda_prop3(:, minIdx);
    Lambdas(:, countt) =  lambda_k;
    estim              =  [int; Beta_b];
    bs(:, countt)      =  estim((im+1):end);
        %------------------
    if strcmp(lambdaR, 'auto')
        [F, H] = fbnd_o(Q, Omega_k, q_k, p, lambda_k);
    else
        [F, H] = fbnd_o_fix_lambR(Q, Omega_k, q_k, p, lamR_prev, lambda_k(1));
    end
    Gradients(:, countt) =  F;
    Hessians(:, countt)  =  H(:);
    %------------------
    if and( H(1,1) >= 0,  det(H)>=0 )
        Ispos(countt) = 1;
    end   
    %------------------
    %_________________________________________________
    
    %-------------------------------------------------
    
    % checking stop conditions
    lambda_prev = [lamQ_prev; lamR_prev];
    if (norm(lambda_k - lambda_prev)/norm(lambda_prev)) < stopCrit || countt >= max_lam_iter
        stopp = 1;
    else
        countt = countt + 1;
    end
    %-------------------------------------------------
    lamQ_prev     =  lambda_k(1);
    lamR_prev     =  lambda_k(2);
    end
    
    % final estimate
    lamQ_final    =  lamQ_prev;
    lamR_final    =  lamR_prev;
    Qt_final      =  [  [ tau*eye(m-1), zeros(m-1,p) ]; [ zeros(p, m-1), lamQ_final*Q + lamR_final*eye(p) ]  ];
    L_final       =  chol(Qt_final);
    L_final_inv   =  L_final^(-1);
    XZL           =  [Xr, Z] * L_final_inv;
    model         =  glm_chosen_family(y, [ones(n,1), XZL]);
    fit           =  penalized(model, @p_ridge, 'penaltywt', [zeros(1,1); ones(p+m-1,1)], 'lambda', 1/n, 'standardize', false);
    int_est       =  fit.beta(1); 
    Beta_ni_b_est =  L_final_inv * (fit.beta(2:end));
    Beta_est      =  [int_est; Beta_ni_b_est( 1:(m-1) )];
    b_est         =  Beta_ni_b_est( m:end );
    
    % confidence interval
    theta_final   =  [X,Z]*[int_est; Beta_ni_b_est];
    Psi           =  diag( psiBis(theta_final) );
    XZ            =  [X, Z];
    Qt_final_ext  =  blkdiag(0, Qt_final);
    estimVar      =  (XZ'*Psi*XZ + Qt_final_ext)^(-1)*XZ'*Psi*XZ*(XZ'*Psi*XZ + Qt_final_ext)^(-1);
    deltaa        =  norminv(1-alpha/2) * sqrt(diag(estimVar));
    estimBand     =  [ [Beta_est; b_est] - deltaa, [Beta_est; b_est] + deltaa ];
    
    % bootstrap confidence interval (optionally)
    if strcmp(ciType, 'both')
        DATASET       = [y, Xr, Z];
        if strcmp(family, 'binomial')
            boot          = @(Dataset) boot_griPEER_tau_binomial(L_final_inv, colsNorm, Dataset);
        else
            boot          = @(Dataset) boot_griPEER_tau_poisson(L_final_inv, colsNorm, Dataset);
        end
        CI_boot       = ( bootci(nboot,{boot, DATASET}, 'Options', options, 'type', type, 'alpha', alpha) )'; 
    end
    
    % WHILE: STOP
end 
Lambdas   =  Lambdas(:, 1:countt);
bs        =  bs(:, 1:countt);
Gradients =  Gradients(:, 1:countt);
Hessians  =  Hessians(:, 1:countt);
Ispos     =  Ispos(1:countt);
%------ warning if final estimate of MSE is not local maxima of loglik ----
if Ispos(countt)== 0
    warning('the final estimate of lambdas may not be a local maximum of loglik')
end

%%
% ______________________________________________________
%|                                                      |
%|--------------  Output of the function  --------------|
%|______________________________________________________|
%
%% OUTPUTS
% standard
out          = struct;
out.beta     = Beta_est;
out.b        = b_est;
out.grads    = Gradients; 
out.hess     = Hessians;
out.ispos    = Ispos;
out.lambs    = Lambdas;
out.bs       = bs;
out.band     = estimBand;

% additional
if strcmp(ciType, 'both')
    out.ciboot   = CI_boot;
end

% temporary
%out.var      = estimVar;
%out.theta    = theta_final;
out.gridd    = gridd;
out.omegasvd = svd(Omega_k);

% 
% h = 0.000001;
% xx = [-1;-1];
% 
% aa = fbnd_o(Q, Omega_k, q_k, p, xx + [1;0]*h)- fbnd_o(Q, Omega_k, q_k, p, xx - [1;0]*h);
% bb = fbnd_o(Q, Omega_k, q_k, p, xx + [0;1]*h)- fbnd_o(Q, Omega_k, q_k, p, xx - [0;1]*h);
% 
% aa1 = aa(1)/(2*h);
% aa2 = aa(2)/(2*h);
% bb1 = bb(2)/(2*h);
% 
% Hnum   = [aa1, aa2; aa2, bb1];
% [~, Htheor] = fbnd_o(Q, Omega_k, q_k, p, xx);
% 
% out.Htheor = Htheor;
% out.Hnum = Hnum;



%______________________________________________________________

end
% ______________________________________________________
%|                                                      |
%|----------------- SUBROUTINES ------------------------|
%|______________________________________________________|
%

%-------------------------------------------------------------

function [F, H] = fbnd_o(Q, Omega, q, p, x)
%==========================================
ax    =  abs(x);
D0    =  ( ax(1)*Q + ax(2)*eye(p) )^(-1);
D     =  ( ax(1)*Q + ax(2)*eye(p) + Omega )^(-1);
D0Q   =  D0*Q;
DQ    =  D*Q;
Dq    =  D*q;

%-------------------------------------------
F     =  zeros(2,1);
F(1)  =  sign(x(1))* ( trace(   DQ - D0Q  )   +   Dq'*Q*Dq );
F(2)  =  sign(x(2))* (  trace(   D - D0  )    +    Dq'*Dq  );

%-------------------------------------------
H1(1, 1)  =  - trace(  DQ^2 - D0Q^2  );
H1(1, 2)  =  - sign(x(1))*sign(x(2))* trace(  (D^2 - D0^2)*Q  );
H1(2, 1)  =    H1(1, 2);
H1(2, 2)  =  - trace(  D^2 - D0^2  );

H2(1, 1) =  -2*q'*DQ^2*D*q;
H2(1, 2) =  -sign(x(1))*sign(x(2))*( Dq'*DQ*Dq + Dq'*Q*D*Dq );
H2(2, 1) =   H2(1, 2);
H2(2, 2) =  -2*Dq'*D*Dq;

H(1,1)   =  H1(1, 1) + H2(1, 1);
H(2,1)   =  H1(2, 1) + H2(2, 1);
H(1,2)   =  H(2,1);
H(2,2)   =  H1(2, 2) + H2(2, 2);

%==========================================
end

%-------------------------------------------------------------

function [F, H] = fbnd_o_fix_lambR(Q, Omega, q, p, lambR, x)
%==========================================
ax    =  [abs(x); lambR];
D0    =  ( ax(1)*Q + ax(2)*eye(p) )^(-1);
D     =  ( ax(1)*Q + ax(2)*eye(p) + Omega )^(-1);
D0Q   =  D0*Q;
DQ    =  D*Q;
Dq    =  D*q;

%-------------------------------------------
F     =  sign(x(1))* ( trace(   DQ - D0Q  )   +   Dq'*Q*Dq );

%-------------------------------------------
H1  =  - trace(  DQ^2 - D0Q^2  );
H2  =  -2*q'*DQ^2*D*q;
H   =  H1 + H2;
%==========================================
end

%-------------------------------------------------------------

function [F, H] = fbnd_zero_lambQ(Omega, q, p, x)
%==========================================
ax    =  abs(x);
D0    =  (ax*eye(p) )^(-1);
D     =  (ax*eye(p) + Omega )^(-1);
Dq    =  D*q;

%-------------------------------------------
F     =  sign(x)* (  trace(   D - D0  )    +    Dq'*Dq  );

%-------------------------------------------
H1  =  -trace(  D^2 - D0^2  );
H2  =  -2*Dq'*D*Dq;
H   =  H1 + H2;

%==========================================
end

% 
% function F = fbnd_o_fix_lambR(Q, Omega, q, p, lambR, x)
% F = trace(  ( (abs(x)*Q + lambR*eye(p) + Omega)^(-1) - (abs(x)*Q + lambR*eye(p) )^(-1) )*Q  ) + q'*(abs(x)*Q + lambR*eye(p) + Omega)^(-1)*Q*(abs(x)*Q + lambR*eye(p) + Omega)^(-1)*q;
% end

%-------------------------------------------------------------

function estimate = boot_griPEER_binomial(L_final_inv, m, colsNorm, Dataset)
y              =   Dataset( :, 1 );
n              =   length(y);
X              =   Dataset( :, 2:(m+1)   );
Z              =   Dataset( :, (m+2):end );
Z              =   zscore(Z)*colsNorm/sqrt(n-1);
X(:,2:end)     =   zscore(X(:,2:end))*colsNorm/sqrt(n-1);
p              =   size(Z,2);

%------------------   optimization problem   ------------------
ZL             =  Z * L_final_inv;
model          =  glm_logistic(y, [X, ZL], 'nointercept');
fit            =  penalized(model, @p_ridge, 'penaltywt', [zeros(m,1); ones(p,1)], 'lambda', 1/n, 'standardize', false); 
estimate       =  [fit.beta( 1:m ); L_final_inv * (fit.beta( (m+1):end) )];

end

%-------------------------------------------------------------

function estimate = boot_griPEER_poisson(L_final_inv, m, colsNorm, Dataset)
y              =   Dataset( :, 1 );
n              =   length(y);
X              =   Dataset( :, 2:(m+1)   );
Z              =   Dataset( :, (m+2):end );
Z              =   zscore(Z)*colsNorm/sqrt(n-1);
X(:,2:end)     =   zscore(X(:,2:end))*colsNorm/sqrt(n-1);
p              =   size(Z,2);

%------------------   optimization problem   ------------------
ZL             =  Z * L_final_inv;
model          =  glm_poisson(y, [X, ZL], 'nointercept');
fit            =  penalized(model, @p_ridge, 'penaltywt', [zeros(m,1); ones(p,1)], 'lambda', 1/n, 'standardize', false); 
estimate       =  [fit.beta( 1:m ); L_final_inv * (fit.beta( (m+1):end) )];

end

%-------------------------------------------------------------

function estimate = boot_griPEER_tau_binomial(L_final_inv, colsNorm, Dataset)
y              =   Dataset( :, 1 );
n              =   length(y);
XZ             =   Dataset( :, 2:end );
XZ             =   zscore(XZ)*colsNorm/sqrt(n-1);
p              =   size(XZ,2);

%------------------   optimization problem   ------------------
XZL            =  XZ*L_final_inv;
model          =  glm_logistic(y, [ones(n,1), XZL], 'nointercept');
fit            =  penalized(model, @p_ridge, 'penaltywt', [zeros(1,1); ones(p,1)], 'lambda', 1/n, 'standardize', false); 
estimate       =  [ fit.beta(1); L_final_inv * (fit.beta( 2:end)) ];

end
%-------------------------------------------------------------

function estimate = boot_griPEER_tau_poisson(L_final_inv, colsNorm, Dataset)
y              =   Dataset( :, 1 );
n              =   length(y);
XZ             =   Dataset( :, 2:end );
XZ             =   zscore(XZ)*colsNorm/sqrt(n-1);
p              =   size(XZ,2);

%------------------   optimization problem   ------------------
XZL            =  XZ*L_final_inv;
model          =  glm_poisson(y, [ones(n,1), XZL], 'nointercept');
fit            =  penalized(model, @p_ridge, 'penaltywt', [zeros(1,1); ones(p,1)], 'lambda', 1/n, 'standardize', false); 
estimate       =  [ fit.beta(1); L_final_inv * (fit.beta( 2:end)) ];

end
