% This script provides the small working example. We model the situation
% where p represents the number of assumed regions in brain for both
% hemispheres (that is why we assume that p is even) and variables for
% homologous regions are correlated (th ecorrelation is regulated by
% hom_corr). 

%% Paths
addpath('D:\users\dbrzyski\Dropbox\LogisticPEER\software\Matlab') %path to the function griPEER

%% Clear workspace
clear

%% Setting seed
rng(1111);

%% SETTINGS
n         = 250;
p         = 60; % assumed to be even in this example
hom_corr  = 0.5; %correlations between homologous variables
lamQtr    = 0.1;   % lambda_Q true 
lamRtr    = 0.001; % lambda_R true 
truInt    = 1;
nIter     = 10;

%% GENERATE Z, X, A and Q 
Zsigma             = [ [eye(p/2, p/2), hom_corr*eye(p/2, p/2)];[hom_corr*eye(p/2, p/2), eye(p/2, p/2)] ];
Zmu                = zeros(p,1); 
Z                  = mvnrnd(Zmu, Zsigma, n);
Z                  = zscore(Z);  
Z                  = Z/sqrt(n-1);  % columns of Z are centered to have mean 0 and scaled to have standard deviation 1
X                  = ones(n,1);  % only intercept in X
A1                 = rand(p/2, p/2);
A1(A1<0.3)         = 0;
A1(A1>0.6)         = 0;
A1                 = triu(A1) + (triu(A1))';
A2                 = rand(p/2, p/2);           
A2(A2<0.3)         = 0;
A2(A2>0.6)         = 0;
A2                 = triu(A2) + (triu(A2))';
A                  = [ [A1, 0.5*eye(p/2,p/2)]; [0.5*eye(p/2,p/2), A2] ];
A                  = A - diag(diag(A));  % the adjancency matrix which imitates the density of connections in human's brain
D                  = diag(sum(A,1));
Q                  = (D^(-0.5))*(D - A)*(D^(-0.5)); %normalized Laplacian of A
Qtilde             = lamQtr*Q + lamRtr*eye(p);
SIGMA              = Qtilde^(-1);

%% GENERATE true b and observations
b_true    =  (mvnrnd(zeros(p,1), SIGMA))';
theta     =  X*truInt + Z*b_true;
BernPr    =  exp(theta)./( 1+exp(theta) );
y         =  binornd(  1, BernPr  );

%% Estimates
griPEER_fit             =   griPEER(y, X, Z, Q);
glm_fit                 =   glmfit( Z, y,'binomial','link','logit');
LogisticRidge_fit       =   griPEER(y, X, Z, eye(p));

griPEER_estimate        =   griPEER_fit.b;
LogisticRidge_estimate  =   LogisticRidge_fit.b;
glm_estimate            =   glm_fit(2:end);


%% Errors
griPEER_error          = (   norm(b_true - griPEER_estimate)/norm(b_true)   )^2;
LogisticRidge_error    = (   norm(b_true - LogisticRidge_estimate)/norm(b_true)   )^2;
glm_error              = (   norm(b_true - glm_estimate)/norm(b_true)   )^2;

%% Display errors
fprintf(['Relative squared error (one iteration) for griPEER is          ', num2str(griPEER_error), '\n']);
fprintf(['Relative squared error (one iteration) for logistic ridge is   ', num2str(LogisticRidge_error), '\n']);
fprintf(['Relative squared error (one iteration) for glm is              ', num2str(glm_error), '\n']);


%% Plots
plot([b_true, griPEER_estimate, LogisticRidge_estimate])
legend('true signal', 'griPEER estimate', 'logRidge estimate', 'Location','southeast')







